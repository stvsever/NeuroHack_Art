from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import re
import shutil
import subprocess
import time
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .config import (
    BROAD_QUERY,
    CACHE_DIR,
    DATA_DIR,
    DEFAULT_ENV,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    END_YEAR,
    ROOT,
    START_YEAR,
    STRIPS,
    THEMES,
    ensure_dirs,
)
from .clusters import enrich_clusters

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
OPENROUTER_EMBEDDINGS = "https://openrouter.ai/api/v1/embeddings"
ABSTRACT_BATCH_CACHE = ROOT / ".build" / "pubmed_batches"
TARGET_QUERIES = {
    "gradient": '("action potential"[Title/Abstract] OR "membrane potential"[Title/Abstract] OR "calcium wave"[Title/Abstract] OR bioelectric*[Title/Abstract] OR "excitable media"[Title/Abstract])',
    "fungal": '(myceli*[Title/Abstract] OR "fungal electrophysiology"[Title/Abstract] OR "fungal electrical"[Title/Abstract] OR "mycorrhizal communication"[Title/Abstract] OR "hyphal network"[Title/Abstract])',
    "elegans": '("Caenorhabditis elegans"[Title/Abstract] OR "C. elegans"[Title/Abstract]) AND (neuron*[Title/Abstract] OR nervous[Title/Abstract] OR behavio*[Title/Abstract])',
    "drosophila": '(drosophila[Title/Abstract] OR "fruit fly"[Title/Abstract]) AND (brain[Title/Abstract] OR neuron*[Title/Abstract] OR connectom*[Title/Abstract] OR behavio*[Title/Abstract])',
    "macaque": '(macaque[Title/Abstract] OR "Macaca mulatta"[Title/Abstract] OR "rhesus monkey"[Title/Abstract]) AND (brain[Title/Abstract] OR neural[Title/Abstract] OR cortex[Title/Abstract])',
    "ai": '(("artificial neural network"[Title/Abstract] OR "deep neural network"[Title/Abstract] OR transformer[Title/Abstract] OR neuroAI[Title/Abstract] OR neuromorphic[Title/Abstract]) AND (brain[Title/Abstract] OR neuroscien*[Title/Abstract] OR cognit*[Title/Abstract] OR neural[Title/Abstract]))',
}

THEME_RULES: dict[str, list[str]] = {
    "molecular": [r"ion channel", r"receptor", r"neurotransmit", r"synap(?:se|tic)", r"astrocy", r"microgl", r"glia", r"gene expression", r"protein", r"cellular"],
    "development": [r"develop", r"plastic", r"axon guidance", r"neurogen", r"critical period", r"regenerat", r"morphogen", r"growth cone"],
    "sensation": [r"visual", r"vision", r"auditory", r"hearing", r"olfact", r"somatosens", r"percept", r"sensory", r"retina"],
    "action": [r"motor", r"movement", r"locomot", r"action selection", r"cerebell", r"basal ganglia", r"sensorimotor", r"gait"],
    "memory": [r"memory", r"learning", r"hippocamp", r"conditioning", r"navigation", r"reinforcement", r"decision mak"],
    "cognition": [r"attention", r"executive", r"language", r"working memory", r"cognitive control", r"reasoning", r"semantic"],
    "affect": [r"emotion", r"affect", r"reward", r"motivation", r"stress", r"social", r"anxiety", r"depress"],
    "sleep": [r"sleep", r"conscious", r"awareness", r"anesthe", r"arousal", r"wakeful", r"dream"],
    "clinical": [r"alzheimer", r"parkinson", r"epilep", r"schizophren", r"stroke", r"dementia", r"multiple sclerosis", r"traumatic brain", r"neurologic(?:al)? disorder", r"psychiatric disorder"],
    "networks": [r"neural coding", r"population dynamics", r"oscillat", r"connectom", r"network", r"information theory", r"computation", r"dynamical system"],
    "methods": [r"magnetic resonance", r"\bfmri\b", r"electrophysi", r"microscop", r"optogen", r"brain.computer interface", r"stimulation", r"algorithm", r"method"],
    "neuroai": [r"artificial neural", r"deep learning", r"machine learning", r"transformer", r"neuroai", r"neuromorphic", r"representation learning", r"brain decoding", r"convolutional network"],
}
MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


@dataclass
class Paper:
    pmid: str
    year: int
    month: int
    date_precision: str
    title: str
    journal: str
    mesh: list[str]
    keywords: list[str]
    publication_types: list[str]
    abstract: str

    @property
    def embedding_text(self) -> str:
        parts = [self.title.strip(), self.abstract.strip()]
        if self.mesh:
            parts.append("MeSH: " + "; ".join(self.mesh[:24]))
        return "\n".join(p for p in parts if p)[:28_000]


def _text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        values[key.strip()] = value
    return values


def _request(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None,
             attempts: int = 6, timeout: int = 90) -> bytes:
    """Use the system TLS stack through curl (the macOS Python bundle lacks local CA roots)."""
    last = "unknown error"
    for attempt in range(attempts):
        command = ["curl", "--http1.1", "--fail-with-body", "--silent", "--show-error", "--max-time", str(timeout)]
        for key, value in (headers or {}).items():
            command.extend(["-H", f"{key}: {value}"])
        if data is not None:
            command.extend(["--data-binary", "@-"])
        command.append(url)
        completed = subprocess.run(command, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if completed.returncode == 0:
            return completed.stdout
        last = completed.stderr.decode("utf-8", errors="replace")[:500]
        time.sleep(min(20.0, 1.2 * (2 ** attempt)))
    raise RuntimeError(f"Network request failed after {attempts} attempts: {last}")


def _eutils_get(endpoint: str, params: dict[str, str | int]) -> bytes:
    query = urllib.parse.urlencode(params)
    payload = _request(f"{EUTILS}/{endpoint}?{query}", headers={"User-Agent": "neurohackademy-eight-forms/1.0"})
    time.sleep(0.36)  # Stay below NCBI's unauthenticated three-request/second ceiling.
    return payload


def _stable_spread(ids: list[str], n: int, year: int) -> list[str]:
    """Select across the relevance-ranked pool instead of taking only its head."""
    if len(ids) <= n:
        return ids
    rng = random.Random(19_760_000 + year)
    bins = np.array_split(np.arange(len(ids)), n)
    selected = [ids[rng.choice(list(map(int, block)))] for block in bins if len(block)]
    return selected[:n]


def _read_search_cache() -> dict[str, dict]:
    path = CACHE_DIR / "pubmed_search_v1.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_search_cache(cache: dict[str, dict]) -> None:
    path = CACHE_DIR / "pubmed_search_v1.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(cache, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)


def _cached_search(term: str, retmax: int, *, refresh: bool, cache: dict[str, dict]) -> tuple[list[str], int, bool]:
    key = hashlib.sha256(f"{term}\n{retmax}\nrelevance".encode()).hexdigest()
    if not refresh and key in cache:
        item = cache[key]
        return [str(value) for value in item["ids"]], int(item["count"]), True
    raw = _eutils_get(
        "esearch.fcgi",
        {
            "db": "pubmed", "term": term, "retmode": "json", "retmax": retmax,
            "sort": "relevance", "tool": "eight_forms_neuroart",
            "email": "neurohackademy-art@users.noreply.github.com",
        },
    )
    result = json.loads(raw)["esearchresult"]
    ids = [str(value) for value in result.get("idlist", [])]
    count = int(result.get("count", 0))
    cache[key] = {"count": count, "ids": ids, "cached_at": datetime.now(timezone.utc).isoformat()}
    _write_search_cache(cache)
    return ids, count, False


def search_pubmed(
    papers_per_year: int,
    targeted_per_form: int = 350,
    *,
    refresh_search: bool = False,
) -> tuple[list[str], dict[int, int], dict[str, int], dict[str, int]]:
    all_ids: list[str] = []
    counts: dict[int, int] = {}
    cache = _read_search_cache()
    cache_hits = 0
    cache_misses = 0
    pool_size = max(papers_per_year * 5, papers_per_year)
    compact_query = " ".join(BROAD_QUERY.split())
    for year in range(START_YEAR, END_YEAR + 1):
        term = f"({compact_query}) AND {year}[Date - Publication]"
        ids, counts[year], hit = _cached_search(term, pool_size, refresh=refresh_search, cache=cache)
        cache_hits += int(hit); cache_misses += int(not hit)
        chosen = _stable_spread(ids, papers_per_year, year)
        all_ids.extend(chosen)
        print(f"PubMed {year}: {counts[year]:>7,} matches, {len(chosen):>2} sampled", flush=True)
    targeted_counts: dict[str, int] = {}
    exclusion = 'hasabstract[text] NOT (editorial[Publication Type] OR letter[Publication Type] OR comment[Publication Type] OR correction[Publication Type])'
    for form, query in TARGET_QUERIES.items():
        target_term = f"({query}) AND ({exclusion}) AND {START_YEAR}:{END_YEAR}[Date - Publication]"
        pool, targeted_counts[form], hit = _cached_search(
            target_term, max(800, targeted_per_form * 5), refresh=refresh_search, cache=cache
        )
        cache_hits += int(hit); cache_misses += int(not hit)
        chosen = _stable_spread(pool, targeted_per_form, 70_000 + list(TARGET_QUERIES).index(form))
        all_ids.extend(chosen)
        print(f"Target {form:>10}: {targeted_counts[form]:>7,} matches, {len(chosen):>3} sampled", flush=True)
    # Preserve first encounter so the broad year-stratified core stays authoritative.
    all_ids = list(dict.fromkeys(all_ids))
    return all_ids, counts, targeted_counts, {"hits": cache_hits, "misses": cache_misses}


def _parse_month(value: str, pmid: str) -> tuple[int, str]:
    clean = value.strip().lower()
    if clean.isdigit():
        month = int(clean)
        if 1 <= month <= 12:
            return month, "month"
    for key, month in MONTHS.items():
        if key in clean[:5]:
            return month, "month"
    # Stable within-year placement when PubMed provides year precision only.
    return (int(hashlib.sha1(pmid.encode()).hexdigest()[:4], 16) % 12) + 1, "year"


def _parse_record(article: ET.Element) -> Paper | None:
    pmid = _text(article.find(".//MedlineCitation/PMID"))
    title = _text(article.find(".//Article/ArticleTitle"))
    abstract_nodes = article.findall(".//Article/Abstract/AbstractText")
    abstract_parts: list[str] = []
    for node in abstract_nodes:
        value = _text(node)
        label = (node.attrib.get("Label") or "").strip()
        if value:
            abstract_parts.append(f"{label}: {value}" if label else value)
    abstract = " ".join(abstract_parts).strip()
    if not pmid or not title or not abstract:
        return None

    year_value = _text(article.find(".//Article/ArticleDate/Year"))
    month_value = _text(article.find(".//Article/ArticleDate/Month"))
    if not year_value:
        year_value = _text(article.find(".//JournalIssue/PubDate/Year"))
        month_value = month_value or _text(article.find(".//JournalIssue/PubDate/Month"))
    if not year_value:
        medline_date = _text(article.find(".//JournalIssue/PubDate/MedlineDate"))
        match = re.search(r"(19|20)\d{2}", medline_date)
        year_value = match.group(0) if match else ""
        month_value = month_value or medline_date
    try:
        year = int(year_value)
    except ValueError:
        return None
    month, precision = _parse_month(month_value, pmid)

    mesh = [_text(node) for node in article.findall(".//MeshHeading/DescriptorName")]
    keywords = [_text(node) for node in article.findall(".//KeywordList/Keyword")]
    pub_types = [_text(node) for node in article.findall(".//PublicationTypeList/PublicationType")]
    journal = _text(article.find(".//Article/Journal/ISOAbbreviation")) or _text(
        article.find(".//Article/Journal/Title")
    )
    return Paper(
        pmid=pmid,
        year=max(START_YEAR, min(END_YEAR, year)),
        month=month,
        date_precision=precision,
        title=re.sub(r"\s+", " ", title),
        journal=re.sub(r"\s+", " ", journal),
        mesh=[m for m in mesh if m],
        keywords=[k for k in keywords if k],
        publication_types=[p for p in pub_types if p],
        abstract=re.sub(r"\s+", " ", abstract),
    )


def _fetch_pubmed_batch(batch: list[str]) -> list[Paper]:
    cache_key = hashlib.sha256(",".join(batch).encode()).hexdigest()
    cache_path = ABSTRACT_BATCH_CACHE / f"{cache_key}.json.gz"
    if cache_path.exists():
        try:
            with gzip.open(cache_path, "rt", encoding="utf-8") as handle:
                cached = json.load(handle)
            return [Paper(**item) for item in cached]
        except (OSError, json.JSONDecodeError, TypeError):
            cache_path.unlink(missing_ok=True)

    def checkpoint(records: list[Paper]) -> list[Paper]:
        ABSTRACT_BATCH_CACHE.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
            json.dump([asdict(record) for record in records], handle, ensure_ascii=False, separators=(",", ":"))
        temporary.replace(cache_path)
        return records

    data = urllib.parse.urlencode(
        {
            "db": "pubmed",
            "id": ",".join(batch),
            "retmode": "xml",
            "rettype": "abstract",
            "tool": "eight_forms_neuroart",
            "email": "neurohackademy-art@users.noreply.github.com",
        }
    ).encode()
    try:
        raw = _request(
            f"{EUTILS}/efetch.fcgi",
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "neurohackademy-eight-forms/1.0"},
            attempts=4,
            timeout=45,
        )
    except RuntimeError as error:
        if len(batch) > 8:
            midpoint = len(batch) // 2
            print(f"NCBI batch retry: splitting {len(batch)} records after {error}", flush=True)
            return checkpoint(_fetch_pubmed_batch(batch[:midpoint]) + _fetch_pubmed_batch(batch[midpoint:]))
        print(f"NCBI batch warning: {len(batch)} inaccessible records skipped after retries", flush=True)
        return checkpoint([])
    root = ET.fromstring(raw)
    parsed = [_parse_record(article) for article in root.findall(".//PubmedArticle")]
    return checkpoint([paper for paper in parsed if paper is not None])


def fetch_pubmed(ids: list[str], batch_size: int = 100) -> list[Paper]:
    batches = [ids[offset : offset + batch_size] for offset in range(0, len(ids), batch_size)]
    completed_records: dict[int, list[Paper]] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(_fetch_pubmed_batch, batch): index for index, batch in enumerate(batches)}
        completed_count = 0
        for future in as_completed(futures):
            batch_index = futures[future]
            completed_records[batch_index] = future.result()
            completed_count += len(batches[batch_index])
            print(f"PubMed abstracts: {min(completed_count, len(ids))}/{len(ids)} streamed", flush=True)
    papers: list[Paper] = []
    for batch_index in range(len(batches)):
        papers.extend(completed_records[batch_index])
    papers.sort(key=lambda p: (p.year, p.month, p.pmid))
    return papers


def _read_embedding_cache() -> dict[str, np.ndarray]:
    path = CACHE_DIR / "embeddings_text_large_v1.npz"
    if not path.exists():
        return {}
    try:
        with np.load(path, allow_pickle=False) as archive:
            hashes = archive["hashes"]
            vectors = archive["vectors"]
            return {str(key): vector.astype(np.float32) for key, vector in zip(hashes, vectors)}
    except (OSError, ValueError, KeyError):
        return {}


def _write_embedding_cache(cache: dict[str, np.ndarray]) -> None:
    path = CACHE_DIR / "embeddings_text_large_v1.npz"
    temporary = path.with_suffix(".tmp")
    hashes = np.asarray(sorted(cache), dtype="<U64")
    vectors = np.stack([cache[key] for key in hashes]).astype(np.float16)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, hashes=hashes, vectors=vectors)
    temporary.replace(path)


def embed_texts(texts: list[str], api_key: str, batch_size: int = 32) -> tuple[np.ndarray, dict[str, float]]:
    cache = _read_embedding_cache()
    keys = [hashlib.sha256(f"{EMBEDDING_MODEL}:{EMBEDDING_DIMENSIONS}:{text}".encode("utf-8")).hexdigest() for text in texts]
    missing_indices = [index for index, key in enumerate(keys) if key not in cache]
    usage = {
        "prompt_tokens": 0.0, "cost_usd": 0.0,
        "cache_hits": float(len(texts) - len(missing_indices)), "cache_misses": float(len(missing_indices)),
    }
    for offset in range(0, len(missing_indices), batch_size):
        indices = missing_indices[offset : offset + batch_size]
        batch = [texts[index] for index in indices]
        body = json.dumps(
            {
                "model": EMBEDDING_MODEL,
                "input": batch,
                "dimensions": EMBEDDING_DIMENSIONS,
                "encoding_format": "float",
                "provider": {"allow_fallbacks": True, "data_collection": "deny"},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        raw = _request(
            OPENROUTER_EMBEDDINGS,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://neurohackademy.org/",
                "X-Title": "Eight Forms of Intelligence",
            },
            timeout=120,
        )
        response = json.loads(raw)
        if response.get("error"):
            raise RuntimeError(f"OpenRouter embedding error: {response['error']}")
        ordered = sorted(response.get("data", []), key=lambda item: int(item.get("index", 0)))
        for index, item in zip(indices, ordered):
            cache[keys[index]] = np.asarray(item["embedding"], dtype=np.float32)
        batch_usage = response.get("usage") or {}
        usage["prompt_tokens"] += float(batch_usage.get("prompt_tokens", 0) or 0)
        usage["cost_usd"] += float(batch_usage.get("cost", 0) or 0)
        _write_embedding_cache(cache)
        print(
            f"OpenRouter embeddings: {min(offset + batch_size, len(missing_indices))}/{len(missing_indices)} new "
            f"({int(usage['cache_hits'])} cached)", flush=True
        )
    if not missing_indices:
        print(f"OpenRouter embeddings: all {len(texts)} loaded from content cache", flush=True)
    matrix = np.stack([cache[key] for key in keys]).astype(np.float32)
    if matrix.shape != (len(texts), EMBEDDING_DIMENSIONS):
        raise RuntimeError(f"Unexpected embedding matrix shape: {matrix.shape}")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix /= np.maximum(norms, 1e-8)
    return matrix, usage


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))


def _rule_evidence(text: str, positive: list[str], negative: list[str]) -> float:
    hits = sum(1 for pattern in positive if re.search(pattern, text, flags=re.I))
    if not hits:
        return 0.0
    score = min(1.0, 0.68 + 0.16 * (hits - 1))
    if any(re.search(pattern, text, flags=re.I) for pattern in negative):
        score *= 0.22
    return score


def classify(
    papers: list[Paper],
    paper_embeddings: np.ndarray,
    strip_embeddings: np.ndarray,
    theme_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    strip_similarity = paper_embeddings @ strip_embeddings.T
    theme_similarity = paper_embeddings @ theme_embeddings.T
    probabilities = np.zeros_like(strip_similarity, dtype=np.float32)
    for row, paper in enumerate(papers):
        evidence_text = " ".join(
            [paper.title, paper.abstract, *paper.mesh, *paper.keywords]
        ).lower()
        sims = strip_similarity[row]
        mean = float(np.mean(sims))
        std = max(float(np.std(sims)), 0.015)
        for col, strip in enumerate(STRIPS):
            semantic = _sigmoid((float(sims[col]) - mean) / (std * 0.9))
            evidence = _rule_evidence(evidence_text, strip["positive"], strip["negative"])
            value = 0.025 + 0.21 * semantic + 0.77 * evidence
            probabilities[row, col] = min(0.995, value)
        if float(probabilities[row].max()) < 0.24:
            probabilities[row, int(np.argmax(sims))] = 0.24
    theme_scores = theme_similarity.copy()
    # Topic labels are explicitly neuroscientific: embedding affinity plus transparent lexical evidence.
    for row, paper in enumerate(papers):
        evidence_text = " ".join([paper.title, paper.abstract, *paper.mesh, *paper.keywords]).lower()
        for col, theme in enumerate(THEMES):
            hits = sum(bool(re.search(pattern, evidence_text, re.I)) for pattern in THEME_RULES[theme[0]])
            theme_scores[row, col] += min(0.24, hits * 0.055)
    primary_theme = np.argmax(theme_scores, axis=1).astype(np.int16)
    secondary_theme = np.argsort(theme_scores, axis=1)[:, -2].astype(np.int16)
    return probabilities, primary_theme, secondary_theme


def project_embeddings(embeddings: np.ndarray, themes: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    pca_dim = min(32, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=pca_dim, random_state=42, svd_solver="randomized")
    reduced = pca.fit_transform(embeddings).astype(np.float32)
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "umap-learn is required. Install requirements.txt or run through make_film.py, "
            "which can use the temporary build vendor."
        ) from exc
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(24, max(5, embeddings.shape[0] - 1)),
        min_dist=0.10,
        spread=1.0,
        metric="cosine",
        random_state=42,
        transform_seed=42,
    )
    raw_xyz = reducer.fit_transform(reduced).astype(np.float32)
    xyz = np.zeros_like(raw_xyz)
    for axis in range(3):
        lo, hi = np.quantile(raw_xyz[:, axis], [0.01, 0.99])
        center = (lo + hi) / 2.0
        scale = max((hi - lo) / 2.0, 1e-6)
        xyz[:, axis] = np.clip((raw_xyz[:, axis] - center) / scale, -1.0, 1.0)

    # Give the global semantic manifold legible topic continents without destroying local neighborhoods.
    angles = np.linspace(0, 2 * np.pi, len(THEMES), endpoint=False) + 0.17
    topic_centers = np.stack([np.cos(angles), np.sin(angles), 0.55 * np.sin(angles * 2.0)], axis=1).astype(np.float32)
    xyz = 0.72 * xyz + 0.34 * topic_centers[themes]
    xyz /= np.maximum(np.max(np.abs(xyz), axis=0, keepdims=True), 1e-6)

    nn = NearestNeighbors(n_neighbors=min(9, len(reduced)), metric="cosine")
    nn.fit(reduced)
    distances, indices = nn.kneighbors(reduced)
    local = np.mean(distances[:, 1:], axis=1)
    lo, hi = np.quantile(local, [0.05, 0.95])
    density = 1.0 - np.clip((local - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    earlier_neighbors: list[list[int]] = []
    for idx, candidates in enumerate(indices):
        previous = [int(value) for value in candidates[1:] if int(value) < idx][:2]
        earlier_neighbors.append(previous)
    return xyz, density.astype(np.float32), earlier_neighbors


def build_timeline(
    papers: list[Paper], xyz: np.ndarray, themes: np.ndarray,
    probabilities: np.ndarray, query_counts: dict[int, int]
) -> list[dict]:
    timeline: list[dict] = []
    previous_centroid = np.zeros(3, dtype=np.float32)
    known_themes: set[int] = set()
    for year in range(START_YEAR, END_YEAR + 1):
        idx = np.array([i for i, paper in enumerate(papers) if paper.year == year], dtype=int)
        if idx.size:
            points = xyz[idx]
            centroid = np.mean(points, axis=0)
            dispersion = float(np.mean(np.linalg.norm(points - centroid, axis=1)))
            counts = np.bincount(themes[idx], minlength=len(THEMES)).astype(np.float64)
            distribution = counts / max(counts.sum(), 1.0)
            entropy = float(-np.sum(distribution[distribution > 0] * np.log2(distribution[distribution > 0])) / math.log2(len(THEMES)))
            births = len(set(map(int, themes[idx])) - known_themes)
            known_themes.update(map(int, themes[idx]))
            strip_activity = probabilities[idx].mean(axis=0)
        else:
            centroid = previous_centroid.copy()
            dispersion = entropy = 0.0
            births = 0
            strip_activity = np.zeros(len(STRIPS), dtype=np.float32)
        velocity = float(np.linalg.norm(centroid - previous_centroid)) if timeline else 0.0
        previous_centroid = centroid
        timeline.append(
            {
                "year": year,
                "query_count": int(query_counts.get(year, 0)),
                "sample_count": int(idx.size),
                "centroid": [round(float(v), 5) for v in centroid],
                "dispersion": round(dispersion, 5),
                "entropy": round(entropy, 5),
                "velocity": round(velocity, 5),
                "theme_births": births,
                "strip_activity": [round(float(v), 4) for v in strip_activity],
            }
        )
    return timeline


def _compact_record(
    paper: Paper,
    xyz: np.ndarray,
    density: float,
    neighbors: list[int],
    strip_probabilities: np.ndarray,
    theme: int,
    secondary_theme: int,
) -> dict:
    strip_map = {
        STRIPS[i]["id"]: round(float(value), 3)
        for i, value in enumerate(strip_probabilities)
        if float(value) >= 0.075
    }
    return {
        "pmid": paper.pmid,
        "year": paper.year,
        "month": paper.month,
        "date_precision": paper.date_precision,
        "title": paper.title,
        "journal": paper.journal,
        "mesh": paper.mesh[:12],
        "keywords": paper.keywords[:8],
        "publication_types": paper.publication_types[:5],
        "theme": THEMES[int(theme)][0],
        "secondary_theme": THEMES[int(secondary_theme)][0],
        "xyz": [round(float(value), 5) for value in xyz],
        "density": round(float(density), 4),
        "neighbors": neighbors,
        "strips": strip_map,
    }


def build_corpus(
    *,
    papers_per_year: int = 104,
    targeted_per_form: int = 350,
    env_file: Path = DEFAULT_ENV,
    force: bool = False,
    refresh_search: bool = False,
) -> tuple[Path, Path]:
    ensure_dirs()
    corpus_path = DATA_DIR / "papers.json.gz"
    manifest_path = DATA_DIR / "run_manifest.json"
    if corpus_path.exists() and manifest_path.exists() and not force:
        print(f"Using existing corpus: {corpus_path}")
        return corpus_path, manifest_path

    env = _load_env(env_file)
    api_key = env.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError(f"OPENROUTER_API_KEY was not found in {env_file}")

    started = datetime.now(timezone.utc)
    ids, query_counts, targeted_counts, search_cache = search_pubmed(
        papers_per_year, targeted_per_form, refresh_search=refresh_search
    )
    papers = fetch_pubmed(ids)
    if len(papers) < (END_YEAR - START_YEAR + 1) * max(4, papers_per_year // 3):
        raise RuntimeError(f"Too few PubMed records were parsed: {len(papers)}")

    anchor_texts = [strip["anchor"] for strip in STRIPS] + [f"{theme[1]}. {theme[2]}" for theme in THEMES]
    anchor_embeddings, anchor_usage = embed_texts(anchor_texts, api_key, batch_size=len(anchor_texts))
    paper_embeddings, paper_usage = embed_texts([paper.embedding_text for paper in papers], api_key)
    # Abstracts exist only in this process. No raw API response or abstract text is written.

    probabilities, themes, secondary = classify(
        papers,
        paper_embeddings,
        anchor_embeddings[: len(STRIPS)],
        anchor_embeddings[len(STRIPS) :],
    )
    xyz, density, neighbors = project_embeddings(paper_embeddings, themes)
    timeline = build_timeline(papers, xyz, themes, probabilities, query_counts)

    records = [
        _compact_record(
            paper, xyz[i], density[i], neighbors[i], probabilities[i], themes[i], secondary[i]
        )
        for i, paper in enumerate(papers)
    ]
    payload = {
        "title": "Eight Forms of Intelligence: 1976–2026",
        "abstracts_retained": False,
        "embedding_vectors_retained": False,
        "strips": [{key: strip[key] for key in ("id", "name", "source_short", "color")} for strip in STRIPS],
        "themes": [
            {"id": theme[0], "name": theme[1], "color": theme[3]} for theme in THEMES
        ],
        "timeline": timeline,
        "papers": records,
    }
    enrich_clusters(payload)
    with gzip.open(corpus_path, "wt", encoding="utf-8", compresslevel=9) as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
    public_path = ROOT / "public" / "data" / "corpus.json"
    public_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    ended = datetime.now(timezone.utc)
    manifest = {
        "generated_at": ended.isoformat(),
        "elapsed_seconds": round((ended - started).total_seconds(), 2),
        "period": {"start": START_YEAR, "end": END_YEAR, "end_is_partial": True},
        "source": {
            "database": "PubMed",
            "api": "NCBI E-utilities ESearch + EFetch",
            "query": " ".join(BROAD_QUERY.split()),
            "sampling": "Per-year relevance pool followed by deterministic spread sampling",
            "requested_per_year": papers_per_year,
            "targeted_per_form": targeted_per_form,
            "targeted_query_counts": targeted_counts,
            "search_cache": search_cache,
            "parsed_records": len(papers),
            "query_counts": {str(year): count for year, count in query_counts.items()},
        },
        "classification": {
            "method": "Transparent MeSH/title/abstract keyword evidence combined with cosine similarity to neuroscience topic and form anchors",
            "generative_llm_used": False,
            "multi_label": True,
        },
        "embedding": {
            "provider": "OpenRouter",
            "model": EMBEDDING_MODEL,
            "dimensions": EMBEDDING_DIMENSIONS,
            "all_abstracts_embedded": True,
            "abstracts_retained": False,
            "vectors_retained": False,
            "prompt_tokens": int(anchor_usage["prompt_tokens"] + paper_usage["prompt_tokens"]),
            "reported_cost_usd": round(anchor_usage["cost_usd"] + paper_usage["cost_usd"], 6),
            "cache_hits": int(anchor_usage["cache_hits"] + paper_usage["cache_hits"]),
            "cache_misses": int(anchor_usage["cache_misses"] + paper_usage["cache_misses"]),
            "cache_format": "content-addressed SHA-256 keys + float16 vectors; no abstract text",
        },
        "projection": {
            "pipeline": "L2 normalization → PCA(32) → global UMAP(3) → 28% neuroscience-topic topology blend",
            "umap": {"neighbors": 24, "min_dist": 0.10, "metric": "cosine", "random_seed": 42},
            "fit_once_globally": True,
        },
        "limitations": [
            "The year-stratified corpus is an artistic sample, not an exhaustive bibliometric dataset.",
            "Strip assignments are unvalidated probabilistic signals and should not be used as scientific labels.",
            "The anatomical forms are morphological boundaries; semantic particles are not claims of anatomical localization.",
            "The 2026 interval is incomplete as of 2026-07-22.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Make accidental post-write retention impossible within the long-running process.
    for paper in papers:
        paper.abstract = ""
    del paper_embeddings, anchor_embeddings
    if ABSTRACT_BATCH_CACHE.exists():
        shutil.rmtree(ABSTRACT_BATCH_CACHE)
    print(f"Retained {len(records)} compact records; abstracts and vectors discarded.")
    return corpus_path, manifest_path


def load_corpus(path: Path | None = None) -> dict:
    target = path or (DATA_DIR / "papers.json.gz")
    with gzip.open(target, "rt", encoding="utf-8") as handle:
        return json.load(handle)
