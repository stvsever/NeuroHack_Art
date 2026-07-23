from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import struct
import subprocess
import tarfile
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors

from .config import ASSET_DIR, BUILD_DIR, ROOT, ensure_dirs


SOURCES = {
    "elegans": {
        "name": "OpenWorm c302 NeuroML",
        "url": "https://codeload.github.com/openworm/c302/tar.gz/refs/heads/master",
        "space": "c302 NeuroML morphology set",
        "license": "LGPL-3.0",
    },
    "drosophila": {
        "name": "FlyWire neuropil meshes v141/v6",
        "url": "https://storage.googleapis.com/flywire_neuropil_meshes/neuropils/neuropil_mesh_v141_v6/mesh/",
        "manifest": "https://storage.googleapis.com/storage/v1/b/flywire_neuropil_meshes/o?prefix=neuropils/neuropil_mesh_v141_v6/mesh/&maxResults=1000",
        "space": "FAFB / FlyWire neuropil space",
        "license": "Source-specific; cite FlyWire",
    },
    "rodent": {
        "name": "Allen Mouse Common Coordinate Framework v3",
        "url": "https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_100.nrrd",
        "space": "Allen CCFv3 (2017), 100 µm annotation",
        "license": "Allen Institute citation policy",
    },
    "macaque": {
        "name": "NIMH Macaque Template v2 symmetric",
        "url": "https://afni.nimh.nih.gov/pub/dist/atlases/macaque/nmt/NMT_v2.0_sym.tgz",
        "space": "NMT v2 symmetric, 0.5 mm",
        "license": "NMT v2 terms / CC BY 4.0 publication",
    },
    "human": {
        "name": "TemplateFlow MNI152NLin2009cAsym",
        "url": "https://templateflow.s3.amazonaws.com/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz",
        "space": "MNI152NLin2009cAsym, resolution 2",
        "license": "TemplateFlow source terms",
    },
}


def run_curl(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "--fail", "--location", "--retry", "4", "--silent", "--show-error", "--output", str(target), url],
        check=True,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    lo, hi = np.quantile(points, [0.005, 0.995], axis=0)
    center = (lo + hi) * 0.5
    scale = max(float(np.max(hi - lo)) * 0.5, 1e-7)
    return np.clip((points - center) / scale, -1.15, 1.15)


def sample_rows(points: np.ndarray, count: int, seed: int) -> np.ndarray:
    if len(points) <= count:
        return points
    rng = np.random.default_rng(seed)
    return points[np.sort(rng.choice(len(points), count, replace=False))]


def knn_edges(points: np.ndarray, neighbors: int = 3, max_factor: float = 2.3) -> np.ndarray:
    if len(points) < 2:
        return np.empty((0, 2), dtype=np.uint16)
    model = NearestNeighbors(n_neighbors=min(neighbors + 1, len(points))).fit(points)
    distances, indices = model.kneighbors(points)
    threshold = float(np.median(distances[:, 1])) * max_factor
    pairs: set[tuple[int, int]] = set()
    for a in range(len(points)):
        for distance, b in zip(distances[a, 1:], indices[a, 1:]):
            b = int(b)
            if distance <= threshold:
                pairs.add((min(a, b), max(a, b)))
    return np.asarray(sorted(pairs), dtype=np.uint16)


def encode_geometry(points: np.ndarray, edges: np.ndarray) -> dict[str, object]:
    points = np.asarray(points, dtype=np.float32)
    quantized = np.rint(np.clip(points, -1.2, 1.2) * 25_000).astype("<i2")
    edge16 = np.asarray(edges, dtype="<u2")
    extents = (points.max(axis=0) - points.min(axis=0)).round(5).tolist()
    return {
        "pointCount": int(len(points)),
        "edgeCount": int(len(edge16)),
        "pointScale": 1 / 25_000,
        "points": base64.b64encode(quantized.tobytes()).decode("ascii"),
        "edges": base64.b64encode(edge16.tobytes()).decode("ascii"),
        "extents": extents,
    }


def parse_nifti(blob: bytes) -> np.ndarray:
    if blob[:2] == b"\x1f\x8b":
        blob = gzip.decompress(blob)
    endian = "<" if struct.unpack_from("<I", blob, 0)[0] == 348 else ">"
    dims = struct.unpack_from(endian + "8h", blob, 40)
    shape = tuple(int(v) for v in dims[1 : dims[0] + 1])
    datatype = struct.unpack_from(endian + "h", blob, 70)[0]
    vox_offset = int(struct.unpack_from(endian + "f", blob, 108)[0])
    dtype_map = {2: "u1", 4: "i2", 8: "i4", 16: "f4", 64: "f8", 256: "i1", 512: "u2", 768: "u4"}
    if datatype not in dtype_map:
        raise ValueError(f"Unsupported NIfTI datatype {datatype}")
    dtype = np.dtype(dtype_map[datatype]).newbyteorder(endian)
    count = int(np.prod(shape))
    return np.frombuffer(blob, dtype=dtype, count=count, offset=vox_offset).reshape(shape, order="F")


def parse_nrrd(blob: bytes) -> np.ndarray:
    split = blob.find(b"\n\n")
    offset = 2
    if split < 0:
        split = blob.find(b"\r\n\r\n")
        offset = 4
    header = blob[:split].decode("ascii", errors="replace")
    fields: dict[str, str] = {}
    for line in header.splitlines():
        if ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
    payload = blob[split + offset :]
    if fields.get("encoding", "raw").lower() in {"gzip", "gz"}:
        payload = gzip.decompress(payload)
    dtype_map = {
        "uchar": "u1", "unsigned char": "u1", "uint8": "u1",
        "short": "i2", "short int": "i2", "int16": "i2",
        "ushort": "u2", "unsigned short": "u2", "uint16": "u2",
        "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4", "unsigned int": "u4",
    }
    dtype = np.dtype(dtype_map[fields["type"].lower()])
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder("<" if fields.get("endian", "little") == "little" else ">")
    shape = tuple(map(int, fields["sizes"].split()))
    return np.frombuffer(payload, dtype=dtype, count=int(np.prod(shape))).reshape(shape, order="F")


def volume_geometry(volume: np.ndarray, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    mask = np.asarray(volume) > 0
    boundary = mask & ~ndimage.binary_erosion(mask, iterations=1)
    coords = np.argwhere(boundary).astype(np.float32)
    points = normalize(sample_rows(coords, count, seed))
    return points, knn_edges(points, 3)


def build_elegans(ref: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    archive = ref / "c302-master.tgz"
    run_curl(SOURCES["elegans"]["url"], archive)
    points: list[list[float]] = []
    edges: list[tuple[int, int]] = []
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if "/NeuroML2/" in m.name and m.name.endswith(".cell.nml")]
        for member in members:
            handle = tar.extractfile(member)
            if handle is None:
                continue
            root = ET.fromstring(handle.read())
            for segment in root.findall(".//{*}segment"):
                proximal = segment.find("{*}proximal")
                distal = segment.find("{*}distal")
                if distal is None:
                    continue
                if proximal is None:
                    proximal = distal
                a = len(points)
                points.append([float(proximal.get(k, "0")) for k in ("x", "y", "z")])
                points.append([float(distal.get(k, "0")) for k in ("x", "y", "z")])
                edges.append((a, a + 1))
    raw = np.asarray(points, dtype=np.float32)
    # Some cell files share the same atlas coordinates. Deduplicate while retaining actual segment endpoints.
    rounded = np.rint(raw * 50).astype(np.int32)
    _, unique = np.unique(rounded, axis=0, return_index=True)
    chosen = np.sort(unique)
    if len(chosen) > 5000:
        chosen = sample_rows(chosen[:, None], 5000, 302).ravel()
    raw = raw[chosen]
    points_out = normalize(raw)
    edges_out = knn_edges(points_out, 2, 3.0)
    meta = {"rawFile": archive.name, "rawSha256": sha256(archive), "morphologyFiles": len(members)}
    return points_out, edges_out, meta


def parse_legacy_mesh(blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(blob) < 16:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint32)
    vertex_count = struct.unpack_from("<I", blob, 0)[0]
    vertex_bytes = 4 + vertex_count * 12
    if vertex_count == 0 or vertex_bytes > len(blob):
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.uint32)
    vertices = np.frombuffer(blob, dtype="<f4", count=vertex_count * 3, offset=4).reshape(-1, 3).copy()
    remainder = len(blob) - vertex_bytes
    faces = np.frombuffer(blob, dtype="<u4", count=remainder // 4, offset=vertex_bytes).reshape(-1, 3).copy()
    return vertices, faces


def build_drosophila(ref: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    manifest_path = ref / "flywire_objects.json"
    run_curl(SOURCES["drosophila"]["manifest"], manifest_path)
    manifest = json.loads(manifest_path.read_text())
    objects = [item for item in manifest.get("items", []) if int(item.get("size", 0)) > 0 and "/mesh/" in item.get("name", "")]
    all_points: list[np.ndarray] = []
    checksums: list[str] = []
    mesh_dir = ref / "flywire_meshes"
    for index, item in enumerate(objects):
        name = item["name"]
        url = "https://storage.googleapis.com/flywire_neuropil_meshes/" + urllib.parse.quote(name, safe="/")
        target = mesh_dir / f"{index:03d}.mesh"
        run_curl(url, target)
        vertices, _ = parse_legacy_mesh(target.read_bytes())
        if len(vertices):
            all_points.append(vertices)
            checksums.append(sha256(target))
    raw = np.concatenate(all_points, axis=0)
    points = normalize(sample_rows(raw, 5000, 1416))
    edges = knn_edges(points, 3, 2.0)
    combined = hashlib.sha256("".join(checksums).encode()).hexdigest()
    return points, edges, {"meshFragments": len(all_points), "rawSha256Combined": combined}


def build_rodent(ref: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    path = ref / "annotation_100.nrrd"
    run_curl(SOURCES["rodent"]["url"], path)
    points, edges = volume_geometry(parse_nrrd(path.read_bytes()), 5000, 100)
    return points, edges, {"rawFile": path.name, "rawSha256": sha256(path)}


def build_macaque(ref: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    path = ref / "NMT_v2.0_sym.tgz"
    run_curl(SOURCES["macaque"]["url"], path)
    member_name = "NMT_v2.0_sym/NMT_v2.0_sym_05mm/NMT_v2.0_sym_05mm_brainmask.nii.gz"
    with tarfile.open(path, "r:gz") as tar:
        handle = tar.extractfile(member_name)
        if handle is None:
            raise FileNotFoundError(member_name)
        volume = parse_nifti(handle.read())
    points, edges = volume_geometry(volume, 5000, 205)
    return points, edges, {"rawFile": path.name, "rawSha256": sha256(path), "member": member_name}


def build_human(ref: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    path = ref / "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
    run_curl(SOURCES["human"]["url"], path)
    points, edges = volume_geometry(parse_nifti(path.read_bytes()), 5000, 2009)
    return points, edges, {"rawFile": path.name, "rawSha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive compact art geometry from authoritative morphology/atlas sources.")
    parser.add_argument("--keep-downloads", action="store_true", help="Keep temporary source downloads for inspection.")
    args = parser.parse_args()
    ensure_dirs()
    ref = BUILD_DIR / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    public = ROOT / "public" / "data"
    public.mkdir(parents=True, exist_ok=True)

    builders = {
        "elegans": build_elegans,
        "drosophila": build_drosophila,
        "rodent": build_rodent,
        "macaque": build_macaque,
        "human": build_human,
    }
    entries: list[dict[str, object]] = [
        {
            "id": "gradient", "kind": "procedural-field", "name": "Excitable neural gradient",
            "space": None, "reason": "A neural gradient is a mechanism, not an atlas-bearing organism.",
        },
        {
            "id": "fungal", "kind": "procedural-network", "name": "Adaptive mycelial graph",
            "space": None, "reason": "Fungi have distributed signalling but no canonical neural template.",
        },
    ]
    for key, builder in builders.items():
        print(f"Deriving {key} geometry…", flush=True)
        points, edges, extra = builder(ref)
        entry = {"id": key, "kind": "source-derived", **SOURCES[key], **encode_geometry(points, edges), **extra}
        entries.append(entry)
        print(f"  {len(points):,} points / {len(edges):,} edges", flush=True)
    entries.append(
        {
            "id": "ai", "kind": "procedural-learning-system", "name": "The learning machine",
            "space": None,
            "reason": "Artificial intelligence has no anatomical atlas; its visible body is a trained topology that evolves from perceptron to attention and mixture-of-experts.",
        }
    )
    result = {
        "version": 2,
        "principle": "Source-derived geometry is morphological scaffolding. Semantic paper positions are artistic territories, never claims of anatomical localization.",
        "entries": entries,
    }
    (public / "templates.json").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    (ASSET_DIR / "template_provenance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.keep_downloads:
        import shutil
        shutil.rmtree(ref)
        try:
            BUILD_DIR.rmdir()
        except OSError:
            pass
    print(f"Wrote {public / 'templates.json'}", flush=True)


if __name__ == "__main__":
    main()
