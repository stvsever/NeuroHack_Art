from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import re
import struct
import subprocess
import tarfile
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image
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
        "name": "American Beaver whole-brain photographs and serial histology atlas, specimen #63-168",
        "url": "https://brains.anatomy.msu.edu/museum/brain/specimens/rodentia/beaver/sections/thumbnail.html",
        "image_base": "https://brains.anatomy.msu.edu/museum/brain/specimens/rodentia/beaver/sections/",
        "whole_brain": "https://brains.anatomy.msu.edu/museum/brain/specimens/rodentia/beaver/brain/Beaver631686clr.jpg",
        "preparation": "https://www.brainmuseum.org/explore/histoprocedures.html",
        "usage": "https://brains.anatomy.msu.edu/copyright.html",
        "space": "Castor canadensis #63-168 photo-calibrated cranial encephalon",
        "license": "Copyrighted source images; free use by permission with collection and NSF credit",
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


def segment_beaver_section(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return the tissue boundary and full mask from one standardized atlas plate."""
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
    red, green, blue = image[..., 0], image[..., 1], image[..., 2]
    chroma = image.max(axis=2) - image.min(axis=2)
    tissue = (
        (chroma > 0.035)
        & (blue > green + 0.018)
        & (red > green - 0.025)
        & (image.mean(axis=2) < 0.975)
    )
    # Atlas text, ruler and credits sit outside this fixed plate region. This
    # crop keeps every stained fragment without letting page furniture become anatomy.
    tissue[:48] = False
    tissue[-92:] = False
    tissue[:, :18] = False
    tissue[:, -18:] = False
    tissue = ndimage.binary_closing(tissue, iterations=2)
    labels, count = ndimage.label(tissue)
    if not count:
        raise ValueError(f"No stained tissue detected in {path.name}")
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max())
    keep = np.flatnonzero((sizes >= max(180, int(largest * 0.0025))))
    keep = keep[keep != 0]
    mask = np.isin(labels, keep)
    filled = np.zeros_like(mask)
    relabeled, component_count = ndimage.label(mask)
    for component in range(1, component_count + 1):
        filled |= ndimage.binary_fill_holes(relabeled == component)
    boundary = filled & ~ndimage.binary_erosion(filled, iterations=2)
    return boundary, filled


def segment_beaver_photograph(image: np.ndarray, crop: tuple[slice, slice]) -> np.ndarray:
    """Segment one specimen view from the collection's fixed whole-brain composite."""
    view = image[crop].astype(np.float32) / 255
    red, green, blue = view[..., 0], view[..., 1], view[..., 2]
    tissue = (red > 0.22) & (green > 0.12) & (red > blue * 0.92)
    tissue = ndimage.binary_closing(tissue, iterations=2)
    labels, count = ndimage.label(tissue)
    if not count:
        raise ValueError("No beaver tissue detected in whole-brain photograph")
    sizes = np.bincount(labels.ravel())
    largest = int(sizes[1:].max())
    keep = np.flatnonzero(sizes >= max(24, int(largest * 0.006)))
    keep = keep[keep != 0]
    return ndimage.binary_fill_holes(np.isin(labels, keep))


def beaver_photo_profiles(path: Path) -> tuple[dict[int, tuple[float, float]], dict[int, tuple[float, float]]]:
    """Return dorsoventral and left-right envelopes in shared photograph pixels."""
    image = np.asarray(Image.open(path).convert("RGB"))
    if image.shape[:2] != (503, 504):
        raise ValueError(f"Unexpected beaver whole-brain photograph dimensions: {image.shape[:2]}")

    # Left lateral and dorsal views are from the same specimen and photographic
    # scale. The posterior limit stops immediately after the cerebellum and
    # short medulla, before the visibly exposed spinal cord.
    lateral_crop = (slice(48, 166), slice(4, 252))
    dorsal_crop = (slice(316, 442), slice(4, 252))
    lateral = segment_beaver_photograph(image, lateral_crop)
    dorsal = segment_beaver_photograph(image, dorsal_crop)
    anterior_limit, posterior_limit = 13, 188

    def profiles(mask: np.ndarray, crop: tuple[slice, slice]) -> dict[int, tuple[float, float]]:
        row_offset = int(crop[0].start or 0)
        column_offset = int(crop[1].start or 0)
        result: dict[int, tuple[float, float]] = {}
        for absolute_column in range(anterior_limit, posterior_limit + 1):
            local_column = absolute_column - column_offset
            rows = np.flatnonzero(mask[:, local_column])
            if len(rows):
                result[absolute_column] = (
                    float(rows.min() + row_offset),
                    float(rows.max() + row_offset),
                )
        return result

    lateral_profiles = profiles(lateral, lateral_crop)
    dorsal_profiles = profiles(dorsal, dorsal_crop)
    shared = sorted(set(lateral_profiles) & set(dorsal_profiles))
    if len(shared) < 150:
        raise ValueError(f"Incomplete beaver photo envelope: only {len(shared)} shared columns")

    def smooth(values: dict[int, tuple[float, float]]) -> dict[int, tuple[float, float]]:
        columns = np.asarray(shared)
        lower = ndimage.gaussian_filter1d(np.asarray([values[x][0] for x in shared]), 1.1)
        upper = ndimage.gaussian_filter1d(np.asarray([values[x][1] for x in shared]), 1.1)
        return {int(x): (float(a), float(b)) for x, a, b in zip(columns, lower, upper)}

    return smooth(lateral_profiles), smooth(dorsal_profiles)


def build_rodent(ref: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    atlas_page = ref / "beaver_picture_atlas.html"
    whole_brain = ref / "Beaver631686clr.jpg"
    run_curl(SOURCES["rodent"]["url"], atlas_page)
    run_curl(SOURCES["rodent"]["whole_brain"], whole_brain)
    section_numbers = sorted({
        int(value)
        for value in re.findall(r"AmBeav63_168Cells(\d+)Lg\.jpg", atlas_page.read_text(errors="replace"))
    })
    if len(section_numbers) < 100:
        raise ValueError(f"Expected a complete beaver atlas series, found {len(section_numbers)} sections")

    section_dir = ref / "beaver_sections"
    checksums: list[str] = []
    section_contours: list[tuple[int, np.ndarray, tuple[int, int]]] = []
    unavailable_sections: list[int] = []
    encephalon_section_limit = 2180
    for number in section_numbers:
        path = section_dir / f"AmBeav63_168Cells{number}Lg.jpg"
        try:
            run_curl(f"{SOURCES['rodent']['image_base']}{path.name}", path)
        except subprocess.CalledProcessError:
            unavailable_sections.append(number)
            continue
        boundary, mask = segment_beaver_section(path)
        rows, columns = np.nonzero(boundary)
        mask_rows, _ = np.nonzero(mask)
        if not len(rows) or not len(mask_rows):
            raise ValueError(f"Empty beaver section after segmentation: {path.name}")
        if number <= encephalon_section_limit:
            section_contours.append((
                number,
                np.column_stack((rows, columns)).astype(np.float32),
                mask.shape,
            ))
        checksums.append(f"{number}:{sha256(path)}")

    lateral_profiles, dorsal_profiles = beaver_photo_profiles(whole_brain)
    photo_columns = sorted(set(lateral_profiles) & set(dorsal_profiles))
    anterior_photo, posterior_photo = photo_columns[0], photo_columns[-1]
    lateral_center = float(np.mean([(a + b) * 0.5 for a, b in lateral_profiles.values()]))
    dorsal_center = float(np.mean([(a + b) * 0.5 for a, b in dorsal_profiles.values()]))
    max_serial_width = max(float(contour[:, 1].max() - contour[:, 1].min()) for _, contour, _ in section_contours)
    max_photo_width = max(bottom - top for top, bottom in dorsal_profiles.values())
    photo_registration_scale = max_photo_width / max(max_serial_width, 1)

    # Each point remains an actual boundary pixel from a coronal beaver section.
    # The specimen photographs calibrate the AP span and one shared in-plane
    # scale. They also provide the subtle dorsal and lateral center curves. No
    # per-section shape fitting or independent-axis stretching is applied.
    contour_points: list[np.ndarray] = []
    first_section = min(number for number, _, _ in section_contours)
    for number, contour, shape in section_contours:
        progress = (number - first_section) / max(encephalon_section_limit - first_section, 1)
        photo_column = int(round(anterior_photo + progress * (posterior_photo - anterior_photo)))
        photo_column = min(photo_columns, key=lambda value: abs(value - photo_column))
        dv_center = (lateral_profiles[photo_column][0] + lateral_profiles[photo_column][1]) * 0.5 - lateral_center
        lr_center = (dorsal_profiles[photo_column][0] + dorsal_profiles[photo_column][1]) * 0.5 - dorsal_center
        row_center = float(np.median(contour[:, 0]))
        column_center = (shape[1] - 1) * 0.5
        sample_count = 85 if 1560 <= number <= 2040 else 40
        sampled = sample_rows(contour, sample_count, 63168 + number)
        contour_points.append(np.column_stack((
            np.full(len(sampled), float(photo_column), dtype=np.float32),
            dv_center - (sampled[:, 0] - row_center) * photo_registration_scale,
            lr_center + (sampled[:, 1] - column_center) * photo_registration_scale,
        )))

    raw = np.concatenate(contour_points, axis=0)
    points = normalize(sample_rows(raw, 5000, 63168))
    edges = knn_edges(points, 4, 2.45)
    combined = hashlib.sha256("\n".join(checksums).encode()).hexdigest()
    return points, edges, {
        "rawSource": "whole-brain photograph and serial JPEG atlas plates; source imagery is not redistributed",
        "rawSha256Combined": combined,
        "wholeBrainSha256": sha256(whole_brain),
        "atlasPageSha256": sha256(atlas_page),
        "specimen": "American Beaver (Castor canadensis) #63-168",
        "plane": "coronal",
        "stain": "thionin cell-body series",
        "indexedSectionCount": len(section_numbers),
        "sectionCount": len(section_contours),
        "sectionRange": [section_numbers[0], section_numbers[-1]],
        "encephalonSectionRange": [first_section, encephalon_section_limit],
        "unavailableSourceSections": unavailable_sections,
        "documentedSectionThicknessMicrons": [25, 40],
        "photoEnvelopeColumns": [anterior_photo, posterior_photo],
        "photoRegistrationScale": round(photo_registration_scale, 8),
        "reconstruction": "spinal-cord exclusion, whole-brain photo calibration, shared-scale serial contours, cerebellar folia oversampling, uniform final scaling",
        "credit": "Comparative Mammalian Brain Collections; source images produced with National Science Foundation support",
    }


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
    parser.add_argument("--only", choices=("elegans", "drosophila", "rodent", "macaque", "human"))
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
    static_entries: list[dict[str, object]] = [
        {
            "id": "gradient", "kind": "procedural-field", "name": "Excitable neural gradient",
            "space": None, "reason": "A neural gradient is a mechanism, not an atlas-bearing organism.",
        },
        {
            "id": "fungal", "kind": "procedural-network", "name": "Adaptive mycelial graph",
            "space": None, "reason": "Fungi have distributed signalling but no canonical neural template.",
        },
    ]
    existing_path = public / "templates.json"
    if args.only and existing_path.exists():
        entries = json.loads(existing_path.read_text(encoding="utf-8"))["entries"]
    else:
        entries = static_entries
    selected = {args.only: builders[args.only]} if args.only else builders
    for key, builder in selected.items():
        print(f"Deriving {key} geometry…", flush=True)
        points, edges, extra = builder(ref)
        entry = {"id": key, "kind": "source-derived", **SOURCES[key], **encode_geometry(points, edges), **extra}
        entries = [candidate for candidate in entries if candidate["id"] != key]
        insert_index = ("gradient", "fungal", "elegans", "drosophila", "rodent", "macaque", "human", "ai").index(key)
        entries.insert(min(insert_index, len(entries)), entry)
        print(f"  {len(points):,} points / {len(edges):,} edges", flush=True)
    if not any(entry["id"] == "ai" for entry in entries):
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
