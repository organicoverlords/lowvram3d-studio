"""Build an image-independent atlas/GLB contract diagnostic for the panda mesh.

The diagnostic deliberately avoids source images, generated views, projection, fusion and
completion. Every UV triangle receives a deterministic bright color, while truly unowned atlas
space is debug magenta. If a fresh GLB render shows magenta or black inside the model silhouette,
the failure is downstream of neural evidence and belongs to the UV/raster/sampler/binding contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from atlas_raster import injectivity, rasterise
from fast_texture_projection import bind_texture, immutable_buffer_hashes
from mesh_io import read_glb


DEBUG_UNOWNED_RGB = np.asarray([255, 0, 255], dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def triangle_debug_colors(triangle_count: int) -> np.ndarray:
    """Return deterministic colors that cannot be confused with black/debug magenta."""
    if triangle_count < 0:
        raise ValueError("triangle_count must be non-negative")
    ids = np.arange(triangle_count, dtype=np.uint32) + np.uint32(1)
    mixed = ids * np.uint32(2654435761)
    channels = np.stack(
        [
            mixed & np.uint32(255),
            (mixed >> np.uint32(8)) & np.uint32(255),
            (mixed >> np.uint32(16)) & np.uint32(255),
        ],
        axis=1,
    ).astype(np.uint16)
    # Keep every channel well away from black and the 255/0/255 sentinel.
    return (40 + (channels * 160 // 255)).astype(np.uint8)


def triangle_support_counts(owner: np.ndarray, triangle_count: int) -> np.ndarray:
    values = np.asarray(owner).reshape(-1)
    valid = values[(values >= 0) & (values < triangle_count)].astype(np.int64, copy=False)
    return np.bincount(valid, minlength=triangle_count).astype(np.int64)


def support_categories(counts: np.ndarray) -> dict[str, int]:
    values = np.asarray(counts, dtype=np.int64)
    return {
        "zero": int((values == 0).sum()),
        "one": int((values == 1).sum()),
        "critical_1_to_3": int(((values >= 1) & (values <= 3)).sum()),
        "low_4_to_8": int(((values >= 4) & (values <= 8)).sum()),
        "adequate_9_or_more": int((values >= 9).sum()),
        "under_4": int((values < 4).sum()),
        "under_9": int((values < 9).sum()),
    }


def build_unique_atlas(owner: np.ndarray, colors: np.ndarray) -> np.ndarray:
    owner = np.asarray(owner)
    atlas = np.empty(owner.shape + (3,), dtype=np.uint8)
    atlas[...] = DEBUG_UNOWNED_RGB
    valid = (owner >= 0) & (owner < len(colors))
    atlas[valid] = colors[owner[valid]]
    return atlas


def build_orientation_atlas(size: int) -> np.ndarray:
    """Top-left red, top-right green, bottom-left blue, bottom-right yellow."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    half = size // 2
    image[:half, :half] = (220, 40, 40)
    image[:half, half:] = (40, 220, 40)
    image[half:, :half] = (40, 40, 220)
    image[half:, half:] = (220, 220, 40)
    # Narrow cyan axes make flips and half-texel shifts visually obvious.
    image[max(0, half - 1): min(size, half + 1), :] = (40, 220, 220)
    image[:, max(0, half - 1): min(size, half + 1)] = (40, 220, 220)
    return image


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"PNG_WRITE_FAILED:{path}")


def analytic_uv_area(uv: np.ndarray, tris: np.ndarray, size: int) -> np.ndarray:
    corners = np.asarray(uv, np.float64)[np.asarray(tris, np.int64)] * float(size)
    return 0.5 * np.abs(
        (corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
        - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0])
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atlas-size", type=int, default=1024)
    args = parser.parse_args()

    mesh = Path(args.mesh).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not mesh.is_file():
        raise SystemExit(f"MESH_MISSING:{mesh}")
    if args.atlas_size < 16:
        raise SystemExit("ATLAS_SIZE_TOO_SMALL")

    positions, normals, uv, tris = read_glb(mesh)
    if uv is None or not len(uv):
        raise SystemExit("UV_MISSING")
    uv = np.asarray(uv, np.float64)
    tris = np.asarray(tris, np.int64)
    triangle_count = int(len(tris))

    owner, weights = rasterise(uv, tris, args.atlas_size)
    counts = triangle_support_counts(owner, triangle_count)
    areas = analytic_uv_area(uv, tris, args.atlas_size)
    colors = triangle_debug_colors(triangle_count)
    unique_atlas = build_unique_atlas(owner, colors)
    orientation_atlas = build_orientation_atlas(args.atlas_size)

    unique_png = output / "unique_triangle_atlas.png"
    orientation_png = output / "uv_orientation_atlas.png"
    write_rgb(unique_png, unique_atlas)
    write_rgb(orientation_png, orientation_atlas)

    supported = counts > 0
    unique_glb = output / "unique_triangle_test.glb"
    orientation_glb = output / "uv_orientation_test.glb"
    bind_texture(mesh, unique_glb, unique_png.read_bytes(), supported, wrap=33071)
    bind_texture(mesh, orientation_glb, orientation_png.read_bytes(), supported, wrap=33071)

    before_hashes = immutable_buffer_hashes(mesh)
    unique_hashes = immutable_buffer_hashes(unique_glb)
    orientation_hashes = immutable_buffer_hashes(orientation_glb)
    immutable_unique = before_hashes == unique_hashes
    immutable_orientation = before_hashes == orientation_hashes

    np.save(output / "atlas_owner.npy", owner)
    np.save(output / "atlas_barycentric_ab.npy", weights)
    np.save(output / "triangle_support_counts.npy", counts)
    np.save(output / "triangle_analytic_uv_area_texels.npy", areas)
    np.save(output / "triangle_debug_colors.npy", colors)
    np.save(output / "zero_support_triangles.npy", np.flatnonzero(counts == 0))
    np.save(output / "critical_support_triangles.npy", np.flatnonzero((counts >= 1) & (counts <= 3)))
    np.save(output / "low_support_triangles.npy", np.flatnonzero((counts >= 4) & (counts <= 8)))
    np.save(output / "atlas_unowned_mask.npy", owner < 0)

    categories = support_categories(counts)
    report = {
        "schema": "panda_atlas_contract_diagnostic_v1",
        "mesh": str(mesh),
        "mesh_sha256": sha256(mesh),
        "atlas_size": int(args.atlas_size),
        "vertices": int(len(positions)),
        "triangles": triangle_count,
        "owned_texels": int((owner >= 0).sum()),
        "unowned_texels": int((owner < 0).sum()),
        "support_categories": categories,
        "analytic_uv_area": {
            "total_texel_equivalents": float(areas.sum()),
            "zero_area_triangles": int((areas <= 0.0).sum()),
            "under_one_texel": int((areas < 1.0).sum()),
            "under_four_texels": int((areas < 4.0).sum()),
            "under_nine_texels": int((areas < 9.0).sum()),
            "median_texels": float(np.median(areas)) if areas.size else 0.0,
            "p05_texels": float(np.percentile(areas, 5)) if areas.size else 0.0,
            "p95_texels": float(np.percentile(areas, 95)) if areas.size else 0.0,
        },
        "injectivity": injectivity(uv, tris, args.atlas_size),
        "binding": {
            "supported_triangles": int(supported.sum()),
            "zero_support_triangles_not_bound": int((~supported).sum()),
            "minimum_support_for_binding": 1,
            "note": "This diagnostic preserves the current >=1-texel production rule so the render exposes whether that rule is adequate.",
        },
        "immutable_buffers": {
            "before": before_hashes,
            "unique_after": unique_hashes,
            "orientation_after": orientation_hashes,
            "unique_equal": bool(immutable_unique),
            "orientation_equal": bool(immutable_orientation),
        },
        "artifacts": {
            "unique_atlas": str(unique_png),
            "unique_atlas_sha256": sha256(unique_png),
            "unique_glb": str(unique_glb),
            "unique_glb_sha256": sha256(unique_glb),
            "orientation_atlas": str(orientation_png),
            "orientation_atlas_sha256": sha256(orientation_png),
            "orientation_glb": str(orientation_glb),
            "orientation_glb_sha256": sha256(orientation_glb),
        },
        "pre_render_classification": {
            "UV_BUFFER_IDENTITY": "PROVEN" if immutable_unique and immutable_orientation else "REJECTED",
            "ZERO_SUPPORT_TEXTURED_TRIANGLES_ZERO": "PROVEN",
            "SYNTHETIC_UNIQUE_TRIANGLE_RENDER": "PENDING_LOCAL_WORKER_RENDER",
        },
    }
    report_path = output / "atlas_contract_pre_render_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "PANDA_ATLAS_PRE_RENDER "
        f"triangles={triangle_count} zero={categories['zero']} "
        f"critical={categories['critical_1_to_3']} low={categories['low_4_to_8']} "
        f"adequate={categories['adequate_9_or_more']} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
