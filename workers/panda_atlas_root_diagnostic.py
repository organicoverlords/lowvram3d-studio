"""Bounded panda UV-atlas support diagnostic.

This worker measures the production texel-centre owner contract and compares it
with a separate positive-area conservative occupancy pass.  Conservative-only
cells are reported as VISIBLE_SOURCE_GAP candidates; they are never promoted to
direct evidence and this script does not write a production GLB.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import platform
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from atlas_raster import rasterise
from conservative_atlas import (
    chart_local_gutter,
    conservative_coverage,
    derive_uv_chart_ids,
    resolve_conservative_support,
)
from fast_texture_projection import bind_texture, triangle_coverage_mask
from mesh_io import read_glb

AREA_BINS = (
    ("zero", 0.0, 0.0),
    ("0_to_0.25", 0.0, 0.25),
    ("0.25_to_0.5", 0.25, 0.5),
    ("0.5_to_1", 0.5, 1.0),
    ("1_to_2", 1.0, 2.0),
    ("2_to_4", 2.0, 4.0),
    ("4_to_16", 4.0, 16.0),
    ("16_plus", 16.0, float("inf")),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def triangle_areas(uv: np.ndarray, tris: np.ndarray, size: int) -> np.ndarray:
    corners = np.asarray(uv, np.float64)[np.asarray(tris, np.int64)] * float(size)
    return 0.5 * np.abs(
        (corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
        - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0])
    )


def stable_owner_rgb(owner: np.ndarray) -> np.ndarray:
    ids = owner.astype(np.int64, copy=False)
    valid = ids >= 0
    values = ids.astype(np.uint64, copy=False)
    rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
    rgb[..., 0] = ((values * np.uint64(73) + np.uint64(41)) & np.uint64(255)).astype(np.uint8)
    rgb[..., 1] = ((values * np.uint64(151) + np.uint64(97)) & np.uint64(255)).astype(np.uint8)
    rgb[..., 2] = ((values * np.uint64(199) + np.uint64(17)) & np.uint64(255)).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def save_mask(path: Path, mask: np.ndarray, foreground: tuple[int, int, int]) -> None:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[np.asarray(mask, dtype=bool)] = foreground
    Image.fromarray(image, mode="RGB").save(path)


def save_support_map(path: Path, center_owner: np.ndarray, conservative_claims: np.ndarray) -> None:
    center = center_owner >= 0
    conservative = conservative_claims > 0
    ambiguous = conservative_claims > 1
    image = np.zeros((*center.shape, 3), dtype=np.uint8)
    image[center] = (40, 210, 90)
    image[conservative & ~center] = (245, 165, 35)
    image[ambiguous & ~center] = (225, 40, 220)
    Image.fromarray(image, mode="RGB").save(path)


def save_histogram(path: Path, counts: list[tuple[str, int]]) -> None:
    width, height = 960, 500
    margin_left, margin_bottom, margin_top = 220, 70, 40
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    maximum = max((count for _, count in counts), default=1)
    row_height = (height - margin_top - margin_bottom) / max(len(counts), 1)
    for index, (label, count) in enumerate(counts):
        y0 = int(margin_top + index * row_height + 5)
        y1 = int(margin_top + (index + 1) * row_height - 5)
        draw.text((10, y0 + 2), label, fill=(20, 20, 20))
        bar_width = int((width - margin_left - 40) * count / max(maximum, 1))
        draw.rectangle((margin_left, y0, margin_left + bar_width, y1), fill=(70, 120, 190))
        draw.text((margin_left + bar_width + 8, y0 + 2), f"{count:,}", fill=(20, 20, 20))
    draw.text((10, height - 34), "Triangle count by analytic UV area in atlas texels²", fill=(20, 20, 20))
    canvas.save(path)


def diagnose(mesh: Path, out_dir: Path, size: int, *, write_support_contract: bool = False) -> dict:
    size_dir = out_dir / f"atlas_{size}"
    size_dir.mkdir(parents=True, exist_ok=True)
    positions, normals, uv, tris = read_glb(mesh)
    if uv is None:
        raise RuntimeError("PANDA_ATLAS_DIAGNOSTIC_NO_UV")

    owner, _weights = rasterise(uv, tris, size)
    valid_owner = owner[owner >= 0].astype(np.int64, copy=False)
    center_counts = np.bincount(valid_owner, minlength=len(tris)).astype(np.int64)
    areas = triangle_areas(uv, tris, size)
    positive = areas > 1e-12
    zero_center = positive & (center_counts == 0)
    zero_ids = np.flatnonzero(zero_center)

    conservative = conservative_coverage(uv, tris, size, triangle_ids=zero_ids)
    recovered = zero_center & (conservative.claims_per_triangle > 0)
    conservative_any = conservative.claim_count > 0
    center_any = owner >= 0
    conservative_only = conservative_any & ~center_any
    collision = conservative.claim_count > 1
    touches_other_owner = conservative_any & center_any

    support = None
    chart_ids = None
    chart_report = None
    if write_support_contract:
        chart_ids, chart_report = derive_uv_chart_ids(uv, tris)
        support = resolve_conservative_support(uv, tris, size, owner, chart_ids, triangle_ids=zero_ids)
        np.save(size_dir / "chart_id_per_triangle.npy", chart_ids)
        np.save(size_dir / "chart_id_per_direct_texel.npy", np.where(owner >= 0, chart_ids[np.maximum(owner, 0)], -1))
        np.save(size_dir / "conservative_owner.npy", support.owner)
        np.save(size_dir / "conservative_barycentric.npy", support.barycentric)
        np.save(size_dir / "conservative_chart_id.npy", support.chart_id)
        np.save(size_dir / "conservative_collision.npy", support.collision)
        np.save(size_dir / "conservative_same_chart_ambiguous.npy", support.same_chart_ambiguous)
        source_owner = owner.copy()
        source_chart = np.where(owner >= 0, chart_ids[np.maximum(owner, 0)], -1).astype(np.int32)
        support_mask = support.owner >= 0
        source_owner[support_mask] = support.owner[support_mask]
        source_chart[support_mask] = support.chart_id[support_mask]
        gutter_owner, gutter_chart, gutter_collision = chart_local_gutter(source_owner, source_chart, radius=8)
        gutter_mask = (owner < 0) & (support.owner < 0) & (gutter_owner >= 0) & ~gutter_collision
        np.save(size_dir / "chart_local_gutter_owner.npy", gutter_owner)
        np.save(size_dir / "chart_local_gutter_chart_id.npy", gutter_chart)
        np.save(size_dir / "chart_local_gutter_mask.npy", gutter_mask)
        np.save(size_dir / "chart_local_gutter_collision.npy", gutter_collision)
        (size_dir / "chart_inventory.json").write_text(json.dumps(chart_report, indent=2), encoding="utf-8")
        support_image = np.zeros((size, size, 3), dtype=np.uint8)
        support_image[owner >= 0] = (40, 200, 70)
        support_image[(support.owner >= 0)] = (240, 150, 30)
        support_image[support.collision] = (220, 40, 40)
        support_image[gutter_mask] = (40, 100, 230)
        support_image[(owner < 0) & (support.owner < 0) & ~support.collision] = (220, 0, 220)
        Image.fromarray(support_image, mode="RGB").save(size_dir / "support_class_atlas.png")
        # Bind the class atlas to the exact mesh for a clean-material synthetic proof.
        png_buffer = io.BytesIO()
        Image.fromarray(support_image, mode="RGB").save(png_buffer, format="PNG")
        covered_triangles = np.zeros(len(tris), dtype=bool)
        direct_triangles = owner[owner >= 0].astype(np.int64, copy=False)
        support_triangles = support.owner[support.owner >= 0].astype(np.int64, copy=False)
        if direct_triangles.size:
            covered_triangles[np.unique(direct_triangles)] = True
        if support_triangles.size:
            covered_triangles[np.unique(support_triangles)] = True
        bind_texture(mesh, size_dir / "support_class.glb", png_buffer.getvalue(), covered_triangles)
        support_summary = {
            "chart_count": int(chart_report["chart_count"]),
            "direct_texels": int((owner >= 0).sum()),
            "conservative_support_texels": int((support.owner >= 0).sum()),
            "same_chart_ambiguous_texels": int(support.same_chart_ambiguous.sum()),
            "cross_chart_collision_texels": int(support.collision.sum()),
            "unresolved_texels": int(((owner < 0) & (support.owner < 0)).sum()),
            "chart_local_gutter_texels": int(gutter_mask.sum()),
            "cross_chart_gutter_collisions": int(gutter_collision.sum()),
            "direct_owner_changed": 0,
            "conservative_provenance_class": "CONSERVATIVE_SURFACE_SUPPORT",
            "cross_chart_support_assignments": 0,
            "visible_uninitialized_sampling": "synthetic render gate required",
        }
        (size_dir / "support_report.json").write_text(json.dumps(support_summary, indent=2), encoding="utf-8")

    histogram: list[tuple[str, int]] = []
    for label, lower, upper in AREA_BINS:
        if label == "zero":
            selected = areas <= 1e-12
        else:
            selected = (areas > lower) & (areas <= upper)
        histogram.append((label, int(selected.sum())))

    Image.fromarray(stable_owner_rgb(owner), mode="RGB").save(size_dir / "center_owner.png")
    centroid_image = np.zeros((size, size, 3), dtype=np.uint8)
    if zero_ids.size:
        centroids = np.asarray(uv, np.float64)[np.asarray(tris, np.int64)[zero_ids]].mean(axis=1)
        cx = np.clip(np.floor(centroids[:, 0] * size).astype(np.int64), 0, size - 1)
        cy = np.clip(np.floor(centroids[:, 1] * size).astype(np.int64), 0, size - 1)
        centroid_image[cy, cx] = (255, 80, 40)
    Image.fromarray(centroid_image, mode="RGB").save(size_dir / "zero_center_triangle_centroids.png")
    save_mask(size_dir / "conservative_only.png", conservative_only, (245, 165, 35))
    save_mask(size_dir / "conservative_collision.png", collision, (225, 40, 220))
    save_support_map(size_dir / "support_classes.png", owner, conservative.claim_count)
    save_histogram(size_dir / "uv_area_histogram.png", histogram)

    with (size_dir / "uv_area_histogram.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["area_bin_texels_squared", "triangle_count"])
        writer.writerows(histogram)

    zero_records = np.column_stack([
        zero_ids,
        areas[zero_ids],
        conservative.claims_per_triangle[zero_ids],
    ]) if zero_ids.size else np.empty((0, 3))
    np.savetxt(
        size_dir / "zero_center_triangles.csv",
        zero_records,
        delimiter=",",
        header="triangle_id,analytic_uv_area_texels_squared,conservative_positive_area_cells",
        comments="",
        fmt=["%d", "%.9g", "%d"],
    )

    summary = {
        "schema": "panda_atlas_root_diagnostic_v1",
        "atlas_size": int(size),
        "triangle_count": int(len(tris)),
        "vertex_count": int(len(positions)),
        "analytic_positive_area_triangles": int(positive.sum()),
        "degenerate_uv_triangles": int((~positive).sum()),
        "center_owned_texels": int(center_any.sum()),
        "center_supported_triangles": int((center_counts > 0).sum()),
        "center_support_zero_positive_area_triangles": int(zero_center.sum()),
        "center_support_zero_area_fraction": float(areas[zero_center].sum() / max(areas[positive].sum(), 1e-12)),
        "zero_center_triangles_recovered_by_positive_area_cell_intersection": int(recovered.sum()),
        "zero_center_triangles_not_recovered": int((zero_center & ~recovered).sum()),
        "conservative_only_texels": int(conservative_only.sum()),
        "conservative_ambiguous_texels": int(collision.sum()),
        "conservative_cells_touching_another_triangle_center_owner": int(touches_other_owner.sum()),
        "maximum_conservative_claims_on_one_texel": int(conservative.claim_count.max()) if conservative.claim_count.size else 0,
        "center_contract": "exact texel-centre owner; unchanged",
        "conservative_contract": "positive-area triangle/pixel-cell intersection for diagnostic occupancy only",
        "conservative_provenance_class": "VISIBLE_SOURCE_GAP_CANDIDATE_NOT_DIRECT_EVIDENCE",
        "production_asset_written": False,
        "production_promotion_authorized": False,
        "support_contract_written": bool(write_support_contract),
        "area_histogram": {label: count for label, count in histogram},
    }
    if support is not None:
        summary.update({
            "uv_chart_count": int(chart_report["chart_count"]),
            "conservative_support_texels": int((support.owner >= 0).sum()),
            "same_chart_ambiguous_texels": int(support.same_chart_ambiguous.sum()),
            "cross_chart_collision_texels": int(support.collision.sum()),
            "unresolved_texels": int(((owner < 0) & (support.owner < 0)).sum()),
            "chart_local_gutter_texels": int(gutter_mask.sum()),
            "cross_chart_gutter_collisions": int(gutter_collision.sum()),
            "direct_owner_changed_by_conservative_pass": 0,
            "cross_chart_support_assignments": 0,
        })
    (size_dir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--write-support-contract", action="store_true")
    args = parser.parse_args()
    args.mesh = args.mesh.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.mesh.is_file():
        raise FileNotFoundError(args.mesh)

    summaries = [diagnose(args.mesh, args.out_dir, size, write_support_contract=args.write_support_contract) for size in args.sizes]
    root = {
        "schema": "panda_atlas_root_diagnostic_bundle_v1",
        "mesh": str(args.mesh),
        "mesh_sha256": sha256(args.mesh),
        "python": sys.version,
        "platform": platform.platform(),
        "sizes": summaries,
        "classification": "DIAGNOSTIC_ONLY_NOT_PRODUCTION_READY",
        "recommended_pipeline_boundary": {
            "direct_observation": "retain exact texel-centre ownership and exact triangle ID",
            "gap_detection": "use separate conservative occupancy only to identify atlas-owned unsampled cells",
            "gap_repair": "same-triangle or same-chart constrained completion; preserve provenance",
            "forbidden": "do not relabel conservative-only cells as ORIGINAL_DIRECT or GENERATED_OBSERVED",
            "support_contract": "direct owner is immutable; conservative support is separate provenance; cross-chart collisions stay unresolved",
        },
    }
    (args.out_dir / "diagnostic_bundle.json").write_text(json.dumps(root, indent=2), encoding="utf-8")
    print(json.dumps(root, indent=2))


if __name__ == "__main__":
    main()
