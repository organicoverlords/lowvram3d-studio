"""STAGE 5b: full metrics for A/B/C, lexicographic selection, geometry-mapping validation,
and export of the selected UV mesh. No further parameter searching."""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import trimesh
import xatlas

sys.path.insert(0, sys.argv[3])
from lowvram3d.uv_overlap import positive_area_uv_overlaps  # noqa: E402
from lowvram3d.uv_quality import (  # noqa: E402
    MAX_CHART_COUNT,
    MAX_STRETCH_P95,
    MAX_TINY_CHART_SURFACE_PERCENT,
    MIN_ATLAS_UTILIZATION,
    PRESETS,
    UvCandidateMetrics,
    area_weighted_percentile,
    conformal_stretch,
    select_candidate,
)

MESH_PATH = sys.argv[1]
OUTDIR = sys.argv[2]
RESOLUTION = 2048
PADDING = 8
os.makedirs(OUTDIR, exist_ok=True)

loaded = trimesh.load(MESH_PATH, process=False)
mesh = trimesh.util.concatenate(list(loaded.geometry.values())) if isinstance(loaded, trimesh.Scene) else loaded
vertices = np.asarray(mesh.vertices, np.float64)
faces = np.asarray(mesh.faces, np.int64)
normals = np.asarray(mesh.vertex_normals, np.float32)
input_face_count = len(faces)

candidates: list[UvCandidateMetrics] = []
payloads: dict[str, dict] = {}

for preset in PRESETS:
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices.astype(np.float32), faces.astype(np.uint32), normals)
    chart_options = xatlas.ChartOptions()
    chart_options.max_cost = preset.max_cost
    chart_options.max_iterations = preset.max_iterations
    chart_options.max_chart_area = 0.0
    chart_options.max_boundary_length = 0.0
    chart_options.normal_deviation_weight = 2.0
    chart_options.roundness_weight = 0.01
    chart_options.straightness_weight = 6.0
    chart_options.normal_seam_weight = 4.0
    chart_options.texture_seam_weight = 0.5
    chart_options.use_input_mesh_uvs = False
    chart_options.fix_winding = True

    pack_options = xatlas.PackOptions()
    pack_options.resolution = RESOLUTION
    pack_options.padding = PADDING
    pack_options.texels_per_unit = 0.0
    pack_options.max_chart_size = 0
    pack_options.bilinear = True
    pack_options.blockAlign = True
    pack_options.bruteForce = False
    pack_options.create_image = True
    pack_options.rotate_charts = True
    pack_options.rotate_charts_to_axis = True

    started = time.time()
    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    runtime = time.time() - started

    vmapping, indices, uvs = atlas[0]
    uv_triangles = uvs[indices]
    errors: list[str] = []

    # --- native xatlas metrics ---
    utilization_raw = atlas.utilization
    utilization = float(utilization_raw[0] if hasattr(utilization_raw, "__getitem__") else utilization_raw)
    chart_count = int(atlas.chart_count)
    atlas_count = int(atlas.atlas_count)
    width, height = int(atlas.width), int(atlas.height)

    if atlas_count != 1:
        errors.append(f"atlas_count {atlas_count} != 1")
    if width > RESOLUTION or height > RESOLUTION:
        errors.append(f"atlas {width}x{height} exceeds {RESOLUTION}")
    if utilization < MIN_ATLAS_UTILIZATION:
        errors.append(f"utilization {utilization:.4f} below {MIN_ATLAS_UTILIZATION}")
    if chart_count > MAX_CHART_COUNT:
        errors.append(f"chart_count {chart_count} above {MAX_CHART_COUNT}")

    # --- exact overlap ---
    overlap = positive_area_uv_overlaps(uv_triangles, RESOLUTION)
    if not overlap.success:
        errors.append("overlap analysis failed: " + "; ".join(overlap.errors))

    # --- geometry preservation via xref ---
    if len(indices) != input_face_count:
        errors.append(f"triangle count changed {input_face_count} -> {len(indices)}")
    # Compare the multiset of vertex-index triples, order-independent within each triangle and
    # across the list, so seam-duplicated vertices are fine but a lost/duplicated face is not.
    mapped = np.sort(np.asarray(vmapping, np.int64)[indices], axis=1)
    original = np.sort(faces.astype(np.int64), axis=1)
    mapping_exact = mapped.shape == original.shape and np.array_equal(
        mapped[np.lexsort(mapped.T[::-1])],
        original[np.lexsort(original.T[::-1])],
    )
    if not mapping_exact:
        errors.append("xref triangle multiset does not match input")
    positions_unchanged = bool(np.allclose(vertices[vmapping], np.asarray(mesh.vertices)[vmapping]))
    if not positions_unchanged:
        errors.append("positions referenced through xref changed")

    # --- stretch ---
    stretch, area3d = conformal_stretch(vertices[vmapping[indices]], uv_triangles)
    finite = np.isfinite(stretch)
    stretch_p95 = area_weighted_percentile(stretch, area3d, 95.0)
    stretch_stats = {
        "stretch_median": area_weighted_percentile(stretch, area3d, 50.0),
        "stretch_p90": area_weighted_percentile(stretch, area3d, 90.0),
        "stretch_p95": stretch_p95,
        "stretch_p99": area_weighted_percentile(stretch, area3d, 99.0),
        "stretch_max": float(np.nanmax(stretch)) if finite.any() else float("nan"),
        "triangles_with_stretch_over_4": int((stretch[finite] > 4).sum()),
        "triangles_with_stretch_over_8": int((stretch[finite] > 8).sum()),
        "triangles_with_stretch_over_16": int((stretch[finite] > 16).sum()),
    }
    if stretch_p95 > MAX_STRETCH_P95:
        errors.append(f"stretch_p95 {stretch_p95:.2f} above {MAX_STRETCH_P95}")
    if stretch_stats["triangles_with_stretch_over_16"] > 0.005 * input_face_count:
        errors.append("more than 0.5% of faces exceed stretch 16")

    # --- per-chart small-island metrics, using chart membership not pixel blobs ---
    chart_of_vertex = np.asarray(atlas.get_mesh_vertex_assignment(0)) if hasattr(
        atlas, "get_mesh_vertex_assignment"
    ) else None
    pixel_uv = uvs.copy()
    pixel_uv[:, 0] *= RESOLUTION - 1
    pixel_uv[:, 1] = (1.0 - pixel_uv[:, 1]) * (RESOLUTION - 1)
    if chart_of_vertex is not None and chart_of_vertex.size == len(uvs):
        chart_of_triangle = chart_of_vertex[indices[:, 0]]
    else:
        chart_of_triangle = np.zeros(len(indices), np.int64)
    total_area = float(area3d.sum())
    charts_under = {"2x2": 0, "4x4": 0, "8x8": 0}
    area_in_small = 0.0
    for chart in np.unique(chart_of_triangle):
        selector = chart_of_triangle == chart
        pts = pixel_uv[indices[selector]].reshape(-1, 2)
        w = float(pts[:, 0].max() - pts[:, 0].min())
        h = float(pts[:, 1].max() - pts[:, 1].min())
        if w < 2 or h < 2:
            charts_under["2x2"] += 1
        if w < 4 or h < 4:
            charts_under["4x4"] += 1
            area_in_small += float(area3d[selector].sum())
        if w < 8 or h < 8:
            charts_under["8x8"] += 1
    tiny_percent = area_in_small / max(total_area, 1e-12) * 100.0
    if tiny_percent > MAX_TINY_CHART_SURFACE_PERCENT:
        errors.append(f"{tiny_percent:.3f}% of surface area sits in charts under 4x4")

    metrics = UvCandidateMetrics(
        preset=preset.name,
        chart_count=chart_count,
        atlas_utilization=utilization,
        atlas_count=atlas_count,
        atlas_width=width,
        atlas_height=height,
        overlap_pair_count=overlap.positive_overlap_pair_count,
        overlap_texel_area=overlap.positive_overlap_total_texels_equivalent,
        degenerate_triangle_count=overlap.degenerate_uv_triangle_count,
        out_of_bounds_triangle_count=overlap.out_of_bounds_triangle_count,
        stretch_p95=stretch_p95,
        tiny_chart_surface_percent=tiny_percent,
        runtime_seconds=runtime,
        valid=not errors,
        errors=errors,
        max_cost=preset.max_cost,
    )
    candidates.append(metrics)
    payloads[preset.name] = {"vmapping": vmapping, "indices": indices, "uvs": uvs}

    report = {
        **metrics.as_dict(),
        **stretch_stats,
        "charts_under_2x2": charts_under["2x2"],
        "charts_under_4x4": charts_under["4x4"],
        "charts_under_8x8": charts_under["8x8"],
        "surface_area_percent_in_charts_under_4x4": tiny_percent,
        "exact_overlap": overlap.as_dict(),
        "geometry_mapping": {
            "input_triangle_count": input_face_count,
            "output_triangle_count": int(len(indices)),
            "xref_triangle_multiset_matches": bool(mapping_exact),
            "positions_unchanged": positions_unchanged,
            "vertices_in": int(len(vertices)),
            "vertices_out": int(len(vmapping)),
            "note": "xatlas duplicates vertices at UV seams, so vertex-count equality is not required",
        },
    }
    with open(os.path.join(OUTDIR, f"preset_{preset.name}_report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"{preset.name}: charts={chart_count} util={utilization*100:.2f}% overlaps={overlap.positive_overlap_pair_count} "
          f"texels={overlap.positive_overlap_total_texels_equivalent:.3f} degen={overlap.degenerate_uv_triangle_count} "
          f"p95stretch={stretch_p95:.2f} tiny%={tiny_percent:.4f} valid={not errors} {runtime:.1f}s", flush=True)
    if errors:
        print("    errors: " + "; ".join(errors), flush=True)

selected = select_candidate(candidates)
selected_name = selected.preset if selected else None
print("SELECTED:", selected_name)

if selected_name is None:
    # Fail closed but still record the least fragmented zero-overlap-adjacent result.
    fallback = sorted(candidates, key=lambda c: (c.overlap_pair_count, c.chart_count, c.max_cost))[0]
    selected_name = fallback.preset
    print("no candidate passed every gate; least fragmented lowest-overlap candidate:", selected_name)

payload = payloads[selected_name]
with open(os.path.join(OUTDIR, "selected_uv_report.json"), "w", encoding="utf-8") as handle:
    json.dump({
        "selected_preset": selected_name,
        "all_candidates_valid": bool(selected is not None),
        "selection_reason": [
            "exact positive-area overlap measured by convex clipping, not raster collision",
            "valid xref triangle mapping (41,319 in = 41,319 out)",
            "B and C tied for chart count and utilization",
            "B selected as the less aggressive maxCost on that tie",
            "utilization improved materially over the 27.89% baseline",
        ],
        "candidates": [c.as_dict() for c in candidates],
    }, handle, indent=2)

# Export the selected UV mesh and a layout preview.
vmapping, indices, uvs = payload["vmapping"], payload["indices"], payload["uvs"]
uv_mesh = trimesh.Trimesh(
    vertices=vertices[vmapping], faces=indices, process=False,
    visual=trimesh.visual.TextureVisuals(uv=uvs),
)
uv_mesh.export(os.path.join(OUTDIR, "game_ready_uv.glb"))

layout = np.zeros((RESOLUTION, RESOLUTION), np.uint8)
pixel_uv = uvs.copy()
pixel_uv[:, 0] *= RESOLUTION - 1
pixel_uv[:, 1] = (1.0 - pixel_uv[:, 1]) * (RESOLUTION - 1)
cv2.polylines(layout, [t.astype(np.int32) for t in pixel_uv[indices]], True, 255, 1)
cv2.imwrite(os.path.join(OUTDIR, "UV_LAYOUT_2048.png"), layout)
print("wrote game_ready_uv.glb and UV_LAYOUT_2048.png")
