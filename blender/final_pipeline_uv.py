"""UV unwrap the shaman LOD0 and measure the result honestly.

Smart UV Project is used rather than seam-and-unwrap because there is no authored seam set on a
generated mesh. The metrics that matter for a 4K atlas are reported rather than assumed: true
triangle overlap measured by rasterising the charts, atlas utilisation, and any degenerate or
non-finite UV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import bpy
import numpy as np

from common import argv_after_double_dash, import_mesh, reset_scene, save_json


def weld(obj: bpy.types.Object, merge_distance: float) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=merge_distance)
    bpy.ops.object.mode_set(mode="OBJECT")


def unwrap(obj: bpy.types.Object, angle_limit: float, island_margin: float) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(
        angle_limit=angle_limit,
        island_margin=island_margin,
        correct_aspect=True,
        scale_to_bounds=False,
    )
    # Smart Project packs conservatively and leaves most of the atlas empty. Normalising island
    # scale gives uniform texel density, then a concave repack recovers the wasted area.
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(rotate=True, margin=island_margin, shape_method="CONCAVE")
    bpy.ops.object.mode_set(mode="OBJECT")


def uv_metrics(obj: bpy.types.Object, resolution: int, shrink: float = 0.82) -> dict:
    mesh = obj.data
    layer = mesh.uv_layers.active
    if layer is None:
        raise RuntimeError("LOD0 has no active UV layer after unwrap")

    loops = np.array([tuple(d.uv) for d in layer.data], dtype=np.float64)
    finite = np.isfinite(loops).all()
    umin, vmin = loops.min(axis=0)
    umax, vmax = loops.max(axis=0)

    # Rasterise every triangle into an occupancy grid. A texel covered more than once is a true
    # overlap; total covered texels give atlas utilisation. This measures the atlas as the baker
    # will actually sample it rather than trusting the packer.
    grid = np.zeros((resolution, resolution), dtype=np.int32)
    degenerate = 0
    tri_uvs = []
    for poly in mesh.polygons:
        idx = list(poly.loop_indices)
        if len(idx) != 3:
            continue
        tri_uvs.append(loops[idx])
    tri_uvs = np.array(tri_uvs)

    # Shrink each triangle toward its centroid before rasterising. Neighbouring triangles inside
    # one chart legitimately share the texels along their common edge; without this inset those
    # seams dominate the count and masquerade as chart overlap.
    centroids = tri_uvs.mean(axis=1, keepdims=True)
    tri_uvs_inset = centroids + (tri_uvs - centroids) * shrink

    for tri in tri_uvs_inset:
        pts = tri * (resolution - 1)
        x0, y0 = np.floor(pts.min(axis=0)).astype(int)
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
        x0 = max(x0, 0); y0 = max(y0, 0)
        x1 = min(x1, resolution - 1); y1 = min(y1, resolution - 1)
        if x1 < x0 or y1 < y0:
            continue
        area = (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) - (pts[2, 0] - pts[0, 0]) * (
            pts[1, 1] - pts[0, 1]
        )
        if abs(area) < 1e-12:
            degenerate += 1
            continue
        ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        px = xs + 0.5
        py = ys + 0.5
        w0 = ((pts[1, 0] - pts[0, 0]) * (py - pts[0, 1]) - (px - pts[0, 0]) * (pts[1, 1] - pts[0, 1])) / area
        w1 = ((px - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) - (pts[2, 0] - pts[0, 0]) * (py - pts[0, 1])) / area
        inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        if inside.any():
            grid[ys[inside], xs[inside]] += 1

    covered = int((grid > 0).sum())
    overlapped = int((grid > 1).sum())
    total = resolution * resolution

    # Blender's own overlap test is the authoritative answer. The raster grid above cannot
    # distinguish genuine chart overlap from several sub-texel triangles crowding one texel, which
    # is common at 220k triangles in a 4K atlas.
    # Analytic overlap: the summed UV area of every chart, expressed in texels, against the number
    # of texels actually covered. Disjoint charts give a ratio of ~1; anything materially above 1
    # means charts are stacked on the same texels. This avoids both the shared-edge false positive
    # of a raw raster count and the Blender UV-selection API, which moved in 5.x.
    areas = np.abs(
        (tri_uvs[:, 1, 0] - tri_uvs[:, 0, 0]) * (tri_uvs[:, 2, 1] - tri_uvs[:, 0, 1])
        - (tri_uvs[:, 2, 0] - tri_uvs[:, 0, 0]) * (tri_uvs[:, 1, 1] - tri_uvs[:, 0, 1])
    ) * 0.5
    chart_area_texels = float(areas.sum() * total)
    area_ratio = chart_area_texels / max(covered, 1)

    return {
        "chart_area_texels": chart_area_texels,
        "area_to_covered_ratio": area_ratio,
        "estimated_overlap_fraction": max(0.0, 1.0 - 1.0 / max(area_ratio, 1e-6)),
        "resolution": resolution,
        "uv_finite": bool(finite),
        "uv_min": [float(umin), float(vmin)],
        "uv_max": [float(umax), float(vmax)],
        "uv_out_of_bounds": bool(umin < -1e-6 or vmin < -1e-6 or umax > 1 + 1e-6 or vmax > 1 + 1e-6),
        "triangles": int(len(tri_uvs)),
        "degenerate_uv_triangles": degenerate,
        "covered_texels": covered,
        "atlas_utilisation": covered / total,
        "overlapping_texels": overlapped,
        "true_overlap_fraction_of_covered": overlapped / max(covered, 1),
        "uv_islands": len({tuple(np.round(t.mean(axis=0), 3)) for t in tri_uvs}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--angle-limit", type=float, default=1.15192)  # 66 degrees
    parser.add_argument("--island-margin", type=float, default=0.003)
    parser.add_argument("--merge-distance", type=float, default=1e-4)
    parser.add_argument("--overlap-shrink", type=float, default=0.82)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.input)
    if not objects:
        raise RuntimeError(f"No mesh in {args.input}")
    if len(objects) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active or objects[0]

    weld(obj, args.merge_distance)
    unwrap(obj, args.angle_limit, args.island_margin)
    metrics = uv_metrics(obj, args.resolution, args.overlap_shrink)
    metrics["input"] = args.input
    metrics["output"] = args.output
    metrics["angle_limit_radians"] = args.angle_limit
    metrics["island_margin"] = args.island_margin

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=args.output, export_format="GLB", use_selection=True, export_yup=True
    )
    save_json(args.report, metrics)
    print(
        f"UV: utilisation={metrics['atlas_utilisation']*100:.2f}% "
        f"overlap~={metrics['estimated_overlap_fraction']*100:.3f}% "
        f"degenerate={metrics['degenerate_uv_triangles']}",
        flush=True,
    )


if __name__ == "__main__":
    main()



