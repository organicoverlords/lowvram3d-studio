from __future__ import annotations

import argparse
import math
from pathlib import Path

import bpy
import numpy as np

from common import (
    argv_after_double_dash,
    export_glb,
    extended_mesh_stats,
    import_mesh,
    reset_scene,
    save_json,
    select_only,
)


def ensure_uv(obj: bpy.types.Object, name: str = "UVMap") -> None:
    if name not in obj.data.uv_layers:
        obj.data.uv_layers.new(name=name)
    obj.data.uv_layers.active = obj.data.uv_layers[name]


def smart_project(objects: list[bpy.types.Object], margin: float, per_object: bool, angle_degrees: float, area_weight: float) -> None:
    groups = [[obj] for obj in objects] if per_object else [objects]
    for group in groups:
        for obj in group:
            ensure_uv(obj)
        select_only(group)
        bpy.context.view_layer.objects.active = group[0]
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(
            angle_limit=math.radians(max(1.0, min(89.0, angle_degrees))),
            island_margin=margin,
            area_weight=max(0.0, min(1.0, area_weight)),
            correct_aspect=True,
            scale_to_bounds=False,
        )
        bpy.ops.object.mode_set(mode="OBJECT")


def make_lightmap_uv(objects: list[bpy.types.Object], margin_divisor: float) -> list[str]:
    created: list[str] = []
    for obj in objects:
        layer = obj.data.uv_layers.get("LightmapUV") or obj.data.uv_layers.new(name="LightmapUV")
        obj.data.uv_layers.active = layer
        select_only([obj])
        bpy.context.view_layer.objects.active = obj
        try:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.lightmap_pack(
                PREF_CONTEXT="ALL_FACES",
                PREF_PACK_IN_ONE=True,
                PREF_NEW_UVLAYER=False,
                PREF_BOX_DIV=12,
                PREF_MARGIN_DIV=margin_divisor,
            )
            bpy.ops.object.mode_set(mode="OBJECT")
            created.append(obj.name)
        except Exception:
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:
                pass
        obj.data.uv_layers.active = obj.data.uv_layers.get("UVMap") or obj.data.uv_layers[0]
    return created


def uv_triangles(objects: list[bpy.types.Object]) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    triangles = []
    for obj in objects:
        uv_layer = obj.data.uv_layers.active
        if not uv_layer:
            continue
        obj.data.calc_loop_triangles()
        for triangle in obj.data.loop_triangles:
            coords = [tuple(float(value) for value in uv_layer.data[index].uv) for index in triangle.loops]
            triangles.append((coords[0], coords[1], coords[2]))
    return triangles


def triangle_area(triangle) -> float:
    (ax, ay), (bx, by), (cx, cy) = triangle
    return abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) * 0.5


def point_in_triangle(px: float, py: float, triangle) -> bool:
    (ax, ay), (bx, by), (cx, cy) = triangle
    v0x, v0y = cx - ax, cy - ay
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = px - ax, py - ay
    denominator = v0x * v1y - v1x * v0y
    if abs(denominator) < 1e-12:
        return False
    u = (v2x * v1y - v1x * v2y) / denominator
    v = (v0x * v2y - v2x * v0y) / denominator
    return u >= 0 and v >= 0 and u + v <= 1


def estimate_uv_metrics(triangles, resolution: int = 128, max_triangles: int = 50_000) -> dict:
    total = len(triangles)
    stride = max(1, math.ceil(total / max_triangles))
    sampled = triangles[::stride]
    counts = np.zeros((resolution, resolution), dtype=np.uint16)
    degenerate = 0
    outside = 0
    summed_area = 0.0
    for triangle in sampled:
        area = triangle_area(triangle)
        if area < 1e-12:
            degenerate += 1
            continue
        summed_area += area
        if any(u < -1e-4 or u > 1.0001 or v < -1e-4 or v > 1.0001 for u, v in triangle):
            outside += 1
        min_x = max(0, int(math.floor(min(p[0] for p in triangle) * resolution)))
        max_x = min(resolution - 1, int(math.ceil(max(p[0] for p in triangle) * resolution)))
        min_y = max(0, int(math.floor(min(p[1] for p in triangle) * resolution)))
        max_y = min(resolution - 1, int(math.ceil(max(p[1] for p in triangle) * resolution)))
        for y in range(min_y, max_y + 1):
            py = (y + 0.5) / resolution
            for x in range(min_x, max_x + 1):
                px = (x + 0.5) / resolution
                if point_in_triangle(px, py, triangle):
                    counts[y, x] = min(65535, counts[y, x] + 1)
    occupied = int(np.count_nonzero(counts))
    overlap = int(np.count_nonzero(counts > 1))
    return {
        "triangles_total": total,
        "triangles_sampled": len(sampled),
        "sample_stride": stride,
        "summed_uv_area_sample": summed_area,
        "degenerate_uv_triangles_sample": degenerate,
        "outside_0_1_triangles_sample": outside,
        "utilization_estimate": occupied / counts.size,
        "overlap_pixel_fraction_estimate": overlap / max(occupied, 1),
        "metric_resolution": resolution,
        "metrics_are_estimates": stride > 1,
    }


def draw_line(canvas: np.ndarray, start, end) -> None:
    height, width, _ = canvas.shape
    x0 = int(round(start[0] * (width - 1)))
    y0 = int(round(start[1] * (height - 1)))
    x1 = int(round(end[0] * (width - 1)))
    y1 = int(round(end[1] * (height - 1)))
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    error = dx - dy
    while True:
        if 0 <= x0 < width and 0 <= y0 < height:
            canvas[height - 1 - y0, x0] = (1.0, 1.0, 1.0, 1.0)
        if x0 == x1 and y0 == y1:
            break
        doubled = error * 2
        if doubled > -dy:
            error -= dy
            x0 += sx
        if doubled < dx:
            error += dx
            y0 += sy


def save_layout(path: str, triangles, resolution: int = 1024) -> None:
    canvas = np.zeros((resolution, resolution, 4), dtype=np.float32)
    canvas[:, :, 3] = 1.0
    stride = max(1, math.ceil(len(triangles) / 100_000))
    for triangle in triangles[::stride]:
        draw_line(canvas, triangle[0], triangle[1])
        draw_line(canvas, triangle[1], triangle[2])
        draw_line(canvas, triangle[2], triangle[0])
    image = bpy.data.images.new("UVLayout", width=resolution, height=resolution, alpha=True)
    image.pixels.foreach_set(canvas.ravel())
    image.filepath_raw = str(Path(path))
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--layout", required=True)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--padding-px", type=int, default=8)
    parser.add_argument("--atlas-mode", choices=("shared", "per_object", "preserve_or_per_object"), default="shared")
    parser.add_argument("--lightmap-uv", action="store_true")
    parser.add_argument("--smart-angle-deg", type=float, default=66.0)
    parser.add_argument("--area-weight", type=float, default=0.0)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.input)
    if not objects:
        raise RuntimeError("No mesh objects imported")
    has_valid_uv = all(obj.data.uv_layers and len(obj.data.uv_layers.active.data) > 0 for obj in objects)
    preserve = args.atlas_mode == "preserve_or_per_object" and has_valid_uv
    if not preserve:
        margin = max(1, args.padding_px) / max(args.texture_size, 1)
        smart_project(objects, margin, per_object=args.atlas_mode != "shared", angle_degrees=args.smart_angle_deg, area_weight=args.area_weight)
    lightmap_objects = make_lightmap_uv(objects, max(0.01, args.texture_size / max(args.padding_px, 1))) if args.lightmap_uv else []
    triangles = uv_triangles(objects)
    metrics = estimate_uv_metrics(triangles)
    save_layout(args.layout, triangles)
    select_only(objects)
    export_glb(args.output)
    errors = []
    if not triangles:
        errors.append("No UV triangles were generated")
    if metrics["degenerate_uv_triangles_sample"] > max(10, metrics["triangles_sampled"] * 0.02):
        errors.append("Too many degenerate UV triangles")
    save_json(
        args.report,
        {
            "success": not errors,
            "backend": "blender_smart_uv",
            "atlas_mode": args.atlas_mode,
            "preserved_existing_uv": preserve,
            "texture_size": args.texture_size,
            "padding_px": args.padding_px,
            "smart_angle_deg": args.smart_angle_deg,
            "area_weight": args.area_weight,
            "metrics": metrics,
            "lightmap_uv_objects": lightmap_objects,
            "mesh_stats": extended_mesh_stats(objects),
            "errors": errors,
            "warnings": ["UV overlap/utilization values are CPU raster estimates, not exact polygon-boolean results."],
        },
    )
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
