from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass

import bmesh
import bpy

from common import (
    argv_after_double_dash,
    extended_mesh_stats,
    import_mesh,
    object_center,
    object_dimensions,
    reset_scene,
    save_json,
    triangle_count,
    world_bounds,
)


@dataclass
class ObjectAnalysis:
    name: str
    triangles: int
    vertices: int
    materials: list[str]
    loose_components: int
    dimensions: tuple[float, float, float]
    center: tuple[float, float, float]
    round_candidate: bool
    small_fragment: bool
    fully_inside_larger_bounds: bool = False
    symmetry_partner: str | None = None
    repeat_group: str | None = None


def connected_component_count(obj: bpy.types.Object) -> int:
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        unseen = set(bm.verts)
        count = 0
        while unseen:
            count += 1
            seed = unseen.pop()
            stack = [seed]
            while stack:
                vert = stack.pop()
                for edge in vert.link_edges:
                    other = edge.other_vert(vert)
                    if other in unseen:
                        unseen.remove(other)
                        stack.append(other)
        return count
    finally:
        bm.free()


def is_round_candidate(obj: bpy.types.Object) -> bool:
    a, b, c = object_dimensions(obj)
    if c <= 1e-8:
        return False
    circular_axes = abs(b - c) / c < 0.28
    thin_axis = a / c < 0.55
    return circular_axes and thin_axis and triangle_count(obj) >= 48


def bounds_contains(outer: bpy.types.Object, inner: bpy.types.Object, margin: float = 1e-5) -> bool:
    omin, omax = world_bounds([outer])
    imin, imax = world_bounds([inner])
    return all(omin[i] - margin <= imin[i] and imax[i] <= omax[i] + margin for i in range(3))


def assign_symmetry(items: list[ObjectAnalysis], center_x: float) -> None:
    for index, item in enumerate(items):
        if item.symmetry_partner:
            continue
        for other in items[index + 1 :]:
            if other.symmetry_partner:
                continue
            dims_error = sum(abs(item.dimensions[i] - other.dimensions[i]) for i in range(3)) / max(sum(item.dimensions), 1e-8)
            mirrored_error = abs((item.center[0] - center_x) + (other.center[0] - center_x))
            scale = max(item.dimensions[2], other.dimensions[2], 1e-6)
            yz_error = math.dist(item.center[1:], other.center[1:]) / scale
            if dims_error < 0.18 and mirrored_error / scale < 0.25 and yz_error < 0.35:
                item.symmetry_partner = other.name
                other.symmetry_partner = item.name
                break



def assign_repeat_groups(items: list[ObjectAnalysis]) -> None:
    groups: dict[tuple, list[ObjectAnalysis]] = {}
    for item in items:
        dims = sorted(max(0.0, float(value)) for value in item.dimensions)
        scale = max(dims[-1], 1e-8)
        shape = tuple(round(value / scale, 2) for value in dims)
        key = (item.triangles, item.vertices, shape, tuple(sorted(item.materials)))
        groups.setdefault(key, []).append(item)
    index = 1
    for group in groups.values():
        if len(group) < 2:
            continue
        name = f"repeat_{index:03d}"
        index += 1
        for item in group:
            item.repeat_group = name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--asset-type", required=True)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.input)
    stats = extended_mesh_stats(objects)
    total_triangles = max(stats["triangles"], 1)
    whole_min, whole_max = world_bounds(objects)
    center_x = float((whole_min.x + whole_max.x) * 0.5)
    analyses: list[ObjectAnalysis] = []
    for obj in objects:
        center = object_center(obj)
        triangles = triangle_count(obj)
        analyses.append(
            ObjectAnalysis(
                name=obj.name,
                triangles=triangles,
                vertices=len(obj.data.vertices),
                materials=[slot.material.name for slot in obj.material_slots if slot.material],
                loose_components=connected_component_count(obj),
                dimensions=object_dimensions(obj),
                center=(float(center.x), float(center.y), float(center.z)),
                round_candidate=is_round_candidate(obj),
                small_fragment=triangles / total_triangles < 0.0005,
            )
        )
    by_name = {obj.name: obj for obj in objects}
    for item in analyses:
        obj = by_name[item.name]
        item.fully_inside_larger_bounds = any(
            other.name != obj.name
            and triangle_count(other) > item.triangles * 4
            and bounds_contains(other, obj)
            for other in objects
        )
    assign_symmetry(analyses, center_x)
    assign_repeat_groups(analyses)
    save_json(
        args.report,
        {
            "success": bool(objects),
            "asset_type": args.asset_type,
            "stats": stats,
            "objects": [asdict(item) for item in analyses],
            "candidate_counts": {
                "round_parts": sum(item.round_candidate for item in analyses),
                "small_fragments": sum(item.small_fragment for item in analyses),
                "inside_candidates": sum(item.fully_inside_larger_bounds for item in analyses),
                "symmetry_pairs": sum(item.symmetry_partner is not None for item in analyses) // 2,
                "repeat_groups": len({item.repeat_group for item in analyses if item.repeat_group}),
            },
            "notes": [
                "Hidden/internal geometry is reported conservatively and is not removed unless explicitly requested.",
                "Semantic labels are geometric/name heuristics unless the experimental semantic backend is separately proven.",
            ],
        },
    )


if __name__ == "__main__":
    main()
