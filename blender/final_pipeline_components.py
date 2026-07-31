"""Analyse and conservatively clean the loose components of the shaman high master.

The Mini Turbo master is a single glTF primitive whose vertices are not welded, so glTF-level
component counts are meaningless (hundreds of thousands of "components" that are really one
surface). Everything here therefore welds by distance first and uses Blender's own connectivity as
the authoritative topology.

Cleanup is deliberately conservative. The shaman's silhouette depends on small, thin, detached-
looking pieces - cords, leaf pendants, a bowl charm, a hollow pod, a lantern, the staff ring - and
a face-count threshold alone would delete exactly those. A component is only removed when it is
both physically tiny and sitting in the empty region above the antler crown, which is where the
generator's stray specks appear and where no source ornament exists.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy
from mathutils import Vector

from common import argv_after_double_dash, import_mesh, reset_scene, save_json


def weld_and_split(objects: list[bpy.types.Object], merge_distance: float) -> list[bpy.types.Object]:
    for obj in objects:
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.remove_doubles(threshold=merge_distance)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="LOOSE")
        bpy.ops.object.mode_set(mode="OBJECT")
    return [o for o in bpy.context.scene.objects if o.type == "MESH"]


def describe(obj: bpy.types.Object) -> dict:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    extent = hi - lo
    centre = (lo + hi) * 0.5
    return {
        "name": obj.name,
        "faces": len(obj.data.polygons),
        "vertices": len(obj.data.vertices),
        "bounds_min": [lo.x, lo.y, lo.z],
        "bounds_max": [hi.x, hi.y, hi.z],
        "extent": [extent.x, extent.y, extent.z],
        "centre": [centre.x, centre.y, centre.z],
        "diagonal": extent.length,
        # A cord or fringe strip is far taller than it is wide; that shape is protected outright.
        "vertical_aspect": extent.z / max(extent.x, extent.y, 1e-6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--report", required=True)
    parser.add_argument("--merge-distance", type=float, default=1e-5)
    parser.add_argument("--analyze-only", action="store_true")
    # Fraction of the model diagonal below which a component counts as "tiny".
    parser.add_argument("--tiny-diagonal-fraction", type=float, default=0.02)
    # Components whose centre sits above this fraction of the model height are in the empty air
    # above the antler crown. Source ornaments all hang at or below the antler pole.
    parser.add_argument("--debris-height-fraction", type=float, default=0.90)
    parser.add_argument("--protect-vertical-aspect", type=float, default=1.8)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    imported = import_mesh(args.input)
    if not imported:
        raise RuntimeError(f"No mesh objects in {args.input}")

    parts = weld_and_split(imported, args.merge_distance)
    stats = [describe(obj) for obj in parts]
    if not stats:
        raise RuntimeError("Welding produced no components")

    lo = Vector((min(s["bounds_min"][i] for s in stats) for i in range(3)))
    hi = Vector((max(s["bounds_max"][i] for s in stats) for i in range(3)))
    model_extent = hi - lo
    model_diagonal = model_extent.length
    model_height = max(model_extent.z, 1e-6)
    largest = max(stats, key=lambda s: s["faces"])

    tiny_cut = args.tiny_diagonal_fraction * model_diagonal
    debris_z = lo.z + args.debris_height_fraction * model_height

    for entry in stats:
        entry["diagonal_fraction"] = entry["diagonal"] / model_diagonal
        entry["height_fraction"] = (entry["centre"][2] - lo.z) / model_height
        reasons = []
        if entry["name"] == largest["name"]:
            reasons.append("main_component")
        if entry["diagonal"] >= tiny_cut:
            reasons.append("large_enough")
        if entry["vertical_aspect"] >= args.protect_vertical_aspect:
            reasons.append("elongated_hanging")
        if entry["centre"][2] <= debris_z:
            reasons.append("within_ornament_height_band")
        entry["keep_reasons"] = reasons
        entry["keep"] = bool(reasons)

    removed = [e for e in stats if not e["keep"]]
    kept = [e for e in stats if e["keep"]]
    total_faces = sum(e["faces"] for e in stats)
    removed_faces = sum(e["faces"] for e in removed)

    report = {
        "input": args.input,
        "merge_distance": args.merge_distance,
        "model_extent": [model_extent.x, model_extent.y, model_extent.z],
        "model_diagonal": model_diagonal,
        "criteria": {
            "tiny_diagonal_fraction": args.tiny_diagonal_fraction,
            "tiny_diagonal_world": tiny_cut,
            "debris_height_fraction": args.debris_height_fraction,
            "debris_height_world_z": debris_z,
            "protect_vertical_aspect": args.protect_vertical_aspect,
            "rule": (
                "remove only components that are simultaneously below the tiny-diagonal cut, not "
                "elongated-vertical, and centred above the antler crown height band"
            ),
        },
        "component_count": len(stats),
        "kept_count": len(kept),
        "removed_count": len(removed),
        "total_faces": total_faces,
        "removed_faces": removed_faces,
        "removed_face_fraction": removed_faces / max(total_faces, 1),
        "main_component_faces": largest["faces"],
        "main_component_face_fraction": largest["faces"] / max(total_faces, 1),
        "removed": sorted(removed, key=lambda e: -e["faces"])[:200],
        "kept_largest": sorted(kept, key=lambda e: -e["faces"])[:60],
    }
    save_json(args.report, report)

    if args.analyze_only or not args.output:
        return

    # Fail closed: refuse to write a cleaned mesh that lost a meaningful share of the model.
    if report["removed_face_fraction"] > 0.02:
        raise RuntimeError(
            f"Cleanup would remove {report['removed_face_fraction']*100:.2f}% of faces; refusing"
        )

    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue
        entry = next((e for e in stats if e["name"] == obj.name), None)
        if entry and not entry["keep"]:
            bpy.data.objects.remove(obj, do_unlink=True)

    survivors = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not survivors:
        raise RuntimeError("Cleanup removed every component")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in survivors:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = survivors[0]
    bpy.ops.object.join()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=args.output,
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_apply=False,
    )


if __name__ == "__main__":
    main()
