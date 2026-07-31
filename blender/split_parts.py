from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import bpy

from common import (
    argv_after_double_dash,
    export_glb,
    extended_mesh_stats,
    import_mesh,
    object_center,
    object_dimensions,
    reset_scene,
    save_json,
    select_only,
    triangle_count,
    world_bounds,
)

ORGANIC_TYPES = {"character", "creature"}
WORLD_TYPES = {"scene", "level"}
ROLE_WORDS = {
    "wheel": ("wheel", "tire", "tyre"),
    "door": ("door", "hatch", "gate"),
    "window": ("window", "glass"),
    "light": ("light", "lamp", "headlight"),
    "roof_cargo": ("roof", "cargo", "luggage", "rack"),
    "weapon": ("weapon", "gun", "rifle", "sword", "shield"),
    "hair": ("hair", "fur", "mane"),
    "armor": ("armor", "armour", "plate", "helmet"),
    "accessory": ("accessory", "strap", "bag", "pack", "pouch"),
    "floor": ("floor", "ground"),
    "ceiling": ("ceiling", "roof"),
    "wall": ("wall",),
    "body": ("body", "torso", "hull", "chassis", "shell"),
}


def clean_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_")
    return value or "part"


def separate_loose(objects: list[bpy.types.Object]) -> None:
    for obj in list(objects):
        select_only([obj])
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.separate(type="LOOSE")
        bpy.ops.object.mode_set(mode="OBJECT")


def round_candidate(obj: bpy.types.Object) -> bool:
    a, b, c = object_dimensions(obj)
    return c > 1e-8 and abs(b - c) / c < 0.30 and a / c < 0.58 and triangle_count(obj) >= 48


def classify_by_name(name: str) -> str | None:
    lowered = name.lower()
    for role, words in ROLE_WORDS.items():
        if any(word in lowered for word in words):
            return role
    return None


def classify(obj: bpy.types.Object, asset_type: str, largest: bpy.types.Object, center) -> str:
    named = classify_by_name(obj.name)
    if named:
        return named
    if asset_type in ORGANIC_TYPES:
        if obj == largest:
            return "deform_body"
        if round_candidate(obj):
            return "accessory"
        return "organic_attachment"
    if asset_type == "vehicle":
        if round_candidate(obj):
            return "wheel"
        if obj == largest:
            return "body"
        local = object_center(obj) - center
        if local.z > 0.25:
            return "roof_cargo"
        return "accessory"
    if asset_type in {"building", "room"}:
        a, b, c = object_dimensions(obj)
        if a < max(b, c) * 0.08:
            return "panel"
        return "shell" if obj == largest else "fixture"
    if asset_type in WORLD_TYPES:
        return "world_object"
    return "body" if obj == largest else "part"


def bounds_contains(outer: bpy.types.Object, inner: bpy.types.Object) -> bool:
    omin, omax = world_bounds([outer])
    imin, imax = world_bounds([inner])
    margin = max((omax - omin)) * 1e-4
    return all(omin[i] - margin <= imin[i] and imax[i] <= omax[i] + margin for i in range(3))


def merge_objects(target: bpy.types.Object, others: list[bpy.types.Object]) -> bpy.types.Object:
    if not others:
        return target
    select_only([target, *others])
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.join()
    return target


def merge_small_fragments(objects: list[bpy.types.Object], threshold: float) -> tuple[list[bpy.types.Object], int]:
    total = max(sum(triangle_count(obj) for obj in objects), 1)
    major = [obj for obj in objects if triangle_count(obj) / total >= threshold]
    small = [obj for obj in objects if obj not in major]
    if not major or not small:
        return objects, 0
    grouped: dict[str, list[bpy.types.Object]] = {obj.name: [] for obj in major}
    for fragment in small:
        center = object_center(fragment)
        parent = min(major, key=lambda obj: (object_center(obj) - center).length)
        grouped[parent.name].append(fragment)
    for parent in major:
        merge_objects(parent, grouped[parent.name])
    return major, len(small)


def remove_enclosed_candidates(objects: list[bpy.types.Object]) -> tuple[list[bpy.types.Object], list[str]]:
    removed: list[str] = []
    ordered = sorted(objects, key=triangle_count, reverse=True)
    keep: list[bpy.types.Object] = []
    for index, obj in enumerate(ordered):
        enclosed = any(
            triangle_count(parent) > triangle_count(obj) * 8 and bounds_contains(parent, obj)
            for parent in ordered[:index]
        )
        if enclosed and triangle_count(obj) < max(64, sum(triangle_count(item) for item in ordered) * 0.0002):
            removed.append(obj.name)
            bpy.data.objects.remove(obj, do_unlink=True)
        else:
            keep.append(obj)
    return keep, removed


def assign_symmetry(objects: list[bpy.types.Object], center_x: float) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for index, obj in enumerate(objects):
        if obj.name in pairs:
            continue
        dims = object_dimensions(obj)
        center = object_center(obj)
        for other in objects[index + 1 :]:
            if other.name in pairs:
                continue
            other_dims = object_dimensions(other)
            other_center = object_center(other)
            dims_error = sum(abs(dims[i] - other_dims[i]) for i in range(3)) / max(sum(dims), 1e-8)
            scale = max(dims[2], other_dims[2], 1e-6)
            mirror_error = abs((center.x - center_x) + (other_center.x - center_x)) / scale
            yz_error = math.dist((center.y, center.z), (other_center.y, other_center.z)) / scale
            if dims_error < 0.18 and mirror_error < 0.25 and yz_error < 0.35:
                pairs[obj.name] = other.name
                pairs[other.name] = obj.name
                break
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--asset-type", default="prop")
    parser.add_argument("--max-parts", type=int, default=96)
    parser.add_argument("--merge-small", action="store_true")
    parser.add_argument("--separate-movable", action="store_true")
    parser.add_argument("--remove-hidden", action="store_true")
    parser.add_argument("--experimental-semantic", action="store_true")
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    original_objects = import_mesh(args.input)
    if not original_objects:
        raise RuntimeError("No mesh objects imported")
    if args.asset_type not in WORLD_TYPES:
        separate_loose(original_objects)
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    removed_hidden: list[str] = []
    if args.remove_hidden:
        objects, removed_hidden = remove_enclosed_candidates(objects)
    merged_fragments = 0
    if args.merge_small and args.asset_type not in {"room", "scene", "level"}:
        objects, merged_fragments = merge_small_fragments(objects, threshold=0.0005)
    objects = sorted(objects, key=triangle_count, reverse=True)
    if len(objects) > args.max_parts:
        keep = objects[: args.max_parts - 1]
        remainder = objects[args.max_parts - 1 :]
        parent = merge_objects(remainder[0], remainder[1:])
        parent.name = "merged_minor_fragments"
        objects = keep + [parent]

    largest = max(objects, key=triangle_count)
    whole_min, whole_max = world_bounds(objects)
    whole_center = (whole_min + whole_max) * 0.5
    symmetry = assign_symmetry(objects, whole_center.x)
    parts_dir = Path(args.parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)
    role_counts: dict[str, int] = {}
    manifest_parts = []
    for obj in objects:
        role = classify(obj, args.asset_type, largest, whole_center)
        role_counts[role] = role_counts.get(role, 0) + 1
        suffix = role_counts[role]
        name = clean_name(role if suffix == 1 else f"{role}_{suffix:02d}")
        old_name = obj.name
        obj.name = name
        movable = args.separate_movable and role in {"wheel", "door", "weapon", "accessory", "roof_cargo", "fixture"}
        obj["lowvram_role"] = role
        obj["lowvram_movable"] = movable
        output = parts_dir / f"{name}.glb"
        select_only([obj])
        export_glb(str(output), selected_only=True)
        minimum, maximum = world_bounds([obj])
        manifest_parts.append(
            {
                "name": name,
                "source_name": old_name,
                "role": role,
                "movable": movable,
                "file": str(output),
                "bounds_min": list(minimum),
                "bounds_max": list(maximum),
                "vertices": len(obj.data.vertices),
                "triangles": triangle_count(obj),
                "materials": [slot.material.name for slot in obj.material_slots if slot.material],
                "symmetry_partner_source": symmetry.get(old_name),
            }
        )
    select_only(objects)
    export_glb(args.output)
    warnings = []
    if args.experimental_semantic:
        warnings.append("Experimental semantic splitting requested but not executed: P3-SAM is not proven on the GTX 1660 SUPER profile.")
    save_json(
        args.manifest,
        {
            "success": True,
            "asset_type": args.asset_type,
            "part_count": len(manifest_parts),
            "parts": manifest_parts,
            "combined_stats": extended_mesh_stats(objects),
            "merged_small_fragments": merged_fragments,
            "removed_hidden_candidates": removed_hidden,
            "semantic_backend": "geometric",
            "semantic_split_proven": False,
            "warnings": warnings,
        },
    )


if __name__ == "__main__":
    main()
