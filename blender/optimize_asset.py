from __future__ import annotations

import argparse
from pathlib import Path

import bpy

from common import (
    apply_transforms,
    argv_after_double_dash,
    export_glb,
    extended_mesh_stats,
    import_mesh,
    object_dimensions,
    reset_scene,
    save_json,
    select_only,
    triangle_count,
)

IMPORTANT_ROLES = {"body", "deform_body", "wheel", "door", "shell", "panel", "weapon"}


def importance(obj: bpy.types.Object, asset_type: str) -> float:
    role = str(obj.get("lowvram_role", ""))
    value = 2.0 if role in IMPORTANT_ROLES else 1.0
    a, b, c = object_dimensions(obj)
    if c > 1e-8 and min(a, b, c) / max(a, b, c) < 0.04:
        value *= 1.8
    if asset_type in {"character", "creature"} and role == "deform_body":
        value *= 1.5
    if bool(obj.get("lowvram_movable", False)):
        value *= 1.25
    return value


def allocate_total(objects: list[bpy.types.Object], target: int, asset_type: str) -> dict[str, int]:
    current = {obj.name: max(triangle_count(obj), 1) for obj in objects}
    weighted = {obj.name: current[obj.name] ** 0.72 * importance(obj, asset_type) for obj in objects}
    total_weight = max(sum(weighted.values()), 1e-8)
    minimum = max(48, min(400, target // max(len(objects) * 4, 1)))
    budgets = {
        obj.name: min(current[obj.name], max(minimum, round(target * weighted[obj.name] / total_weight)))
        for obj in objects
    }
    overflow = sum(budgets.values()) - target
    for obj in sorted(objects, key=lambda item: budgets[item.name] - minimum, reverse=True):
        if overflow <= 0:
            break
        reduction = min(overflow, max(0, budgets[obj.name] - minimum))
        budgets[obj.name] -= reduction
        overflow -= reduction
    return budgets


def allocate_per_object(objects: list[bpy.types.Object], cap: int, asset_type: str) -> dict[str, int]:
    budgets = {}
    for obj in objects:
        current = max(triangle_count(obj), 1)
        role_scale = min(1.5, max(0.5, importance(obj, asset_type)))
        budgets[obj.name] = min(current, max(48, round(cap * role_scale)))
    return budgets


def apply_modifier(obj: bpy.types.Object, modifier) -> bool:
    select_only([obj])
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.modifier_apply(modifier=modifier.name)
        return True
    except Exception:
        try:
            obj.modifiers.remove(modifier)
        except Exception:
            pass
        return False


def dissolve_flat_surfaces(obj: bpy.types.Object, angle_degrees: float) -> bool:
    if angle_degrees <= 0:
        return False
    modifier = obj.modifiers.new("PlanarDissolve", "DECIMATE")
    modifier.decimate_type = "DISSOLVE"
    modifier.angle_limit = angle_degrees * 0.017453292519943295
    try:
        modifier.delimit = {"MATERIAL", "UV", "NORMAL", "SEAM", "SHARP"}
    except Exception:
        try:
            modifier.delimit = {"MATERIAL", "UV", "NORMAL"}
        except Exception:
            pass
    return apply_modifier(obj, modifier)


def collapse_to_budget(obj: bpy.types.Object, budget: int, preserve_symmetry: bool) -> dict:
    before = triangle_count(obj)
    if before <= budget:
        return {"before": before, "after": before, "ratio": 1.0, "strategy": "unchanged"}
    ratio = max(0.005, min(1.0, budget / before))
    modifier = obj.modifiers.new("GameReadyCollapse", "DECIMATE")
    modifier.decimate_type = "COLLAPSE"
    modifier.ratio = ratio
    modifier.use_collapse_triangulate = True
    if preserve_symmetry:
        try:
            modifier.use_symmetry = True
            modifier.symmetry_axis = "X"
        except Exception:
            pass
    applied = apply_modifier(obj, modifier)
    return {
        "before": before,
        "after": triangle_count(obj),
        "ratio": ratio,
        "strategy": "collapse" if applied else "modifier_failed",
    }


def create_lod(objects: list[bpy.types.Object], path: Path, ratio: float, asset_type: str) -> dict:
    duplicates: list[bpy.types.Object] = []
    for obj in objects:
        duplicate = obj.copy()
        duplicate.data = obj.data.copy()
        bpy.context.collection.objects.link(duplicate)
        duplicate.name = f"{obj.name}_{path.stem.upper()}"
        collapse_to_budget(
            duplicate,
            max(24, round(triangle_count(obj) * ratio)),
            preserve_symmetry=asset_type in {"character", "creature", "vehicle"},
        )
        duplicates.append(duplicate)
    select_only(duplicates)
    export_glb(str(path), selected_only=True)
    stats = extended_mesh_stats(duplicates)
    for duplicate in duplicates:
        bpy.data.objects.remove(duplicate, do_unlink=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--asset-type", required=True)
    parser.add_argument("--target-triangles", type=int, required=True)
    parser.add_argument("--target-min", type=int, required=True)
    parser.add_argument("--target-max", type=int, required=True)
    parser.add_argument("--budget-mode", choices=("total", "per_object"), default="total")
    parser.add_argument("--per-object-target", type=int, default=15000)
    parser.add_argument("--planar-angle-deg", type=float, default=0.0)
    parser.add_argument("--lod1", default="")
    parser.add_argument("--lod2", default="")
    parser.add_argument("--lod-ratios", default="0.55,0.25")
    parser.add_argument("--lod-count", type=int, default=1)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.input)
    if not objects:
        raise RuntimeError("No mesh objects imported")
    apply_transforms(objects)
    before = extended_mesh_stats(objects)
    for obj in objects:
        dissolve_flat_surfaces(obj, args.planar_angle_deg)
    budgets = (
        allocate_per_object(objects, args.per_object_target, args.asset_type)
        if args.budget_mode == "per_object"
        else allocate_total(objects, args.target_triangles, args.asset_type)
    )
    object_reports = {}
    for obj in objects:
        object_reports[obj.name] = collapse_to_budget(
            obj,
            budgets[obj.name],
            preserve_symmetry=args.asset_type in {"character", "creature", "vehicle"},
        )
        obj.data.validate(clean_customdata=False)
        obj.data.update()
    after = extended_mesh_stats(objects)
    warnings = []
    if args.budget_mode == "total" and after["triangles"] < args.target_min * 0.75:
        warnings.append("Triangle count undershot the requested range; inspect thin parts and silhouette.")
    if args.budget_mode == "total" and after["triangles"] > args.target_max * 1.25:
        warnings.append("Triangle count remains above tolerance because per-object minimum budgets were preserved.")
    select_only(objects)
    export_glb(args.output)
    ratios = [float(value) for value in args.lod_ratios.split(",") if value.strip()]
    lod_paths = [args.lod1, args.lod2]
    lods = []
    for index in range(min(args.lod_count, len(ratios), len(lod_paths))):
        if not lod_paths[index]:
            continue
        path = Path(lod_paths[index])
        path.parent.mkdir(parents=True, exist_ok=True)
        lods.append({"level": index + 1, "ratio": ratios[index], "path": str(path), "stats": create_lod(objects, path, ratios[index], args.asset_type)})
    save_json(
        args.stats,
        {
            "success": True,
            "backend": "blender_headless_decimate",
            "asset_type": args.asset_type,
            "budget_mode": args.budget_mode,
            "requested": {"target": args.target_triangles, "min": args.target_min, "max": args.target_max, "per_object_target": args.per_object_target},
            "planar_angle_deg": args.planar_angle_deg,
            "before": before,
            "after": after,
            "object_budgets": budgets,
            "objects": object_reports,
            "lods": lods,
            "warnings": warnings,
        },
    )


if __name__ == "__main__":
    main()
