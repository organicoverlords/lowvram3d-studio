"""Generate bounded descending geometry candidates from one clean-master import."""
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
    reset_scene,
    save_json,
    select_only,
)
from optimize_asset import allocate_per_object, allocate_total, collapse_to_budget, dissolve_flat_surfaces


def duplicate_objects(objects: list[bpy.types.Object], suffix: str) -> list[bpy.types.Object]:
    duplicates = []
    for source in objects:
        duplicate = source.copy()
        duplicate.data = source.data.copy()
        duplicate.name = f"{source.name}_{suffix}"
        bpy.context.collection.objects.link(duplicate)
        duplicates.append(duplicate)
    return duplicates


def remove_objects(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data and data.users == 0:
            bpy.data.meshes.remove(data)


def symmetry_verified(obj: bpy.types.Object) -> bool:
    """Never assume symmetry from asset type; asymmetric gear and poses are common."""
    return bool(obj.get("lowvram_symmetry_verified", False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--asset-type", required=True)
    parser.add_argument("--budgets", required=True, help="Comma-separated descending total face budgets")
    parser.add_argument("--budget-mode", choices=("total", "per_object"), default="total")
    parser.add_argument("--per-object-target", type=int, default=150_000)
    parser.add_argument("--planar-angle-deg", type=float, default=0.0)
    args = parser.parse_args(argv_after_double_dash())

    budgets = [int(value) for value in args.budgets.split(",") if value.strip()]
    if not budgets or any(value <= 0 for value in budgets):
        raise RuntimeError("At least one positive candidate budget is required")
    if budgets != sorted(set(budgets), reverse=True):
        raise RuntimeError("Candidate budgets must be unique and descending")

    reset_scene()
    master_objects = import_mesh(args.input)
    if not master_objects:
        raise RuntimeError("No mesh objects imported")
    apply_transforms(master_objects)
    master_stats = extended_mesh_stats(master_objects)
    master_triangles = max(int(master_stats["triangles"]), 1)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = []

    for index, target in enumerate(budgets, start=1):
        suffix = f"C{index:02d}"
        objects = duplicate_objects(master_objects, suffix)
        for obj in objects:
            dissolve_flat_surfaces(obj, args.planar_angle_deg)

        ratio = min(1.0, max(0.0, target / master_triangles))
        per_object_cap = max(48, round(args.per_object_target * ratio))
        object_budgets = (
            allocate_per_object(objects, per_object_cap, args.asset_type)
            if args.budget_mode == "per_object"
            else allocate_total(objects, target, args.asset_type)
        )
        object_reports = {}
        for obj in objects:
            object_reports[obj.name] = collapse_to_budget(
                obj,
                object_budgets[obj.name],
                preserve_symmetry=symmetry_verified(obj),
            )
            obj.data.validate(clean_customdata=False)
            obj.data.update()

        path = output_dir / f"candidate_{index:02d}_{target}.glb"
        select_only(objects)
        export_glb(str(path), selected_only=True)
        stats = extended_mesh_stats(objects)
        candidates.append(
            {
                "index": index,
                "target_faces": target,
                "target_ratio": ratio,
                "per_object_cap": per_object_cap if args.budget_mode == "per_object" else None,
                "path": str(path),
                "stats": stats,
                "object_budgets": object_budgets,
                "objects": object_reports,
            }
        )
        print(
            f"LOD_CANDIDATE index={index} target={target} actual={stats['triangles']} "
            f"mode={args.budget_mode} path={path}",
            flush=True,
        )
        remove_objects(objects)

    save_json(
        args.manifest,
        {
            "success": True,
            "input": args.input,
            "asset_type": args.asset_type,
            "budget_mode": args.budget_mode,
            "master_stats": master_stats,
            "budgets": budgets,
            "candidates": candidates,
        },
    )


if __name__ == "__main__":
    main()
