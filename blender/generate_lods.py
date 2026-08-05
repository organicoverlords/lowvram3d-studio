"""Generate lower LODs without rewriting the already-measured LOD0 candidate."""
from __future__ import annotations

import argparse
from pathlib import Path

import bpy

from common import argv_after_double_dash, export_glb, extended_mesh_stats, import_mesh, reset_scene, save_json, select_only, triangle_count
from optimize_asset import collapse_to_budget


def remove_objects(objects: list[bpy.types.Object]) -> None:
    for obj in objects:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if data and data.users == 0:
            bpy.data.meshes.remove(data)


def generate_lod(
    source_objects: list[bpy.types.Object],
    path: Path,
    ratio: float,
) -> dict:
    duplicates: list[bpy.types.Object] = []
    object_reports = {}
    for source in source_objects:
        duplicate = source.copy()
        duplicate.data = source.data.copy()
        duplicate.name = f"{source.name}_{path.stem.upper()}"
        bpy.context.collection.objects.link(duplicate)
        report = collapse_to_budget(
            duplicate,
            max(24, round(triangle_count(source) * ratio)),
            preserve_symmetry=bool(source.get("lowvram_symmetry_verified", False)),
        )
        duplicate.data.validate(clean_customdata=False)
        duplicate.data.update()
        duplicates.append(duplicate)
        object_reports[source.name] = report
    path.parent.mkdir(parents=True, exist_ok=True)
    select_only(duplicates)
    export_glb(str(path), selected_only=True)
    stats = extended_mesh_stats(duplicates)
    remove_objects(duplicates)
    return {"path": str(path), "ratio": ratio, "stats": stats, "objects": object_reports}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--lod1", default="")
    parser.add_argument("--lod2", default="")
    parser.add_argument("--lod-ratios", default="0.5,0.2")
    parser.add_argument("--lod-count", type=int, default=2)
    args = parser.parse_args(argv_after_double_dash())

    ratios = [float(value) for value in args.lod_ratios.split(",") if value.strip()]
    if any(value <= 0.0 or value >= 1.0 for value in ratios):
        raise RuntimeError("LOD ratios must be greater than zero and less than one")

    reset_scene()
    source_objects = import_mesh(args.input)
    if not source_objects:
        raise RuntimeError("No mesh objects imported")
    before = extended_mesh_stats(source_objects)
    outputs = [args.lod1, args.lod2]
    lods = []
    for index in range(min(args.lod_count, len(ratios), len(outputs))):
        if outputs[index]:
            lods.append(generate_lod(source_objects, Path(outputs[index]), ratios[index]))

    save_json(
        args.report,
        {
            "success": True,
            "input": args.input,
            "lod0_preserved_external_to_script": True,
            "source_stats": before,
            "lods": lods,
        },
    )


if __name__ == "__main__":
    main()
