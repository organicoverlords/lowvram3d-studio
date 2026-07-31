from __future__ import annotations

import argparse
import math
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
    shade_smooth,
    weld_vertices,
)


def validate_objects(objects: list[bpy.types.Object]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not objects:
        errors.append("No mesh objects were imported")
        return errors, warnings
    for obj in objects:
        if not obj.data.vertices or not obj.data.polygons:
            warnings.append(f"{obj.name}: empty mesh")
        # bpy_prop_collection rejects a slice with a step, so sample by index instead.
        vertices = obj.data.vertices
        step = max(1, len(vertices) // 10000)
        for index in range(0, len(vertices), step):
            if not all(math.isfinite(float(value)) for value in vertices[index].co):
                errors.append(f"{obj.name}: non-finite vertex coordinates")
                break
    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--apply-transforms", action="store_true")
    args = parser.parse_args(argv_after_double_dash())

    source = Path(args.input)
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Input mesh is missing or empty: {source}")
    reset_scene()
    objects = import_mesh(str(source))
    if args.apply_transforms:
        apply_transforms(objects)
    smoothed = shade_smooth(objects)
    weld = weld_vertices(objects)
    weld["faces_smoothed"] = smoothed
    errors, warnings = validate_objects(objects)
    stats = extended_mesh_stats(objects)
    if stats["triangles"] <= 0:
        errors.append("Imported mesh has no triangles")
    if not stats["finite_bounds"]:
        errors.append("Imported mesh has invalid bounds")
    export_glb(args.output)
    report = {
        "success": not errors,
        "source": str(source),
        "canonical_high_glb": str(Path(args.output)),
        "stats": stats,
        "weld": weld,
        "errors": errors,
        "warnings": warnings,
        "textures_found": len(stats["texture_paths"]),
    }
    save_json(args.report, report)
    if errors:
        raise RuntimeError("; ".join(errors))


if __name__ == "__main__":
    main()
