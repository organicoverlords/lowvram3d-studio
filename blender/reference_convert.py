"""Convert a reference model to GLB for measurement, without touching the original.

The source is opened read-only and the result is written to a destination the caller supplies, which
must be outside the reference library. Conversion exists so formats the metric workers cannot parse
(FBX, OBJ, PLY, STL) can still contribute to a geometry-quality baseline; it is not a licence to
edit, clean or re-export anything in place.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy

from common import argv_after_double_dash, reset_scene, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv_after_double_dash())

    source = Path(args.input).resolve()
    destination = Path(args.output).resolve()
    if not source.is_file():
        raise SystemExit(f"missing input {source}")
    if destination.parent == source.parent:
        raise SystemExit("refusing to write the conversion beside the reference it came from")

    reset_scene()
    suffix = source.suffix.lower()
    if suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(source))
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=str(source))
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=str(source))
    elif suffix == ".stl":
        bpy.ops.wm.stl_import(filepath=str(source))
    else:
        raise SystemExit(f"unsupported reference format {suffix}")

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit(f"no mesh imported from {source}")
    triangles = sum(len(o.data.loop_triangles) for o in meshes
                    if (o.data.calc_loop_triangles() or True))

    destination.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(destination), export_format="GLB",
        use_selection=False, export_apply=True,
    )

    record = {
        "input": str(source),
        "input_bytes": source.stat().st_size,
        "output": str(destination),
        "output_bytes": destination.stat().st_size if destination.exists() else 0,
        "mesh_objects": len(meshes),
        "triangles": int(triangles),
        "original_modified": False,
    }
    if args.report:
        save_json(args.report, record)
    print(f"REFERENCE_CONVERTED {source.name} -> {destination.name} "
          f"meshes={len(meshes)} triangles={triangles}", flush=True)


if __name__ == "__main__":
    main()
