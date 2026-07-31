from __future__ import annotations

import argparse
import re
from pathlib import Path

import bpy

from common import argv_after_double_dash, export_glb, extended_mesh_stats, import_mesh, reset_scene, save_json, select_only


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value).strip("_") or "texture"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maps-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv_after_double_dash())

    maps_dir = Path(args.maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)
    reset_scene()
    objects = import_mesh(args.input)
    if not objects:
        raise RuntimeError("No mesh objects imported")
    materials = {slot.material for obj in objects for slot in obj.material_slots if slot.material}
    saved = []
    missing = []
    for material in materials:
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or not node.image:
                continue
            image = node.image
            extension = ".png"
            target = maps_dir / f"{safe_name(material.name)}__{safe_name(image.name)}{extension}"
            try:
                image.filepath_raw = str(target)
                image.file_format = "PNG"
                image.save()
                saved.append({"material": material.name, "image": image.name, "path": str(target)})
            except Exception as exc:
                missing.append({"material": material.name, "image": image.name, "error": str(exc)})
    if not saved:
        raise RuntimeError("Preserve-materials mode requires at least one readable source image texture")
    if not all(obj.data.uv_layers for obj in objects):
        raise RuntimeError("Preserve-materials mode requires UVs on every mesh object")
    select_only(objects)
    export_glb(args.output, selected_only=True)
    report = {
        "success": not missing,
        "backend": "blender_preserve_existing_materials",
        "output": args.output,
        "mesh_stats": extended_mesh_stats(objects),
        "material_count": len(materials),
        "saved_textures": saved,
        "texture_errors": missing,
    }
    save_json(args.report, report)
    save_json(maps_dir / "maps_manifest.json", report)
    if missing:
        raise RuntimeError(f"Some source textures could not be preserved: {missing}")


if __name__ == "__main__":
    main()
