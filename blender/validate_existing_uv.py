"""Fresh-process validation for an immutable existing textured UV mesh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bpy
import numpy as np

from common import argv_after_double_dash, import_mesh, reset_scene, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--require-texture", action="store_true")
    args = parser.parse_args(argv_after_double_dash())
    source = Path(args.input)
    report = {"input": str(source), "success": False, "gates": {}}
    try:
        if not source.is_file() or source.stat().st_size == 0:
            raise RuntimeError("UV_SOURCE_MISSING")
        reset_scene()
        objects = import_mesh(str(source))
        if not objects:
            raise RuntimeError("UV_SOURCE_NO_MESH")
        uv_layers = []
        finite = True
        in_bounds = True
        material_count = 0
        texture_nodes = 0
        packed_images = 0
        triangles = 0
        for obj in objects:
            triangles += sum(max(len(poly.vertices) - 2, 0) for poly in obj.data.polygons)
            uv_layers.extend(layer.name for layer in obj.data.uv_layers)
            for layer in obj.data.uv_layers:
                values = np.asarray([tuple(loop.uv) for loop in layer.data], dtype=np.float64)
                finite &= bool(np.isfinite(values).all())
                in_bounds &= bool((values >= -1e-6).all() and (values <= 1.000001).all())
            material_count += sum(slot.material is not None for slot in obj.material_slots)
            for material in obj.data.materials:
                if not material or not material.use_nodes:
                    continue
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE" and node.image:
                        texture_nodes += 1
                        packed_images += int(node.image.packed_file is not None)
        report["gates"] = {
            "fresh_import": True, "uv_layer_exists": bool(uv_layers),
            "uv_finite": finite, "uv_in_bounds": in_bounds,
            "material_present": material_count > 0,
            "texture_resolves": texture_nodes > 0,
            "packed_texture": packed_images > 0,
            "triangles": triangles, "uv_layers": sorted(set(uv_layers)),
        }
        required = ["fresh_import", "uv_layer_exists", "uv_finite", "uv_in_bounds", "material_present"]
        if args.require_texture:
            required += ["texture_resolves", "packed_texture"]
        report["success"] = all(report["gates"].get(key) for key in required)
    except Exception as exc:
        report["error"] = repr(exc)
    save_json(str(args.report), report)
    raise SystemExit(0 if report["success"] else 1)


if __name__ == "__main__":
    main()
