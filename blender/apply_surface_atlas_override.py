"""Route selected triangle consumers through a separate safe atlas."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import argv_after_double_dash, import_mesh, reset_scene, save_json


def clone_with_image(material, image):
    clone = material.copy()
    clone.name = f"{material.name}_surface_safe"
    clone.use_nodes = True
    nodes, links = clone.node_tree.nodes, clone.node_tree.links
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return clone
    socket = bsdf.inputs.get("Base Color")
    node = socket.links[0].from_node if socket and socket.is_linked else None
    if node is None or node.type != "TEX_IMAGE":
        node = nodes.new("ShaderNodeTexImage")
        node.name = "SurfaceSafeBaseColor"
        links.new(node.outputs["Color"], socket)
    node.image = image
    return clone


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-glb", required=True)
    p.add_argument("--safe-atlas", required=True)
    p.add_argument("--triangle-ids", required=True)
    p.add_argument("--output-glb", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.input_glb)
    image = bpy.data.images.load(str(Path(args.safe_atlas)), check_existing=False)
    image.pack()
    ids = set(int(v) for v in np.load(args.triangle_ids).reshape(-1).tolist())
    mesh_objects = [obj for obj in objects if obj.type == "MESH"]
    target_objects = [obj for obj in mesh_objects if ids and max(ids) < len(obj.data.polygons)]
    if not target_objects:
        raise RuntimeError("triangle IDs do not match any imported mesh polygon index space")
    mappings = {}
    routed = 0
    for obj in target_objects:
        for poly in obj.data.polygons:
            if poly.index not in ids:
                continue
            old = int(poly.material_index)
            if old not in mappings:
                material = obj.material_slots[old].material
                obj.data.materials.append(clone_with_image(material, image))
                mappings[old] = len(obj.data.materials) - 1
            poly.material_index = mappings[old]
            routed += 1

    output = Path(args.output_glb)
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(output), export_format="GLB", use_selection=False, export_apply=False)
    save_json(args.report, {
        "schema": "surface_atlas_override_v1",
        "input_glb": str(args.input_glb),
        "output_glb": str(output),
        "triangle_id_count": len(ids),
        "target_objects": [obj.name for obj in target_objects],
        "routed_polygons": routed,
        "material_mappings": mappings,
        "geometry_uv_unchanged": True,
    })


if __name__ == "__main__":
    main()
