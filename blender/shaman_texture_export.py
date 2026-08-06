"""Stage 6, step 4: build the Unreal-compatible Principled material and export the textured GLB.

One material slot, three images: base colour, tangent normal, and an ORM pack split by a Separate
Color node so R/G/B drive occlusion, roughness and metallic. All three are packed into the .blend
and embedded in the GLB, so the delivery has no external texture dependencies to resolve.

The UV layer is restored from the GLB when the importer drops it - the atlas mesh carries no
material, and Blender only materialises the TEXCOORD sets some material samples.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

import bpy

from common import argv_after_double_dash, export_glb, extended_mesh_stats, import_mesh, reset_scene, save_json
from final_pipeline_bake import ensure_uv_layer

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from lowvram3d.anchor_provenance import AnchorProvenanceError, geometry_sha256, load_anchor_provenance, provenance_record


def build_material(basecolor: Path, normal: Path, orm: Path):
    material = bpy.data.materials.new("ShamanPBR")
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()

    uv = tree.nodes.new("ShaderNodeUVMap")
    bsdf = tree.nodes.new("ShaderNodeBsdfPrincipled")
    output = tree.nodes.new("ShaderNodeOutputMaterial")

    def image_node(path: Path, non_colour: bool):
        node = tree.nodes.new("ShaderNodeTexImage")
        image = bpy.data.images.load(str(path))
        if non_colour:
            image.colorspace_settings.name = "Non-Color"
        image.pack()
        node.image = image
        tree.links.new(uv.outputs["UV"], node.inputs["Vector"])
        return node

    base_node = image_node(basecolor, non_colour=False)
    normal_node = image_node(normal, non_colour=True)
    orm_node = image_node(orm, non_colour=True)

    normal_map = tree.nodes.new("ShaderNodeNormalMap")
    normal_map.space = "TANGENT"
    tree.links.new(normal_node.outputs["Color"], normal_map.inputs["Color"])

    separate = tree.nodes.new("ShaderNodeSeparateColor")
    tree.links.new(orm_node.outputs["Color"], separate.inputs["Color"])

    tree.links.new(base_node.outputs["Color"], bsdf.inputs["Base Color"])
    tree.links.new(separate.outputs["Green"], bsdf.inputs["Roughness"])
    tree.links.new(separate.outputs["Blue"], bsdf.inputs["Metallic"])
    tree.links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    tree.links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material, {"basecolor": base_node, "normal": normal_node, "orm": orm_node}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--normal", required=True)
    parser.add_argument("--orm", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--anchor-receipt", required=True)
    parser.add_argument("--expected-source-sha256", default="")
    args = parser.parse_args(argv_after_double_dash())

    try:
        _receipt, receipt_hash, anchor_ids = load_anchor_provenance(
            args.anchor_receipt, expected_source_sha256=args.expected_source_sha256 or None
        )
    except AnchorProvenanceError as exc:
        save_json(args.manifest, {"success": False, "failure_codes": [exc.code], "failure_detail": exc.detail})
        raise SystemExit(2)

    reset_scene()
    objects = import_mesh(args.mesh)
    if not objects:
        raise RuntimeError(f"no mesh imported from {args.mesh}")

    def geometry_hash(items) -> str:
        vertices, triangles, base = [], [], 0
        for item in items:
            vertices.extend([list(item.matrix_world @ vertex.co) for vertex in item.data.vertices])
            for polygon in item.data.polygons:
                indices = [base + int(v) for v in polygon.vertices]
                triangles.extend([[indices[0], indices[i], indices[i + 1]] for i in range(1, len(indices) - 1)])
            base += len(item.data.vertices)
        return geometry_sha256(np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64))

    input_geometry_hash = geometry_hash(objects)
    for obj in objects:
        source = ensure_uv_layer(obj, args.mesh)
        print(f"UV_LAYER {obj.name} {source}", flush=True)

    material, nodes = build_material(Path(args.basecolor), Path(args.normal), Path(args.orm))
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        obj.data.update()

    Path(args.output_glb).parent.mkdir(parents=True, exist_ok=True)
    export_glb(args.output_glb, selected_only=False)
    exported_stats = extended_mesh_stats(objects)
    Path(args.output_blend).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend))

    reset_scene()
    imported = import_mesh(args.output_glb)
    if not imported:
        save_json(args.manifest, {"success": False, "failure_codes": ["GEOMETRY_MUTATION"], "failure_detail": "fresh textured GLB import produced no mesh"})
        raise SystemExit(2)
    output_geometry_hash = geometry_hash(imported)
    geometry_unchanged = output_geometry_hash == input_geometry_hash
    if not geometry_unchanged:
        save_json(args.manifest, {"success": False, "failure_codes": ["GEOMETRY_MUTATION"], "failure_detail": "fresh textured GLB geometry hash differs from input"})
        raise SystemExit(2)

    slots = sorted({slot.material.name for obj in objects for slot in obj.material_slots if slot.material})
    manifest = {
        "material_name": material.name,
        "material_slot_count": len(slots),
        "material_slots": slots,
        "shader": "Principled BSDF",
        "textures": {
            "base_color": {"path": args.basecolor, "colorspace": "sRGB", "packed": True},
            "normal": {"path": args.normal, "colorspace": "Non-Color", "space": "TANGENT", "packed": True},
            "orm": {"path": args.orm, "colorspace": "Non-Color", "packed": True,
                    "channels": {"R": "ambient_occlusion", "G": "roughness", "B": "metallic"}},
        },
        "packed_images": [image.name for image in bpy.data.images if image.packed_file],
        "output_glb": args.output_glb,
        "output_blend": args.output_blend,
        "mesh_stats": exported_stats,
        "fresh_import": {"mesh_objects": len(imported), "geometry_hash": output_geometry_hash},
        "anchor_receipt_sha256": receipt_hash,
        "anchor_ids": sorted(anchor_ids),
        "input_geometry_sha256": input_geometry_hash,
        "output_geometry_sha256": output_geometry_hash,
        "geometry_unchanged": geometry_unchanged,
        "provenance": provenance_record(
            receipt_sha256=receipt_hash, anchor_ids=anchor_ids,
            input_geometry_sha256=input_geometry_hash,
            output_geometry_sha256=output_geometry_hash,
            geometry_unchanged=geometry_unchanged,
        ),
        "success": True,
    }
    save_json(args.manifest, manifest)
    print(f"TEXTURE_EXPORT slots={len(slots)} packed={len(manifest['packed_images'])}", flush=True)


if __name__ == "__main__":
    main()
