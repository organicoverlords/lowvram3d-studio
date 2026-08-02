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

import bpy

from common import argv_after_double_dash, export_glb, extended_mesh_stats, import_mesh, reset_scene, save_json
from final_pipeline_bake import ensure_uv_layer


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
    parser.add_argument("--atlas-size", type=int, default=0)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.mesh)
    if not objects:
        raise RuntimeError(f"no mesh imported from {args.mesh}")
    for obj in objects:
        source = ensure_uv_layer(obj, args.mesh)
        print(f"UV_LAYER {obj.name} {source}", flush=True)

    material, nodes = build_material(Path(args.basecolor), Path(args.normal), Path(args.orm))
    if args.atlas_size:
        base_image = nodes["basecolor"].image
        actual = (int(base_image.size[0]), int(base_image.size[1]))
        expected = (int(args.atlas_size), int(args.atlas_size))
        if actual != expected:
            raise RuntimeError(
                "ATLAS_RESOLUTION_CONTRACT_MISMATCH: "
                f"shaman_texture_export received {actual}, expected {expected}"
            )
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        obj.data.update()

    Path(args.output_glb).parent.mkdir(parents=True, exist_ok=True)
    export_glb(args.output_glb, selected_only=False)
    Path(args.output_blend).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend))

    slots = sorted({slot.material.name for obj in objects for slot in obj.material_slots if slot.material})
    manifest = {
        "material_name": material.name,
        "material_slot_count": len(slots),
        "material_slots": slots,
        "shader": "Principled BSDF",
        "atlas_resolution": [int(nodes["basecolor"].image.size[0]), int(nodes["basecolor"].image.size[1])],
        "atlas_resolution_contract": {
            "requested": int(args.atlas_size) if args.atlas_size else None,
            "saved": [int(nodes["basecolor"].image.size[0]), int(nodes["basecolor"].image.size[1])],
            "passed": not bool(args.atlas_size) or tuple(nodes["basecolor"].image.size) == (int(args.atlas_size), int(args.atlas_size)),
        },
        "textures": {
            "base_color": {"path": args.basecolor, "colorspace": "sRGB", "packed": True},
            "normal": {"path": args.normal, "colorspace": "Non-Color", "space": "TANGENT", "packed": True},
            "orm": {"path": args.orm, "colorspace": "Non-Color", "packed": True,
                    "channels": {"R": "ambient_occlusion", "G": "roughness", "B": "metallic"}},
        },
        "packed_images": [image.name for image in bpy.data.images if image.packed_file],
        "output_glb": args.output_glb,
        "output_blend": args.output_blend,
        "mesh_stats": extended_mesh_stats(objects),
    }
    save_json(args.manifest, manifest)
    print(f"TEXTURE_EXPORT slots={len(slots)} packed={len(manifest['packed_images'])}", flush=True)


if __name__ == "__main__":
    main()
