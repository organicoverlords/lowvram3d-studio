"""Raster texture route, stage 3: assign the rastered atlas as a Principled BSDF base colour and
export the textured GLB. Uses the cleaned mesh from raster_cleanup_extract.py (not the original
input) so exported face/vertex counts match what was actually rastered.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import bpy

from common import (
    argv_after_double_dash,
    export_glb,
    extended_mesh_stats,
    import_mesh,
    reset_scene,
    save_json,
    shade_smooth,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleaned-mesh", required=True)
    parser.add_argument("--atlas", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--texture", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.cleaned_mesh)
    if not objects:
        raise RuntimeError("No mesh imported")

    texture_path = Path(args.texture)
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    texture_path.write_bytes(Path(args.atlas).read_bytes())

    mat = bpy.data.materials.new("RasterAtlas")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    uvmap = nt.nodes.new("ShaderNodeUVMap")
    img_node = nt.nodes.new("ShaderNodeTexImage")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    out_node = nt.nodes.new("ShaderNodeOutputMaterial")
    img = bpy.data.images.load(str(texture_path))
    img.pack()
    img_node.image = img
    nt.links.new(uvmap.outputs["UV"], img_node.inputs["Vector"])
    nt.links.new(img_node.outputs["Color"], bsdf.inputs["Base Color"])
    nt.links.new(bsdf.outputs["BSDF"], out_node.inputs["Surface"])

    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
        if not obj.data.uv_layers:
            raise RuntimeError(f"{obj.name} lost its UV layer before export")

    # Marching-cubes output is flat-shaded, which reads as hard polygonal banding across the vest
    # and hood once a texture is applied. Smooth shading is a normals-only change: it does not
    # move a single vertex, so silhouette, face count and UVs are untouched.
    smoothed = shade_smooth(objects)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    export_glb(args.output, selected_only=False)

    if args.report:
        save_json(args.report, {
            "success": True,
            "backend": "raster_uv_atlas_projection_export",
            "output": str(args.output),
            "texture": str(texture_path),
            "smooth_shaded_polygons": smoothed,
            "output_stats": extended_mesh_stats(objects),
        })

    print(f"RASTER_EXPORT {args.output} faces={sum(len(o.data.polygons) for o in objects)}", flush=True)


if __name__ == "__main__":
    main()
