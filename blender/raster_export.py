"""Raster texture route, stage 3: assign the rastered atlas as a Principled BSDF base colour and
export the textured GLB. Uses the cleaned mesh from raster_cleanup_extract.py (not the original
input) so exported face/vertex counts match what was actually rastered.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import bpy
import sys
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from lowvram3d.anchor_provenance import (
    AnchorProvenanceError,
    GEOMETRY_HASH_FRAME,
    geometry_sha256,
    load_anchor_provenance,
    provenance_record,
)

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
    parser.add_argument("--anchor-receipt", default="")
    parser.add_argument("--expected-source-sha256", default="")
    parser.add_argument("--expected-input-geometry-sha256", default="")
    parser.add_argument("--require-anchor-provenance", action="store_true")
    args = parser.parse_args(argv_after_double_dash())

    if args.require_anchor_provenance and not args.anchor_receipt:
        if args.report:
            save_json(args.report, {"success": False, "failure_codes": ["ANCHOR_RECEIPT_MISSING"]})
        raise SystemExit(2)

    receipt_hash = None
    anchor_ids = []
    provenance_verified = False
    try:
        if args.anchor_receipt:
            _receipt, receipt_hash, anchor_ids = load_anchor_provenance(
                args.anchor_receipt, expected_source_sha256=args.expected_source_sha256 or None
            )
            provenance_verified = True
    except AnchorProvenanceError as exc:
        if args.report:
            save_json(args.report, {"success": False, "failure_codes": [exc.code], "failure_detail": exc.detail})
        raise SystemExit(2)

    reset_scene()
    objects = import_mesh(args.cleaned_mesh)
    if not objects:
        raise RuntimeError("No mesh imported")

    def geometry_hash(items=None) -> str:
        items = objects if items is None else items
        vertices = []
        triangles = []
        base = 0
        for item in items:
            vertices.extend([list(item.matrix_world @ vertex.co) for vertex in item.data.vertices])
            verts = [base + int(v) for v in polygon.vertices]
            triangles.extend([[verts[0], verts[i], verts[i + 1]] for i in range(1, len(verts) - 1)])
            base += len(item.data.vertices)
        return geometry_sha256(np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64))

    input_geometry_hash = geometry_hash()
    if args.expected_input_geometry_sha256 and input_geometry_hash != args.expected_input_geometry_sha256:
        if args.report:
            save_json(args.report, {
                "success": False,
                "failure_codes": ["GEOMETRY_MUTATION"],
                "failure_detail": "export input geometry hash differs from verified cleanup geometry",
                "provenance": {
                    "geometry_hash_frame": GEOMETRY_HASH_FRAME,
                    "input_geometry_sha256": input_geometry_hash,
                    "expected_input_geometry_sha256": args.expected_input_geometry_sha256,
                    "geometry_unchanged": False,
                    "provenance_verified": provenance_verified,
                },
            })
        raise SystemExit(2)

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
    exported_stats = extended_mesh_stats(objects)
    reset_scene()
    imported = import_mesh(args.output)
    if not imported:
        raise RuntimeError("fresh raster export import produced no mesh")
    output_geometry_hash = geometry_hash(imported)
    if output_geometry_hash != input_geometry_hash:
        if args.report:
            save_json(args.report, {
                "success": False,
                "failure_codes": ["GEOMETRY_MUTATION"],
                "failure_detail": "fresh raster export geometry hash differs from input",
                "provenance": {
                    "geometry_hash_frame": GEOMETRY_HASH_FRAME,
                    "input_geometry_sha256": input_geometry_hash,
                    "output_geometry_sha256": output_geometry_hash,
                    "geometry_unchanged": False,
                    "provenance_verified": provenance_verified,
                },
            })
        raise SystemExit(2)

    if args.report:
        save_json(args.report, {
            "success": True,
            "backend": "raster_uv_atlas_projection_export",
            "output": str(args.output),
            "texture": str(texture_path),
            "smooth_shaded_polygons": smoothed,
            "output_stats": exported_stats,
            "fresh_import": {"mesh_objects": len(imported), "geometry_hash": output_geometry_hash},
            "provenance": ({**provenance_record(
                receipt_sha256=receipt_hash, anchor_ids=anchor_ids,
                input_geometry_sha256=input_geometry_hash,
                output_geometry_sha256=output_geometry_hash,
                geometry_unchanged=input_geometry_hash == output_geometry_hash,
            ), "provenance_verified": provenance_verified} if provenance_verified else {
                "geometry_hash_frame": GEOMETRY_HASH_FRAME,
                "provenance_verified": False,
                "input_geometry_sha256": input_geometry_hash,
                "output_geometry_sha256": output_geometry_hash,
                "geometry_unchanged": input_geometry_hash == output_geometry_hash,
            }),
        })

    print(f"RASTER_EXPORT {args.output} faces={sum(len(o.data.polygons) for o in objects)}", flush=True)


if __name__ == "__main__":
    main()
