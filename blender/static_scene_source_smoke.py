"""Cheap static-scene source-ingest smoke test; does not invent 3D geometry."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

from common import argv_after_double_dash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv_after_double_dash())
    bpy.ops.wm.read_factory_settings(use_empty=True)
    image = bpy.data.images.load(args.image, check_existing=False)
    width, height = image.size
    aspect = width / max(height, 1)
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, 0, 0))
    plane = bpy.context.object
    plane.name = "BarnSourceReferenceCard"
    plane.scale = (aspect, 1.0, 1.0)
    material = bpy.data.materials.new("BarnSourceReferenceMaterial")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    texture = nodes.new("ShaderNodeTexImage")
    texture.image = image
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    output = nodes.new("ShaderNodeOutputMaterial")
    links.new(texture.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    plane.data.materials.append(material)

    camera_data = bpy.data.cameras.new("SourceIngestCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 2.25
    camera = bpy.data.objects.new("SourceIngestCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0, 0, 5)
    camera.rotation_euler = (0, 0, 0)
    camera.rotation_euler = (Vector((0, 0, 0)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = 720
    scene.render.resolution_y = max(1, round(720 / aspect))
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview = output_dir / "source_ingest_front.png"
    scene.render.filepath = str(preview)
    bpy.ops.render.render(write_still=True)
    blend = output_dir / "barn_source_ingest.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    report = {
        "classification": "PROVEN",
        "stage": "SOURCE_INGEST_REFERENCE_ONLY",
        "source_image": str(Path(args.image).resolve()),
        "blend": str(blend),
        "preview": str(preview),
        "object_count": 1,
        "mesh_object_count": 1,
        "armature_count": 0,
        "animation_count": 0,
        "route": "building/static_environment",
        "geometry_generated": False,
        "background_preserved_as_reference": True,
        "barn_preservation": "NOT_PROVEN_2D_REFERENCE_ONLY",
        "tree_preservation": "NOT_PROVEN_2D_REFERENCE_ONLY",
        "background_contamination": "NOT_ASSESSED_ON_3D_GEOMETRY",
        "next_stage": "requires a real 3D candidate or approved generation run"
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("STATIC_SOURCE_SMOKE=PROVEN", flush=True)
    print("GEOMETRY_GENERATED=false", flush=True)


if __name__ == "__main__":
    main()
