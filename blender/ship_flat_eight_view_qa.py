"""Fresh-import flat-material eight-view proof for bounded ship geometry candidates."""
from __future__ import annotations

import argparse
from pathlib import Path

import bpy

from common import argv_after_double_dash, import_mesh, reset_scene, save_json, world_bounds
from shaman_texture_review import add_lights, analyse, place_camera, setup_world


VIEWS = (
    ("front", 0.0, 0.0),
    ("right", 90.0, 0.0),
    ("rear", 180.0, 0.0),
    ("left", 270.0, 0.0),
    ("top", 0.0, 55.0),
    ("bottom", 0.0, -55.0),
    ("front_three_quarter", 35.0, 8.0),
    ("rear_three_quarter", 215.0, 8.0),
)


def add_flat_material(objects: list[bpy.types.Object]) -> None:
    material = bpy.data.materials.new("ShipGeometryFlatProof")
    material.diffuse_color = (0.32, 0.38, 0.44, 1.0)
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled:
        principled.inputs["Base Color"].default_value = (0.32, 0.38, 0.44, 1.0)
        principled.inputs["Roughness"].default_value = 0.82
        principled.inputs["Metallic"].default_value = 0.0
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.use_smooth = True
        obj.data.update()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--front-direction", choices=("+z", "-z"), default="+z")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=4)
    args = parser.parse_args(argv_after_double_dash())

    glb = Path(args.glb)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reset_scene()
    objects = import_mesh(str(glb))
    if not objects:
        raise RuntimeError("FRESH_IMPORT_NO_MESH")

    minimum, maximum = world_bounds(objects)
    centre = (minimum + maximum) * 0.5
    radius = max((maximum - minimum).length * 0.5, 1e-3)
    add_flat_material(objects)
    setup_world()
    add_lights(radius)
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH" if "BLENDER_WORKBENCH" in scene.render.bl_rna.properties["engine"].enum_items.keys() else scene.render.engine
    # Workbench is deterministic and flat; use transparent PNGs for silhouette occupancy.
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    camera_data = bpy.data.cameras.new("ship_flat_qa_camera")
    camera = bpy.data.objects.new("ship_flat_qa_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    front_sign = -1.0 if args.front_direction == "+z" else 1.0
    views = {}
    for name, yaw, pitch in VIEWS:
        place_camera(camera, centre, radius, yaw, pitch, 1.0, centre, front_sign)
        path = output_dir / f"{name}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        views[name] = {"path": str(path), **analyse(path)}
        print(f"SHIP_FLAT_VIEW {name} {path}", flush=True)

    report = {
        "schema": "ship_flat_eight_view_qa_v1",
        "classification": "PROVEN_FRESH_IMPORT_AND_FLAT_EIGHT_VIEW_RENDER",
        "glb": str(glb),
        "fresh_import": True,
        "mesh_objects": len(objects),
        "front_direction_gltf": args.front_direction,
        "render_engine": scene.render.engine,
        "resolution": args.resolution,
        "samples": args.samples,
        "views": views,
    }
    save_json(str(args.report), report)
    print(f"SHIP_FLAT_QA_DONE views={len(views)} report={args.report}", flush=True)


if __name__ == "__main__":
    main()
