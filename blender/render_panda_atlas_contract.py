"""Fresh-import and render a GLB as unlit Base Color from six orthographic views.

This script is intentionally independent of the production render scene. It removes lighting,
normal-map, roughness, metallic and occlusion ambiguity so visible debug-magenta or black pixels
can only come from the bound Base Color/material/UV contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def args_after_separator() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def parse_args() -> tuple[Path, Path, str, int]:
    values = args_after_separator()
    if len(values) < 3:
        raise SystemExit("usage: blender --background --python render_panda_atlas_contract.py -- INPUT_GLB OUTPUT_DIR PREFIX [RESOLUTION]")
    resolution = int(values[3]) if len(values) > 3 else 768
    return Path(values[0]).resolve(), Path(values[1]).resolve(), values[2], resolution


def imported_meshes() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def material_image(material: bpy.types.Material):
    if not material.use_nodes or not material.node_tree:
        return None
    nodes = material.node_tree.nodes
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is not None:
        base = principled.inputs.get("Base Color")
        if base and base.is_linked:
            image_node = next((link.from_node for link in base.links if link.from_node.type == "TEX_IMAGE"), None)
            if image_node is not None and image_node.image is not None:
                return image_node.image
    return next(
        (node.image for node in nodes if node.type == "TEX_IMAGE" and node.image is not None),
        None,
    )


def replace_with_unlit(material: bpy.types.Material) -> dict:
    material.use_nodes = True
    image = material_image(material)
    fallback = tuple(float(v) for v in material.diffuse_color[:4])
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    emission.inputs["Strength"].default_value = 1.0
    if image is not None:
        try:
            image.colorspace_settings.name = "Non-Color"
        except Exception:
            pass
        texture = nodes.new("ShaderNodeTexImage")
        texture.image = image
        texture.interpolation = "Linear"
        texture.extension = "EXTEND"
        links.new(texture.outputs["Color"], emission.inputs["Color"])
    else:
        emission.inputs["Color"].default_value = fallback
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return {
        "material": material.name,
        "image": image.name if image is not None else None,
        "fallback_rgba": list(fallback),
    }


def world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for obj in objects for corner in obj.bound_box]
    if not corners:
        raise RuntimeError("NO_MESH_BOUNDS")
    minimum = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maximum = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return minimum, maximum


def setup_scene(resolution: int):
    scene = bpy.context.scene
    engines = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    scene.render.use_file_extension = True
    return scene


def render_views(scene, objects: list[bpy.types.Object], output_dir: Path, prefix: str) -> list[dict]:
    minimum, maximum = world_bounds(objects)
    center = (minimum + maximum) * 0.5
    extents = maximum - minimum
    radius = max(float(extents.x), float(extents.y), float(extents.z), 1e-4) * 0.5
    distance = radius * 4.0
    ortho_scale = radius * 2.35

    camera_data = bpy.data.cameras.new("AtlasContractCamera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ortho_scale
    camera = bpy.data.objects.new("AtlasContractCamera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    views = [
        ("front", Vector((0.0, -1.0, 0.0))),
        ("left", Vector((-1.0, 0.0, 0.0))),
        ("right", Vector((1.0, 0.0, 0.0))),
        ("rear", Vector((0.0, 1.0, 0.0))),
        ("top", Vector((0.0, 0.0, 1.0))),
        ("bottom", Vector((0.0, 0.0, -1.0))),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for semantic, offset in views:
        camera.location = center + offset * distance
        direction = center - camera.location
        # Use a different track-up axis at the poles to avoid a singular camera roll.
        track_up = "Y" if abs(float(offset.z)) < 0.9 else "X"
        camera.rotation_euler = direction.to_track_quat("-Z", track_up).to_euler()
        path = output_dir / f"{prefix}_{semantic}.png"
        scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)
        records.append(
            {
                "semantic": semantic,
                "path": str(path),
                "camera_location": list(camera.location),
                "camera_rotation_euler": list(camera.rotation_euler),
                "ortho_scale": float(camera_data.ortho_scale),
            }
        )
    return records


def main() -> None:
    input_glb, output_dir, prefix, resolution = parse_args()
    if not input_glb.is_file():
        raise SystemExit(f"GLB_MISSING:{input_glb}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(input_glb))
    objects = imported_meshes()
    if not objects:
        raise RuntimeError("GLB_IMPORT_HAS_NO_MESH")

    material_records = []
    seen = set()
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if material is None or material.name in seen:
                continue
            seen.add(material.name)
            material_records.append(replace_with_unlit(material))

    scene = setup_scene(resolution)
    renders = render_views(scene, objects, output_dir, prefix)
    report = {
        "schema": "panda_atlas_contract_fresh_render_v1",
        "input_glb": str(input_glb),
        "resolution": resolution,
        "mesh_objects": [obj.name for obj in objects],
        "materials": material_records,
        "renders": renders,
        "classification": "FRESH_IMPORT_UNLIT_RENDERED",
    }
    report_path = output_dir / f"{prefix}_render_receipt.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"ATLAS_CONTRACT_RENDER_DONE prefix={prefix} views={len(renders)} output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
