from __future__ import annotations

import argparse
from pathlib import Path

import bpy
from mathutils import Vector

from common import argv_after_double_dash, configure_render, create_camera, export_fbx, export_glb, import_mesh, mesh_objects, mesh_stats, reset_scene, save_json, select_only, world_bounds


def add_collision(objects):
    names = []
    for obj in objects:
        duplicate = obj.copy(); duplicate.data = obj.data.copy(); bpy.context.collection.objects.link(duplicate)
        duplicate.name = f"UCX_{obj.name}_00"; select_only([duplicate]); bpy.context.view_layer.objects.active = duplicate
        try:
            bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.select_all(action="SELECT"); bpy.ops.mesh.convex_hull(); bpy.ops.object.mode_set(mode="OBJECT")
        except RuntimeError:
            try: bpy.ops.object.mode_set(mode="OBJECT")
            except Exception: pass
        duplicate.hide_render = True; names.append(duplicate.name)
    return names


def add_sockets(objects):
    minimum, maximum = world_bounds(objects); center = (minimum + maximum) * 0.5
    sockets = {"SOCKET_root": Vector((center.x, center.y, minimum.z)), "SOCKET_center": center, "SOCKET_top": Vector((center.x, center.y, maximum.z))}
    for name, location in sockets.items():
        empty = bpy.data.objects.new(name, None); empty.empty_display_type = "PLAIN_AXES"; empty.location = location; bpy.context.collection.objects.link(empty)
    return list(sockets)


def create_lod_files(objects, output_dir):
    outputs = []
    for level, ratio in ((1, 0.5), (2, 0.2)):
        duplicates = []
        for obj in objects:
            dup = obj.copy(); dup.data = obj.data.copy(); bpy.context.collection.objects.link(dup); dup.name = f"{obj.name}_LOD{level}"
            modifier = dup.modifiers.new("LODDecimate", "DECIMATE"); modifier.ratio = ratio
            select_only([dup]); bpy.context.view_layer.objects.active = dup; bpy.ops.object.modifier_apply(modifier=modifier.name); duplicates.append(dup)
        select_only(duplicates); path = output_dir / f"asset_LOD{level}.glb"; export_glb(str(path), selected_only=True); outputs.append(str(path))
        for dup in duplicates: bpy.data.objects.remove(dup, do_unlink=True)
    return outputs


def render_preview(path):
    objects = mesh_objects(); minimum, maximum = world_bounds(objects); center = (minimum + maximum) * 0.5
    scale = max(*(maximum - minimum), 1.0)
    camera = create_camera("PreviewCamera", (center.x + scale * 1.8, center.y - scale * 2.2, center.z + scale * 1.2), scale * 2.0)
    bpy.context.scene.camera = camera
    light_data = bpy.data.lights.new("Key", type="AREA"); light_data.energy = 1200; light_data.shape = "DISK"; light_data.size = scale * 2
    light = bpy.data.objects.new("Key", light_data); bpy.context.collection.objects.link(light); light.location = (center.x - scale, center.y - scale, center.z + scale * 2)
    configure_render(768, 768, transparent=True); bpy.context.scene.render.filepath = str(path); bpy.ops.render.render(write_still=True)


def validate_clean(path):
    reset_scene(); objects = import_mesh(path); stats = mesh_stats(objects)
    images = [image for image in bpy.data.images if image.source == "FILE" or image.packed_file]
    materials = [material for material in bpy.data.materials if material.use_nodes]
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    return {
        "success": bool(objects) and stats["finite_bounds"] and stats["uv_layers"] > 0 and bool(materials),
        "mesh_count": len(objects), "material_count": len(materials), "image_count": len(images),
        "uv_layers": stats["uv_layers"], "face_count": stats["faces"], "armature_count": len(armatures),
        "action_count": len(bpy.data.actions), "finite_bounds": stats["finite_bounds"],
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation", required=True); parser.add_argument("--preview", required=True); parser.add_argument("--manifest", required=True)
    args = parser.parse_args(argv_after_double_dash()); output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    reset_scene(); objects = import_mesh(args.input); collision_names = add_collision(objects); socket_names = add_sockets(objects); lods = create_lod_files(objects, output_dir)
    final_glb, final_fbx = output_dir / "asset.glb", output_dir / "asset.fbx"
    select_only([obj for obj in bpy.context.scene.objects if obj.type in {"MESH", "ARMATURE", "EMPTY"}]); export_glb(str(final_glb), selected_only=True); export_fbx(str(final_fbx), selected_only=True)
    render_preview(Path(args.preview)); validation = validate_clean(str(final_glb))
    validation.update({"glb": str(final_glb), "fbx": str(final_fbx), "lods": lods, "collisions": collision_names, "sockets": socket_names})
    save_json(args.validation, validation)
    save_json(args.manifest, {
        "engine_targets": ["Unreal Engine", "Unity", "Godot"], "units": "meters", "up_axis": "Z", "forward_axis": "-Y",
        "primary_glb": str(final_glb), "skeletal_or_static_fbx": str(final_fbx), "lods": lods,
        "collisions": collision_names, "sockets": socket_names, "validation": str(args.validation),
        "notes": ["Inspect organic skin weights and semantic part names before shipping.", "LOD files are separate and can be assigned by the game-engine importer."]
    })
    if not validation["success"]: raise RuntimeError(f"Clean import validation failed: {validation}")


if __name__ == "__main__": main()
