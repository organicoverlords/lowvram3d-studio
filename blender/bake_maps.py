from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import bpy

from common import (
    argv_after_double_dash,
    enable_cycles_gpu,
    export_glb,
    import_mesh,
    reset_scene,
    select_only,
)


def ensure_material(obj):
    material = obj.active_material
    if material is None:
        material = bpy.data.materials.new("GamePBR"); material.use_nodes = True; obj.data.materials.append(material)
    material.use_nodes = True
    return material


def make_image(name, path, size, color):
    image = bpy.data.images.new(name, width=size, height=size, alpha=False, float_buffer=False)
    image.generated_color = color; image.filepath_raw = str(path); image.file_format = "PNG"
    return image


def active_target(material, image):
    nodes = material.node_tree.nodes
    target = nodes.new("ShaderNodeTexImage"); target.image = image; target.select = True; nodes.active = target
    return target


def bake_image(obj, material, image, bake_type, margin=16):
    active_target(material, image); select_only([obj]); bpy.context.view_layer.objects.active = obj
    enable_cycles_gpu()
    bpy.ops.object.bake(type=bake_type, margin=margin); image.save()


def export_basecolor(material, path: Path, size: int):
    candidate = None
    for node in material.node_tree.nodes:
        if node.type == "TEX_IMAGE" and node.image and node.image.name not in {"AmbientOcclusion", "Normal", "Roughness", "Metallic"}:
            candidate = node.image; break
    if candidate:
        try:
            candidate.save_render(str(path)); return candidate
        except Exception:
            source = Path(bpy.path.abspath(candidate.filepath)) if candidate.filepath else None
            if source and source.is_file(): shutil.copy2(source, path); return candidate
    fallback = make_image("BaseColor", path, size, (0.5, 0.5, 0.5, 1.0)); fallback.save(); return fallback


def bake_high_to_low_normal(low, material, high_path: str, image) -> bool:
    if not high_path or not Path(high_path).is_file(): return False
    try:
        high_objects = import_mesh(high_path)
        active_target(material, image)
        select_only(high_objects + [low]); bpy.context.view_layer.objects.active = low
        enable_cycles_gpu()
        bpy.context.scene.render.bake.use_selected_to_active = True
        bpy.context.scene.render.bake.cage_extrusion = 0.02
        bpy.ops.object.bake(type="NORMAL", use_selected_to_active=True, margin=16)
        image.save()
        for obj in high_objects: bpy.data.objects.remove(obj, do_unlink=True)
        return True
    except Exception:
        return False


def connect_map(material, image, kind):
    nodes, links = material.node_tree.nodes, material.node_tree.links
    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None) or nodes.new("ShaderNodeBsdfPrincipled")
    output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
    if output is None: output = nodes.new("ShaderNodeOutputMaterial"); links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    texture = nodes.new("ShaderNodeTexImage"); texture.image = image
    if kind == "normal":
        texture.image.colorspace_settings.name = "Non-Color"; normal = nodes.new("ShaderNodeNormalMap")
        links.new(texture.outputs["Color"], normal.inputs["Color"]); links.new(normal.outputs["Normal"], principled.inputs["Normal"])
    elif kind == "roughness":
        texture.image.colorspace_settings.name = "Non-Color"; links.new(texture.outputs["Color"], principled.inputs["Roughness"])
    elif kind == "metallic":
        texture.image.colorspace_settings.name = "Non-Color"; links.new(texture.outputs["Color"], principled.inputs["Metallic"])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--input", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--maps-dir", required=True); parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--material-hint", default="organic"); parser.add_argument("--high-poly", default="")
    args = parser.parse_args(argv_after_double_dash()); maps_dir = Path(args.maps_dir); maps_dir.mkdir(parents=True, exist_ok=True)
    reset_scene(); objects = import_mesh(args.input)
    if not objects: raise RuntimeError("No mesh imported")
    obj, material = objects[0], ensure_material(objects[0])
    basecolor = export_basecolor(material, maps_dir / "basecolor.png", args.size)
    ao = make_image("AmbientOcclusion", maps_dir / "ambient_occlusion.png", args.size, (1, 1, 1, 1))
    try: bake_image(obj, material, ao, "AO")
    except Exception: ao.save()
    normal = make_image("Normal", maps_dir / "normal.png", args.size, (0.5, 0.5, 1.0, 1.0))
    if not bake_high_to_low_normal(obj, material, args.high_poly, normal): normal.save()
    metallic_value = 0.8 if any(word in args.material_hint.lower() for word in ("metal", "armor", "robot", "vehicle")) else 0.0
    roughness_value = 0.35 if metallic_value > 0 else 0.65
    roughness = make_image("Roughness", maps_dir / "roughness.png", args.size, (roughness_value,) * 3 + (1.0,))
    metallic = make_image("Metallic", maps_dir / "metallic.png", args.size, (metallic_value,) * 3 + (1.0,))
    roughness.save(); metallic.save(); connect_map(material, normal, "normal"); connect_map(material, roughness, "roughness"); connect_map(material, metallic, "metallic")
    export_glb(args.output)


if __name__ == "__main__": main()
