from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_beggars_meme_scene_faceverse_v10 as v10  # noqa: E402


v8 = v10.v8
v7 = v8.v7
v6 = v8.v6
v5 = v8.v5
base = v8.base
_exact_model_space_face = v8._original_create_model_space_face_mesh
_original_build_refined_character = v8._original_build_refined_character

MAIN_HEAD_END = 13916
RIGHT_EYE_END = 15456
TONGUE_END = 15846
TOTAL_VERTICES = 19546
PROJECTION_MATERIAL_NAME = "MAT_Antinous_CameraProjected_V11"
PROJECTION_STRENGTH_NODE = "ProjectionStrength"


def set_input(node: bpy.types.Node, name: str, value: Any) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def simple_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float,
    specular: float,
) -> bpy.types.Material:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        raise RuntimeError(f"Principled BSDF missing for {name}")
    set_input(bsdf, "Base Color", color)
    set_input(bsdf, "Roughness", roughness)
    set_input(bsdf, "Specular IOR Level", specular)
    set_input(bsdf, "Metallic", 0.0)
    return material


def projected_skin_material_v11() -> bpy.types.Material:
    if v10._projection_image_path is None:
        raise RuntimeError("V11 projection image was not loaded")
    material = bpy.data.materials.new(PROJECTION_MATERIAL_NAME)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    uv_map = nodes.new("ShaderNodeUVMap")
    uv_map.uv_map = "ProjectedUV"
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.name = "ProjectedKeyframeTexture"
    image = bpy.data.images.load(str(v10._projection_image_path), check_existing=True)
    image.colorspace_settings.name = "sRGB"
    image.pack()
    image_node.image = image
    image_node.extension = "CLIP"
    image_node.interpolation = "Linear"

    native = nodes.new("ShaderNodeAttribute")
    native.attribute_name = "Col"
    native_balance = nodes.new("ShaderNodeHueSaturation")
    native_balance.inputs["Saturation"].default_value = 0.96
    native_balance.inputs["Value"].default_value = 0.60

    projected_balance = nodes.new("ShaderNodeHueSaturation")
    projected_balance.inputs["Saturation"].default_value = 0.84
    projected_balance.inputs["Value"].default_value = 0.76

    mask = nodes.new("ShaderNodeAttribute")
    mask.attribute_name = "ProjMask"
    strength = nodes.new("ShaderNodeValue")
    strength.name = PROJECTION_STRENGTH_NODE
    strength.outputs[0].default_value = 0.60
    multiply = nodes.new("ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 62.0
    noise.inputs["Detail"].default_value = 2.2
    noise.inputs["Roughness"].default_value = 0.60
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.018
    bump.inputs["Distance"].default_value = 0.010

    set_input(bsdf, "Roughness", 0.58)
    set_input(bsdf, "Specular IOR Level", 0.15)
    set_input(bsdf, "Subsurface Weight", 0.006)

    links.new(uv_map.outputs["UV"], image_node.inputs["Vector"])
    links.new(image_node.outputs["Color"], projected_balance.inputs["Color"])
    links.new(native.outputs["Color"], native_balance.inputs["Color"])
    links.new(mask.outputs["Fac"], multiply.inputs[0])
    links.new(strength.outputs[0], multiply.inputs[1])
    links.new(multiply.outputs[0], mix.inputs[0])
    links.new(native_balance.outputs["Color"], mix.inputs[1])
    links.new(projected_balance.outputs["Color"], mix.inputs[2])
    links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def apply_projection_attributes_v11(face: bpy.types.Object) -> None:
    if v10._projection_uv is None or v10._projection_mask is None:
        raise RuntimeError("V11 projection arrays are unavailable")
    mesh = face.data
    if len(mesh.vertices) != TOTAL_VERTICES:
        raise RuntimeError(f"Unexpected v11 vertex count: {len(mesh.vertices)}")

    old_uv = mesh.uv_layers.get("ProjectedUV")
    if old_uv is not None:
        mesh.uv_layers.remove(old_uv)
    uv_layer = mesh.uv_layers.new(name="ProjectedUV")
    for loop in mesh.loops:
        uv_layer.data[loop.index].uv = v10._projection_uv[loop.vertex_index]

    old_mask = mesh.color_attributes.get("ProjMask")
    if old_mask is not None:
        mesh.color_attributes.remove(old_mask)
    mask_attribute = mesh.color_attributes.new(
        name="ProjMask",
        type="FLOAT_COLOR",
        domain="POINT",
    )
    mask_rgba = np.ones((TOTAL_VERTICES, 4), dtype=np.float32)
    mask_rgba[:, :3] = v10._projection_mask[:, None]
    mask_attribute.data.foreach_set("color", mask_rgba.reshape(-1).tolist())

    projected = projected_skin_material_v11()
    dark_mouth = simple_material(
        "MAT_Antinous_InternalMouth_V11",
        (0.018, 0.0025, 0.0018, 1.0),
        0.70,
        0.06,
    )
    ivory = simple_material(
        "MAT_Antinous_Teeth_Ivory_V11",
        (0.50, 0.39, 0.26, 1.0),
        0.40,
        0.20,
    )
    if len(mesh.materials) == 0:
        mesh.materials.append(projected)
    else:
        mesh.materials[0] = projected
    mesh.materials.append(dark_mouth)
    mouth_index = len(mesh.materials) - 1
    mesh.materials.append(ivory)
    ivory_index = len(mesh.materials) - 1

    mouth_polygons = 0
    tooth_polygons = 0
    for polygon in mesh.polygons:
        indices = [int(index) for index in polygon.vertices]
        polygon.material_index = 0
        if indices and all(TONGUE_END > index >= RIGHT_EYE_END for index in indices):
            polygon.material_index = mouth_index
            mouth_polygons += 1
        elif indices and all(index >= TONGUE_END for index in indices):
            polygon.material_index = ivory_index
            tooth_polygons += 1
    if mouth_polygons <= 0 or tooth_polygons <= 0:
        raise RuntimeError(
            f"V11 component assignment failed: mouth={mouth_polygons} teeth={tooth_polygons}"
        )
    face["v11_exact_projection_geometry"] = True
    print(
        "BLENDER_FACEVERSE_V11_EXACT_UV_GEOMETRY=PROVEN "
        f"MOUTH_POLYGONS={mouth_polygons} TOOTH_POLYGONS={tooth_polygons}"
    )


def create_face_v11(*args, **kwargs):
    face, follow, targets = _exact_model_space_face(*args, **kwargs)
    apply_projection_attributes_v11(face)
    return face, follow, targets


def build_character_v11(follow: bpy.types.Object, colors_rgb: np.ndarray) -> dict[str, object]:
    result = _original_build_refined_character(follow, colors_rgb)
    removed = 0
    kept_hair_cap = False
    for obj in list(bpy.data.objects):
        if obj.name == "CHAR_Antinous_HairCap":
            kept_hair_cap = True
            continue
        if (
            obj.name.startswith("HAIR_")
            or obj.name.startswith("FACIALHAIR_")
            or obj.name.startswith("EYE_V8_")
            or obj.name.startswith("EYE_V9_")
            or obj.name.startswith("MOUTH_V9_")
            or obj.name.startswith("CHAR_Antinous_FittedHairShell")
        ):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    if not kept_hair_cap:
        raise RuntimeError("V11 rear hair cap was not created by the proven character builder")
    result["hair_cap"] = bpy.data.objects.get("CHAR_Antinous_HairCap")
    result["strand_count"] = 0
    result["moustache_hair_count"] = 0
    result["sideburn_hair_count"] = 0
    result["removed_procedural_face_objects"] = removed
    result["variant"] = "FACEVERSE_EXACT_UV_PROJECTION_REAR_CAP_V11"
    print(
        "BLENDER_FACEVERSE_V11_REAR_CAP_ONLY=PROVEN "
        f"REMOVED_OBJECTS={removed}"
    )
    return result


def resolve_output_dir() -> Path:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    for index, value in enumerate(argv):
        if value == "--output-dir" and index + 1 < len(argv):
            return Path(argv[index + 1]).resolve()
    raise RuntimeError("Could not resolve v11 output directory")


def render_variants(output_dir: Path) -> list[dict[str, Any]]:
    face = bpy.data.objects.get("CHAR_Antinous")
    if face is None or len(face.data.materials) < 1:
        raise RuntimeError("V11 face or material slot zero is missing after reload")
    material = face.data.materials[0]
    if material is None or not material.use_nodes or material.node_tree is None:
        raise RuntimeError("V11 material slot zero is invalid after reload")
    strength_node = material.node_tree.nodes.get(PROJECTION_STRENGTH_NODE)
    if strength_node is None:
        raise RuntimeError(f"V11 projection node missing from slot zero: {material.name}")
    print(f"BLENDER_FACEVERSE_V11_RELOADED_MATERIAL=PROVEN NAME={material.name}")

    scene = bpy.context.scene
    camera = bpy.data.objects.get("CAM_Hero")
    if camera is None:
        raise RuntimeError("V11 hero camera is missing")
    scene.camera = camera
    scene.frame_set(48)
    variants: list[dict[str, Any]] = []
    for strength in (0.45, 0.60, 0.75):
        strength_node.outputs[0].default_value = strength
        output_path = output_dir / f"projection_blend_{int(round(strength * 100)):03d}.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        if not output_path.is_file() or output_path.stat().st_size < 50000:
            raise RuntimeError(f"V11 projection render missing or too small: {output_path}")
        variants.append(
            {
                "strength": strength,
                "render": output_path.name,
                "bytes": output_path.stat().st_size,
            }
        )
        print(f"BLENDER_FACEVERSE_V11_PROJECTION_VARIANT=PROVEN STRENGTH={strength:.2f}")
    strength_node.outputs[0].default_value = 0.60
    bpy.ops.wm.save_as_mainfile(
        filepath=str(output_dir / "beggars_photoreal_recreation.blend")
    )
    return variants


def patch_receipt(output_dir: Path, variants: list[dict[str, Any]]) -> None:
    receipt_path = output_dir / "scene_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["character_variant"] = "FACEVERSE_EXACT_UV_PROJECTION_REAR_CAP_V11"
    receipt["visual_changes_v11"] = {
        "uv_and_geometry_coordinate_match": True,
        "post_projection_anatomy_deformation": False,
        "rear_hair_cap_preserved": True,
        "hair_wires_removed": True,
        "facial_hair_geometry_removed": True,
        "dark_internal_mouth_material": True,
        "ivory_tooth_material": True,
        "projection_strengths_tested": [0.45, 0.60, 0.75],
        "selected_strength_in_blend": 0.60,
        "source_frame_plane_used": False,
    }
    receipt["projection_variants"] = variants
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    v10.load_projection_inputs()
    v5.create_model_space_face_mesh = create_face_v11
    v6.build_refined_character = build_character_v11
    result = int(v6.main())
    output_dir = resolve_output_dir()
    variants = render_variants(output_dir)
    patch_receipt(output_dir, variants)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
