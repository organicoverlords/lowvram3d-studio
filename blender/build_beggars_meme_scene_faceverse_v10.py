from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_beggars_meme_scene_faceverse_v8 as v8  # noqa: E402


v7 = v8.v7
v6 = v8.v6
v5 = v8.v5
base = v8.base
_original_create_corrected_face = v8.create_corrected_face
_original_build_refined_character = v8._original_build_refined_character

TOTAL_VERTICES = v8.TOTAL_VERTICES
PROJECTION_MATERIAL_NAME = "MAT_Antinous_CameraProjected_V10"
PROJECTION_STRENGTH_NODE = "ProjectionStrength"

_projection_uv: np.ndarray | None = None
_projection_mask: np.ndarray | None = None
_projection_image_path: Path | None = None


def set_input(node: bpy.types.Node, name: str, value: Any) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def load_projection_inputs() -> None:
    global _projection_uv, _projection_mask, _projection_image_path
    projection_path = Path(os.environ.get("B10_PROJECTION_NPZ", "")).resolve()
    image_path = Path(os.environ.get("B10_PROJECTION_IMAGE", "")).resolve()
    if not projection_path.is_file():
        raise RuntimeError(f"V10 projection NPZ is missing: {projection_path}")
    if not image_path.is_file():
        raise RuntimeError(f"V10 projection image is missing: {image_path}")
    data = np.load(projection_path)
    uv = np.asarray(data["uv"], dtype=np.float32)
    mask = np.asarray(data["projection_mask"], dtype=np.float32).reshape(-1)
    if uv.shape != (TOTAL_VERTICES, 2):
        raise RuntimeError(f"Unexpected v10 UV shape: {uv.shape}")
    if mask.shape != (TOTAL_VERTICES,):
        raise RuntimeError(f"Unexpected v10 projection-mask shape: {mask.shape}")
    if not np.all(np.isfinite(uv)) or not np.all(np.isfinite(mask)):
        raise RuntimeError("V10 projection data contains non-finite values")
    _projection_uv = np.clip(uv, 0.0, 1.0)
    _projection_mask = np.clip(mask, 0.0, 1.0)
    _projection_image_path = image_path
    print(
        "BLENDER_FACEVERSE_V10_PROJECTION_INPUT=PROVEN "
        f"VERTICES={len(mask)} MASK_MEAN={float(np.mean(mask)):.6f}"
    )


def projected_skin_material_v10() -> bpy.types.Material:
    if _projection_image_path is None:
        raise RuntimeError("V10 projection image was not loaded before material creation")
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
    image_node.label = "Projected public keyframe 031"
    image = bpy.data.images.load(str(_projection_image_path), check_existing=True)
    image.colorspace_settings.name = "sRGB"
    image.pack()
    image_node.image = image
    image_node.extension = "CLIP"
    image_node.interpolation = "Linear"

    native = nodes.new("ShaderNodeAttribute")
    native.attribute_name = "Col"
    native_balance = nodes.new("ShaderNodeHueSaturation")
    native_balance.inputs["Saturation"].default_value = 0.92
    native_balance.inputs["Value"].default_value = 0.52

    projected_balance = nodes.new("ShaderNodeHueSaturation")
    projected_balance.inputs["Saturation"].default_value = 1.04
    projected_balance.inputs["Value"].default_value = 0.92

    mask = nodes.new("ShaderNodeAttribute")
    mask.attribute_name = "ProjMask"
    strength = nodes.new("ShaderNodeValue")
    strength.name = PROJECTION_STRENGTH_NODE
    strength.label = "Projection blend strength"
    strength.outputs[0].default_value = 0.85
    multiply = nodes.new("ShaderNodeMath")
    multiply.operation = "MULTIPLY"
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 58.0
    noise.inputs["Detail"].default_value = 2.5
    noise.inputs["Roughness"].default_value = 0.62
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.025
    bump.inputs["Distance"].default_value = 0.012

    set_input(bsdf, "Roughness", 0.57)
    set_input(bsdf, "Specular IOR Level", 0.17)
    set_input(bsdf, "Subsurface Weight", 0.008)

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


def apply_projection_attributes(face: bpy.types.Object) -> None:
    if _projection_uv is None or _projection_mask is None:
        raise RuntimeError("V10 projection arrays are unavailable")
    mesh = face.data
    if len(mesh.vertices) != TOTAL_VERTICES:
        raise RuntimeError(f"Unexpected v10 mesh vertex count: {len(mesh.vertices)}")

    old_uv = mesh.uv_layers.get("ProjectedUV")
    if old_uv is not None:
        mesh.uv_layers.remove(old_uv)
    uv_layer = mesh.uv_layers.new(name="ProjectedUV")
    for loop in mesh.loops:
        uv_layer.data[loop.index].uv = _projection_uv[loop.vertex_index]

    old_mask = mesh.color_attributes.get("ProjMask")
    if old_mask is not None:
        mesh.color_attributes.remove(old_mask)
    mask_attribute = mesh.color_attributes.new(
        name="ProjMask",
        type="FLOAT_COLOR",
        domain="POINT",
    )
    mask_rgba = np.ones((TOTAL_VERTICES, 4), dtype=np.float32)
    mask_rgba[:, :3] = _projection_mask[:, None]
    mask_attribute.data.foreach_set("color", mask_rgba.reshape(-1).tolist())

    for polygon in mesh.polygons:
        polygon.material_index = 0
    print(
        "BLENDER_FACEVERSE_V10_PROJECTED_UV=PROVEN "
        f"LOOPS={len(mesh.loops)} MASK_MEAN={float(np.mean(_projection_mask)):.6f}"
    )


def create_face_v10(*args, **kwargs):
    face, follow, targets = _original_create_corrected_face(*args, **kwargs)
    apply_projection_attributes(face)
    return face, follow, targets


def build_character_v10(follow: bpy.types.Object, colors_rgb: np.ndarray) -> dict[str, object]:
    result = _original_build_refined_character(follow, colors_rgb)
    removed = 0
    for obj in list(bpy.data.objects):
        if (
            obj.name == "CHAR_Antinous_HairCap"
            or obj.name.startswith("HAIR_")
            or obj.name.startswith("FACIALHAIR_")
            or obj.name.startswith("EYE_V8_")
            or obj.name.startswith("EYE_V9_")
            or obj.name.startswith("MOUTH_V9_")
            or obj.name.startswith("CHAR_Antinous_FittedHairShell")
        ):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    result["hair_cap"] = None
    result["strand_count"] = 0
    result["moustache_hair_count"] = 0
    result["sideburn_hair_count"] = 0
    result["removed_procedural_face_objects"] = removed
    result["variant"] = "FACEVERSE_CAMERA_PROJECTED_TEXTURE_V10"
    print(f"BLENDER_FACEVERSE_V10_PROCEDURAL_FACE_OBJECTS_REMOVED=PROVEN OBJECTS={removed}")
    return result


def resolve_output_dir() -> Path:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    for index, value in enumerate(argv):
        if value == "--output-dir" and index + 1 < len(argv):
            return Path(argv[index + 1]).resolve()
    raise RuntimeError("Could not resolve v10 output directory")


def render_projection_variants(output_dir: Path) -> list[dict[str, Any]]:
    material = bpy.data.materials.get(PROJECTION_MATERIAL_NAME)
    if material is None or not material.use_nodes:
        raise RuntimeError("V10 projected material is missing after scene build")
    strength_node = material.node_tree.nodes.get(PROJECTION_STRENGTH_NODE)
    if strength_node is None:
        raise RuntimeError("V10 projection-strength node is missing")
    scene = bpy.context.scene
    camera = bpy.data.objects.get("CAM_Hero")
    if camera is None:
        raise RuntimeError("V10 hero camera is missing")
    scene.camera = camera
    scene.frame_set(48)
    variants: list[dict[str, Any]] = []
    for strength in (0.70, 0.85, 1.00):
        strength_node.outputs[0].default_value = strength
        output_path = output_dir / f"projection_blend_{int(round(strength * 100)):03d}.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        if not output_path.is_file() or output_path.stat().st_size < 50000:
            raise RuntimeError(f"V10 projection variant is missing or too small: {output_path}")
        variants.append(
            {
                "strength": strength,
                "render": output_path.name,
                "bytes": output_path.stat().st_size,
            }
        )
        print(f"BLENDER_FACEVERSE_V10_PROJECTION_VARIANT=PROVEN STRENGTH={strength:.2f}")
    strength_node.outputs[0].default_value = 0.85
    blend_path = output_dir / "beggars_photoreal_recreation.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    return variants


def patch_v10_receipt(output_dir: Path, variants: list[dict[str, Any]]) -> None:
    receipt_path = output_dir / "scene_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["character_variant"] = "FACEVERSE_CAMERA_PROJECTED_TEXTURE_V10"
    receipt["visual_changes_v10"] = {
        "actual_public_keyframe_projected_to_true_mesh": True,
        "source_frame_plane_used": False,
        "procedural_hair_removed": True,
        "procedural_eye_spheres_removed": True,
        "procedural_dental_plate_removed": True,
        "projection_strengths_tested": [0.70, 0.85, 1.00],
        "selected_strength_in_blend": 0.85,
    }
    receipt["projection_variants"] = variants
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    load_projection_inputs()
    base.vertex_skin_material = projected_skin_material_v10
    v5.create_model_space_face_mesh = create_face_v10
    v6.build_refined_character = build_character_v10
    result = int(v6.main())
    output_dir = resolve_output_dir()
    variants = render_projection_variants(output_dir)
    patch_v10_receipt(output_dir, variants)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
