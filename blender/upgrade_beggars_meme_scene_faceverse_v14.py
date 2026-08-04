from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy


CARD_NAME = "CHAR_Antinous_CameraSourceHead_V14"
CARD_MATERIAL = "MAT_Antinous_CameraSourceHead_V14"
SELECTED = {"scale": 0.96, "x": -0.20, "y": 0.25, "z": -10.20}


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cutout", required=True)
    parser.add_argument("--cutout-report", required=True)
    parser.add_argument("--base-receipt", required=True)
    parser.add_argument("--frame", type=int, default=48)
    return parser.parse_args(argv)


def set_input(node: bpy.types.Node, name: str, value: Any) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def create_card_mesh(name: str, aspect: float) -> bpy.types.Mesh:
    width = 2.30
    height = width / aspect
    vertices = [
        (-width * 0.5, -height * 0.5, 0.0),
        (width * 0.5, -height * 0.5, 0.0),
        (width * 0.5, height * 0.5, 0.0),
        (-width * 0.5, height * 0.5, 0.0),
    ]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3)])
    mesh.update(calc_edges=True)
    uv_layer = mesh.uv_layers.new(name="SourceHeadUV")
    uv_values = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    for loop in mesh.loops:
        uv_layer.data[loop.index].uv = uv_values[loop.vertex_index]
    return mesh


def create_opaque_material(image: bpy.types.Image) -> bpy.types.Material:
    material = bpy.data.materials.get(CARD_MATERIAL) or bpy.data.materials.new(CARD_MATERIAL)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    uv_map = nodes.new("ShaderNodeUVMap")
    uv_map.uv_map = "SourceHeadUV"
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.name = "FeatheredPublicReferenceHead"
    image_node.image = image
    image_node.extension = "CLIP"
    image_node.interpolation = "Linear"
    background = nodes.new("ShaderNodeRGB")
    background.outputs[0].default_value = (0.014, 0.0035, 0.0025, 1.0)
    composite = nodes.new("ShaderNodeMixRGB")
    composite.blend_type = "MIX"

    set_input(principled, "Roughness", 1.0)
    set_input(principled, "Specular IOR Level", 0.0)
    set_input(principled, "Metallic", 0.0)
    set_input(principled, "Emission Strength", 0.90)

    links.new(uv_map.outputs["UV"], image_node.inputs["Vector"])
    links.new(image_node.outputs["Alpha"], composite.inputs[0])
    links.new(background.outputs[0], composite.inputs[1])
    links.new(image_node.outputs["Color"], composite.inputs[2])
    links.new(composite.outputs["Color"], principled.inputs["Base Color"])
    emission = principled.inputs.get("Emission Color")
    if emission is not None:
        links.new(composite.outputs["Color"], emission)
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return material


def hide_rejected_head_geometry() -> list[str]:
    exact_names = {
        "CHAR_Antinous",
        "CHAR_Antinous_HairCap",
        "CHAR_Antinous_FittedHairShell",
        "CHAR_Antinous_Neck",
    }
    prefixes = ("HAIR_", "FACIALHAIR_", "EYE_", "MOUTH_")
    hidden: list[str] = []
    for obj in bpy.data.objects:
        if obj.name in exact_names or obj.name.startswith(prefixes):
            obj.hide_render = True
            obj.hide_set(True)
            hidden.append(obj.name)
    if "CHAR_Antinous" not in hidden:
        raise RuntimeError("V14 could not hide the rejected synthetic face")
    return sorted(hidden)


def create_camera_card(image: bpy.types.Image) -> bpy.types.Object:
    camera = bpy.data.objects.get("CAM_Hero")
    if camera is None:
        raise RuntimeError("V14 requires CAM_Hero")
    existing = bpy.data.objects.get(CARD_NAME)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)

    aspect = float(image.size[0]) / float(image.size[1])
    mesh = create_card_mesh("MESH_Antinous_CameraSourceHead_V14", aspect)
    card = bpy.data.objects.new(CARD_NAME, mesh)
    bpy.context.collection.objects.link(card)
    card.data.materials.append(create_opaque_material(image))
    card.parent = camera
    card.location = (SELECTED["x"], SELECTED["y"], SELECTED["z"])
    card.rotation_euler = (0.0, 0.0, 0.0)
    card.scale = (SELECTED["scale"], SELECTED["scale"], 1.0)
    card["v14_opaque_camera_card"] = True
    card["v14_image_aspect"] = aspect
    card["v14_camera_local_selected"] = dict(SELECTED)
    return card


def ensure_render(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 50000:
        raise RuntimeError(f"V14 render missing or implausibly small: {path}")


def render_variants(output_dir: Path, card: bpy.types.Object, frame: int) -> list[dict[str, Any]]:
    scene = bpy.context.scene
    camera = bpy.data.objects.get("CAM_Hero")
    if camera is None:
        raise RuntimeError("V14 hero camera missing")
    scene.camera = camera
    scene.frame_set(frame)

    tests = [
        (0.90, -0.20, 0.25, "scale_090"),
        (0.96, -0.20, 0.25, "scale_096"),
        (1.02, -0.20, 0.25, "scale_102"),
        (0.96, -0.35, 0.25, "left_035"),
        (0.96, -0.05, 0.25, "right_005"),
        (0.96, -0.20, 0.10, "low_010"),
        (0.96, -0.20, 0.40, "high_040"),
    ]
    variants: list[dict[str, Any]] = []
    for scale, x_value, y_value, label in tests:
        card.scale = (scale, scale, 1.0)
        card.location = (x_value, y_value, SELECTED["z"])
        output_path = output_dir / f"camera_head_{label}.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        ensure_render(output_path)
        variants.append(
            {
                "scale": scale,
                "camera_local_x": x_value,
                "camera_local_y": y_value,
                "camera_local_z": SELECTED["z"],
                "render": output_path.name,
                "bytes": output_path.stat().st_size,
            }
        )
        print(
            "BLENDER_FACEVERSE_V14_CAMERA_HEAD_VARIANT=PROVEN "
            f"SCALE={scale:.2f} X={x_value:.2f} Y={y_value:.2f}"
        )

    card.scale = (SELECTED["scale"], SELECTED["scale"], 1.0)
    card.location = (SELECTED["x"], SELECTED["y"], SELECTED["z"])
    return variants


def patch_receipt(
    output_dir: Path,
    base_receipt_path: Path,
    cutout_report_path: Path,
    hidden: list[str],
    variants: list[dict[str, Any]],
) -> None:
    receipt = json.loads(base_receipt_path.read_text(encoding="utf-8"))
    cutout_report = json.loads(cutout_report_path.read_text(encoding="utf-8"))
    receipt["classification"] = "USER_VISUAL_REVIEW_REQUIRED"
    receipt["character_variant"] = "FACEVERSE_BODY_PLUS_OPAQUE_CAMERA_HEAD_V14"
    receipt["claim_policy"] = (
        "The v14 opaque camera-space head, Blender body/set integration, renders and save/reload "
        "are machine-proven. Visual quality remains NOT_PROVEN until direct review."
    )
    receipt["visual_changes_v14"] = {
        "transparent_shader_route_rejected": True,
        "opaque_scene_matched_matte": True,
        "camera_space_anchor": True,
        "rejected_face_hair_and_neck_hidden": True,
        "hidden_objects": hidden,
        "public_reference_aspect_preserved": True,
        "selected_camera_local_transform": dict(SELECTED),
        "procedural_torso_set_and_lighting_preserved": True,
        "animation_rendered": False,
        "cutout_report": cutout_report,
    }
    receipt["v14_variants"] = variants
    (output_dir / "scene_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "v14_visual_report.json").write_text(
        json.dumps(
            {
                "classification": "USER_VISUAL_REVIEW_REQUIRED",
                "machine_status": "PROVEN",
                "visual_status": "NOT_PROVEN",
                "card_object": CARD_NAME,
                "selected": dict(SELECTED),
                "hidden_objects": hidden,
                "variants": variants,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cutout_path = Path(args.cutout).resolve()
    cutout_report_path = Path(args.cutout_report).resolve()
    base_receipt_path = Path(args.base_receipt).resolve()
    for path in (cutout_path, cutout_report_path, base_receipt_path):
        if not path.is_file():
            raise RuntimeError(f"V14 input missing: {path}")

    image = bpy.data.images.load(str(cutout_path), check_existing=True)
    image.colorspace_settings.name = "sRGB"
    image.alpha_mode = "STRAIGHT"
    image.pack()
    hidden = hide_rejected_head_geometry()
    card = create_camera_card(image)
    variants = render_variants(output_dir, card, args.frame)
    patch_receipt(output_dir, base_receipt_path, cutout_report_path, hidden, variants)

    blend_path = output_dir / "beggars_photoreal_recreation.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    if not blend_path.is_file() or blend_path.stat().st_size < 1000000:
        raise RuntimeError("V14 Blender file missing or implausibly small")

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    reloaded = bpy.data.objects.get(CARD_NAME)
    camera = bpy.data.objects.get("CAM_Hero")
    face = bpy.data.objects.get("CHAR_Antinous")
    if reloaded is None or camera is None or reloaded.parent != camera:
        raise RuntimeError("V14 camera card did not survive save/reload")
    if face is None or not face.hide_render:
        raise RuntimeError("V14 hidden synthetic face did not survive save/reload")
    if not any(image.packed_file is not None for image in bpy.data.images):
        raise RuntimeError("V14 source image was not packed")

    scene = bpy.context.scene
    scene.camera = camera
    scene.frame_set(args.frame)
    final_path = output_dir / "hero_camera_head_render.png"
    scene.render.filepath = str(final_path)
    bpy.ops.render.render(write_still=True)
    ensure_render(final_path)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    print(
        "BLENDER_FACEVERSE_V14_OPAQUE_CAMERA_HEAD=PROVEN "
        f"IMAGE_SIZE={image.size[0]}x{image.size[1]} HIDDEN={len(hidden)} "
        f"SELECTED={json.dumps(SELECTED, sort_keys=True)}"
    )
    print("BLENDER_FACEVERSE_V14_SAVE_RELOAD_STILL=PROVEN")
    print("BEGGARS_FACEVERSE_V14=USER_VISUAL_REVIEW_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
