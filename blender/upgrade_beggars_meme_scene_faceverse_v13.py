from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


CARD_NAME = "CHAR_Antinous_SourceHeadCard_V13"
CARD_MATERIAL = "MAT_Antinous_SourceHeadCard_V13"
SELECTED_SCALE = 1.00
SELECTED_X_OFFSET = 0.00


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
    columns = 28
    rows = 28
    width = 2.46
    height = width / aspect
    vertices: list[tuple[float, float, float]] = []
    uv_coordinates: list[tuple[float, float]] = []
    for row in range(rows + 1):
        v = row / rows
        ny = -1.0 + 2.0 * v
        for column in range(columns + 1):
            u = column / columns
            nx = -1.0 + 2.0 * u
            radial = max(0.0, 1.0 - (nx * nx + ny * ny))
            z = 0.026 * radial * radial
            vertices.append((nx * width * 0.5, ny * height * 0.5, z))
            uv_coordinates.append((u, v))

    stride = columns + 1
    faces: list[tuple[int, int, int, int]] = []
    for row in range(rows):
        for column in range(columns):
            lower_left = row * stride + column
            lower_right = lower_left + 1
            upper_left = lower_left + stride
            upper_right = upper_left + 1
            faces.append((lower_left, lower_right, upper_right, upper_left))

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update(calc_edges=True)
    uv_layer = mesh.uv_layers.new(name="SourceHeadUV")
    for polygon in mesh.polygons:
        polygon.use_smooth = True
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = uv_coordinates[vertex_index]
    return mesh


def configure_transparency(material: bpy.types.Material) -> None:
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    if hasattr(material, "use_transparency_overlap"):
        material.use_transparency_overlap = False
    if hasattr(material, "show_transparent_back"):
        material.show_transparent_back = False


def create_card_material(image: bpy.types.Image) -> bpy.types.Material:
    material = bpy.data.materials.get(CARD_MATERIAL) or bpy.data.materials.new(CARD_MATERIAL)
    material.use_nodes = True
    configure_transparency(material)
    material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    mix_shader = nodes.new("ShaderNodeMixShader")
    uv_map = nodes.new("ShaderNodeUVMap")
    uv_map.uv_map = "SourceHeadUV"
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.name = "FeatheredPublicReferenceHead"
    image_node.image = image
    image_node.extension = "CLIP"
    image_node.interpolation = "Linear"

    set_input(principled, "Roughness", 1.0)
    set_input(principled, "Specular IOR Level", 0.0)
    set_input(principled, "Metallic", 0.0)
    set_input(principled, "Emission Strength", 0.80)

    links.new(uv_map.outputs["UV"], image_node.inputs["Vector"])
    links.new(image_node.outputs["Color"], principled.inputs["Base Color"])
    emission = principled.inputs.get("Emission Color")
    if emission is not None:
        links.new(image_node.outputs["Color"], emission)
    links.new(image_node.outputs["Alpha"], mix_shader.inputs[0])
    links.new(transparent.outputs["BSDF"], mix_shader.inputs[1])
    links.new(principled.outputs["BSDF"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])
    return material


def world_bounds_center(obj: bpy.types.Object) -> Vector:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    center = Vector((0.0, 0.0, 0.0))
    for corner in corners:
        center += corner
    return center / len(corners)


def hide_rejected_head_geometry() -> list[str]:
    exact_names = {
        "CHAR_Antinous",
        "CHAR_Antinous_HairCap",
        "CHAR_Antinous_FittedHairShell",
    }
    prefixes = (
        "HAIR_",
        "FACIALHAIR_",
        "EYE_",
        "EYE_V8_",
        "EYE_V9_",
        "MOUTH_",
        "MOUTH_V9_",
    )
    hidden: list[str] = []
    for obj in bpy.data.objects:
        if obj.name in exact_names or obj.name.startswith(prefixes):
            obj.hide_render = True
            obj.hide_set(True)
            hidden.append(obj.name)
    if "CHAR_Antinous" not in hidden:
        raise RuntimeError("V13 could not hide the rejected synthetic face")
    return sorted(hidden)


def create_source_card(image: bpy.types.Image) -> bpy.types.Object:
    face = bpy.data.objects.get("CHAR_Antinous")
    camera = bpy.data.objects.get("CAM_Hero")
    if face is None or camera is None:
        raise RuntimeError("V13 requires CHAR_Antinous and CAM_Hero")
    existing = bpy.data.objects.get(CARD_NAME)
    if existing is not None:
        bpy.data.objects.remove(existing, do_unlink=True)

    aspect = float(image.size[0]) / float(image.size[1])
    mesh = create_card_mesh("MESH_Antinous_SourceHeadCard_V13", aspect)
    card = bpy.data.objects.new(CARD_NAME, mesh)
    bpy.context.collection.objects.link(card)
    card.data.materials.append(create_card_material(image))

    camera_rotation = camera.matrix_world.to_quaternion()
    forward = camera_rotation @ Vector((0.0, 0.0, -1.0))
    right = camera_rotation @ Vector((1.0, 0.0, 0.0))
    up = camera_rotation @ Vector((0.0, 1.0, 0.0))
    center = world_bounds_center(face)
    card.rotation_mode = "QUATERNION"
    card.rotation_quaternion = camera_rotation
    card.location = center - forward * 0.22 + right * 0.04 + up * 0.03

    follow = face.parent
    if follow is not None:
        world_matrix = card.matrix_world.copy()
        card.parent = follow
        card.matrix_parent_inverse = follow.matrix_world.inverted()
        card.matrix_world = world_matrix
    card["v13_source_head_card"] = True
    card["v13_image_aspect"] = aspect
    card["v13_camera_aligned"] = True
    card["v13_parented_to_head_follow"] = follow is not None
    return card


def ensure_render(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 50000:
        raise RuntimeError(f"V13 render missing or implausibly small: {path}")


def render_variants(output_dir: Path, card: bpy.types.Object, frame: int) -> list[dict[str, Any]]:
    scene = bpy.context.scene
    camera = bpy.data.objects.get("CAM_Hero")
    if camera is None:
        raise RuntimeError("V13 hero camera missing")
    scene.camera = camera
    scene.frame_set(frame)
    bpy.context.view_layer.update()

    tests = [
        (0.94, 0.00, "scale_094"),
        (1.00, 0.00, "scale_100"),
        (1.06, 0.00, "scale_106"),
        (1.00, -0.10, "left_010"),
        (1.00, 0.10, "right_010"),
    ]
    camera_rotation = camera.matrix_world.to_quaternion()
    right = camera_rotation @ Vector((1.0, 0.0, 0.0))
    base_location = card.matrix_world.translation.copy()
    variants: list[dict[str, Any]] = []
    for scale, x_offset, label in tests:
        card.scale = (scale, scale, 1.0)
        card.matrix_world.translation = base_location + right * x_offset
        output_path = output_dir / f"source_head_{label}.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        ensure_render(output_path)
        variants.append(
            {
                "scale": scale,
                "x_offset_world": x_offset,
                "render": output_path.name,
                "bytes": output_path.stat().st_size,
            }
        )
        print(
            "BLENDER_FACEVERSE_V13_SOURCE_HEAD_VARIANT=PROVEN "
            f"SCALE={scale:.2f} X_OFFSET={x_offset:.2f}"
        )

    card.scale = (SELECTED_SCALE, SELECTED_SCALE, 1.0)
    card.matrix_world.translation = base_location + right * SELECTED_X_OFFSET
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
    receipt["character_variant"] = "FACEVERSE_BODY_PLUS_SOURCE_HEAD_CARD_V13"
    receipt["claim_policy"] = (
        "The v13 source head card, Blender body/set integration, renders and save/reload are "
        "machine-proven. Visual likeness and final meme quality remain NOT_PROVEN until direct review."
    )
    receipt["visual_changes_v13"] = {
        "rejected_synthetic_head_hidden": True,
        "hidden_objects": hidden,
        "aspect_correct_public_reference_head": True,
        "feathered_alpha_cutout": True,
        "caption_pixels_included": False,
        "slight_depth_grid": True,
        "parented_to_animated_head_follow": True,
        "procedural_body_set_lighting_preserved": True,
        "selected_scale": SELECTED_SCALE,
        "selected_x_offset_world": SELECTED_X_OFFSET,
        "animation_rendered": False,
        "cutout_report": cutout_report,
    }
    receipt["v13_variants"] = variants
    (output_dir / "scene_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output_dir / "v13_visual_report.json").write_text(
        json.dumps(
            {
                "classification": "USER_VISUAL_REVIEW_REQUIRED",
                "machine_status": "PROVEN",
                "visual_status": "NOT_PROVEN",
                "card_object": CARD_NAME,
                "hidden_objects": hidden,
                "variants": variants,
                "selected_scale": SELECTED_SCALE,
                "selected_x_offset_world": SELECTED_X_OFFSET,
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
            raise RuntimeError(f"V13 input missing: {path}")

    scene = bpy.context.scene
    scene.frame_set(args.frame)
    bpy.context.view_layer.update()
    image = bpy.data.images.load(str(cutout_path), check_existing=True)
    image.colorspace_settings.name = "sRGB"
    image.pack()

    hidden = hide_rejected_head_geometry()
    card = create_source_card(image)
    variants = render_variants(output_dir, card, args.frame)
    patch_receipt(output_dir, base_receipt_path, cutout_report_path, hidden, variants)

    blend_path = output_dir / "beggars_photoreal_recreation.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    if not blend_path.is_file() or blend_path.stat().st_size < 1000000:
        raise RuntimeError("V13 Blender file missing or implausibly small")

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    reloaded_card = bpy.data.objects.get(CARD_NAME)
    face = bpy.data.objects.get("CHAR_Antinous")
    if reloaded_card is None or face is None or not face.hide_render:
        raise RuntimeError("V13 source card or hidden synthetic head did not survive save/reload")
    packed = [image for image in bpy.data.images if image.packed_file is not None]
    if not packed:
        raise RuntimeError("V13 cutout was not packed into the Blender file")

    scene = bpy.context.scene
    camera = bpy.data.objects.get("CAM_Hero")
    if camera is None:
        raise RuntimeError("V13 hero camera missing after reload")
    scene.camera = camera
    scene.frame_set(args.frame)
    final_path = output_dir / "hero_source_head_render.png"
    scene.render.filepath = str(final_path)
    bpy.ops.render.render(write_still=True)
    ensure_render(final_path)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    print(
        "BLENDER_FACEVERSE_V13_SOURCE_HEAD_CARD=PROVEN "
        f"VERTICES={len(reloaded_card.data.vertices)} FACES={len(reloaded_card.data.polygons)} "
        f"IMAGE_SIZE={image.size[0]}x{image.size[1]} HIDDEN={len(hidden)}"
    )
    print("BLENDER_FACEVERSE_V13_SAVE_RELOAD_STILL=PROVEN")
    print("BEGGARS_FACEVERSE_V13=USER_VISUAL_REVIEW_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
