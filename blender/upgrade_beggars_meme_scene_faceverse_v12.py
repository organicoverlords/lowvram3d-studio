from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


CARD_NAME = "CHAR_Antinous_SourceAlignedFaceShell_V12"
MATTE_NAME = "CHAR_Antinous_SourceAlignedFaceMatte_V12"
CARD_MATERIAL_NAME = "MAT_Antinous_SourceAlignedFaceShell_V12"
MATTE_MATERIAL_NAME = "MAT_Antinous_SourceAlignedFaceMatte_V12"
SELECTED_SCALE = 1.0


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--base-receipt", required=True)
    parser.add_argument("--crop-x0", type=int, default=138)
    parser.add_argument("--crop-y0", type=int, default=18)
    parser.add_argument("--crop-x1", type=int, default=407)
    parser.add_argument("--crop-y1", type=int, default=249)
    parser.add_argument("--frame", type=int, default=48)
    return parser.parse_args(argv)


def set_input(node: bpy.types.Node, name: str, value: Any) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def smoothstep(edge0: float, edge1: float, value: float) -> float:
    if edge1 <= edge0:
        raise ValueError("smoothstep edges must be increasing")
    t = max(0.0, min(1.0, (value - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def shell_mask(nx: float, ny: float) -> float:
    ellipse = math.sqrt((nx / 0.96) ** 2 + ((ny - 0.02) / 1.04) ** 2)
    head = 1.0 - smoothstep(0.83, 1.0, ellipse)
    neck_vertical = 1.0 - smoothstep(-0.62, -0.36, ny)
    neck_horizontal = 1.0 - smoothstep(0.38, 0.66, abs(nx))
    neck = neck_vertical * neck_horizontal
    return max(0.0, min(1.0, max(head, neck)))


def make_shell_mesh(
    name: str,
    width: float,
    height: float,
    depth: float,
    columns: int,
    rows: int,
    crop: tuple[int, int, int, int],
    image_size: tuple[int, int],
) -> bpy.types.Mesh:
    vertices: list[tuple[float, float, float]] = []
    masks: list[float] = []
    grid_coordinates: list[tuple[float, float]] = []
    for row in range(rows + 1):
        ny = -1.0 + 2.0 * row / rows
        for column in range(columns + 1):
            nx = -1.0 + 2.0 * column / columns
            ellipse = (nx / 0.98) ** 2 + ((ny - 0.02) / 1.05) ** 2
            bulge = depth * max(0.0, 1.0 - ellipse) ** 1.35
            vertices.append((nx * width * 0.5, ny * height * 0.5, bulge))
            masks.append(shell_mask(nx, ny))
            grid_coordinates.append((nx, ny))

    faces: list[tuple[int, int, int, int]] = []
    stride = columns + 1
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

    x0, y0, x1, y1 = crop
    image_width, image_height = image_size
    uv_layer = mesh.uv_layers.new(name="SourceCropUV")
    for polygon in mesh.polygons:
        polygon.use_smooth = True
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            nx, ny = grid_coordinates[vertex_index]
            source_x = x0 + ((nx + 1.0) * 0.5) * (x1 - x0)
            source_y = y1 - ((ny + 1.0) * 0.5) * (y1 - y0)
            u = source_x / image_width
            v = 1.0 - source_y / image_height
            uv_layer.data[loop_index].uv = (u, v)

    mask_attribute = mesh.color_attributes.new(
        name="SourceShellMask",
        type="FLOAT_COLOR",
        domain="POINT",
    )
    rgba: list[float] = []
    for mask in masks:
        rgba.extend((mask, mask, mask, mask))
    mask_attribute.data.foreach_set("color", rgba)
    return mesh


def configure_transparency(material: bpy.types.Material) -> None:
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    if hasattr(material, "use_transparency_overlap"):
        material.use_transparency_overlap = False
    if hasattr(material, "show_transparent_back"):
        material.show_transparent_back = False


def source_shell_material(image: bpy.types.Image) -> bpy.types.Material:
    material = bpy.data.materials.get(CARD_MATERIAL_NAME) or bpy.data.materials.new(
        CARD_MATERIAL_NAME
    )
    material.use_nodes = True
    configure_transparency(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    mix_shader = nodes.new("ShaderNodeMixShader")
    uv_map = nodes.new("ShaderNodeUVMap")
    uv_map.uv_map = "SourceCropUV"
    image_node = nodes.new("ShaderNodeTexImage")
    image_node.name = "PublicReferenceSourceFrame"
    image_node.image = image
    image_node.extension = "CLIP"
    image_node.interpolation = "Linear"
    balance = nodes.new("ShaderNodeHueSaturation")
    balance.inputs["Saturation"].default_value = 0.94
    balance.inputs["Value"].default_value = 0.93
    mask = nodes.new("ShaderNodeAttribute")
    mask.attribute_name = "SourceShellMask"

    set_input(principled, "Roughness", 0.72)
    set_input(principled, "Specular IOR Level", 0.10)
    set_input(principled, "Metallic", 0.0)
    set_input(principled, "Emission Strength", 0.20)

    links.new(uv_map.outputs["UV"], image_node.inputs["Vector"])
    links.new(image_node.outputs["Color"], balance.inputs["Color"])
    links.new(balance.outputs["Color"], principled.inputs["Base Color"])
    emission = principled.inputs.get("Emission Color")
    if emission is not None:
        links.new(balance.outputs["Color"], emission)
    links.new(mask.outputs["Fac"], mix_shader.inputs[0])
    links.new(transparent.outputs["BSDF"], mix_shader.inputs[1])
    links.new(principled.outputs["BSDF"], mix_shader.inputs[2])
    links.new(mix_shader.outputs["Shader"], output.inputs["Surface"])
    return material


def matte_material() -> bpy.types.Material:
    material = bpy.data.materials.get(MATTE_MATERIAL_NAME) or bpy.data.materials.new(
        MATTE_MATERIAL_NAME
    )
    material.use_nodes = True
    configure_transparency(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    mix_shader = nodes.new("ShaderNodeMixShader")
    mask = nodes.new("ShaderNodeAttribute")
    mask.attribute_name = "SourceShellMask"
    set_input(principled, "Base Color", (0.095, 0.022, 0.014, 1.0))
    set_input(principled, "Roughness", 0.92)
    set_input(principled, "Specular IOR Level", 0.02)
    links.new(mask.outputs["Fac"], mix_shader.inputs[0])
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


def create_shell_objects(
    source_image: bpy.types.Image,
    crop: tuple[int, int, int, int],
) -> tuple[bpy.types.Object, bpy.types.Object]:
    face = bpy.data.objects.get("CHAR_Antinous")
    camera = bpy.data.objects.get("CAM_Hero")
    if face is None or camera is None:
        raise RuntimeError("V12 requires CHAR_Antinous and CAM_Hero")

    for name in (CARD_NAME, MATTE_NAME):
        existing = bpy.data.objects.get(name)
        if existing is not None:
            bpy.data.objects.remove(existing, do_unlink=True)

    width = 2.14
    height = 2.40
    mesh = make_shell_mesh(
        "MESH_Antinous_SourceAlignedFaceShell_V12",
        width=width,
        height=height,
        depth=0.105,
        columns=48,
        rows=52,
        crop=crop,
        image_size=(int(source_image.size[0]), int(source_image.size[1])),
    )
    card = bpy.data.objects.new(CARD_NAME, mesh)
    bpy.context.collection.objects.link(card)
    card.data.materials.append(source_shell_material(source_image))

    matte = bpy.data.objects.new(MATTE_NAME, mesh.copy())
    bpy.context.collection.objects.link(matte)
    matte.data.materials.append(matte_material())

    center = world_bounds_center(face)
    camera_rotation = camera.matrix_world.to_quaternion()
    forward = camera_rotation @ Vector((0.0, 0.0, -1.0))
    up = camera_rotation @ Vector((0.0, 1.0, 0.0))
    target = center - forward * 0.095 + up * 0.075

    card.rotation_mode = "QUATERNION"
    card.rotation_quaternion = camera_rotation
    card.location = target
    matte.rotation_mode = "QUATERNION"
    matte.rotation_quaternion = camera_rotation
    matte.location = target + forward * 0.035
    matte.scale = (1.035, 1.035, 1.0)

    card["v12_source_crop"] = list(crop)
    card["v12_source_image"] = source_image.name
    card["v12_camera_aligned"] = True
    card["v12_2_5d_depth"] = 0.105
    matte["v12_occlusion_matte"] = True
    return card, matte


def ensure_render(path: Path) -> None:
    if not path.is_file() or path.stat().st_size < 50000:
        raise RuntimeError(f"V12 render missing or implausibly small: {path}")


def render_variants(
    output_dir: Path,
    card: bpy.types.Object,
    matte: bpy.types.Object,
    frame: int,
) -> list[dict[str, Any]]:
    scene = bpy.context.scene
    camera = bpy.data.objects.get("CAM_Hero")
    if camera is None:
        raise RuntimeError("V12 hero camera is missing")
    scene.camera = camera
    scene.frame_set(frame)
    bpy.context.view_layer.update()

    variants: list[dict[str, Any]] = []
    for scale in (0.94, 1.00, 1.06):
        card.scale = (scale, scale, 1.0)
        matte.scale = (scale * 1.035, scale * 1.035, 1.0)
        output_path = output_dir / f"source_shell_{int(round(scale * 100)):03d}.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        ensure_render(output_path)
        variants.append(
            {
                "scale": scale,
                "render": output_path.name,
                "bytes": output_path.stat().st_size,
            }
        )
        print(f"BLENDER_FACEVERSE_V12_SOURCE_SHELL_VARIANT=PROVEN SCALE={scale:.2f}")

    card.scale = (SELECTED_SCALE, SELECTED_SCALE, 1.0)
    matte.scale = (SELECTED_SCALE * 1.035, SELECTED_SCALE * 1.035, 1.0)
    return variants


def patch_receipt(
    output_dir: Path,
    base_receipt_path: Path,
    source_image: Path,
    crop: tuple[int, int, int, int],
    variants: list[dict[str, Any]],
) -> None:
    receipt = json.loads(base_receipt_path.read_text(encoding="utf-8"))
    receipt["classification"] = "USER_VISUAL_REVIEW_REQUIRED"
    receipt["character_variant"] = "FACEVERSE_V11_PLUS_SOURCE_ALIGNED_2_5D_SHELL_V12"
    receipt["claim_policy"] = (
        "The v12 shell, renders and save/reload are machine-proven. Visual likeness, "
        "photoreal match and meme timing remain NOT_PROVEN until direct visual review."
    )
    receipt["visual_changes_v12"] = {
        "source_aligned_face_and_hair_shell": True,
        "underlying_true_3d_face_preserved": True,
        "camera_aligned_dense_grid": [49, 53],
        "shell_depth_world": 0.105,
        "soft_head_and_neck_mask": True,
        "occlusion_matte": True,
        "source_crop_pixels": list(crop),
        "source_frame_embedded_in_blend": True,
        "source_frame_path_at_build": str(source_image),
        "selected_scale": SELECTED_SCALE,
        "scales_tested": [0.94, 1.00, 1.06],
        "animation_rendered": False,
    }
    receipt["v12_variants"] = variants
    (output_dir / "scene_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "v12_visual_report.json").write_text(
        json.dumps(
            {
                "classification": "USER_VISUAL_REVIEW_REQUIRED",
                "machine_status": "PROVEN",
                "visual_status": "NOT_PROVEN",
                "card_object": CARD_NAME,
                "matte_object": MATTE_NAME,
                "source_crop_pixels": list(crop),
                "variants": variants,
                "selected_scale": SELECTED_SCALE,
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
    source_image_path = Path(args.source_image).resolve()
    base_receipt_path = Path(args.base_receipt).resolve()
    if not source_image_path.is_file():
        raise RuntimeError(f"V12 source image is missing: {source_image_path}")
    if not base_receipt_path.is_file():
        raise RuntimeError(f"V12 base receipt is missing: {base_receipt_path}")

    scene = bpy.context.scene
    scene.frame_set(args.frame)
    bpy.context.view_layer.update()

    image = bpy.data.images.load(str(source_image_path), check_existing=True)
    image.colorspace_settings.name = "sRGB"
    image.pack()
    crop = (args.crop_x0, args.crop_y0, args.crop_x1, args.crop_y1)
    if not (0 <= crop[0] < crop[2] <= image.size[0]):
        raise RuntimeError(f"Invalid v12 horizontal crop {crop} for image {tuple(image.size)}")
    if not (0 <= crop[1] < crop[3] <= image.size[1]):
        raise RuntimeError(f"Invalid v12 vertical crop {crop} for image {tuple(image.size)}")

    card, matte = create_shell_objects(image, crop)
    variants = render_variants(output_dir, card, matte, args.frame)
    patch_receipt(output_dir, base_receipt_path, source_image_path, crop, variants)

    blend_path = output_dir / "beggars_photoreal_recreation.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    if not blend_path.is_file() or blend_path.stat().st_size < 1000000:
        raise RuntimeError("V12 Blender file is missing or implausibly small")

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    reloaded_card = bpy.data.objects.get(CARD_NAME)
    reloaded_matte = bpy.data.objects.get(MATTE_NAME)
    if reloaded_card is None or reloaded_matte is None:
        raise RuntimeError("V12 shell objects did not survive save/reload")
    if not any(image.packed_file is not None for image in bpy.data.images):
        raise RuntimeError("V12 source image did not survive as packed data")

    scene = bpy.context.scene
    camera = bpy.data.objects.get("CAM_Hero")
    if camera is None:
        raise RuntimeError("V12 hero camera missing after reload")
    scene.camera = camera
    scene.frame_set(args.frame)
    hero_path = output_dir / "hero_source_shell_render.png"
    scene.render.filepath = str(hero_path)
    bpy.ops.render.render(write_still=True)
    ensure_render(hero_path)
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    print(
        "BLENDER_FACEVERSE_V12_SOURCE_ALIGNED_SHELL=PROVEN "
        f"VERTICES={len(reloaded_card.data.vertices)} FACES={len(reloaded_card.data.polygons)} "
        f"CROP={list(crop)}"
    )
    print("BLENDER_FACEVERSE_V12_SAVE_RELOAD_STILL=PROVEN")
    print("BEGGARS_FACEVERSE_V12=USER_VISUAL_REVIEW_REQUIRED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
