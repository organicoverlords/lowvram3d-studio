from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Build the photoreal beggars meme recreation scene.")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--render-engine", choices=("eevee", "cycles"), default="eevee")
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def set_input(node: bpy.types.Node, name: str, value) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def principled_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float,
    metallic: float = 0.0,
    emission: tuple[float, float, float, float] | None = None,
    emission_strength: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        raise RuntimeError("Principled BSDF node is unavailable")
    set_input(bsdf, "Base Color", color)
    set_input(bsdf, "Roughness", roughness)
    set_input(bsdf, "Metallic", metallic)
    set_input(bsdf, "Specular IOR Level", 0.33)
    if emission is not None:
        set_input(bsdf, "Emission Color", emission)
        set_input(bsdf, "Emission Strength", emission_strength)
    return material


def vertex_skin_material() -> bpy.types.Material:
    material = bpy.data.materials.new("MAT_Antinous_Face_VertexSkin")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "Col"
    set_input(bsdf, "Roughness", 0.46)
    set_input(bsdf, "Specular IOR Level", 0.28)
    set_input(bsdf, "Subsurface Weight", 0.055)
    set_input(bsdf, "Emission Strength", 0.09)
    links.new(attribute.outputs["Color"], bsdf.inputs["Base Color"])
    emission_socket = bsdf.inputs.get("Emission Color")
    if emission_socket is not None:
        links.new(attribute.outputs["Color"], emission_socket)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def create_uv_sphere(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    segments: int = 48,
    ring_count: int = 24,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=ring_count,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def create_cube(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    bevel: float = 0.0,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    if bevel > 0:
        modifier = obj.modifiers.new("SoftEdges", "BEVEL")
        modifier.width = bevel
        modifier.segments = 3
    return obj


def create_curve(
    name: str,
    points: list[tuple[float, float, float]],
    material: bpy.types.Material,
    bevel_depth: float,
    parent: bpy.types.Object | None = None,
) -> bpy.types.Object:
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 2
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for bezier_point, coordinate in zip(spline.bezier_points, points):
        bezier_point.co = coordinate
        bezier_point.handle_left_type = "AUTO"
        bezier_point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    curve.materials.append(material)
    if parent is not None:
        obj.parent = parent
    return obj


def add_point_light(
    name: str,
    location: tuple[float, float, float],
    energy: float,
    color: tuple[float, float, float],
    radius: float,
) -> bpy.types.Object:
    light_data = bpy.data.lights.new(name=name, type="POINT")
    light_data.energy = energy
    light_data.color = color
    light_data.shadow_soft_size = radius
    light_obj = bpy.data.objects.new(name, light_data)
    light_obj.location = location
    bpy.context.collection.objects.link(light_obj)
    return light_obj


def add_area_light(
    name: str,
    location: tuple[float, float, float],
    energy: float,
    color: tuple[float, float, float],
    size: float,
    target: Vector,
) -> bpy.types.Object:
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = energy
    light_data.color = color
    light_data.shape = "DISK"
    light_data.size = size
    light_obj = bpy.data.objects.new(name, light_data)
    light_obj.location = location
    bpy.context.collection.objects.link(light_obj)
    look_at(light_obj, target)
    return light_obj


def transform_vertices(
    vertices_image: np.ndarray,
    center_x: float,
    center_y: float,
    pixel_scale: float,
    base_depth: float,
) -> np.ndarray:
    output = np.empty((vertices_image.shape[1], 3), dtype=np.float32)
    output[:, 0] = (vertices_image[0] - center_x) * pixel_scale
    output[:, 2] = (center_y - vertices_image[1]) * pixel_scale
    output[:, 1] = -(vertices_image[2] - base_depth) * pixel_scale * 0.72
    return output


def create_face_mesh(
    vertices_sequence: np.ndarray,
    triangles: np.ndarray,
    colors_rgb: np.ndarray,
    boxes: np.ndarray,
    image_size: tuple[int, int],
    scene_frame_count: int,
) -> tuple[bpy.types.Object, bpy.types.Object, list[int]]:
    keyframe_index = int(len(vertices_sequence) * 0.62)
    keyframe_index = min(max(keyframe_index, 0), len(vertices_sequence) - 1)
    key_box = boxes[keyframe_index]
    center_x = float((key_box[0] + key_box[2]) * 0.5)
    center_y = float((key_box[1] + key_box[3]) * 0.5)
    face_height_pixels = max(float(key_box[3] - key_box[1]), 1.0)
    pixel_scale = 2.30 / face_height_pixels
    base_depth = float(np.median(vertices_sequence[keyframe_index, 2]))

    transformed = np.stack(
        [
            transform_vertices(frame_vertices, center_x, center_y, pixel_scale, base_depth)
            for frame_vertices in vertices_sequence
        ],
        axis=0,
    )

    maximum_shape_keys = 42
    if len(transformed) <= maximum_shape_keys:
        selected_indices = list(range(len(transformed)))
    else:
        selected_indices = sorted(
            set(int(round(value)) for value in np.linspace(0, len(transformed) - 1, maximum_shape_keys))
        )

    base_vertices = transformed[selected_indices[0]]
    mesh = bpy.data.meshes.new("MESH_Antinous_ReconstructedFace")
    mesh.from_pydata(base_vertices.tolist(), [], triangles.tolist())
    mesh.update()
    face = bpy.data.objects.new("CHAR_Antinous", mesh)
    bpy.context.collection.objects.link(face)
    for polygon in mesh.polygons:
        polygon.use_smooth = True

    color_attribute = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    rgba = np.ones((colors_rgb.shape[0], 4), dtype=np.float32)
    rgba[:, :3] = np.clip(colors_rgb, 0.0, 1.0)
    color_attribute.data.foreach_set("color", rgba.reshape(-1).tolist())
    mesh.materials.append(vertex_skin_material())

    solidify = face.modifiers.new("FaceShell", "SOLIDIFY")
    solidify.thickness = 0.012
    solidify.offset = -1.0

    face.shape_key_add(name="Basis")
    target_frames: list[int] = []
    key_blocks = []
    for sequence_index in selected_indices:
        key = face.shape_key_add(name=f"Track_{sequence_index:04d}")
        key.data.foreach_set("co", transformed[sequence_index].reshape(-1).tolist())
        key.value = 0.0
        target = 1 + int(round(sequence_index / max(len(transformed) - 1, 1) * (scene_frame_count - 1)))
        target_frames.append(target)
        key_blocks.append(key)

    for index, (key, target) in enumerate(zip(key_blocks, target_frames)):
        previous_target = target_frames[index - 1] if index > 0 else 1
        next_target = target_frames[index + 1] if index + 1 < len(target_frames) else scene_frame_count
        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=previous_target)
        key.value = 1.0
        key.keyframe_insert(data_path="value", frame=target)
        key.value = 0.0
        key.keyframe_insert(data_path="value", frame=next_target)

    if face.data.shape_keys and face.data.shape_keys.animation_data and face.data.shape_keys.animation_data.action:
        for fcurve in face.data.shape_keys.animation_data.action.fcurves:
            for keyframe_point in fcurve.keyframe_points:
                keyframe_point.interpolation = "LINEAR"

    follow = bpy.data.objects.new("RIG_HeadFollow", None)
    bpy.context.collection.objects.link(follow)
    follow.empty_display_type = "SPHERE"
    follow.empty_display_size = 0.18

    base_box_height = max(float(key_box[3] - key_box[1]), 1.0)
    for sequence_index, target in zip(selected_indices, target_frames):
        box = boxes[sequence_index]
        cx = float((box[0] + box[2]) * 0.5)
        cy = float((box[1] + box[3]) * 0.5)
        scale_ratio = max(0.85, min(1.15, float(box[3] - box[1]) / base_box_height))
        follow.location = (
            (cx - center_x) * pixel_scale,
            0.0,
            (center_y - cy) * pixel_scale,
        )
        follow.scale = (scale_ratio, scale_ratio, scale_ratio)
        follow.keyframe_insert(data_path="location", frame=target)
        follow.keyframe_insert(data_path="scale", frame=target)

    if follow.animation_data and follow.animation_data.action:
        for fcurve in follow.animation_data.action.fcurves:
            for keyframe_point in fcurve.keyframe_points:
                keyframe_point.interpolation = "BEZIER"

    return face, follow, target_frames


def build_character(follow: bpy.types.Object, colors_rgb: np.ndarray) -> dict[str, bpy.types.Object]:
    black_cloth = principled_material("MAT_Black_AncientCloth", (0.008, 0.006, 0.008, 1.0), 0.72)
    dark_hair = principled_material("MAT_Dark_Hair", (0.012, 0.006, 0.004, 1.0), 0.36)
    gold = principled_material("MAT_Aged_Gold", (0.34, 0.12, 0.025, 1.0), 0.28, metallic=0.72)
    skin_average = np.clip(np.median(colors_rgb, axis=0), 0.05, 0.95)
    neck_skin = principled_material(
        "MAT_Neck_Skin",
        (float(skin_average[0]), float(skin_average[1]), float(skin_average[2]), 1.0),
        0.52,
    )

    hair_cap = create_uv_sphere(
        "CHAR_Antinous_HairCap",
        (0.0, 0.34, 0.42),
        (1.04, 0.62, 1.16),
        dark_hair,
        segments=64,
        ring_count=32,
    )
    hair_cap.parent = follow

    neck = create_uv_sphere(
        "CHAR_Antinous_Neck",
        (0.02, 0.36, -1.20),
        (0.58, 0.46, 0.83),
        neck_skin,
    )
    neck.parent = follow

    torso = create_uv_sphere(
        "CHAR_Antinous_Torso",
        (0.08, 0.55, -2.10),
        (1.92, 0.72, 1.16),
        black_cloth,
        segments=64,
        ring_count=32,
    )
    torso.parent = follow

    shoulder_left = create_uv_sphere(
        "CHAR_Antinous_Shoulder_L",
        (-1.60, 0.52, -1.70),
        (0.90, 0.62, 0.72),
        black_cloth,
    )
    shoulder_left.parent = follow
    shoulder_right = create_uv_sphere(
        "CHAR_Antinous_Shoulder_R",
        (1.60, 0.52, -1.70),
        (0.90, 0.62, 0.72),
        black_cloth,
    )
    shoulder_right.parent = follow

    neckline_points = [
        (-0.78, -0.08, -1.46),
        (-0.42, -0.18, -1.65),
        (0.00, -0.20, -1.72),
        (0.42, -0.18, -1.65),
        (0.78, -0.08, -1.46),
    ]
    trim = create_curve("COSTUME_GoldNeckTrim", neckline_points, gold, 0.028, parent=follow)

    random.seed(20260804)
    strands = []
    for index in range(86):
        angle = random.uniform(-math.pi * 0.92, math.pi * 0.92)
        radial = random.uniform(0.76, 1.02)
        start_x = math.sin(angle) * radial * 0.88
        start_z = 0.52 + math.cos(angle) * radial * 0.84
        start_y = 0.00 + random.uniform(-0.06, 0.08)
        curl = random.uniform(-0.16, 0.16)
        end_x = start_x + curl
        end_z = start_z - random.uniform(0.16, 0.42)
        end_y = start_y - random.uniform(0.04, 0.15)
        middle = (
            (start_x + end_x) * 0.5 + random.uniform(-0.10, 0.10),
            (start_y + end_y) * 0.5 - 0.04,
            (start_z + end_z) * 0.5 + random.uniform(-0.08, 0.08),
        )
        strand = create_curve(
            f"HAIR_Strand_{index:03d}",
            [(start_x, start_y, start_z), middle, (end_x, end_y, end_z)],
            dark_hair,
            random.uniform(0.007, 0.013),
            parent=follow,
        )
        strands.append(strand)

    return {
        "hair_cap": hair_cap,
        "neck": neck,
        "torso": torso,
        "trim": trim,
        "strand_count": len(strands),
    }


def build_banquet_set() -> dict[str, int]:
    stone = principled_material("MAT_Warm_Stone", (0.10, 0.045, 0.018, 1.0), 0.82)
    wood = principled_material("MAT_Dark_Wood", (0.055, 0.018, 0.008, 1.0), 0.62)
    bronze = principled_material("MAT_Bronze", (0.20, 0.055, 0.014, 1.0), 0.34, metallic=0.65)
    silhouette = principled_material("MAT_Background_Suitor", (0.025, 0.010, 0.008, 1.0), 0.80)
    ember = principled_material(
        "MAT_Fire_Bokeh",
        (1.0, 0.12, 0.01, 1.0),
        0.25,
        emission=(1.0, 0.055, 0.004, 1.0),
        emission_strength=18.0,
    )

    create_cube("SET_BanquetHall", (0.0, 8.0, 0.0), (8.0, 0.25, 4.5), stone)
    create_cube("PROP_FeastTable", (0.0, 5.3, -2.4), (5.0, 1.1, 0.18), wood, bevel=0.10)

    for x in (-5.6, -3.7, 3.7, 5.6):
        bpy.ops.mesh.primitive_cylinder_add(vertices=48, radius=0.48, depth=7.2, location=(x, 7.0, 0.2))
        column = bpy.context.object
        column.name = f"SET_Column_{x:+.1f}"
        column.data.materials.append(stone)

    for index, x in enumerate((-3.3, -1.9, 1.8, 3.2, 4.5)):
        body = create_uv_sphere(
            f"BG_Suitor_{index:02d}",
            (x, 5.9 + (index % 2) * 0.7, -0.95),
            (0.58, 0.44, 1.25),
            silhouette,
            segments=24,
            ring_count=12,
        )
        head = create_uv_sphere(
            f"BG_SuitorHead_{index:02d}",
            (x, 5.75 + (index % 2) * 0.7, 0.55),
            (0.42, 0.38, 0.52),
            silhouette,
            segments=24,
            ring_count=12,
        )
        body.rotation_euler[2] = random.uniform(-0.24, 0.24)
        head.rotation_euler[2] = body.rotation_euler[2]

    bokeh_positions = [
        (-4.8, 6.0, 2.5),
        (-3.0, 7.8, 1.8),
        (-1.8, 5.8, 2.8),
        (1.9, 6.6, 2.2),
        (3.2, 7.4, 2.9),
        (4.9, 5.9, 1.7),
        (5.7, 8.0, 3.0),
    ]
    for index, position in enumerate(bokeh_positions):
        create_uv_sphere(
            f"FX_FireBokeh_{index:02d}",
            position,
            (0.10, 0.10, 0.16),
            ember,
            segments=20,
            ring_count=10,
        )
        add_point_light(
            f"LIGHT_Torch_{index:02d}",
            position,
            energy=420.0,
            color=(1.0, 0.12, 0.018),
            radius=0.42,
        )

    for index, x in enumerate((-3.8, -2.3, 2.5, 4.0)):
        bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=0.20, depth=0.52, location=(x, 3.9, -1.95))
        goblet = bpy.context.object
        goblet.name = f"PROP_Goblet_{index:02d}"
        goblet.data.materials.append(bronze)

    return {"background_suitors": 5, "bokeh_lights": len(bokeh_positions), "goblets": 4}


def build_foreground_beggar() -> bpy.types.Object:
    cloth = principled_material("MAT_Beggar_Cloth", (0.025, 0.018, 0.013, 1.0), 0.92)
    root = bpy.data.objects.new("CHAR_Beggar", None)
    bpy.context.collection.objects.link(root)
    root.location = (-2.75, -4.2, -0.55)
    root.rotation_euler = (0.08, -0.16, -0.20)

    hood = create_uv_sphere("CHAR_Beggar_Hood", (0.0, 0.0, 1.25), (0.92, 0.72, 1.05), cloth)
    shoulders = create_uv_sphere("CHAR_Beggar_Shoulders", (0.0, 0.15, -0.10), (1.75, 0.82, 1.20), cloth)
    hood.parent = root
    shoulders.parent = root
    return root


def configure_camera_and_lighting(face_target: Vector) -> tuple[bpy.types.Object, bpy.types.Object]:
    camera_data = bpy.data.cameras.new("CAM_Hero")
    camera_data.lens = 85.0
    camera_data.sensor_width = 36.0
    camera_data.dof.use_dof = True
    camera_data.dof.aperture_fstop = 1.55
    camera_data.dof.aperture_blades = 9
    camera = bpy.data.objects.new("CAM_Hero", camera_data)
    camera.location = (0.10, -8.15, -0.05)
    bpy.context.collection.objects.link(camera)
    look_at(camera, face_target)

    focus = bpy.data.objects.new("FOCUS_Antinous_Eyes", None)
    focus.location = (0.05, -0.08, 0.22)
    bpy.context.collection.objects.link(focus)
    camera_data.dof.focus_object = focus

    wide_data = bpy.data.cameras.new("CAM_Wide")
    wide_data.lens = 52.0
    wide_data.sensor_width = 36.0
    wide_data.dof.use_dof = True
    wide_data.dof.focus_object = focus
    wide_data.dof.aperture_fstop = 2.4
    wide = bpy.data.objects.new("CAM_Wide", wide_data)
    wide.location = (0.1, -10.7, 0.15)
    bpy.context.collection.objects.link(wide)
    look_at(wide, Vector((0.0, 0.3, -0.35)))

    add_area_light(
        "LIGHT_WarmKey",
        (-3.2, -3.5, 3.8),
        energy=1050.0,
        color=(1.0, 0.17, 0.045),
        size=3.1,
        target=face_target,
    )
    add_area_light(
        "LIGHT_WarmRim",
        (3.3, 0.4, 2.9),
        energy=620.0,
        color=(1.0, 0.055, 0.015),
        size=2.0,
        target=face_target,
    )
    add_area_light(
        "LIGHT_SoftFill",
        (0.0, -2.0, 4.8),
        energy=90.0,
        color=(0.25, 0.08, 0.04),
        size=5.0,
        target=face_target,
    )
    return camera, wide


def configure_render(scene: bpy.types.Scene, config: dict, engine: str) -> None:
    width, height = config["creative_target"]["resolution"]
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.resolution_percentage = 100
    scene.render.fps = int(config["creative_target"]["fps"])
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.render.engine = "CYCLES" if engine == "cycles" else "BLENDER_EEVEE_NEXT"
    if engine == "cycles":
        scene.cycles.samples = 96
        scene.cycles.use_denoising = True
    else:
        scene.render.use_file_extension = True
    scene.render.image_settings.color_depth = "8"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass
    scene.world.color = (0.003, 0.001, 0.0005)


def render_still(scene: bpy.types.Scene, camera: bpy.types.Object, frame: int, path: Path) -> None:
    scene.camera = camera
    scene.frame_set(frame)
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def render_animation(scene: bpy.types.Scene, camera: bpy.types.Object, path: Path) -> None:
    scene.camera = camera
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.filepath = str(path)
    bpy.ops.render.render(animation=True)


def validate_reload(blend_path: Path, required_objects: list[str]) -> dict:
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    missing = [name for name in required_objects if bpy.data.objects.get(name) is None]
    scene = bpy.context.scene
    report = {
        "blend_reloaded": True,
        "missing_required_objects": missing,
        "frame_start": int(scene.frame_start),
        "frame_end": int(scene.frame_end),
        "camera": scene.camera.name if scene.camera else None,
        "object_count": len(bpy.data.objects),
    }
    if missing:
        raise RuntimeError(f"Save/reload validation failed; missing objects: {missing}")
    return report


def main() -> int:
    args = parse_args()
    sequence_path = Path(args.sequence).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not sequence_path.is_file():
        raise SystemExit(f"Reconstructed sequence is missing: {sequence_path}")
    if not config_path.is_file():
        raise SystemExit(f"Scene config is missing: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = np.load(sequence_path)
    vertices = np.asarray(data["vertices_smoothed"], dtype=np.float32)
    triangles = np.asarray(data["triangles"], dtype=np.int32)
    colors_rgb = np.asarray(data["colors_rgb"], dtype=np.float32)
    boxes = np.asarray(data["boxes"], dtype=np.float32)
    image_size_array = np.asarray(data["image_size"], dtype=np.int32)
    image_size = (int(image_size_array[0]), int(image_size_array[1]))

    clear_scene()
    scene = bpy.context.scene
    configure_render(scene, config, args.render_engine)
    scene.frame_start = 1
    scene.frame_end = int(config["creative_target"]["duration_frames"])

    face, follow, target_frames = create_face_mesh(
        vertices,
        triangles,
        colors_rgb,
        boxes,
        image_size,
        scene.frame_end,
    )
    character_info = build_character(follow, colors_rgb)
    set_info = build_banquet_set()
    build_foreground_beggar()
    hero_camera, wide_camera = configure_camera_and_lighting(Vector((0.0, 0.0, 0.05)))
    scene.camera = hero_camera

    blend_path = output_dir / "beggars_photoreal_recreation.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    reload_report = validate_reload(blend_path, config["acceptance"]["required_objects"])

    scene = bpy.context.scene
    hero_camera = bpy.data.objects["CAM_Hero"]
    wide_camera = bpy.data.objects["CAM_Wide"]
    proof_frames = [
        max(1, int(round(scene.frame_end * 0.28))),
        max(1, int(round(scene.frame_end * 0.62))),
        max(1, int(round(scene.frame_end * 0.86))),
    ]
    for index, frame in enumerate(proof_frames):
        render_still(scene, hero_camera, frame, output_dir / f"proof_frame_{index + 1:02d}.png")
    render_still(scene, hero_camera, proof_frames[1], output_dir / "hero_clean_render.png")
    render_still(scene, wide_camera, proof_frames[1], output_dir / "wide_scene_proof.png")
    silent_video = output_dir / "beggars_photoreal_recreation_silent.mp4"
    render_animation(scene, hero_camera, silent_video)

    receipt = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "claim_policy": "The scene, animation, renders and save/reload are machine-proven. Meme likeness and photoreal match remain NOT_PROVEN until user review.",
        "scene_id": config["scene_id"],
        "source_of_truth": config["worker_policy"]["source_of_truth"],
        "render_engine": scene.render.engine,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "fps": scene.render.fps,
        "frame_range": [scene.frame_start, scene.frame_end],
        "tracked_input_frames": int(vertices.shape[0]),
        "animated_shape_keys": len(target_frames),
        "face_vertices": int(vertices.shape[2]),
        "face_triangles": int(triangles.shape[0]),
        "hair_strands": int(character_info["strand_count"]),
        "set": set_info,
        "required_objects": config["acceptance"]["required_objects"],
        "reload_validation": reload_report,
        "outputs": {
            "blend": str(blend_path),
            "hero_render": str(output_dir / "hero_clean_render.png"),
            "wide_render": str(output_dir / "wide_scene_proof.png"),
            "silent_video": str(silent_video),
            "proof_frames": [str(output_dir / f"proof_frame_{index + 1:02d}.png") for index in range(3)],
        },
        "reference_media_packaged": False,
        "actor_voice_cloned": False,
    }
    receipt_path = output_dir / "scene_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
