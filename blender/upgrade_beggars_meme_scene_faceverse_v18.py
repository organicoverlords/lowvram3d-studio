from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable

import bpy
import numpy as np
from mathutils import Vector


MAIN_HEAD_END = 13916
LEFT_EYE_END = 14686
RIGHT_EYE_END = 15456
TONGUE_END = 15846
TOTAL_VERTICES = 19546
FRAME = 48


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1 :] if "--" in argv else []
    parser = argparse.ArgumentParser(
        description="Upgrade the exact-UV FaceVerse v11 scene with true-3D anatomical details."
    )
    parser.add_argument("--input-blend", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--baseline-image", required=True)
    return parser.parse_args(argv)


def set_input(node: bpy.types.Node, name: str, value) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float,
    specular: float,
) -> bpy.types.Material:
    existing = bpy.data.materials.get(name)
    mat = existing or bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    set_input(bsdf, "Base Color", color)
    set_input(bsdf, "Roughness", roughness)
    set_input(bsdf, "Specular IOR Level", specular)
    set_input(bsdf, "Metallic", 0.0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def hair_material() -> bpy.types.Material:
    mat = bpy.data.materials.get("MAT_Antinous_Hair_V18") or bpy.data.materials.new(
        "MAT_Antinous_Hair_V18"
    )
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 34.0
    noise.inputs["Detail"].default_value = 4.0
    noise.inputs["Roughness"].default_value = 0.68
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.002, 0.0005, 0.00025, 1.0)
    ramp.color_ramp.elements[1].color = (0.030, 0.007, 0.0025, 1.0)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.010
    set_input(bsdf, "Roughness", 0.50)
    set_input(bsdf, "Specular IOR Level", 0.16)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat


def parent_keep_world(obj: bpy.types.Object, parent: bpy.types.Object | None) -> None:
    if parent is None:
        return
    world = obj.matrix_world.copy()
    obj.parent = parent
    obj.matrix_world = world


def create_ellipsoid(
    name: str,
    location: Iterable[float],
    scale: Iterable[float],
    mat: bpy.types.Material,
    parent: bpy.types.Object | None,
    rotation_y: float = 0.0,
    segments: int = 32,
    rings: int = 16,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        location=tuple(float(v) for v in location),
        rotation=(0.0, rotation_y, 0.0),
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = tuple(float(v) for v in scale)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(mat)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    parent_keep_world(obj, parent)
    return obj


def create_curve(
    name: str,
    points: list[tuple[float, float, float]],
    mat: bpy.types.Material,
    bevel_depth: float,
    parent: bpy.types.Object | None,
) -> bpy.types.Object:
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = float(bevel_depth)
    curve.bevel_resolution = 2
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, coordinate in zip(spline.bezier_points, points):
        point.co = coordinate
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    curve.materials.append(mat)
    parent_keep_world(obj, parent)
    return obj


def evaluated_world_coordinates(face: bpy.types.Object) -> np.ndarray:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = face.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    try:
        coordinates = np.asarray(
            [tuple(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices],
            dtype=np.float32,
        )
    finally:
        evaluated.to_mesh_clear()
    if coordinates.shape != (TOTAL_VERTICES, 3):
        raise RuntimeError(f"Unexpected evaluated FaceVerse coordinates: {coordinates.shape}")
    return coordinates


def append_material(mesh: bpy.types.Mesh, mat: bpy.types.Material) -> int:
    for index, existing in enumerate(mesh.materials):
        if existing == mat:
            return index
    mesh.materials.append(mat)
    return len(mesh.materials) - 1


def assign_component_materials(face: bpy.types.Object) -> dict[str, int]:
    sclera = material("MAT_Antinous_Sclera_V18", (0.72, 0.66, 0.58, 1.0), 0.32, 0.28)
    mouth = material("MAT_Antinous_MouthCavity_V18", (0.016, 0.002, 0.0015, 1.0), 0.76, 0.04)
    sclera_index = append_material(face.data, sclera)
    mouth_index = append_material(face.data, mouth)
    eye_polygons = 0
    mouth_polygons = 0
    hidden_teeth = 0
    for polygon in face.data.polygons:
        indices = [int(index) for index in polygon.vertices]
        if indices and all(MAIN_HEAD_END <= index < RIGHT_EYE_END for index in indices):
            polygon.material_index = sclera_index
            eye_polygons += 1
        elif indices and all(RIGHT_EYE_END <= index < TONGUE_END for index in indices):
            polygon.material_index = mouth_index
            mouth_polygons += 1
        elif indices and all(index >= TONGUE_END for index in indices):
            polygon.material_index = mouth_index
            hidden_teeth += 1
    if eye_polygons <= 0 or mouth_polygons <= 0 or hidden_teeth <= 0:
        raise RuntimeError(
            "V18 component assignment failed: "
            f"eyes={eye_polygons} mouth={mouth_polygons} teeth={hidden_teeth}"
        )
    return {
        "eye_polygons": eye_polygons,
        "mouth_polygons": mouth_polygons,
        "hidden_segmented_tooth_polygons": hidden_teeth,
    }


def repair_projection_mask(face: bpy.types.Object, coordinates: np.ndarray) -> dict[str, int]:
    attribute = face.data.color_attributes.get("ProjMask")
    if attribute is None or attribute.domain != "POINT":
        raise RuntimeError("V18 requires the v11 POINT-domain ProjMask attribute")
    main = coordinates[:MAIN_HEAD_END]
    minimum = np.min(main, axis=0)
    maximum = np.max(main, axis=0)
    extent = np.maximum(maximum - minimum, 1e-6)
    center = (minimum + maximum) * 0.5
    normalized_x = np.abs((main[:, 0] - center[0]) / (extent[0] * 0.5))
    normalized_y = (main[:, 1] - minimum[1]) / extent[1]
    normalized_z = (main[:, 2] - minimum[2]) / extent[2]

    values = np.ones(TOTAL_VERTICES, dtype=np.float32)
    scalp = (normalized_z > (0.64 - 0.08 * normalized_x)) & (normalized_y > 0.08)
    values[:MAIN_HEAD_END][scalp] = 0.0

    eye_ranges = ((MAIN_HEAD_END, LEFT_EYE_END), (LEFT_EYE_END, RIGHT_EYE_END))
    eye_reduced = 0
    for start, end in eye_ranges:
        eye = coordinates[start:end]
        eye_center = np.mean(eye, axis=0)
        eye_extent = np.maximum(np.ptp(eye, axis=0), 1e-6)
        delta = (main - eye_center[None, :]) / np.asarray(
            [eye_extent[0] * 1.10, extent[1] * 0.20, eye_extent[2] * 1.30],
            dtype=np.float32,
        )[None, :]
        region = np.sum(delta**2, axis=1) < 1.0
        values[:MAIN_HEAD_END][region] = np.minimum(values[:MAIN_HEAD_END][region], 0.18)
        eye_reduced += int(np.sum(region))

    teeth = coordinates[TONGUE_END:TOTAL_VERTICES]
    mouth_center = np.mean(teeth, axis=0)
    mouth_delta = (main - mouth_center[None, :]) / np.asarray(
        [extent[0] * 0.34, extent[1] * 0.22, extent[2] * 0.18], dtype=np.float32
    )[None, :]
    mouth_region = np.sum(mouth_delta**2, axis=1) < 1.0
    values[:MAIN_HEAD_END][mouth_region] = np.minimum(
        values[:MAIN_HEAD_END][mouth_region], 0.34
    )

    rgba = np.ones((TOTAL_VERTICES, 4), dtype=np.float32)
    rgba[:, :3] = values[:, None]
    attribute.data.foreach_set("color", rgba.reshape(-1).tolist())

    material_slot = face.data.materials[0]
    if material_slot and material_slot.use_nodes and material_slot.node_tree:
        strength = material_slot.node_tree.nodes.get("ProjectionStrength")
        if strength is not None:
            strength.outputs[0].default_value = 0.50
        projected_balance = next(
            (
                node
                for node in material_slot.node_tree.nodes
                if node.bl_idname == "ShaderNodeHueSaturation"
                and node.name != "Hue Saturation Value"
            ),
            None,
        )
        if projected_balance is not None:
            projected_balance.inputs["Saturation"].default_value = 0.78
    return {
        "scalp_vertices_masked": int(np.sum(scalp)),
        "eye_rim_vertices_reduced": eye_reduced,
        "mouth_vertices_reduced": int(np.sum(mouth_region)),
    }


def build_hair_shell(
    face: bpy.types.Object,
    follow: bpy.types.Object | None,
    coordinates: np.ndarray,
    mat: bpy.types.Material,
) -> dict[str, int]:
    main = coordinates[:MAIN_HEAD_END]
    minimum = np.min(main, axis=0)
    maximum = np.max(main, axis=0)
    extent = np.maximum(maximum - minimum, 1e-6)
    center = (minimum + maximum) * 0.5
    nx = np.abs((main[:, 0] - center[0]) / (extent[0] * 0.5))
    ny = (main[:, 1] - minimum[1]) / extent[1]
    nz = (main[:, 2] - minimum[2]) / extent[2]
    scalp = ((nz > (0.62 - 0.10 * nx)) & (ny > 0.10)) | (
        (nx > 0.74) & (nz > 0.34) & (ny > 0.08)
    )
    vertex_mask = np.zeros(TOTAL_VERTICES, dtype=bool)
    vertex_mask[:MAIN_HEAD_END] = scalp

    selected_polygons: list[tuple[int, ...]] = []
    selected_vertices: set[int] = set()
    for polygon in face.data.polygons:
        indices = tuple(int(index) for index in polygon.vertices)
        if indices and all(index < MAIN_HEAD_END and vertex_mask[index] for index in indices):
            selected_polygons.append(indices)
            selected_vertices.update(indices)
    if len(selected_polygons) < 800:
        raise RuntimeError(f"V18 fitted hair shell too sparse: {len(selected_polygons)} polygons")

    old_indices = sorted(selected_vertices)
    remap = {old: new for new, old in enumerate(old_indices)}
    shell_vertices = coordinates[old_indices].copy()
    directions = shell_vertices - center[None, :]
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-6)
    shell_vertices += directions * (0.012 * float(extent[2]))
    shell_faces = [tuple(remap[index] for index in polygon) for polygon in selected_polygons]

    mesh = bpy.data.meshes.new("MESH_Antinous_FittedHairShell_V18")
    mesh.from_pydata(shell_vertices.tolist(), [], shell_faces)
    mesh.update()
    shell = bpy.data.objects.new("CHAR_Antinous_FittedHairShell_V18", mesh)
    bpy.context.collection.objects.link(shell)
    shell.data.materials.append(mat)
    for polygon in shell.data.polygons:
        polygon.use_smooth = True
    solidify = shell.modifiers.new("HairThickness", "SOLIDIFY")
    solidify.thickness = 0.010
    solidify.offset = 0.15
    bevel = shell.modifiers.new("HairlineSoftening", "BEVEL")
    bevel.width = 0.006
    bevel.segments = 2
    parent_keep_world(shell, follow)

    front_y = float(minimum[1]) - 0.006
    locks = 0
    for index, position in enumerate(np.linspace(-0.72, 0.72, 13)):
        x = float(center[0] + position * extent[0] * 0.50)
        z = float(minimum[2] + (0.76 - 0.07 * abs(position)) * extent[2])
        create_ellipsoid(
            f"HAIR_V18_HairlineLock_{index:02d}",
            (x, front_y, z),
            (extent[0] * 0.035, extent[1] * 0.018, extent[2] * 0.050),
            mat,
            follow,
            rotation_y=0.20 * math.sin(index * 0.8),
            segments=24,
            rings=12,
        )
        locks += 1
    return {"shell_polygons": len(shell_faces), "hairline_locks": locks}


def build_eye_details(
    coordinates: np.ndarray,
    follow: bpy.types.Object | None,
) -> dict[str, int]:
    iris_mat = material("MAT_Antinous_Iris_V18", (0.035, 0.011, 0.004, 1.0), 0.26, 0.34)
    pupil_mat = material("MAT_Antinous_Pupil_V18", (0.0004, 0.0004, 0.0004, 1.0), 0.20, 0.18)
    catchlight_mat = material("MAT_Antinous_Catchlight_V18", (0.90, 0.86, 0.78, 1.0), 0.16, 0.42)
    created = 0
    for side, (start, end) in zip(
        ("L", "R"),
        ((MAIN_HEAD_END, LEFT_EYE_END), (LEFT_EYE_END, RIGHT_EYE_END)),
    ):
        eye = coordinates[start:end]
        center = np.mean(eye, axis=0)
        width = max(float(np.ptp(eye[:, 0])), 1e-4)
        height = max(float(np.ptp(eye[:, 2])), 1e-4)
        front_y = float(np.min(eye[:, 1])) - 0.010
        iris = create_ellipsoid(
            f"EYE_V18_Iris_{side}",
            (float(center[0]), front_y, float(center[2])),
            (width * 0.145, 0.012, height * 0.165),
            iris_mat,
            follow,
            segments=28,
            rings=14,
        )
        create_ellipsoid(
            f"EYE_V18_Pupil_{side}",
            (float(center[0]), front_y - 0.008, float(center[2])),
            (width * 0.058, 0.007, height * 0.068),
            pupil_mat,
            follow,
            segments=24,
            rings=12,
        )
        create_ellipsoid(
            f"EYE_V18_Catchlight_{side}",
            (
                float(center[0] - width * 0.025),
                front_y - 0.014,
                float(center[2] + height * 0.035),
            ),
            (width * 0.018, 0.003, height * 0.020),
            catchlight_mat,
            follow,
            segments=16,
            rings=8,
        )
        iris["eye_side"] = side
        created += 3
    return {"eye_detail_objects": created}


def build_dental_strip(
    coordinates: np.ndarray,
    follow: bpy.types.Object | None,
) -> dict[str, int]:
    main = coordinates[:MAIN_HEAD_END]
    head_extent = np.maximum(np.ptp(main, axis=0), 1e-6)
    teeth = coordinates[TONGUE_END:TOTAL_VERTICES]
    center = np.mean(teeth, axis=0)
    minimum = np.min(teeth, axis=0)
    maximum = np.max(teeth, axis=0)
    width = min(float(maximum[0] - minimum[0]), float(head_extent[0] * 0.52))
    front_y = float(np.min(teeth[:, 1])) - 0.010

    dark = material("MAT_Antinous_MouthBackdrop_V18", (0.010, 0.0012, 0.0010, 1.0), 0.80, 0.02)
    ivory = material("MAT_Antinous_DentalIvory_V18", (0.68, 0.57, 0.42, 1.0), 0.46, 0.16)
    create_ellipsoid(
        "MOUTH_V18_Backdrop",
        (float(center[0]), front_y + 0.028, float(center[2])),
        (width * 0.56, head_extent[1] * 0.045, head_extent[2] * 0.085),
        dark,
        follow,
        segments=36,
        rings=18,
    )

    span = width * 0.43
    segments = 10
    front_vertices: list[tuple[float, float, float]] = []
    back_vertices: list[tuple[float, float, float]] = []
    left = teeth[teeth[:, 0] <= center[0]]
    right = teeth[teeth[:, 0] > center[0]]
    slope = 0.0
    if len(left) and len(right):
        dx = float(np.mean(right[:, 0]) - np.mean(left[:, 0]))
        if abs(dx) > 1e-6:
            slope = float((np.mean(right[:, 2]) - np.mean(left[:, 2])) / dx)
    for index in range(segments + 1):
        t = index / segments
        x = float(center[0] - span + 2.0 * span * t)
        normalized = (x - center[0]) / max(span, 1e-6)
        top = float(center[2] + slope * (x - center[0]) + head_extent[2] * (0.026 - 0.018 * normalized**2))
        bottom = top - float(head_extent[2] * (0.058 + 0.006 * (1.0 - normalized**2)))
        front_vertices.extend(((x, front_y - 0.006, top), (x, front_y - 0.006, bottom)))
        back_vertices.extend(((x, front_y + 0.024, top), (x, front_y + 0.024, bottom)))
    vertices = front_vertices + back_vertices
    faces: list[tuple[int, int, int, int]] = []
    row = (segments + 1) * 2
    for index in range(segments):
        a = index * 2
        b = a + 2
        faces.append((a, b, b + 1, a + 1))
        faces.append((row + a + 1, row + b + 1, row + b, row + a))
        faces.append((a + 1, b + 1, row + b + 1, row + a + 1))
        faces.append((a, row + a, row + b, b))
    faces.append((0, 1, row + 1, row))
    last = segments * 2
    faces.append((last, row + last, row + last + 1, last + 1))
    mesh = bpy.data.meshes.new("MESH_Antinous_DentalStrip_V18")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    strip = bpy.data.objects.new("MOUTH_V18_DentalStrip", mesh)
    bpy.context.collection.objects.link(strip)
    strip.data.materials.append(ivory)
    bevel = strip.modifiers.new("DentalEdgeSoftening", "BEVEL")
    bevel.width = float(head_extent[2] * 0.010)
    bevel.segments = 3
    parent_keep_world(strip, follow)
    return {"dental_strip_segments": segments}


def build_facial_hair(
    coordinates: np.ndarray,
    follow: bpy.types.Object | None,
    mat: bpy.types.Material,
) -> dict[str, int]:
    main = coordinates[:MAIN_HEAD_END]
    extent = np.maximum(np.ptp(main, axis=0), 1e-6)
    teeth = coordinates[TONGUE_END:TOTAL_VERTICES]
    mouth = np.mean(teeth, axis=0)
    face_front = float(np.min(main[:, 1])) - 0.006

    eye_left = np.mean(coordinates[MAIN_HEAD_END:LEFT_EYE_END], axis=0)
    eye_right = np.mean(coordinates[LEFT_EYE_END:RIGHT_EYE_END], axis=0)
    eye_dx = float(eye_right[0] - eye_left[0])
    eye_slope = 0.0 if abs(eye_dx) < 1e-6 else float((eye_right[2] - eye_left[2]) / eye_dx)

    eyebrows = 0
    for side, eye in (("L", eye_left), ("R", eye_right)):
        sign = -1.0 if side == "L" else 1.0
        half_width = extent[0] * 0.105
        inner = float(eye[0] - sign * half_width)
        outer = float(eye[0] + sign * half_width)
        base_z = float(eye[2] + extent[2] * 0.105)
        points = [
            (inner, face_front, base_z + eye_slope * (inner - eye[0])),
            (float(eye[0]), face_front - 0.003, base_z + extent[2] * 0.018),
            (outer, face_front, base_z + eye_slope * (outer - eye[0]) - extent[2] * 0.010),
        ]
        create_curve(
            f"FACIALHAIR_V18_Eyebrow_{side}",
            points,
            mat,
            extent[0] * 0.0065,
            follow,
        )
        eyebrows += 1

    moustache = 0
    for side in (-1.0, 1.0):
        for index in range(8):
            t = (index + 0.5) / 8.0
            start_x = float(mouth[0] + side * extent[0] * (0.018 + 0.105 * t))
            end_x = float(start_x + side * extent[0] * (0.045 + 0.025 * t))
            z = float(mouth[2] + extent[2] * (0.090 - 0.020 * t))
            create_curve(
                f"FACIALHAIR_V18_Moustache_{moustache:02d}",
                [
                    (start_x, face_front - 0.004, z),
                    (
                        (start_x + end_x) * 0.5,
                        face_front - 0.010,
                        z - extent[2] * 0.012,
                    ),
                    (end_x, face_front - 0.003, z - extent[2] * (0.022 + 0.010 * t)),
                ],
                mat,
                extent[0] * 0.0038,
                follow,
            )
            moustache += 1
    return {"eyebrows": eyebrows, "moustache_strands": moustache}


def add_robe_bridge(follow: bpy.types.Object | None) -> dict[str, int]:
    cloth = material("MAT_Antinous_RobeBridge_V18", (0.012, 0.009, 0.014, 1.0), 0.84, 0.06)
    trim = material("MAT_Antinous_RobeTrim_V18", (0.30, 0.11, 0.022, 1.0), 0.40, 0.18)
    bridge = create_ellipsoid(
        "COSTUME_V18_RobeBridge",
        (0.0, 0.34, -1.26),
        (0.88, 0.48, 0.34),
        cloth,
        follow,
        segments=48,
        rings=24,
    )
    bridge["true_3d_robe_bridge"] = True
    create_curve(
        "COSTUME_V18_RobeTrim",
        [(-0.62, -0.05, -1.23), (0.0, -0.13, -1.34), (0.62, -0.05, -1.23)],
        trim,
        0.018,
        follow,
    )
    return {"robe_bridge_objects": 2}


def remove_conflicting_objects() -> int:
    prefixes = (
        "CHAR_Antinous_HairCap",
        "CHAR_Antinous_FittedHairShell",
        "HAIR_",
        "FACIALHAIR_",
        "EYE_V",
        "MOUTH_V",
        "COSTUME_V18_",
    )
    removed = 0
    for obj in list(bpy.data.objects):
        if obj.name.startswith(prefixes):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    return removed


def render(scene: bpy.types.Scene, camera: bpy.types.Object, path: Path) -> None:
    scene.camera = camera
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    if not path.is_file() or path.stat().st_size < 50000:
        raise RuntimeError(f"V18 render missing or too small: {path}")


def configure_render_quality(scene: bpy.types.Scene) -> None:
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    input_blend = Path(args.input_blend).resolve()
    output_dir = Path(args.output_dir).resolve()
    reference_image = Path(args.reference_image).resolve()
    baseline_image = Path(args.baseline_image).resolve()
    for required in (input_blend, reference_image, baseline_image):
        if not required.is_file():
            raise SystemExit(f"V18 input is missing: {required}")
    output_dir.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.open_mainfile(filepath=str(input_blend))
    scene = bpy.context.scene
    scene.frame_set(FRAME)
    configure_render_quality(scene)
    face = bpy.data.objects.get("CHAR_Antinous")
    follow = bpy.data.objects.get("RIG_HeadFollow")
    hero = bpy.data.objects.get("CAM_Hero")
    wide = bpy.data.objects.get("CAM_Wide")
    if face is None or hero is None or wide is None:
        raise RuntimeError("V18 required face/cameras are absent from the v11 blend")

    removed = remove_conflicting_objects()
    coordinates = evaluated_world_coordinates(face)
    component_report = assign_component_materials(face)
    mask_report = repair_projection_mask(face, coordinates)
    hair_mat = hair_material()
    hair_report = build_hair_shell(face, follow, coordinates, hair_mat)
    eye_report = build_eye_details(coordinates, follow)
    dental_report = build_dental_strip(coordinates, follow)
    facial_hair_report = build_facial_hair(coordinates, follow, hair_mat)
    robe_report = add_robe_bridge(follow)

    shutil.copy2(reference_image, output_dir / "public_reference.png")
    shutil.copy2(baseline_image, output_dir / "v11_baseline.png")

    projection_material = face.data.materials[0]
    strength = None
    if projection_material and projection_material.use_nodes and projection_material.node_tree:
        strength = projection_material.node_tree.nodes.get("ProjectionStrength")
    variants = []
    for value in (0.42, 0.50, 0.58):
        if strength is not None:
            strength.outputs[0].default_value = value
        output_path = output_dir / f"v18_projection_{int(round(value * 100)):03d}.png"
        render(scene, hero, output_path)
        variants.append({"projection_strength": value, "render": output_path.name})
    if strength is not None:
        strength.outputs[0].default_value = 0.50
    hero_path = output_dir / "hero_v18.png"
    render(scene, hero, hero_path)
    wide_path = output_dir / "wide_v18.png"
    render(scene, wide, wide_path)

    blend_path = output_dir / "beggars_true3d_v18.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    missing = [
        name
        for name in (
            "CHAR_Antinous",
            "CHAR_Antinous_FittedHairShell_V18",
            "EYE_V18_Iris_L",
            "EYE_V18_Iris_R",
            "MOUTH_V18_DentalStrip",
            "COSTUME_V18_RobeBridge",
            "CAM_Hero",
        )
        if bpy.data.objects.get(name) is None
    ]
    if missing:
        raise RuntimeError(f"V18 save/reload validation failed: {missing}")

    report = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "route": "FACEVERSE_V11_EXACT_UV_TRUE_3D_ANATOMICAL_REPAIR_V18",
        "source_frame_plane_used": False,
        "source_head_overlay_used": False,
        "external_compositor_used": False,
        "face_vertices": TOTAL_VERTICES,
        "frame": FRAME,
        "removed_conflicting_objects": removed,
        "component_materials": component_report,
        "projection_mask": mask_report,
        "hair": hair_report,
        "eyes": eye_report,
        "dentition": dental_report,
        "facial_hair": facial_hair_report,
        "robe_bridge": robe_report,
        "projection_variants": variants,
        "outputs": {
            "blend": blend_path.name,
            "hero": hero_path.name,
            "wide": wide_path.name,
            "reference": "public_reference.png",
            "baseline": "v11_baseline.png",
        },
        "claim_policy": (
            "The geometry, material reassignment, true-3D anatomical additions, still renders, "
            "and save/reload are machine-proven. Likeness remains pending visual inspection."
        ),
    }
    (output_dir / "v18_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
