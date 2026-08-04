from __future__ import annotations

import json
import math
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
_original_build_character_v8 = v8.build_character_v8

MAIN_HEAD_END = v8.MAIN_HEAD_END
LEFT_EYE_END = v8.LEFT_EYE_END
RIGHT_EYE_END = v8.RIGHT_EYE_END
TONGUE_END = v8.TONGUE_END
TOTAL_VERTICES = v8.TOTAL_VERTICES


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


def textured_skin_material_v9() -> bpy.types.Material:
    material = bpy.data.materials.new("MAT_Antinous_Face_VertexSkin_V9")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "Col"
    hue = nodes.new("ShaderNodeHueSaturation")
    hue.inputs["Saturation"].default_value = 1.08
    hue.inputs["Value"].default_value = 0.46
    multiply = nodes.new("ShaderNodeMixRGB")
    multiply.blend_type = "MULTIPLY"
    multiply.inputs[0].default_value = 0.58
    multiply.inputs[2].default_value = (0.72, 0.34, 0.22, 1.0)
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 45.0
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.66
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.042
    bump.inputs["Distance"].default_value = 0.018

    set_input(bsdf, "Roughness", 0.53)
    set_input(bsdf, "Specular IOR Level", 0.20)
    set_input(bsdf, "Subsurface Weight", 0.015)
    links.new(attribute.outputs["Color"], hue.inputs["Color"])
    links.new(hue.outputs["Color"], multiply.inputs[1])
    links.new(multiply.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def hair_material_v9() -> bpy.types.Material:
    material = bpy.data.materials.new("MAT_Antinous_FittedHair_V9")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 28.0
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.74
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.0022, 0.00065, 0.00025, 1.0)
    ramp.color_ramp.elements[1].color = (0.018, 0.0040, 0.0012, 1.0)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.17
    bump.inputs["Distance"].default_value = 0.018
    set_input(bsdf, "Roughness", 0.58)
    set_input(bsdf, "Specular IOR Level", 0.14)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def create_ellipsoid(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    follow: bpy.types.Object,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    segments: int = 36,
    rings: int = 18,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        location=location,
        rotation=rotation,
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj.parent = follow
    return obj


def create_face_v9(*args, **kwargs):
    face, follow, targets = _original_create_corrected_face(*args, **kwargs)
    dark_mouth = simple_material(
        "MAT_Antinous_InternalMouth_V9",
        (0.020, 0.0030, 0.0020, 1.0),
        0.72,
        0.08,
    )
    face.data.materials.append(dark_mouth)
    mouth_index = len(face.data.materials) - 1
    hidden_tooth_polygons = 0
    for polygon in face.data.polygons:
        indices = [int(index) for index in polygon.vertices]
        if indices and all(index >= TONGUE_END for index in indices):
            polygon.material_index = mouth_index
            hidden_tooth_polygons += 1
    if hidden_tooth_polygons <= 0:
        raise RuntimeError("V9 could not replace segmented tooth polygons")
    face["v9_hidden_tooth_polygons"] = hidden_tooth_polygons
    print(
        "BLENDER_FACEVERSE_V9_SEGMENTED_TEETH_HIDDEN=PROVEN "
        f"POLYGONS={hidden_tooth_polygons}"
    )
    return face, follow, targets


def remove_v8_groom() -> int:
    removed = 0
    for obj in list(bpy.data.objects):
        if (
            obj.name.startswith("CHAR_Antinous_FittedHairShell_V8")
            or obj.name.startswith("HAIR_V8_")
            or obj.name.startswith("EYE_V8_")
        ):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    return removed


def create_hair_v9(face: bpy.types.Object, follow: bpy.types.Object) -> dict[str, int]:
    coordinates = np.asarray([vertex.co[:] for vertex in face.data.vertices], dtype=np.float32)
    if coordinates.shape != (TOTAL_VERTICES, 3):
        raise RuntimeError(f"Unexpected v9 face coordinates: {coordinates.shape}")
    main = coordinates[:MAIN_HEAD_END]
    x = main[:, 0]
    y = main[:, 1]
    z = main[:, 2]
    lateral = np.clip(np.abs(x) / max(float(np.max(np.abs(x))), 1e-6), 0.0, 1.0)
    hairline = 0.31 - 0.10 * lateral
    scalp_mask = (z >= hairline) & (y >= -0.80)
    side_mask = (
        (np.abs(x) >= 0.62)
        & (z >= -0.08)
        & (z <= 0.46)
        & (y >= -0.64)
    )
    vertex_mask = np.zeros(TOTAL_VERTICES, dtype=bool)
    vertex_mask[:MAIN_HEAD_END] = scalp_mask | side_mask

    selected_polygons: list[tuple[int, ...]] = []
    selected_vertices: set[int] = set()
    for polygon in face.data.polygons:
        polygon_vertices = tuple(int(index) for index in polygon.vertices)
        if polygon_vertices and all(vertex_mask[index] for index in polygon_vertices):
            selected_polygons.append(polygon_vertices)
            selected_vertices.update(polygon_vertices)
    if len(selected_polygons) < 1600:
        raise RuntimeError(f"V9 hair shell too sparse: polygons={len(selected_polygons)}")

    old_indices = sorted(selected_vertices)
    remap = {old_index: new_index for new_index, old_index in enumerate(old_indices)}
    shell_vertices = coordinates[old_indices].copy()
    center = np.asarray([0.0, 0.12, 0.18], dtype=np.float32)
    directions = shell_vertices - center[None, :]
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-6)
    shell_vertices += directions * 0.020
    shell_faces = [tuple(remap[index] for index in polygon) for polygon in selected_polygons]

    material = hair_material_v9()
    mesh = bpy.data.meshes.new("MESH_Antinous_FittedHairShell_V9")
    mesh.from_pydata(shell_vertices.tolist(), [], shell_faces)
    mesh.update()
    shell = bpy.data.objects.new("CHAR_Antinous_FittedHairShell_V9", mesh)
    bpy.context.collection.objects.link(shell)
    shell.parent = follow
    shell.data.materials.append(material)
    for polygon in shell.data.polygons:
        polygon.use_smooth = True
    subdivision = shell.modifiers.new("FittedHairSubdivision", "SUBSURF")
    subdivision.subdivision_type = "CATMULL_CLARK"
    subdivision.levels = 1
    subdivision.render_levels = 1
    solidify = shell.modifiers.new("FittedHairThickness", "SOLIDIFY")
    solidify.thickness = 0.008
    solidify.offset = 0.0
    bevel = shell.modifiers.new("FittedHairEdgeSoftening", "BEVEL")
    bevel.width = 0.009
    bevel.segments = 3

    minimum = np.min(main, axis=0)
    maximum = np.max(main, axis=0)
    extent = maximum - minimum
    center_x = float((minimum[0] + maximum[0]) * 0.5)
    front_y = float(minimum[1]) - 0.012
    lock_specs = (
        (-0.54, 0.58, 0.12, 0.038, 0.090, -0.18),
        (-0.31, 0.62, 0.13, 0.038, 0.095, -0.10),
        (-0.08, 0.64, 0.14, 0.038, 0.100, -0.03),
        (0.16, 0.63, 0.14, 0.038, 0.098, 0.05),
        (0.39, 0.60, 0.13, 0.038, 0.092, 0.12),
        (0.58, 0.55, 0.11, 0.038, 0.085, 0.18),
    )
    lock_count = 0
    for index, (nx, nz, sx, sy, sz, rz) in enumerate(lock_specs):
        create_ellipsoid(
            f"HAIR_V9_HairlineLock_{index:02d}",
            (
                center_x + nx * float(extent[0]) * 0.50,
                front_y,
                minimum[2] + nz * float(extent[2]),
            ),
            (
                sx * float(extent[0]),
                sy * float(extent[1]),
                sz * float(extent[2]),
            ),
            material,
            follow,
            rotation=(0.0, rz, rz * 0.35),
            segments=28,
            rings=14,
        )
        lock_count += 1

    print(
        "BLENDER_FACEVERSE_V9_HAIR=PROVEN "
        f"SHELL_POLYGONS={len(shell_faces)} HAIRLINE_LOCKS={lock_count}"
    )
    return {"shell_polygons": len(shell_faces), "hairline_locks": lock_count}


def create_eye_details_v9(face: bpy.types.Object, follow: bpy.types.Object) -> dict[str, int]:
    coordinates = np.asarray([vertex.co[:] for vertex in face.data.vertices], dtype=np.float32)
    iris = simple_material(
        "MAT_Antinous_IrisBrown_V9",
        (0.020, 0.0045, 0.0015, 1.0),
        0.24,
        0.30,
    )
    pupil = simple_material(
        "MAT_Antinous_Pupil_V9",
        (0.0005, 0.0005, 0.0005, 1.0),
        0.20,
        0.20,
    )
    created = 0
    for side, (start, end) in zip(
        ("L", "R"),
        ((MAIN_HEAD_END, LEFT_EYE_END), (LEFT_EYE_END, RIGHT_EYE_END)),
    ):
        eye = coordinates[start:end]
        center = np.mean(eye, axis=0)
        front_y = float(np.min(eye[:, 1])) - 0.010
        x_extent = float(np.ptp(eye[:, 0]))
        z_extent = float(np.ptp(eye[:, 2]))
        create_ellipsoid(
            f"EYE_V9_Iris_{side}",
            (float(center[0]), front_y, float(center[2])),
            (x_extent * 0.13, 0.014, z_extent * 0.15),
            iris,
            follow,
            segments=28,
            rings=14,
        )
        create_ellipsoid(
            f"EYE_V9_Pupil_{side}",
            (float(center[0]), front_y - 0.009, float(center[2])),
            (x_extent * 0.050, 0.008, z_extent * 0.058),
            pupil,
            follow,
            segments=24,
            rings=12,
        )
        created += 2
    print(f"BLENDER_FACEVERSE_V9_EYES=PROVEN OBJECTS={created}")
    return {"objects": created}


def create_dental_plate_v9(face: bpy.types.Object, follow: bpy.types.Object) -> bpy.types.Object:
    coordinates = np.asarray([vertex.co[:] for vertex in face.data.vertices], dtype=np.float32)
    teeth = coordinates[TONGUE_END:TOTAL_VERTICES]
    center = np.mean(teeth, axis=0)
    minimum = np.min(teeth, axis=0)
    maximum = np.max(teeth, axis=0)
    extent = maximum - minimum
    ivory = simple_material(
        "MAT_Antinous_SmoothDentalIvory_V9",
        (0.56, 0.44, 0.29, 1.0),
        0.34,
        0.24,
    )
    plate = create_ellipsoid(
        "MOUTH_V9_SmoothDentalPlate",
        (
            float(center[0]),
            float(minimum[1]) - 0.008,
            float(center[2]) + float(extent[2]) * 0.01,
        ),
        (
            float(extent[0]) * 0.43,
            0.024,
            float(extent[2]) * 0.23,
        ),
        ivory,
        follow,
        rotation=(0.0, 0.0, 0.0),
        segments=48,
        rings=20,
    )
    print(
        "BLENDER_FACEVERSE_V9_DENTAL_PLATE=PROVEN "
        f"WIDTH={float(extent[0]) * 0.86:.5f} HEIGHT={float(extent[2]) * 0.46:.5f}"
    )
    return plate


def build_character_v9(follow: bpy.types.Object, colors_rgb: np.ndarray) -> dict[str, object]:
    result = _original_build_character_v8(follow, colors_rgb)
    removed = remove_v8_groom()
    face = bpy.data.objects.get("CHAR_Antinous")
    if face is None:
        raise RuntimeError("Face object missing before v9 treatment")
    hair = create_hair_v9(face, follow)
    eyes = create_eye_details_v9(face, follow)
    dental_plate = create_dental_plate_v9(face, follow)
    result["hair_cap"] = bpy.data.objects.get("CHAR_Antinous_FittedHairShell_V9")
    result["strand_count"] = 0
    result["moustache_hair_count"] = 0
    result["sideburn_hair_count"] = 0
    result["removed_v8_objects"] = removed
    result["hair_v9"] = hair
    result["eye_details_v9"] = eyes
    result["dental_plate_v9"] = dental_plate.name
    result["variant"] = "FACEVERSE_MODEL_SPACE_REFINED_HAIRLINE_DENTAL_V9"
    print(f"BLENDER_FACEVERSE_V9_V8_OBJECTS_REMOVED=PROVEN OBJECTS={removed}")
    return result


def patch_v9_receipt() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    output_dir = None
    for index, value in enumerate(argv):
        if value == "--output-dir" and index + 1 < len(argv):
            output_dir = Path(argv[index + 1]).resolve()
            break
    if output_dir is None:
        raise RuntimeError("Could not resolve v9 output directory")
    receipt_path = output_dir / "scene_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["character_variant"] = "FACEVERSE_MODEL_SPACE_REFINED_HAIRLINE_DENTAL_V9"
    receipt["visual_changes_v9"] = {
        "giant_fringe_masses_removed": True,
        "lower_subdivided_scalp_shell": True,
        "small_hairline_locks": 6,
        "skin_value_reduced": 0.46,
        "warm_skin_multiply": True,
        "segmented_tooth_geometry_hidden": True,
        "smooth_dental_plate": True,
        "smaller_iris_and_pupil_geometry": True,
        "animation_rendered": False,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    base.vertex_skin_material = textured_skin_material_v9
    v5.create_model_space_face_mesh = create_face_v9
    v6.build_refined_character = build_character_v9
    result = int(v6.main())
    patch_v9_receipt()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
