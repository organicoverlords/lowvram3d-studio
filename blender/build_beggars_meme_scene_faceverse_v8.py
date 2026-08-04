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

import build_beggars_meme_scene_faceverse_v7 as v7  # noqa: E402


v6 = v7.v6
v5 = v7.v5
base = v7.base
_original_create_model_space_face_mesh = v5.create_model_space_face_mesh
_original_build_refined_character = v6.build_refined_character

MAIN_HEAD_END = 13916
LEFT_EYE_END = 14686
RIGHT_EYE_END = 15456
TONGUE_END = 15846
TOTAL_VERTICES = 19546


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


def textured_skin_material_v8() -> bpy.types.Material:
    material = bpy.data.materials.new("MAT_Antinous_Face_VertexSkin_V8")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "Col"
    balance = nodes.new("ShaderNodeVectorMath")
    balance.operation = "MULTIPLY"
    balance.inputs[1].default_value = (0.66, 0.50, 0.44)
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 42.0
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.68
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.055
    bump.inputs["Distance"].default_value = 0.022
    set_input(bsdf, "Roughness", 0.56)
    set_input(bsdf, "Specular IOR Level", 0.19)
    set_input(bsdf, "Subsurface Weight", 0.018)
    links.new(attribute.outputs["Color"], balance.inputs[0])
    links.new(balance.outputs["Vector"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def hair_material_v8() -> bpy.types.Material:
    material = bpy.data.materials.new("MAT_Antinous_FittedHair_V8")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 22.0
    noise.inputs["Detail"].default_value = 4.5
    noise.inputs["Roughness"].default_value = 0.72
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.0035, 0.0010, 0.00035, 1.0)
    ramp.color_ramp.elements[1].color = (0.024, 0.0060, 0.0018, 1.0)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.20
    bump.inputs["Distance"].default_value = 0.025
    set_input(bsdf, "Roughness", 0.54)
    set_input(bsdf, "Specular IOR Level", 0.18)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def deform_coordinates(coordinates: np.ndarray) -> np.ndarray:
    result = np.asarray(coordinates, dtype=np.float32).copy()
    main = result[:MAIN_HEAD_END]
    x = main[:, 0]
    y = main[:, 1]
    z = main[:, 2]

    upper = np.clip((z - 0.40) / 0.72, 0.0, 1.0)
    z[:] = np.where(z > 0.40, 0.40 + (z - 0.40) * 0.80, z)
    x[:] *= 1.0 - upper * 0.025

    lower = np.clip((-z - 0.20) / 0.82, 0.0, 1.0)
    x[:] *= 1.0 - lower * 0.085
    z[:] = np.where(z < -0.20, -0.20 + (z + 0.20) * 0.91, z)

    nose_mask = (
        (np.abs(x) < 0.32)
        & (z > -0.32)
        & (z < 0.42)
        & (y < -0.50)
    )
    y[nose_mask] = -0.50 + (y[nose_mask] + 0.50) * 0.88
    result[:MAIN_HEAD_END] = main
    return result


def deform_all_shape_keys(face: bpy.types.Object) -> None:
    if face.data.shape_keys is None:
        raise RuntimeError("FaceVerse v8 requires shape keys")
    key_blocks = list(face.data.shape_keys.key_blocks)
    if len(key_blocks) < 2:
        raise RuntimeError(f"FaceVerse v8 shape-key count is implausible: {len(key_blocks)}")
    for key_block in key_blocks:
        coordinates = np.asarray([point.co[:] for point in key_block.data], dtype=np.float32)
        if coordinates.shape != (TOTAL_VERTICES, 3):
            raise RuntimeError(
                f"Unexpected v8 shape-key coordinates for {key_block.name}: {coordinates.shape}"
            )
        corrected = deform_coordinates(coordinates)
        for point, coordinate in zip(key_block.data, corrected):
            point.co = coordinate
    print(f"BLENDER_FACEVERSE_V8_ANATOMY_DEFORMATION=PROVEN SHAPE_KEYS={len(key_blocks)}")


def assign_component_materials(face: bpy.types.Object) -> dict[str, int]:
    sclera = simple_material(
        "MAT_Antinous_Sclera_V8",
        (0.34, 0.29, 0.23, 1.0),
        0.28,
        0.36,
    )
    ivory = simple_material(
        "MAT_Antinous_Teeth_Ivory_V8",
        (0.48, 0.37, 0.24, 1.0),
        0.38,
        0.25,
    )
    face.data.materials.append(sclera)
    sclera_index = len(face.data.materials) - 1
    face.data.materials.append(ivory)
    ivory_index = len(face.data.materials) - 1
    eye_polygons = 0
    tooth_polygons = 0
    for polygon in face.data.polygons:
        indices = [int(index) for index in polygon.vertices]
        if indices and all(MAIN_HEAD_END <= index < RIGHT_EYE_END for index in indices):
            polygon.material_index = sclera_index
            eye_polygons += 1
        elif indices and all(index >= TONGUE_END for index in indices):
            polygon.material_index = ivory_index
            tooth_polygons += 1
    if eye_polygons <= 0 or tooth_polygons <= 0:
        raise RuntimeError(
            f"V8 component assignment failed: eyes={eye_polygons} teeth={tooth_polygons}"
        )
    print(
        "BLENDER_FACEVERSE_V8_COMPONENT_MATERIALS=PROVEN "
        f"EYE_POLYGONS={eye_polygons} TOOTH_POLYGONS={tooth_polygons}"
    )
    return {"eye_polygons": eye_polygons, "tooth_polygons": tooth_polygons}


def create_corrected_face(*args, **kwargs):
    face, follow, targets = _original_create_model_space_face_mesh(*args, **kwargs)
    deform_all_shape_keys(face)
    assignment = assign_component_materials(face)
    face["v8_eye_polygons"] = assignment["eye_polygons"]
    face["v8_tooth_polygons"] = assignment["tooth_polygons"]
    return face, follow, targets


def create_ellipsoid(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    follow: bpy.types.Object,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=40,
        ring_count=20,
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


def create_fitted_hair(face: bpy.types.Object, follow: bpy.types.Object) -> dict[str, int]:
    coordinates = np.asarray([vertex.co[:] for vertex in face.data.vertices], dtype=np.float32)
    if coordinates.shape != (TOTAL_VERTICES, 3):
        raise RuntimeError(f"Unexpected v8 face coordinates: {coordinates.shape}")
    main = coordinates[:MAIN_HEAD_END]
    x = main[:, 0]
    y = main[:, 1]
    z = main[:, 2]
    lateral = np.clip(np.abs(x) / 0.90, 0.0, 1.0)
    hairline = 0.49 - 0.17 * lateral
    scalp_mask = (z >= hairline) & (y >= -0.82)
    side_mask = (
        (np.abs(x) >= 0.64)
        & (z >= -0.13)
        & (z <= 0.54)
        & (y >= -0.67)
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
    if len(selected_polygons) < 1200:
        raise RuntimeError(
            f"V8 fitted hair shell is too sparse: polygons={len(selected_polygons)}"
        )

    old_indices = sorted(selected_vertices)
    remap = {old_index: new_index for new_index, old_index in enumerate(old_indices)}
    shell_vertices = coordinates[old_indices].copy()
    center = np.asarray([0.0, 0.13, 0.22], dtype=np.float32)
    directions = shell_vertices - center[None, :]
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-6)
    shell_vertices += directions * 0.026
    shell_faces = [tuple(remap[index] for index in polygon) for polygon in selected_polygons]

    material = hair_material_v8()
    mesh = bpy.data.meshes.new("MESH_Antinous_FittedHairShell_V8")
    mesh.from_pydata(shell_vertices.tolist(), [], shell_faces)
    mesh.update()
    shell = bpy.data.objects.new("CHAR_Antinous_FittedHairShell_V8", mesh)
    bpy.context.collection.objects.link(shell)
    shell.parent = follow
    shell.data.materials.append(material)
    for polygon in shell.data.polygons:
        polygon.use_smooth = True
    solidify = shell.modifiers.new("FittedHairThickness", "SOLIDIFY")
    solidify.thickness = 0.010
    solidify.offset = 0.0
    bevel = shell.modifiers.new("FittedHairEdgeSoftening", "BEVEL")
    bevel.width = 0.012
    bevel.segments = 3

    minimum = np.min(main, axis=0)
    maximum = np.max(main, axis=0)
    extent = maximum - minimum
    center_x = float((minimum[0] + maximum[0]) * 0.5)
    front_y = float(minimum[1]) - 0.018
    top_z = float(maximum[2])
    fringe_specs = (
        (-0.56, 0.60, 0.27, 0.105, 0.18, -0.16),
        (-0.29, 0.67, 0.30, 0.105, 0.19, -0.08),
        (-0.02, 0.71, 0.32, 0.105, 0.20, 0.02),
        (0.27, 0.68, 0.30, 0.105, 0.19, 0.10),
        (0.53, 0.59, 0.25, 0.105, 0.17, 0.17),
    )
    fringe_count = 0
    for index, (nx, nz, sx, sy, sz, rz) in enumerate(fringe_specs):
        create_ellipsoid(
            f"HAIR_V8_FrontFringe_{index:02d}",
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
            rotation=(0.0, rz, rz * 0.45),
        )
        fringe_count += 1

    print(
        "BLENDER_FACEVERSE_V8_FITTED_HAIR=PROVEN "
        f"SHELL_POLYGONS={len(shell_faces)} FRINGE_MASSES={fringe_count}"
    )
    return {"shell_polygons": len(shell_faces), "fringe_masses": fringe_count}


def create_eye_details(face: bpy.types.Object, follow: bpy.types.Object) -> dict[str, int]:
    coordinates = np.asarray([vertex.co[:] for vertex in face.data.vertices], dtype=np.float32)
    iris_material = simple_material(
        "MAT_Antinous_IrisBrown_V8",
        (0.035, 0.009, 0.0025, 1.0),
        0.22,
        0.34,
    )
    pupil_material = simple_material(
        "MAT_Antinous_Pupil_V8",
        (0.001, 0.001, 0.001, 1.0),
        0.18,
        0.25,
    )
    ranges = ((MAIN_HEAD_END, LEFT_EYE_END), (LEFT_EYE_END, RIGHT_EYE_END))
    created = 0
    for side, (start, end) in zip(("L", "R"), ranges):
        eye = coordinates[start:end]
        center = np.mean(eye, axis=0)
        front_y = float(np.min(eye[:, 1])) - 0.018
        x_extent = float(np.ptp(eye[:, 0]))
        z_extent = float(np.ptp(eye[:, 2]))
        create_ellipsoid(
            f"EYE_V8_Iris_{side}",
            (float(center[0]), front_y, float(center[2])),
            (x_extent * 0.16, 0.020, z_extent * 0.18),
            iris_material,
            follow,
        )
        create_ellipsoid(
            f"EYE_V8_Pupil_{side}",
            (float(center[0]), front_y - 0.017, float(center[2])),
            (x_extent * 0.072, 0.012, z_extent * 0.082),
            pupil_material,
            follow,
        )
        created += 2
    print(f"BLENDER_FACEVERSE_V8_EYE_DETAILS=PROVEN OBJECTS={created}")
    return {"objects": created}


def build_character_v8(follow: bpy.types.Object, colors_rgb: np.ndarray) -> dict[str, object]:
    result = _original_build_refined_character(follow, colors_rgb)
    removed = 0
    for obj in list(bpy.data.objects):
        if (
            obj.name == "CHAR_Antinous_HairCap"
            or obj.name.startswith("HAIR_")
            or obj.name.startswith("FACIALHAIR_")
            or obj.name.startswith("CHAR_Antinous_HairShell")
        ):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    face = bpy.data.objects.get("CHAR_Antinous")
    if face is None:
        raise RuntimeError("Face object is missing before v8 correction")
    hair = create_fitted_hair(face, follow)
    eyes = create_eye_details(face, follow)
    result["hair_cap"] = bpy.data.objects.get("CHAR_Antinous_FittedHairShell_V8")
    result["strand_count"] = 0
    result["moustache_hair_count"] = 0
    result["sideburn_hair_count"] = 0
    result["removed_prior_groom_objects"] = removed
    result["hair_v8"] = hair
    result["eye_details_v8"] = eyes
    result["variant"] = "FACEVERSE_MODEL_SPACE_ANATOMY_HAIR_EYES_TEETH_V8"
    print(f"BLENDER_FACEVERSE_V8_PRIOR_GROOM_REMOVED=PROVEN OBJECTS={removed}")
    return result


def patch_v8_receipt() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    output_dir = None
    for index, value in enumerate(argv):
        if value == "--output-dir" and index + 1 < len(argv):
            output_dir = Path(argv[index + 1]).resolve()
            break
    if output_dir is None:
        raise RuntimeError("Could not resolve v8 output directory")
    receipt_path = output_dir / "scene_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["character_variant"] = "FACEVERSE_MODEL_SPACE_ANATOMY_HAIR_EYES_TEETH_V8"
    receipt["visual_changes_v8"] = {
        "forehead_compressed": True,
        "jaw_and_chin_tapered": True,
        "nose_depth_reduced": True,
        "lower_fitted_hair_shell": True,
        "broad_front_fringe_masses": 5,
        "wire_hair_curves": False,
        "facial_hair_geometry": False,
        "neutral_sclera_material": True,
        "iris_and_pupil_geometry": True,
        "uniform_ivory_tooth_material": True,
        "darker_warmer_vertex_skin": True,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    base.vertex_skin_material = textured_skin_material_v8
    v5.create_model_space_face_mesh = create_corrected_face
    v6.build_refined_character = build_character_v8
    result = int(v6.main())
    patch_v8_receipt()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
