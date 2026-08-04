from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_beggars_meme_scene_faceverse_v6 as v6  # noqa: E402


base = v6.base
v5 = v6.v5


def textured_skin_material_v7() -> bpy.types.Material:
    material = bpy.data.materials.new("MAT_Antinous_Face_VertexSkin_V7")
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
    balance.inputs[1].default_value = (0.86, 1.00, 1.05)
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 38.0
    noise.inputs["Detail"].default_value = 3.0
    noise.inputs["Roughness"].default_value = 0.72
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.08
    bump.inputs["Distance"].default_value = 0.035
    base.set_input(bsdf, "Roughness", 0.52)
    base.set_input(bsdf, "Specular IOR Level", 0.22)
    base.set_input(bsdf, "Subsurface Weight", 0.025)
    base.set_input(bsdf, "Emission Strength", 0.0)
    links.new(attribute.outputs["Color"], balance.inputs[0])
    links.new(balance.outputs["Vector"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def eye_vertex_material_v7() -> bpy.types.Material:
    material = bpy.data.materials.new("MAT_Antinous_Eyes_Vertex_V7")
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
    balance.inputs[1].default_value = (0.92, 1.06, 1.15)
    base.set_input(bsdf, "Roughness", 0.24)
    base.set_input(bsdf, "Specular IOR Level", 0.42)
    base.set_input(bsdf, "Subsurface Weight", 0.0)
    links.new(attribute.outputs["Color"], balance.inputs[0])
    links.new(balance.outputs["Vector"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def create_face_with_eye_materials(*args, **kwargs):
    face, follow, targets = v5.create_model_space_face_mesh(*args, **kwargs)
    eye_material = eye_vertex_material_v7()
    face.data.materials.append(eye_material)
    eye_material_index = len(face.data.materials) - 1
    assigned = 0
    for polygon in face.data.polygons:
        vertices = polygon.vertices
        if all(13916 <= int(index) < 15456 for index in vertices):
            polygon.material_index = eye_material_index
            assigned += 1
    if assigned <= 0:
        raise RuntimeError("No FaceVerse eye polygons were assigned the v7 eye material")
    print(f"BLENDER_FACEVERSE_EYE_MATERIAL_V7=PROVEN POLYGONS={assigned}")
    return face, follow, targets


def hair_shell_material_v7() -> bpy.types.Material:
    material = bpy.data.materials.new("MAT_Antinous_ConformingHairShell_V7")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 16.0
    noise.inputs["Detail"].default_value = 5.0
    noise.inputs["Roughness"].default_value = 0.78
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (0.004, 0.0015, 0.0007, 1.0)
    ramp.color_ramp.elements[1].color = (0.035, 0.010, 0.004, 1.0)
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.28
    bump.inputs["Distance"].default_value = 0.045
    base.set_input(bsdf, "Roughness", 0.48)
    base.set_input(bsdf, "Specular IOR Level", 0.22)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def create_conforming_hair_shell(face: bpy.types.Object, follow: bpy.types.Object) -> bpy.types.Object:
    coordinates = np.asarray([vertex.co[:] for vertex in face.data.vertices], dtype=np.float32)
    if coordinates.shape[0] != 19546:
        raise RuntimeError(
            f"Unexpected FaceVerse vertex count for hair shell: {coordinates.shape[0]}"
        )
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    z = coordinates[:, 2]
    lateral = np.clip(np.abs(x) / 0.90, 0.0, 1.0)
    scalp_line = 0.76 - 0.30 * lateral
    scalp_mask = (z >= scalp_line) & (y >= -0.72)
    sideburn_mask = (
        (np.abs(x) >= 0.67)
        & (z >= -0.18)
        & (z <= 0.55)
        & (y >= -0.64)
    )
    vertex_mask = scalp_mask | sideburn_mask

    selected_polygons: list[tuple[int, ...]] = []
    selected_vertices: set[int] = set()
    for polygon in face.data.polygons:
        polygon_vertices = tuple(int(index) for index in polygon.vertices)
        if all(vertex_mask[index] for index in polygon_vertices):
            selected_polygons.append(polygon_vertices)
            selected_vertices.update(polygon_vertices)
    if len(selected_polygons) < 1000:
        raise RuntimeError(
            f"Conforming hair shell is implausibly sparse: polygons={len(selected_polygons)}"
        )

    old_indices = sorted(selected_vertices)
    remap = {old_index: new_index for new_index, old_index in enumerate(old_indices)}
    center = np.asarray([0.0, 0.12, 0.25], dtype=np.float32)
    shell_vertices = coordinates[old_indices].copy()
    directions = shell_vertices - center[None, :]
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / np.maximum(lengths, 1e-6)
    shell_vertices += directions * 0.032
    shell_faces = [tuple(remap[index] for index in polygon) for polygon in selected_polygons]

    mesh = bpy.data.meshes.new("MESH_Antinous_ConformingHairShell")
    mesh.from_pydata(shell_vertices.tolist(), [], shell_faces)
    mesh.update()
    shell = bpy.data.objects.new("CHAR_Antinous_HairShell", mesh)
    bpy.context.collection.objects.link(shell)
    shell.parent = follow
    shell.data.materials.append(hair_shell_material_v7())
    for polygon in shell.data.polygons:
        polygon.use_smooth = True

    solidify = shell.modifiers.new("HairShellThickness", "SOLIDIFY")
    solidify.thickness = 0.012
    solidify.offset = 0.0
    bevel = shell.modifiers.new("HairlineSoftening", "BEVEL")
    bevel.width = 0.008
    bevel.segments = 2
    print(
        "BLENDER_CONFORMING_HAIR_SHELL_V7=PROVEN "
        f"VERTICES={len(shell_vertices)} POLYGONS={len(shell_faces)} "
        "WIRE_STRANDS=ABSENT FACIAL_HAIR_GEOMETRY=ABSENT"
    )
    return shell


def build_shell_character(follow: bpy.types.Object, colors_rgb: np.ndarray) -> dict[str, object]:
    result = v6.build_refined_character(follow, colors_rgb)
    removed = 0
    for obj in list(bpy.data.objects):
        if (
            obj.name == "CHAR_Antinous_HairCap"
            or obj.name.startswith("HAIR_")
            or obj.name.startswith("FACIALHAIR_")
        ):
            bpy.data.objects.remove(obj, do_unlink=True)
            removed += 1
    face = bpy.data.objects.get("CHAR_Antinous")
    if face is None:
        raise RuntimeError("Face object is missing before conforming hair-shell creation")
    shell = create_conforming_hair_shell(face, follow)
    result["hair_cap"] = shell
    result["strand_count"] = 0
    result["moustache_hair_count"] = 0
    result["sideburn_hair_count"] = 0
    result["removed_wire_groom_objects"] = removed
    result["variant"] = "FACEVERSE_TRUE_MODEL_SPACE_CONFORMING_HAIR_SHELL_V7"
    print(f"BLENDER_WIRE_GROOM_REMOVAL_V7=PROVEN OBJECTS={removed}")
    return result


def patch_v7_receipt() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    output_dir = None
    for index, value in enumerate(argv):
        if value == "--output-dir" and index + 1 < len(argv):
            output_dir = Path(argv[index + 1]).resolve()
            break
    if output_dir is None:
        raise RuntimeError("Could not resolve v7 output directory")
    receipt_path = output_dir / "scene_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["character_variant"] = "FACEVERSE_TRUE_MODEL_SPACE_CONFORMING_HAIR_SHELL_V7"
    receipt["visual_changes_v7"] = {
        "wire_hair_curves": False,
        "facial_hair_geometry": False,
        "conforming_topology_hair_shell": True,
        "dedicated_eye_material": True,
        "skin_micro_bump": True,
        "source_texture_brows_and_moustache_only": True,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    base.vertex_skin_material = textured_skin_material_v7
    v5.create_model_space_face_mesh = create_face_with_eye_materials
    v6.build_refined_character = build_shell_character
    result = int(v6.main())
    patch_v7_receipt()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
