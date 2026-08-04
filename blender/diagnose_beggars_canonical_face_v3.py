from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

import diagnose_beggars_canonical_face_v2 as base


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--tracked-sequence", required=True)
    parser.add_argument("--keyframe-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bfm-pkl", default=str(base.DEFAULT_BFM))
    return parser.parse_args(argv)


def projective_texture_material(image_path: Path) -> bpy.types.Material:
    image = bpy.data.images.load(str(image_path), check_existing=False)
    image.colorspace_settings.name = "sRGB"

    material = bpy.data.materials.new("MAT_Antinous_ProjectiveFace_V3")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    texture = nodes.new("ShaderNodeTexImage")
    uv = nodes.new("ShaderNodeUVMap")
    texture.image = image
    texture.interpolation = "Linear"
    texture.extension = "CLIP"
    uv.uv_map = "SourceProjection"

    base.set_input(bsdf, "Roughness", 0.52)
    base.set_input(bsdf, "Specular IOR Level", 0.22)
    base.set_input(bsdf, "Subsurface Weight", 0.018)
    base.set_input(bsdf, "Emission Strength", 0.055)
    links.new(uv.outputs["UV"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    emission = bsdf.inputs.get("Emission Color")
    if emission is not None:
        links.new(texture.outputs["Color"], emission)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def create_projective_face(
    vertices: np.ndarray,
    triangles: np.ndarray,
    projected_vertices: np.ndarray,
    image_size: tuple[int, int],
    image_path: Path,
    root: bpy.types.Object,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new("MESH_Canonical_Antinous_V3")
    mesh.from_pydata(vertices.tolist(), [], triangles.tolist())
    mesh.update()
    face = bpy.data.objects.new("CHAR_Antinous", mesh)
    bpy.context.collection.objects.link(face)
    face.parent = root
    for polygon in mesh.polygons:
        polygon.use_smooth = True

    width, height = image_size
    projected = np.asarray(projected_vertices, dtype=np.float32)
    if projected.shape == (3, len(vertices)):
        x = projected[0]
        y = projected[1]
    elif projected.shape == (len(vertices), 3):
        x = projected[:, 0]
        y = projected[:, 1]
    else:
        raise RuntimeError(f"Unexpected projected vertex shape: {projected.shape}")
    uv_per_vertex = np.stack(
        (
            np.clip(x / max(width - 1, 1), 0.0, 1.0),
            np.clip(1.0 - y / max(height - 1, 1), 0.0, 1.0),
        ),
        axis=1,
    ).astype(np.float32)
    uv_layer = mesh.uv_layers.new(name="SourceProjection")
    for loop in mesh.loops:
        uv_layer.data[loop.index].uv = uv_per_vertex[loop.vertex_index]

    mesh.materials.append(projective_texture_material(image_path))
    solidify = face.modifiers.new("FaceShell", "SOLIDIFY")
    solidify.thickness = 0.005
    solidify.offset = 0.0
    return face


def update_face(face: bpy.types.Object, vertices: np.ndarray) -> None:
    face.data.vertices.foreach_set("co", vertices.astype(np.float32).reshape(-1).tolist())
    face.data.update()


def hide_objects(objects: list[bpy.types.Object], hidden: bool) -> None:
    for obj in objects:
        obj.hide_render = hidden


def tune_features(features: dict[str, object]) -> None:
    # The projective face texture already supplies the eyes and lips. Keep only
    # restrained 3D support behind openings and move all rear closure behind the face.
    features["cranium"].hide_render = True

    eyes = list(features["eyes"])
    for index, obj in enumerate(eyes):
        obj.scale *= 0.58 if index % 3 == 0 else 0.48
        obj.location.y += 0.075
    hide_objects(eyes, True)

    cavity = features["cavity"]
    cavity.scale.x *= 0.88
    cavity.scale.z *= 0.78
    cavity.location.y += 0.055

    teeth = list(features["teeth"])
    for obj in teeth:
        obj.scale.x *= 0.72
        obj.scale.y *= 0.72
        obj.scale.z *= 0.52
        obj.location.y += 0.070
        obj.location.z -= 0.015

    tongue = features["tongue"]
    tongue.scale *= 0.72
    tongue.location.y += 0.060

    for ear in features["ears"]:
        ear.scale *= 0.78
        ear.location.y += 0.10

    scalp = features["scalp"]
    scalp.scale.x *= 0.95
    scalp.scale.y *= 0.74
    scalp.scale.z *= 0.78
    scalp.location.y += 0.16
    scalp.location.z += 0.03

    strands = list(features["strands"])
    for strand in strands:
        strand.hide_render = True

    neck = features["neck"]
    neck.scale.x *= 0.82
    neck.scale.y *= 0.85
    neck.location.y += 0.08
    torso = features["torso"]
    torso.scale.x *= 0.86
    torso.scale.y *= 0.86


def configure_scene(bounds: dict[str, list[float]]) -> tuple[bpy.types.Scene, bpy.types.Object]:
    xmin, xmax = bounds["x"]
    ymin, ymax = bounds["y"]
    zmin, zmax = bounds["z"]
    width = xmax - xmin
    height = zmax - zmin
    center = Vector(((xmin + xmax) * 0.5, (ymin + ymax) * 0.5, (zmin + zmax) * 0.5))

    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.world.color = (0.003, 0.002, 0.002)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass

    camera_data = bpy.data.cameras.new("CAM_Diagnostic_V3")
    camera_data.lens = 72.0
    camera_data.sensor_width = 36.0
    camera_data.dof.use_dof = False
    camera = bpy.data.objects.new("CAM_Diagnostic_V3", camera_data)
    distance = max(height * 3.82, 9.25)
    camera.location = (center.x, ymin - distance, center.z + height * 0.015)
    bpy.context.collection.objects.link(camera)
    target = Vector((center.x, center.y, center.z + height * 0.015))
    base.look_at(camera, target)
    scene.camera = camera

    base.add_area_light(
        "LIGHT_ProjectiveKey_V3",
        (center.x - width * 1.20, ymin - height * 1.15, center.z + height * 1.20),
        330.0,
        (1.0, 0.62, 0.42),
        3.4,
        target,
    )
    base.add_area_light(
        "LIGHT_ProjectiveFill_V3",
        (center.x + width * 1.20, ymin - height * 0.75, center.z + height * 0.35),
        110.0,
        (0.30, 0.22, 0.20),
        3.2,
        target,
    )
    base.add_area_light(
        "LIGHT_ProjectiveRim_V3",
        (center.x + width * 0.90, ymax + height * 0.75, center.z + height * 0.95),
        210.0,
        (1.0, 0.24, 0.10),
        2.4,
        target,
    )
    return scene, camera


def converted_quaternion(rotation: np.ndarray) -> Quaternion:
    axis_map = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    converted = axis_map @ np.asarray(rotation, dtype=np.float64) @ axis_map.T
    u, _, vt = np.linalg.svd(converted)
    converted = u @ vt
    if np.linalg.det(converted) < 0.0:
        u[:, -1] *= -1.0
        converted = u @ vt
    return Matrix(converted.tolist()).to_quaternion()


def apply_pose(root: bpy.types.Object, rotation: np.ndarray, strength: float) -> float:
    target = converted_quaternion(rotation)
    angle = float(target.angle)
    root.rotation_mode = "QUATERNION"
    root.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0)).slerp(target, strength)
    return math.degrees(angle)


def select_expression_indices(
    landmarks_sequence: np.ndarray,
    rotations: np.ndarray,
    keyframe_index: int,
) -> tuple[int, int, dict[str, list[float]]]:
    upper = np.mean(landmarks_sequence[:, 61:64, 2], axis=1)
    lower = np.mean(landmarks_sequence[:, 65:68, 2], axis=1)
    gap = np.abs(upper - lower)
    width = np.ptp(landmarks_sequence[:, 48:60, 0], axis=1)
    angles = np.asarray([math.degrees(converted_quaternion(r).angle) for r in rotations])

    mild_score = gap - angles * 0.0025
    grin_score = width + gap * 0.45 - angles * 0.0040
    mild_index = int(np.argmax(mild_score))
    grin_index = int(np.argmax(grin_score))
    if angles[grin_index] > 28.0:
        allowed = np.flatnonzero(angles <= 28.0)
        if len(allowed):
            grin_index = int(allowed[np.argmax(grin_score[allowed])])
    return mild_index, grin_index, {
        "mouth_gap": gap.astype(float).tolist(),
        "mouth_width": width.astype(float).tolist(),
        "pose_angle_degrees": angles.astype(float).tolist(),
    }


def render(scene: bpy.types.Scene, output: Path) -> None:
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def main() -> int:
    args = parse_args()
    sequence_path = Path(args.sequence).resolve()
    tracked_path = Path(args.tracked_sequence).resolve()
    keyframe_image = Path(args.keyframe_image).resolve()
    output_dir = Path(args.output_dir).resolve()
    bfm_path = Path(args.bfm_pkl).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for required in (sequence_path, tracked_path, keyframe_image, bfm_path):
        if not required.is_file():
            raise SystemExit(f"Required v3 diagnostic input is missing: {required}")

    canonical_data = np.load(sequence_path)
    tracked_data = np.load(tracked_path)
    vertices_sequence = np.asarray(canonical_data["canonical_vertices"], dtype=np.float32)
    triangles = np.asarray(canonical_data["triangles"], dtype=np.int32)
    colors_rgb = np.asarray(canonical_data["colors_rgb"], dtype=np.float32)
    rotations = np.asarray(canonical_data["relative_rotations"], dtype=np.float32)
    keyframe_index = int(np.asarray(canonical_data["keyframe_index"]).reshape(-1)[0])
    projected_keyframe = np.asarray(tracked_data["vertices_raw"], dtype=np.float32)[keyframe_index]
    image_size_raw = np.asarray(tracked_data["image_size"], dtype=np.int32).reshape(-1)
    image_size = (int(image_size_raw[0]), int(image_size_raw[1]))
    landmark_indices = base.load_landmark_indices(bfm_path, vertices_sequence.shape[1])

    base.clear_scene()
    root = bpy.data.objects.new("RIG_CanonicalHead_V3", None)
    bpy.context.collection.objects.link(root)
    root.rotation_mode = "QUATERNION"
    root.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)

    face = create_projective_face(
        vertices_sequence[keyframe_index],
        triangles,
        projected_keyframe,
        image_size,
        keyframe_image,
        root,
    )
    features = base.build_landmark_features(
        vertices_sequence[keyframe_index], colors_rgb, landmark_indices, root
    )
    tune_features(features)
    scene, _ = configure_scene(features["bounds"])

    outputs: list[str] = []
    base.set_feature_visibility(features, False)
    output = output_dir / "v3_01_projective_face.png"
    render(scene, output)
    outputs.append(output.name)

    base.set_feature_visibility(features, True, include_hair=False)
    hide_objects(list(features["eyes"]), True)
    output = output_dir / "v3_02_face_teeth_no_hair.png"
    render(scene, output)
    outputs.append(output.name)

    base.set_feature_visibility(features, True, include_hair=True)
    hide_objects(list(features["eyes"]), True)
    hide_objects(list(features["strands"]), True)
    output = output_dir / "v3_03_face_teeth_hair.png"
    render(scene, output)
    outputs.append(output.name)

    landmarks_sequence = vertices_sequence[:, landmark_indices]
    mild_index, grin_index, metrics = select_expression_indices(
        landmarks_sequence, rotations, keyframe_index
    )

    update_face(face, vertices_sequence[mild_index])
    mild_angle = apply_pose(root, rotations[mild_index], 0.40)
    output = output_dir / f"v3_04_mild_pose_{mild_index:02d}.png"
    render(scene, output)
    outputs.append(output.name)

    update_face(face, vertices_sequence[grin_index])
    grin_angle = apply_pose(root, rotations[grin_index], 0.46)
    output = output_dir / f"v3_05_grin_pose_{grin_index:02d}.png"
    render(scene, output)
    outputs.append(output.name)

    report = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "claim": "Projective-textured still diagnostic only; likeness and animation remain unproven until visual review.",
        "route": "POSE_DECOUPLED_CANONICAL_3DDFA_PROJECTIVE_TEXTURE_V3",
        "diagnostic_workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "vertex_count": int(vertices_sequence.shape[1]),
        "triangle_count": int(triangles.shape[0]),
        "landmark_count": int(len(landmark_indices)),
        "keyframe_index": keyframe_index,
        "mild_index": mild_index,
        "grin_index": grin_index,
        "mild_source_pose_angle_degrees": mild_angle,
        "grin_source_pose_angle_degrees": grin_angle,
        "source_image_size": list(image_size),
        "source_image_packaged": False,
        "outputs": outputs,
        "selection_metrics": metrics,
    }
    (output_dir / "diagnostic_v3.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "selection_metrics"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
