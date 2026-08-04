from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

THREEDDFA_COMMIT = "1b6c67601abffc1e9f248b291708aef0e43b55ae"
DEFAULT_BFM = Path(
    rf"C:\AI\LowVRAM3D-cache\beggars-scene-v2\models\3DDFA_V2-{THREEDDFA_COMMIT}\configs\bfm_noneck_v3.pkl"
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bfm-pkl", default=str(DEFAULT_BFM))
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            collection.remove(block)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def set_input(node: bpy.types.Node, name: str, value) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def principled_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float,
    metallic: float = 0.0,
    subsurface: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        raise RuntimeError("Principled BSDF is unavailable")
    set_input(bsdf, "Base Color", color)
    set_input(bsdf, "Roughness", roughness)
    set_input(bsdf, "Metallic", metallic)
    set_input(bsdf, "Specular IOR Level", 0.30)
    set_input(bsdf, "Subsurface Weight", subsurface)
    return material


def vertex_color_material() -> bpy.types.Material:
    material = bpy.data.materials.new("MAT_CanonicalFace_VertexColor_V2")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    try:
        color_node = nodes.new("ShaderNodeVertexColor")
        color_node.layer_name = "Col"
        color_output = color_node.outputs.get("Color")
    except RuntimeError:
        color_node = nodes.new("ShaderNodeAttribute")
        color_node.attribute_name = "Col"
        color_output = color_node.outputs.get("Color")
    if color_output is None:
        raise RuntimeError("Vertex-color output is unavailable")
    set_input(bsdf, "Roughness", 0.48)
    set_input(bsdf, "Specular IOR Level", 0.25)
    set_input(bsdf, "Subsurface Weight", 0.025)
    links.new(color_output, bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def create_uv_sphere(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    parent: bpy.types.Object | None = None,
    segments: int = 48,
    rings: int = 24,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        location=location,
    )
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj.data.materials.append(material)
    obj.parent = parent
    return obj


def create_curve(
    name: str,
    points: list[tuple[float, float, float]],
    material: bpy.types.Material,
    radius: float,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    data = bpy.data.curves.new(name=name, type="CURVE")
    data.dimensions = "3D"
    data.bevel_depth = radius
    data.bevel_resolution = 2
    spline = data.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, coordinate in zip(spline.bezier_points, points):
        point.co = coordinate
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    data.materials.append(material)
    obj.parent = parent
    return obj


def create_rounded_tooth(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    return create_uv_sphere(name, location, scale, material, parent, segments=32, rings=16)


def add_area_light(
    name: str,
    location: tuple[float, float, float],
    energy: float,
    color: tuple[float, float, float],
    size: float,
    target: Vector,
) -> bpy.types.Object:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.color = color
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    bpy.context.collection.objects.link(obj)
    look_at(obj, target)
    return obj


def derive_landmark_indices(raw: np.ndarray, vertex_count: int) -> np.ndarray:
    values = np.asarray(raw, dtype=np.int64).reshape(-1)
    candidates: list[np.ndarray] = []
    if len(values) >= 68:
        candidates.append(values[:68].copy())
    if len(values) >= 204 and len(values) % 3 == 0:
        candidates.append(values[: len(values) // 3].copy())
        candidates.append((values[::3] // 3).copy())
    for candidate in candidates:
        if len(candidate) < 68:
            continue
        candidate = candidate[:68]
        if candidate.min() >= 1 and candidate.max() == vertex_count:
            candidate = candidate - 1
        if candidate.min() >= 0 and candidate.max() < vertex_count and len(np.unique(candidate)) >= 60:
            return candidate.astype(np.int32)
    raise RuntimeError(
        f"Could not derive 68 dense vertex landmarks from BFM keypoints; raw_count={len(values)} vertex_count={vertex_count}"
    )


def load_landmark_indices(path: Path, vertex_count: int) -> np.ndarray:
    if not path.is_file():
        raise RuntimeError(f"BFM model is missing: {path}")
    with path.open("rb") as handle:
        try:
            data = pickle.load(handle, encoding="latin1")
        except TypeError:
            handle.seek(0)
            data = pickle.load(handle)
    if "keypoints" not in data:
        raise RuntimeError("BFM model does not contain keypoints")
    return derive_landmark_indices(np.asarray(data["keypoints"]), vertex_count)


def create_face(
    vertices: np.ndarray,
    triangles: np.ndarray,
    colors_rgb: np.ndarray,
    root: bpy.types.Object,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new("MESH_Canonical_Antinous_V2")
    mesh.from_pydata(vertices.tolist(), [], triangles.tolist())
    mesh.update()
    obj = bpy.data.objects.new("CHAR_Antinous", mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = root
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    attribute = mesh.color_attributes.new(name="Col", type="FLOAT_COLOR", domain="POINT")
    rgba = np.ones((len(colors_rgb), 4), dtype=np.float32)
    rgba[:, :3] = np.clip(colors_rgb, 0.0, 1.0)
    attribute.data.foreach_set("color", rgba.reshape(-1).tolist())
    mesh.materials.append(vertex_color_material())
    solidify = obj.modifiers.new("FaceShell", "SOLIDIFY")
    solidify.thickness = 0.006
    solidify.offset = 0.0
    return obj


def update_face(obj: bpy.types.Object, vertices: np.ndarray) -> None:
    obj.data.vertices.foreach_set("co", vertices.astype(np.float32).reshape(-1).tolist())
    obj.data.update()


def group_center(landmarks: np.ndarray, indices: list[int]) -> np.ndarray:
    return np.mean(landmarks[np.asarray(indices, dtype=np.int32)], axis=0)


def build_landmark_features(
    vertices: np.ndarray,
    colors_rgb: np.ndarray,
    landmark_indices: np.ndarray,
    root: bpy.types.Object,
) -> dict[str, object]:
    landmarks = vertices[landmark_indices]
    xmin, xmax = float(np.min(vertices[:, 0])), float(np.max(vertices[:, 0]))
    ymin, ymax = float(np.min(vertices[:, 1])), float(np.max(vertices[:, 1]))
    zmin, zmax = float(np.min(vertices[:, 2])), float(np.max(vertices[:, 2]))
    width = xmax - xmin
    depth = max(ymax - ymin, 0.20)
    height = zmax - zmin
    center_x = (xmin + xmax) * 0.5
    center_z = (zmin + zmax) * 0.5

    central_skin_indices = np.concatenate((landmark_indices[27:36], landmark_indices[48:60]))
    skin_rgb = np.clip(np.median(colors_rgb[central_skin_indices], axis=0), 0.08, 0.92)
    skin = principled_material(
        "MAT_CraniumSkin_V2",
        (float(skin_rgb[0]), float(skin_rgb[1]), float(skin_rgb[2]), 1.0),
        0.50,
        subsurface=0.025,
    )
    sclera = principled_material("MAT_EyeWhite_V2", (0.76, 0.70, 0.62, 1.0), 0.32)
    iris = principled_material("MAT_Iris_V2", (0.045, 0.022, 0.012, 1.0), 0.24)
    pupil = principled_material("MAT_Pupil_V2", (0.003, 0.002, 0.002, 1.0), 0.18)
    teeth = principled_material("MAT_Teeth_V2", (0.78, 0.70, 0.58, 1.0), 0.36)
    mouth_dark = principled_material("MAT_MouthCavity_V2", (0.055, 0.006, 0.004, 1.0), 0.82)
    tongue = principled_material("MAT_Tongue_V2", (0.22, 0.025, 0.018, 1.0), 0.65)
    hair = principled_material("MAT_Hair_V2", (0.009, 0.004, 0.002, 1.0), 0.43)
    cloth = principled_material("MAT_Cloth_V2", (0.004, 0.003, 0.004, 1.0), 0.80)

    cranium = create_uv_sphere(
        "CHAR_Antinous_Cranium_V2",
        (center_x, ymax + depth * 0.28, zmin + height * 0.60),
        (width * 0.47, depth * 0.58, height * 0.49),
        skin,
        root,
        64,
        32,
    )

    eye_objects: list[bpy.types.Object] = []
    eye_groups = (("L", list(range(36, 42))), ("R", list(range(42, 48))))
    for label, group in eye_groups:
        points = landmarks[group]
        center = np.mean(points, axis=0)
        eye_width = max(float(np.ptp(points[:, 0])), width * 0.10)
        eye_height = max(float(np.ptp(points[:, 2])), height * 0.035)
        white_center = (float(center[0]), float(center[1] + depth * 0.070), float(center[2]))
        white = create_uv_sphere(
            f"EYE_{label}_White_V2",
            white_center,
            (eye_width * 0.46, depth * 0.105, eye_height * 0.62),
            sclera,
            root,
            40,
            20,
        )
        iris_center = (float(center[0]), float(center[1] - depth * 0.005), float(center[2]))
        iris_obj = create_uv_sphere(
            f"EYE_{label}_Iris_V2",
            iris_center,
            (eye_width * 0.145, depth * 0.030, eye_height * 0.46),
            iris,
            root,
            32,
            16,
        )
        pupil_obj = create_uv_sphere(
            f"EYE_{label}_Pupil_V2",
            (iris_center[0], iris_center[1] - depth * 0.012, iris_center[2]),
            (eye_width * 0.060, depth * 0.013, eye_height * 0.22),
            pupil,
            root,
            24,
            12,
        )
        eye_objects.extend((white, iris_obj, pupil_obj))

    inner_mouth = landmarks[60:68]
    mouth_center = np.mean(inner_mouth, axis=0)
    mouth_width = max(float(np.ptp(inner_mouth[:, 0])), width * 0.18)
    mouth_height = max(float(np.ptp(inner_mouth[:, 2])), height * 0.05)
    cavity = create_uv_sphere(
        "MOUTH_Cavity_V2",
        (float(mouth_center[0]), float(mouth_center[1] + depth * 0.095), float(mouth_center[2])),
        (mouth_width * 0.54, depth * 0.085, mouth_height * 0.72),
        mouth_dark,
        root,
        48,
        20,
    )
    upper_z = float(np.mean(landmarks[61:64, 2]) - mouth_height * 0.06)
    left_x = float(landmarks[60, 0] + mouth_width * 0.12)
    right_x = float(landmarks[64, 0] - mouth_width * 0.12)
    teeth_objects: list[bpy.types.Object] = []
    for index, x in enumerate(np.linspace(left_x, right_x, 7)):
        tooth = create_rounded_tooth(
            f"TOOTH_Upper_{index:02d}_V2",
            (float(x), float(mouth_center[1] + depth * 0.018), upper_z),
            (mouth_width * 0.070, depth * 0.055, mouth_height * 0.19),
            teeth,
            root,
        )
        teeth_objects.append(tooth)
    tongue_obj = create_uv_sphere(
        "MOUTH_Tongue_V2",
        (float(mouth_center[0]), float(mouth_center[1] + depth * 0.050), float(np.mean(landmarks[65:68, 2]))),
        (mouth_width * 0.36, depth * 0.050, mouth_height * 0.20),
        tongue,
        root,
        40,
        16,
    )

    ear_objects: list[bpy.types.Object] = []
    for label, point in (("L", landmarks[0]), ("R", landmarks[16])):
        ear = create_uv_sphere(
            f"EAR_{label}_V2",
            (float(point[0]), float(point[1] + depth * 0.22), float(point[2] + height * 0.10)),
            (width * 0.050, depth * 0.19, height * 0.095),
            skin,
            root,
            32,
            16,
        )
        ear_objects.append(ear)

    scalp = create_uv_sphere(
        "CHAR_Antinous_HairCap_V2",
        (center_x, ymax + depth * 0.40, zmin + height * 0.78),
        (width * 0.50, depth * 0.47, height * 0.39),
        hair,
        root,
        64,
        32,
    )

    random.seed(20260804)
    strands: list[bpy.types.Object] = []
    hairline_z = float(np.mean(landmarks[17:27, 2]) + height * 0.13)
    for index in range(52):
        side = -1.0 if index % 2 == 0 else 1.0
        fraction = random.uniform(0.05, 0.92)
        start_x = center_x + side * width * random.uniform(0.08, 0.43)
        start_z = zmin + height * random.uniform(0.76, 1.01)
        start_y = ymax + depth * random.uniform(0.20, 0.44)
        end_x = start_x + side * width * random.uniform(-0.02, 0.07)
        end_z = max(hairline_z, start_z - height * random.uniform(0.12, 0.32))
        end_y = ymax + depth * random.uniform(0.04, 0.16)
        middle = (
            (start_x + end_x) * 0.5 + random.uniform(-0.025, 0.025) * width,
            (start_y + end_y) * 0.5 + depth * 0.02,
            (start_z + end_z) * 0.5 + random.uniform(-0.025, 0.025) * height,
        )
        strands.append(
            create_curve(
                f"HAIR_Landmark_{index:03d}_V2",
                [(start_x, start_y, start_z), middle, (end_x, end_y, end_z)],
                hair,
                max(width * 0.0038, 0.0035),
                root,
            )
        )

    neck = create_uv_sphere(
        "CHAR_Antinous_Neck_V2",
        (center_x, ymax + depth * 0.20, zmin - height * 0.13),
        (width * 0.20, depth * 0.34, height * 0.24),
        skin,
        root,
        48,
        24,
    )
    torso = create_uv_sphere(
        "CHAR_Antinous_Torso_V2",
        (center_x, ymax + depth * 0.38, zmin - height * 0.58),
        (width * 0.70, depth * 0.62, height * 0.43),
        cloth,
        root,
        64,
        32,
    )

    return {
        "landmarks": landmarks,
        "bounds": {"x": [xmin, xmax], "y": [ymin, ymax], "z": [zmin, zmax]},
        "cranium": cranium,
        "eyes": eye_objects,
        "cavity": cavity,
        "teeth": teeth_objects,
        "tongue": tongue_obj,
        "ears": ear_objects,
        "scalp": scalp,
        "strands": strands,
        "neck": neck,
        "torso": torso,
    }


def set_feature_visibility(features: dict[str, object], visible: bool, include_hair: bool = True) -> None:
    single_keys = ("cranium", "cavity", "tongue", "neck", "torso")
    for key in single_keys:
        obj = features[key]
        if isinstance(obj, bpy.types.Object):
            obj.hide_render = not visible
    for key in ("eyes", "teeth", "ears"):
        for obj in features[key]:
            obj.hide_render = not visible
    features["scalp"].hide_render = not (visible and include_hair)
    for obj in features["strands"]:
        obj.hide_render = not (visible and include_hair)


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
    scene.world.color = (0.004, 0.002, 0.001)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass

    camera_data = bpy.data.cameras.new("CAM_Diagnostic_V2")
    camera_data.lens = 68.0
    camera_data.sensor_width = 36.0
    camera_data.dof.use_dof = False
    camera = bpy.data.objects.new("CAM_Diagnostic_V2", camera_data)
    distance = max(height * 3.55, 8.8)
    camera.location = (center.x, ymin - distance, center.z + height * 0.015)
    bpy.context.collection.objects.link(camera)
    look_at(camera, Vector((center.x, center.y, center.z + height * 0.01)))
    scene.camera = camera

    target = Vector((center.x, center.y, center.z + height * 0.03))
    add_area_light(
        "LIGHT_Key_V2",
        (center.x - width * 1.35, ymin - height * 1.25, center.z + height * 1.35),
        560.0,
        (1.0, 0.42, 0.20),
        3.2,
        target,
    )
    add_area_light(
        "LIGHT_Fill_V2",
        (center.x + width * 1.20, ymin - height * 0.90, center.z + height * 0.45),
        165.0,
        (0.28, 0.16, 0.12),
        3.0,
        target,
    )
    add_area_light(
        "LIGHT_Rim_V2",
        (center.x + width * 1.00, ymax + height * 0.85, center.z + height * 1.05),
        360.0,
        (1.0, 0.18, 0.06),
        2.2,
        target,
    )
    return scene, camera


def apply_relative_pose(root: bpy.types.Object, rotation: np.ndarray) -> None:
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
    if np.linalg.det(converted) < 0:
        u[:, -1] *= -1
        converted = u @ vt
    root.rotation_mode = "QUATERNION"
    root.rotation_quaternion = Matrix(converted.tolist()).to_quaternion()


def render(scene: bpy.types.Scene, output: Path) -> None:
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def main() -> int:
    args = parse_args()
    sequence_path = Path(args.sequence).resolve()
    output_dir = Path(args.output_dir).resolve()
    bfm_path = Path(args.bfm_pkl).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not sequence_path.is_file():
        raise SystemExit(f"Canonical sequence is missing: {sequence_path}")

    data = np.load(sequence_path)
    vertices_sequence = np.asarray(data["canonical_vertices"], dtype=np.float32)
    triangles = np.asarray(data["triangles"], dtype=np.int32)
    colors_rgb = np.asarray(data["colors_rgb"], dtype=np.float32)
    relative_rotations = np.asarray(data["relative_rotations"], dtype=np.float32)
    keyframe_index = int(np.asarray(data["keyframe_index"]).reshape(-1)[0])
    landmark_indices = load_landmark_indices(bfm_path, vertices_sequence.shape[1])

    clear_scene()
    root = bpy.data.objects.new("RIG_CanonicalHead_V2", None)
    bpy.context.collection.objects.link(root)
    face = create_face(vertices_sequence[keyframe_index], triangles, colors_rgb, root)
    features = build_landmark_features(
        vertices_sequence[keyframe_index], colors_rgb, landmark_indices, root
    )
    scene, _ = configure_scene(features["bounds"])

    outputs: list[str] = []
    root.rotation_mode = "QUATERNION"
    root.rotation_quaternion = (1.0, 0.0, 0.0, 0.0)

    set_feature_visibility(features, False)
    output = output_dir / "v2_01_canonical_face_only.png"
    render(scene, output)
    outputs.append(output.name)

    set_feature_visibility(features, True, include_hair=False)
    output = output_dir / "v2_02_landmark_features.png"
    render(scene, output)
    outputs.append(output.name)

    set_feature_visibility(features, True, include_hair=True)
    output = output_dir / "v2_03_hair_and_features.png"
    render(scene, output)
    outputs.append(output.name)

    landmarks_sequence = vertices_sequence[:, landmark_indices]
    upper = np.mean(landmarks_sequence[:, 61:64, 2], axis=1)
    lower = np.mean(landmarks_sequence[:, 65:68, 2], axis=1)
    mouth_gap = upper - lower
    mouth_width = np.ptp(landmarks_sequence[:, 48:60, 0], axis=1)
    open_index = int(np.argmax(mouth_gap))
    grin_index = int(np.argmax(mouth_width + mouth_gap * 0.55))

    update_face(face, vertices_sequence[open_index])
    apply_relative_pose(root, relative_rotations[open_index])
    output = output_dir / f"v2_04_open_pose_{open_index:02d}.png"
    render(scene, output)
    outputs.append(output.name)

    update_face(face, vertices_sequence[grin_index])
    apply_relative_pose(root, relative_rotations[grin_index])
    output = output_dir / f"v2_05_grin_pose_{grin_index:02d}.png"
    render(scene, output)
    outputs.append(output.name)

    report = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "claim": "Landmark-driven still diagnostic only; likeness and animation remain unproven until visual review.",
        "route": "POSE_DECOUPLED_CANONICAL_3DDFA_BFM68_LANDMARK_FEATURES_V2",
        "diagnostic_workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "vertex_count": int(vertices_sequence.shape[1]),
        "triangle_count": int(triangles.shape[0]),
        "landmark_count": int(len(landmark_indices)),
        "keyframe_index": keyframe_index,
        "open_index": open_index,
        "grin_index": grin_index,
        "outputs": outputs,
        "bfm_path": str(bfm_path),
    }
    (output_dir / "diagnostic_v2.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
