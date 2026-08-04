from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--output-dir", required=True)
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
) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        raise RuntimeError("Principled BSDF is unavailable")
    set_input(bsdf, "Base Color", color)
    set_input(bsdf, "Roughness", roughness)
    set_input(bsdf, "Metallic", metallic)
    set_input(bsdf, "Specular IOR Level", 0.32)
    return material


def vertex_color_material() -> bpy.types.Material:
    material = bpy.data.materials.new("MAT_CanonicalFace_VertexColor")
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
        raise RuntimeError("Vertex-color shader output is unavailable")
    set_input(bsdf, "Roughness", 0.50)
    set_input(bsdf, "Specular IOR Level", 0.28)
    set_input(bsdf, "Subsurface Weight", 0.035)
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
    curve_data = bpy.data.curves.new(name=name, type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.bevel_depth = radius
    curve_data.bevel_resolution = 2
    spline = curve_data.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for control, coordinate in zip(spline.bezier_points, points):
        control.co = coordinate
        control.handle_left_type = "AUTO"
        control.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve_data)
    bpy.context.collection.objects.link(obj)
    curve_data.materials.append(material)
    obj.parent = parent
    return obj


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


def create_face(
    vertices: np.ndarray,
    triangles: np.ndarray,
    colors_rgb: np.ndarray,
    root: bpy.types.Object,
) -> bpy.types.Object:
    mesh = bpy.data.meshes.new("MESH_Canonical_Antinous")
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
    solidify.thickness = 0.008
    solidify.offset = 0.0
    return obj


def build_features(
    vertices: np.ndarray,
    colors_rgb: np.ndarray,
    root: bpy.types.Object,
) -> dict[str, object]:
    xmin, xmax = float(np.min(vertices[:, 0])), float(np.max(vertices[:, 0]))
    ymin, ymax = float(np.min(vertices[:, 1])), float(np.max(vertices[:, 1]))
    zmin, zmax = float(np.min(vertices[:, 2])), float(np.max(vertices[:, 2]))
    width = xmax - xmin
    height = zmax - zmin
    depth = max(ymax - ymin, 0.30)
    center_x = (xmin + xmax) * 0.5
    front_y = ymin

    skin_median = np.clip(np.median(colors_rgb, axis=0), 0.08, 0.92)
    skin = principled_material(
        "MAT_CraniumSkin",
        (float(skin_median[0]), float(skin_median[1]), float(skin_median[2]), 1.0),
        0.52,
    )
    sclera = principled_material("MAT_EyeWhite", (0.72, 0.62, 0.54, 1.0), 0.30)
    iris = principled_material("MAT_Iris", (0.025, 0.015, 0.010, 1.0), 0.22)
    teeth = principled_material("MAT_Teeth", (0.82, 0.70, 0.56, 1.0), 0.36)
    hair = principled_material("MAT_Hair", (0.012, 0.006, 0.004, 1.0), 0.42)
    cloth = principled_material("MAT_Cloth", (0.006, 0.004, 0.005, 1.0), 0.78)

    head_center_z = zmin + height * 0.56
    cranium = create_uv_sphere(
        "CHAR_Antinous_Cranium",
        (center_x, ymax + depth * 0.18, head_center_z),
        (width * 0.45, depth * 0.74, height * 0.47),
        skin,
        parent=root,
        segments=64,
        rings=32,
    )

    eye_z = zmin + height * 0.62
    eye_dx = width * 0.205
    eye_y = front_y - depth * 0.015
    eye_scale = (width * 0.070, depth * 0.18, height * 0.046)
    eyes = []
    for side, x in (("L", center_x - eye_dx), ("R", center_x + eye_dx)):
        white = create_uv_sphere(
            f"EYE_{side}_White",
            (x, eye_y, eye_z),
            eye_scale,
            sclera,
            parent=root,
            segments=40,
            rings=20,
        )
        pupil = create_uv_sphere(
            f"EYE_{side}_Iris",
            (x, eye_y - depth * 0.105, eye_z),
            (width * 0.024, depth * 0.035, height * 0.020),
            iris,
            parent=root,
            segments=32,
            rings=16,
        )
        eyes.extend((white, pupil))

    mouth_z = zmin + height * 0.305
    tooth_band = create_uv_sphere(
        "MOUTH_UpperTeeth",
        (center_x + width * 0.015, front_y - depth * 0.025, mouth_z),
        (width * 0.205, depth * 0.105, height * 0.052),
        teeth,
        parent=root,
        segments=48,
        rings=20,
    )

    left_ear = create_uv_sphere(
        "EAR_L",
        (xmin + width * 0.025, ymax + depth * 0.02, zmin + height * 0.54),
        (width * 0.075, depth * 0.30, height * 0.115),
        skin,
        parent=root,
    )
    right_ear = create_uv_sphere(
        "EAR_R",
        (xmax - width * 0.025, ymax + depth * 0.02, zmin + height * 0.54),
        (width * 0.075, depth * 0.30, height * 0.115),
        skin,
        parent=root,
    )

    scalp = create_uv_sphere(
        "CHAR_Antinous_HairCap",
        (center_x, ymax + depth * 0.30, zmin + height * 0.76),
        (width * 0.47, depth * 0.62, height * 0.37),
        hair,
        parent=root,
        segments=64,
        rings=32,
    )

    random.seed(20260804)
    strands: list[bpy.types.Object] = []
    for index in range(38):
        angle = random.uniform(-1.15, 1.15)
        start_x = center_x + math.sin(angle) * width * random.uniform(0.16, 0.40)
        start_z = zmin + height * random.uniform(0.70, 0.96)
        start_y = ymax + depth * random.uniform(0.22, 0.42)
        direction = -1.0 if start_x < center_x else 1.0
        end = (
            start_x + direction * width * random.uniform(0.01, 0.07),
            start_y - depth * random.uniform(0.08, 0.20),
            start_z - height * random.uniform(0.10, 0.28),
        )
        middle = (
            (start_x + end[0]) * 0.5 + random.uniform(-0.035, 0.035) * width,
            (start_y + end[1]) * 0.5,
            (start_z + end[2]) * 0.5 + random.uniform(-0.025, 0.025) * height,
        )
        strands.append(
            create_curve(
                f"HAIR_Canonical_{index:03d}",
                [(start_x, start_y, start_z), middle, end],
                hair,
                max(width * 0.0042, 0.004),
                root,
            )
        )

    neck = create_uv_sphere(
        "CHAR_Antinous_Neck",
        (center_x, ymax + depth * 0.12, zmin - height * 0.12),
        (width * 0.21, depth * 0.42, height * 0.25),
        skin,
        parent=root,
    )
    torso = create_uv_sphere(
        "CHAR_Antinous_Torso",
        (center_x, ymax + depth * 0.35, zmin - height * 0.56),
        (width * 0.72, depth * 0.72, height * 0.45),
        cloth,
        parent=root,
        segments=64,
        rings=32,
    )

    return {
        "bounds": {
            "x": [xmin, xmax],
            "y": [ymin, ymax],
            "z": [zmin, zmax],
        },
        "cranium": cranium,
        "eyes": eyes,
        "teeth": tooth_band,
        "ears": [left_ear, right_ear],
        "scalp": scalp,
        "strands": strands,
        "neck": neck,
        "torso": torso,
    }


def set_group_visibility(features: dict[str, object], visible: bool) -> None:
    for key in ("cranium", "teeth", "scalp", "neck", "torso"):
        obj = features[key]
        if isinstance(obj, bpy.types.Object):
            obj.hide_render = not visible
    for key in ("eyes", "ears", "strands"):
        for obj in features[key]:
            obj.hide_render = not visible


def configure_scene() -> tuple[bpy.types.Scene, bpy.types.Object]:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.world.color = (0.003, 0.0015, 0.001, 1.0)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass

    camera_data = bpy.data.cameras.new("CAM_Diagnostic")
    camera_data.lens = 78.0
    camera_data.sensor_width = 36.0
    camera_data.dof.use_dof = False
    camera = bpy.data.objects.new("CAM_Diagnostic", camera_data)
    camera.location = (0.0, -7.6, 0.02)
    bpy.context.collection.objects.link(camera)
    look_at(camera, Vector((0.0, 0.0, -0.03)))
    scene.camera = camera

    add_area_light(
        "LIGHT_Key",
        (-3.1, -3.4, 3.3),
        780.0,
        (1.0, 0.28, 0.10),
        3.0,
        Vector((0.0, 0.0, 0.0)),
    )
    add_area_light(
        "LIGHT_Fill",
        (2.8, -2.4, 1.2),
        300.0,
        (0.32, 0.12, 0.07),
        2.6,
        Vector((0.0, 0.0, 0.0)),
    )
    add_area_light(
        "LIGHT_Rim",
        (2.2, 1.5, 3.4),
        520.0,
        (1.0, 0.12, 0.035),
        2.0,
        Vector((0.0, 0.0, 0.0)),
    )
    return scene, camera


def render(scene: bpy.types.Scene, path: Path) -> None:
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    if not path.is_file() or path.stat().st_size < 100_000:
        raise RuntimeError(f"Diagnostic render is missing or implausibly small: {path}")


def main() -> int:
    args = parse_args()
    sequence_path = Path(args.sequence).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not sequence_path.is_file():
        raise SystemExit(f"Canonical sequence is missing: {sequence_path}")

    data = np.load(sequence_path)
    vertices_sequence = np.asarray(data["canonical_vertices"], dtype=np.float32)
    triangles = np.asarray(data["triangles"], dtype=np.int32)
    colors_rgb = np.asarray(data["colors_rgb"], dtype=np.float32)
    keyframe_index = int(np.asarray(data["keyframe_index"]).reshape(-1)[0])
    selected_indices = sorted(
        {
            keyframe_index,
            int(round((len(vertices_sequence) - 1) * 0.62)),
            int(round((len(vertices_sequence) - 1) * 0.86)),
        }
    )

    clear_scene()
    scene, _ = configure_scene()
    root = bpy.data.objects.new("RIG_CanonicalHead", None)
    bpy.context.collection.objects.link(root)
    root.rotation_euler = (
        math.radians(-3.0),
        math.radians(-8.0),
        math.radians(7.0),
    )

    face = create_face(vertices_sequence[keyframe_index], triangles, colors_rgb, root)
    features = build_features(vertices_sequence[keyframe_index], colors_rgb, root)

    set_group_visibility(features, False)
    render(scene, output_dir / "variant_a_canonical_face_only.png")

    set_group_visibility(features, True)
    render(scene, output_dir / "variant_b_canonical_keyframe_features.png")

    outputs = [
        "variant_a_canonical_face_only.png",
        "variant_b_canonical_keyframe_features.png",
    ]
    for sequence_index in selected_indices:
        face.data.vertices.foreach_set(
            "co", vertices_sequence[sequence_index].reshape(-1).tolist()
        )
        face.data.update()
        filename = f"variant_expression_{sequence_index:02d}.png"
        render(scene, output_dir / filename)
        outputs.append(filename)

    report = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "diagnostic_workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "route": "POSE_DECOUPLED_CANONICAL_3DDFA_WITH_EXPLICIT_FEATURES",
        "keyframe_index": keyframe_index,
        "selected_expression_indices": selected_indices,
        "vertex_count": int(vertices_sequence.shape[1]),
        "triangle_count": int(triangles.shape[0]),
        "outputs": outputs,
        "feature_bounds": features["bounds"],
        "claim": "Still renders only; animation and likeness are not proven.",
    }
    (output_dir / "diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
