from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame", type=int, default=60)
    return parser.parse_args(argv)


def choose_eevee(scene: bpy.types.Scene) -> str:
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = candidate
            return candidate
        except (TypeError, ValueError):
            continue
    raise RuntimeError("No compatible Eevee engine is available")


def object_snapshot(obj: bpy.types.Object) -> dict:
    materials = []
    if getattr(obj, "data", None) is not None:
        materials = [material.name for material in obj.data.materials if material]
    return {
        "name": obj.name,
        "type": obj.type,
        "location": [float(value) for value in obj.location],
        "rotation_euler": [float(value) for value in obj.rotation_euler],
        "scale": [float(value) for value in obj.scale],
        "hide_render": bool(obj.hide_render),
        "materials": materials,
    }


def bounds_world(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return (
        Vector((min(point.x for point in corners), min(point.y for point in corners), min(point.z for point in corners))),
        Vector((max(point.x for point in corners), max(point.y for point in corners), max(point.z for point in corners))),
    )


def principled(name: str, color: tuple[float, float, float, float], roughness: float) -> bpy.types.Material:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        raise RuntimeError(f"Principled BSDF missing for {name}")
    if bsdf.inputs.get("Base Color"):
        bsdf.inputs["Base Color"].default_value = color
    if bsdf.inputs.get("Roughness"):
        bsdf.inputs["Roughness"].default_value = roughness
    if bsdf.inputs.get("Specular IOR Level"):
        bsdf.inputs["Specular IOR Level"].default_value = 0.18
    return material


def create_robe_bust(
    name: str,
    center_x: float,
    front_y: float,
    top_z: float,
    width: float,
    height: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    top_half = width * 0.16
    shoulder_half = width * 0.61
    bottom_half = width * 0.84
    bottom_z = top_z - height
    shoulder_z = top_z - height * 0.17
    back_y = front_y + height * 0.28
    vertices = [
        (center_x - top_half, front_y, top_z),
        (center_x + top_half, front_y, top_z),
        (center_x + shoulder_half, front_y, shoulder_z),
        (center_x + bottom_half, front_y, bottom_z),
        (center_x - bottom_half, front_y, bottom_z),
        (center_x - shoulder_half, front_y, shoulder_z),
        (center_x - top_half, back_y, top_z),
        (center_x + top_half, back_y, top_z),
        (center_x + shoulder_half, back_y, shoulder_z),
        (center_x + bottom_half, back_y, bottom_z),
        (center_x - bottom_half, back_y, bottom_z),
        (center_x - shoulder_half, back_y, shoulder_z),
    ]
    faces = [
        (0, 1, 2, 3, 4, 5),
        (11, 10, 9, 8, 7, 6),
        (0, 6, 7, 1),
        (1, 7, 8, 2),
        (2, 8, 9, 3),
        (3, 9, 10, 4),
        (4, 10, 11, 5),
        (5, 11, 6, 0),
    ]
    mesh = bpy.data.meshes.new(f"MESH_{name}")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(material)
    bevel = obj.modifiers.new("RobeSoftForm", "BEVEL")
    bevel.width = width * 0.045
    bevel.segments = 6
    return obj


def create_curve(
    name: str,
    points: list[Vector],
    bevel_depth: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.resolution_u = 4
    data.bevel_depth = bevel_depth
    data.bevel_resolution = 3
    spline = data.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, coordinate in zip(spline.bezier_points, points):
        point.co = coordinate
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    data.materials.append(material)
    return obj


def render(scene: bpy.types.Scene, output: Path) -> None:
    scene.render.filepath = str(output)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    bpy.ops.render.render(write_still=True)


def hide_original_character() -> None:
    exact = {
        "CHAR_Antinous_HairCap",
        "CHAR_Antinous_Neck",
        "CHAR_Antinous_Torso",
        "CHAR_Antinous_Shoulder_L",
        "CHAR_Antinous_Shoulder_R",
        "COSTUME_GoldNeckTrim",
        "COSTUME_HighCollar_L",
        "COSTUME_HighCollar_R",
    }
    for obj in bpy.data.objects:
        if obj.name in exact or obj.name.startswith(("HAIR_Strand_", "HAIR_Wave", "FACIALHAIR_")):
            obj.hide_render = True
            obj.hide_viewport = True


def main() -> int:
    args = parse_args()
    blend_path = Path(args.blend).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not blend_path.is_file():
        raise SystemExit(f"Blend file is missing: {blend_path}")

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    scene = bpy.context.scene
    engine = choose_eevee(scene)
    scene.frame_set(args.frame)
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    camera = bpy.data.objects.get("CAM_Hero")
    face_plate = bpy.data.objects.get("CHAR_Antinous_FacePlate")
    if camera is None or face_plate is None:
        raise RuntimeError("CAM_Hero or CHAR_Antinous_FacePlate is missing")
    scene.camera = camera

    image_report = []
    for image in bpy.data.images:
        if image.size[0] < 1024 or image.size[1] < 1024:
            continue
        if "sprite" not in image.name.lower() and "face" not in image.name.lower():
            continue
        destination = output_dir / f"packed_{image.name.replace(' ', '_').replace('/', '_')}.png"
        old_path = image.filepath_raw
        old_format = image.file_format
        try:
            image.filepath_raw = str(destination)
            image.file_format = "PNG"
            image.save()
        finally:
            image.filepath_raw = old_path
            image.file_format = old_format
        image_report.append({"name": image.name, "size": list(image.size), "saved": destination.name})

    render(scene, output_dir / "variant_00_current.png")
    hide_original_character()
    render(scene, output_dir / "variant_01_plate_only.png")

    minimum, maximum = bounds_world(face_plate)
    center = (minimum + maximum) * 0.5
    height = max(maximum.z - minimum.z, 0.1)
    width = max(maximum.x - minimum.x, 0.1)

    robe = principled("MAT_FACEPLATE_V10_ROBE", (0.005, 0.0045, 0.008, 1.0), 0.88)
    dark_trim = principled("MAT_FACEPLATE_V10_DARK_TRIM", (0.016, 0.009, 0.014, 1.0), 0.74)

    # The packed alpha footprint shows the visible chin approximately 18% above
    # the plate bottom. A five-percent plate drop and an 11.5-percent raised
    # robe top meet at that visible boundary without covering the face.
    face_plate.location.z -= height * 0.050
    robe_top = minimum.z + height * 0.115
    robe_front_y = center.y + height * 0.16
    robe_obj = create_robe_bust(
        "DIAG_V10_RobeBust",
        center.x,
        robe_front_y,
        robe_top,
        width * 1.72,
        height * 1.18,
        robe,
    )
    render(scene, output_dir / "variant_02_natural_bust.png")

    collar_y = center.y - height * 0.005
    collar_center_z = robe_top - height * 0.045
    collar_left = create_curve(
        "DIAG_V10_VCollar_L",
        [
            Vector((center.x - width * 0.25, collar_y, collar_center_z + height * 0.025)),
            Vector((center.x - width * 0.09, collar_y, collar_center_z - height * 0.030)),
            Vector((center.x, collar_y, collar_center_z - height * 0.085)),
        ],
        width * 0.014,
        dark_trim,
    )
    collar_right = create_curve(
        "DIAG_V10_VCollar_R",
        [
            Vector((center.x, collar_y, collar_center_z - height * 0.085)),
            Vector((center.x + width * 0.09, collar_y, collar_center_z - height * 0.030)),
            Vector((center.x + width * 0.25, collar_y, collar_center_z + height * 0.025)),
        ],
        width * 0.014,
        dark_trim,
    )
    face_plate.scale.x *= 1.04
    face_plate.scale.z *= 1.04
    camera.data.dof.use_dof = False
    render(scene, output_dir / "variant_03_larger_face.png")

    diagnostic_objects = [robe_obj, collar_left, collar_right]
    report = {
        "classification": "VISUAL_DIAGNOSTIC",
        "source_blend": str(blend_path),
        "frame": args.frame,
        "engine": engine,
        "face_plate_bounds": [list(minimum), list(maximum)],
        "images": image_report,
        "alpha_alignment": {
            "median_visible_bottom_from_plate_bottom_fraction": 0.18,
            "plate_drop_fraction": 0.05,
            "robe_top_above_plate_bottom_fraction": 0.115,
        },
        "diagnostic_objects": [object_snapshot(obj) for obj in diagnostic_objects],
        "variant_policy": {
            "variant_01": "derived animated face plate only",
            "variant_02": "robe top aligned to measured visible chin boundary; exposed neck absent",
            "variant_03": "variant 02 with a subtle dark V neckline and four-percent larger face",
        },
    }
    (output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "faceplate_v3_diagnostic.blend"))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
