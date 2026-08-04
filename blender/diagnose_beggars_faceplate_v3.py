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


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def object_snapshot(obj: bpy.types.Object) -> dict:
    materials = []
    if getattr(obj, "data", None) is not None:
        for material in getattr(obj.data, "materials", ()):
            if material is not None:
                materials.append(material.name)
    return {
        "name": obj.name,
        "type": obj.type,
        "location": [float(value) for value in obj.location],
        "scale": [float(value) for value in obj.scale],
        "hide_render": bool(obj.hide_render),
        "materials": materials,
    }


def bounds_world(obj: bpy.types.Object) -> tuple[Vector, Vector]:
    corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    minimum = Vector((min(point.x for point in corners), min(point.y for point in corners), min(point.z for point in corners)))
    maximum = Vector((max(point.x for point in corners), max(point.y for point in corners), max(point.z for point in corners)))
    return minimum, maximum


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
    if bsdf.inputs.get("Subsurface Weight"):
        bsdf.inputs["Subsurface Weight"].default_value = 0.025 if "SKIN" in name else 0.0
    return material


def sphere(name: str, location: Vector, scale: tuple[float, float, float], material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def silhouette_mesh(
    name: str,
    center: Vector,
    width: float,
    height: float,
    y: float,
    material: bpy.types.Material,
) -> bpy.types.Object:
    outline = [
        (-0.54, -0.20),
        (-0.50, 0.24),
        (-0.38, 0.58),
        (-0.18, 0.76),
        (0.05, 0.82),
        (0.28, 0.70),
        (0.46, 0.43),
        (0.54, 0.08),
        (0.48, -0.30),
        (0.29, -0.48),
        (0.00, -0.54),
        (-0.30, -0.44),
    ]
    vertices = [(center.x, y, center.z)]
    vertices.extend(
        (center.x + width * x, y, center.z + height * z) for x, z in outline
    )
    faces = []
    for index in range(len(outline)):
        current = index + 1
        following = (index + 1) % len(outline) + 1
        faces.append((0, current, following))
    mesh = bpy.data.meshes.new(f"MESH_{name}")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(material)
    return obj


def area_light(name: str, location: Vector, energy: float, color: tuple[float, float, float], size: float, target: Vector) -> bpy.types.Object:
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.color = color
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    obj.location = location
    bpy.context.collection.objects.link(obj)
    look_at(obj, target)
    return obj


def render(scene: bpy.types.Scene, output: Path) -> None:
    scene.render.filepath = str(output)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    bpy.ops.render.render(write_still=True)


def hide_original_character_shell() -> None:
    names = [
        "CHAR_Antinous_HairCap",
        "CHAR_Antinous_Neck",
        "CHAR_Antinous_Torso",
        "CHAR_Antinous_Shoulder_L",
        "CHAR_Antinous_Shoulder_R",
        "COSTUME_GoldNeckTrim",
    ]
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is not None:
            obj.hide_render = True
            obj.hide_viewport = True
    for obj in bpy.data.objects:
        if obj.name.startswith(("HAIR_Strand_", "HAIR_Wave", "FACIALHAIR_")):
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
    hide_original_character_shell()
    render(scene, output_dir / "variant_01_plate_only.png")

    minimum, maximum = bounds_world(face_plate)
    center = (minimum + maximum) * 0.5
    height = max(maximum.z - minimum.z, 0.1)
    width = max(maximum.x - minimum.x, 0.1)
    front_y = center.y

    robe = principled("MAT_FACEPLATE_V6_ROBE", (0.003, 0.0035, 0.005, 1.0), 0.88)
    skin = principled("MAT_FACEPLATE_V6_SKIN", (0.135, 0.064, 0.055, 1.0), 0.60)
    hair = principled("MAT_FACEPLATE_V6_HAIR", (0.002, 0.001, 0.0007, 1.0), 0.62)

    face_plate.location.y -= height * 0.035
    neck = sphere(
        "DIAG_V6_Neck",
        Vector((center.x, front_y + height * 0.34, minimum.z - height * 0.22)),
        (width * 0.18, height * 0.11, height * 0.29),
        skin,
    )
    torso = sphere(
        "DIAG_V6_Torso",
        Vector((center.x, front_y + height * 0.52, minimum.z - height * 0.82)),
        (width * 1.00, height * 0.31, height * 0.59),
        robe,
    )
    shoulder_left = sphere(
        "DIAG_V6_Shoulder_L",
        Vector((center.x - width * 0.77, front_y + height * 0.48, minimum.z - height * 0.69)),
        (width * 0.47, height * 0.23, height * 0.33),
        robe,
    )
    shoulder_right = sphere(
        "DIAG_V6_Shoulder_R",
        Vector((center.x + width * 0.77, front_y + height * 0.48, minimum.z - height * 0.69)),
        (width * 0.47, height * 0.23, height * 0.33),
        robe,
    )
    render(scene, output_dir / "variant_02_natural_bust.png")

    backplate = silhouette_mesh(
        "DIAG_V6_HairHeadSilhouette",
        center=Vector((center.x, front_y, center.z + height * 0.03)),
        width=width * 1.04,
        height=height * 1.04,
        y=front_y + height * 0.055,
        material=hair,
    )
    ear_left = sphere(
        "DIAG_V6_Ear_L",
        Vector((center.x - width * 0.50, front_y + height * 0.10, center.z - height * 0.01)),
        (width * 0.075, height * 0.042, height * 0.125),
        skin,
    )
    ear_right = sphere(
        "DIAG_V6_Ear_R",
        Vector((center.x + width * 0.50, front_y + height * 0.10, center.z - height * 0.01)),
        (width * 0.075, height * 0.042, height * 0.125),
        skin,
    )
    area_light(
        "DIAG_V6_FrontFill",
        Vector((center.x + width * 1.2, front_y - height * 2.4, center.z + height * 1.1)),
        260.0,
        (0.72, 0.78, 1.0),
        height * 2.0,
        Vector((center.x, center.y, center.z - height * 0.18)),
    )
    camera.data.dof.use_dof = False
    render(scene, output_dir / "variant_03_larger_face.png")

    diagnostic_objects = [neck, torso, shoulder_left, shoulder_right, backplate, ear_left, ear_right]
    report = {
        "classification": "VISUAL_DIAGNOSTIC",
        "source_blend": str(blend_path),
        "frame": args.frame,
        "engine": engine,
        "face_plate_bounds": [list(minimum), list(maximum)],
        "images": image_report,
        "diagnostic_objects": [object_snapshot(obj) for obj in diagnostic_objects],
        "variant_policy": {
            "variant_01": "derived face plate only",
            "variant_02": "face plate with true 3D neck and robe only",
            "variant_03": "variant 02 plus a thin dark head/hair silhouette and small ears behind the face",
        },
    }
    (output_dir / "diagnostic.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "faceplate_v3_diagnostic.blend"))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
