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
        for material in getattr(obj.data, "materials", ()):
            if material is not None:
                materials.append(material.name)
    return {
        "name": obj.name,
        "type": obj.type,
        "parent": obj.parent.name if obj.parent else None,
        "location": [float(value) for value in obj.location],
        "rotation_euler": [float(value) for value in obj.rotation_euler],
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
        bsdf.inputs["Specular IOR Level"].default_value = 0.22
    return material


def sphere(name: str, location: Vector, scale: tuple[float, float, float], material: bpy.types.Material) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def render(scene: bpy.types.Scene, output: Path) -> None:
    scene.render.filepath = str(output)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    bpy.ops.render.render(write_still=True)


def hide_named(names: list[str]) -> None:
    for name in names:
        obj = bpy.data.objects.get(name)
        if obj is not None:
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
    if camera is None:
        raise RuntimeError("CAM_Hero is missing")
    scene.camera = camera

    face_plate = bpy.data.objects.get("CHAR_Antinous_FacePlate")
    if face_plate is None:
        raise RuntimeError("CHAR_Antinous_FacePlate is missing")

    candidate_images = [
        image
        for image in bpy.data.images
        if image.size[0] >= 1024
        and image.size[1] >= 1024
        and ("sprite" in image.name.lower() or "face" in image.name.lower())
    ]
    image_report = []
    for image in candidate_images:
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

    report = {
        "classification": "VISUAL_DIAGNOSTIC",
        "source_blend": str(blend_path),
        "frame": args.frame,
        "engine": engine,
        "face_plate_bounds": [list(bounds_world(face_plate)[0]), list(bounds_world(face_plate)[1])],
        "images": image_report,
        "objects": [
            object_snapshot(obj)
            for obj in bpy.data.objects
            if obj.name.startswith(("CHAR_Antinous", "RIG_Head", "HAIR_", "COSTUME_"))
        ],
    }

    render(scene, output_dir / "variant_00_current.png")

    hide_named([
        "CHAR_Antinous_HairCap",
        "CHAR_Antinous_Neck",
        "COSTUME_GoldNeckTrim",
    ])
    for obj in bpy.data.objects:
        if obj.name.startswith("HAIR_Strand_"):
            obj.hide_render = True
            obj.hide_viewport = True
    render(scene, output_dir / "variant_01_plate_only.png")

    minimum, maximum = bounds_world(face_plate)
    center = (minimum + maximum) * 0.5
    height = max(maximum.z - minimum.z, 0.1)
    width = max(maximum.x - minimum.x, 0.1)
    front_y = center.y

    robe = principled("MAT_FACEPLATE_V3_ROBE", (0.006, 0.005, 0.008, 1.0), 0.82)
    skin = principled("MAT_FACEPLATE_V3_SKIN", (0.20, 0.085, 0.050, 1.0), 0.58)
    hair = principled("MAT_FACEPLATE_V3_HAIR", (0.006, 0.0025, 0.0015, 1.0), 0.48)

    neck = sphere(
        "DIAG_V3_Neck",
        Vector((center.x, front_y + height * 0.18, minimum.z - height * 0.20)),
        (width * 0.20, height * 0.12, height * 0.28),
        skin,
    )
    torso = sphere(
        "DIAG_V3_Torso",
        Vector((center.x, front_y + height * 0.42, minimum.z - height * 0.78)),
        (width * 0.95, height * 0.34, height * 0.58),
        robe,
    )
    shoulder_left = sphere(
        "DIAG_V3_Shoulder_L",
        Vector((center.x - width * 0.76, front_y + height * 0.38, minimum.z - height * 0.66)),
        (width * 0.48, height * 0.28, height * 0.34),
        robe,
    )
    shoulder_right = sphere(
        "DIAG_V3_Shoulder_R",
        Vector((center.x + width * 0.76, front_y + height * 0.38, minimum.z - height * 0.66)),
        (width * 0.48, height * 0.28, height * 0.34),
        robe,
    )

    hair_specs = [
        (-0.42, 0.30, 0.78, 0.34, 0.17, 0.32),
        (-0.18, 0.34, 0.96, 0.36, 0.18, 0.30),
        (0.10, 0.34, 1.02, 0.37, 0.18, 0.30),
        (0.36, 0.31, 0.88, 0.34, 0.17, 0.32),
        (-0.52, 0.28, 0.47, 0.23, 0.15, 0.34),
        (0.52, 0.28, 0.52, 0.23, 0.15, 0.34),
    ]
    for index, (rx, ry, rz, sx, sy, sz) in enumerate(hair_specs):
        sphere(
            f"DIAG_V3_Hair_{index:02d}",
            Vector((center.x + width * rx, front_y + height * ry, minimum.z + height * rz)),
            (width * sx, height * sy, height * sz),
            hair,
        )

    render(scene, output_dir / "variant_02_natural_bust.png")

    face_plate.scale.x *= 1.10
    face_plate.scale.z *= 1.06
    face_plate.location.z -= height * 0.015
    camera_data = camera.data
    camera_data.dof.use_dof = False
    render(scene, output_dir / "variant_03_larger_face.png")

    report["diagnostic_objects"] = [
        object_snapshot(obj)
        for obj in (neck, torso, shoulder_left, shoulder_right)
    ]
    (output_dir / "diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(output_dir / "faceplate_v3_diagnostic.blend"))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
