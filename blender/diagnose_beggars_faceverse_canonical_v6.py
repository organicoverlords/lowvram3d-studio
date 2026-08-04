from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from mathutils import Vector


VARIANTS = (
    ("01_raw_canonical", False, False, False),
    ("02_cranium_corrected", True, False, False),
    ("03_corrected_groomed", True, True, False),
    ("04_corrected_groomed_uniform_teeth", True, True, True),
)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(
        description="Render bounded canonical FaceVerse head variants for visual acceptance."
    )
    parser.add_argument("--canonical", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def delete_variant_objects() -> None:
    for obj in list(bpy.data.objects):
        if obj.get("beggars_variant_object"):
            bpy.data.objects.remove(obj, do_unlink=True)


def look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def set_input(node: bpy.types.Node, name: str, value: Any) -> None:
    socket = node.inputs.get(name)
    if socket is not None:
        socket.default_value = value


def principled_material(
    name: str,
    color: tuple[float, float, float, float],
    roughness: float,
    metallic: float = 0.0,
) -> bpy.types.Material:
    material = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        raise RuntimeError("Principled BSDF node is unavailable")
    set_input(bsdf, "Base Color", color)
    set_input(bsdf, "Roughness", roughness)
    set_input(bsdf, "Metallic", metallic)
    set_input(bsdf, "Specular IOR Level", 0.30)
    return material


def vertex_color_material() -> bpy.types.Material:
    material = bpy.data.materials.get("MAT_FaceVerse_SourceColor") or bpy.data.materials.new(
        "MAT_FaceVerse_SourceColor"
    )
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    attribute = nodes.new("ShaderNodeAttribute")
    attribute.attribute_name = "Col"
    set_input(bsdf, "Roughness", 0.48)
    set_input(bsdf, "Specular IOR Level", 0.27)
    set_input(bsdf, "Subsurface Weight", 0.035)
    links.new(attribute.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def mark_variant(obj: bpy.types.Object) -> bpy.types.Object:
    obj["beggars_variant_object"] = True
    return obj


def create_uv_sphere(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material: bpy.types.Material,
    rotation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    segments: int = 40,
    rings: int = 20,
) -> bpy.types.Object:
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=segments,
        ring_count=rings,
        location=location,
        rotation=rotation,
    )
    obj = mark_variant(bpy.context.object)
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    return obj


def correct_cranium(vertices: np.ndarray, main_end: int) -> np.ndarray:
    corrected = np.asarray(vertices, dtype=np.float32).copy()
    threshold = -0.50
    top = corrected[:main_end, 1] < threshold
    original_y = corrected[:main_end, 1].copy()
    top_extent = max(threshold - float(np.min(original_y)), 1e-6)
    influence = np.clip((threshold - original_y) / top_extent, 0.0, 1.0)
    corrected_main = corrected[:main_end]
    corrected_main[top, 1] = threshold + (corrected_main[top, 1] - threshold) * 0.78
    corrected_main[:, 0] *= 1.0 - influence[:, None].reshape(-1) * 0.035
    corrected[:main_end] = corrected_main
    return corrected


def transform_to_blender(vertices: np.ndarray, scale: float = 0.90) -> np.ndarray:
    source = np.asarray(vertices, dtype=np.float32)
    transformed = np.empty_like(source)
    transformed[:, 0] = source[:, 0] * scale
    transformed[:, 1] = source[:, 2] * scale
    transformed[:, 2] = -source[:, 1] * scale
    return transformed


def create_head_mesh(
    name: str,
    canonical_vertices: np.ndarray,
    triangles: np.ndarray,
    colors: np.ndarray,
    component_boundaries: np.ndarray,
    uniform_teeth: bool,
) -> tuple[bpy.types.Object, np.ndarray]:
    transformed = transform_to_blender(canonical_vertices)
    main_end = int(component_boundaries[0])
    main_center_x = float(
        (np.min(transformed[:main_end, 0]) + np.max(transformed[:main_end, 0])) * 0.5
    )
    transformed[:, 0] -= main_center_x

    mesh = bpy.data.meshes.new(f"MESH_{name}")
    mesh.from_pydata(transformed.tolist(), [], triangles.tolist())
    mesh.update()
    obj = mark_variant(bpy.data.objects.new(name, mesh))
    bpy.context.collection.objects.link(obj)
    for polygon in mesh.polygons:
        polygon.use_smooth = True

    color_attribute = mesh.color_attributes.new(
        name="Col",
        type="FLOAT_COLOR",
        domain="POINT",
    )
    rgba = np.ones((len(colors), 4), dtype=np.float32)
    rgba[:, :3] = np.clip(colors, 0.0, 1.0)
    color_attribute.data.foreach_set("color", rgba.reshape(-1).tolist())
    mesh.materials.append(vertex_color_material())

    if uniform_teeth:
        tooth_material = principled_material(
            "MAT_Simplified_Ivory_Teeth",
            (0.72, 0.62, 0.48, 1.0),
            0.42,
        )
        mesh.materials.append(tooth_material)
        tooth_start = int(component_boundaries[3])
        for polygon, triangle in zip(mesh.polygons, triangles):
            if np.any(np.asarray(triangle) >= tooth_start):
                polygon.material_index = 1
                polygon.use_smooth = True

    return obj, transformed


def add_compact_groom(transformed: np.ndarray, main_end: int) -> dict[str, int]:
    main = transformed[:main_end]
    minimum = np.min(main, axis=0)
    maximum = np.max(main, axis=0)
    extent = maximum - minimum
    center_x = float((minimum[0] + maximum[0]) * 0.5)
    back_y = float(maximum[1])
    top_z = float(maximum[2])

    hair = principled_material(
        "MAT_Compact_Dark_Brown_Hair",
        (0.014, 0.0045, 0.0025, 1.0),
        0.39,
    )

    cap = create_uv_sphere(
        "HAIR_CompactRearScalp",
        (
            center_x,
            back_y - float(extent[1]) * 0.22,
            top_z - float(extent[2]) * 0.38,
        ),
        (
            float(extent[0]) * 0.43,
            float(extent[1]) * 0.24,
            float(extent[2]) * 0.38,
        ),
        hair,
        segments=56,
        rings=28,
    )

    wave_count = 0
    for index, normalized_x in enumerate(np.linspace(-0.78, 0.78, 9)):
        arch = 1.0 - normalized_x**2
        x = center_x + float(normalized_x) * float(extent[0]) * 0.40
        y = back_y - float(extent[1]) * (0.42 + 0.05 * arch)
        z = top_z - float(extent[2]) * (0.11 + 0.09 * abs(normalized_x))
        create_uv_sphere(
            f"HAIR_TopWave_{index:02d}",
            (x, y, z),
            (
                float(extent[0]) * 0.135,
                float(extent[1]) * 0.105,
                float(extent[2]) * 0.115,
            ),
            hair,
            rotation=(0.0, 0.16 * math.sin(index * 0.9), 0.22 * normalized_x),
            segments=32,
            rings=16,
        )
        wave_count += 1

    side_count = 0
    for side in (-1.0, 1.0):
        for row in range(4):
            create_uv_sphere(
                f"HAIR_SideCurl_{'L' if side < 0 else 'R'}_{row:02d}",
                (
                    center_x + side * float(extent[0]) * (0.43 - row * 0.012),
                    back_y - float(extent[1]) * 0.40,
                    top_z - float(extent[2]) * (0.28 + row * 0.16),
                ),
                (
                    float(extent[0]) * 0.11,
                    float(extent[1]) * 0.09,
                    float(extent[2]) * 0.12,
                ),
                hair,
                rotation=(0.0, side * 0.18, side * 0.12),
                segments=28,
                rings=14,
            )
            side_count += 1

    return {"cap": 1 if cap else 0, "top_waves": wave_count, "side_curls": side_count}


def add_context_geometry(transformed: np.ndarray, main_end: int) -> None:
    main = transformed[:main_end]
    minimum = np.min(main, axis=0)
    maximum = np.max(main, axis=0)
    extent = maximum - minimum
    center_x = float((minimum[0] + maximum[0]) * 0.5)
    chin_z = float(minimum[2])
    back_y = float(maximum[1])

    skin = principled_material("MAT_Diagnostic_Neck", (0.27, 0.07, 0.045, 1.0), 0.56)
    cloth = principled_material("MAT_Diagnostic_Tunic", (0.009, 0.006, 0.008, 1.0), 0.80)
    create_uv_sphere(
        "CTX_Neck",
        (center_x, back_y - float(extent[1]) * 0.52, chin_z - float(extent[2]) * 0.21),
        (float(extent[0]) * 0.27, float(extent[1]) * 0.24, float(extent[2]) * 0.28),
        skin,
    )
    create_uv_sphere(
        "CTX_Shoulders",
        (center_x, back_y - float(extent[1]) * 0.30, chin_z - float(extent[2]) * 0.62),
        (float(extent[0]) * 0.92, float(extent[1]) * 0.42, float(extent[2]) * 0.38),
        cloth,
        segments=48,
        rings=24,
    )


def add_area_light(
    name: str,
    location: tuple[float, float, float],
    energy: float,
    color: tuple[float, float, float],
    size: float,
    target: Vector,
) -> None:
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = energy
    light_data.color = color
    light_data.shape = "DISK"
    light_data.size = size
    light = bpy.data.objects.new(name, light_data)
    light.location = location
    bpy.context.collection.objects.link(light)
    look_at(light, target)


def configure_scene() -> bpy.types.Object:
    scene = bpy.context.scene
    selected_engine = None
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE"):
        try:
            scene.render.engine = candidate
            selected_engine = candidate
            break
        except TypeError:
            continue
    if selected_engine is None:
        raise RuntimeError("No compatible Eevee render engine is available")
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.world.color = (0.006, 0.002, 0.0015)
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass

    camera_data = bpy.data.cameras.new("CAM_CanonicalHeroDiagnostic")
    camera_data.lens = 65.0
    camera_data.sensor_width = 36.0
    camera_data.dof.use_dof = False
    camera = bpy.data.objects.new("CAM_CanonicalHeroDiagnostic", camera_data)
    camera.location = (0.0, -9.0, 0.12)
    bpy.context.collection.objects.link(camera)
    look_at(camera, Vector((0.0, -0.15, 0.10)))
    scene.camera = camera

    add_area_light(
        "LIGHT_DiagnosticWarmKey",
        (-3.2, -4.2, 3.6),
        820.0,
        (1.0, 0.24, 0.10),
        3.0,
        Vector((0.0, 0.0, 0.15)),
    )
    add_area_light(
        "LIGHT_DiagnosticNeutralFill",
        (3.2, -3.5, 1.6),
        330.0,
        (0.36, 0.42, 0.55),
        2.8,
        Vector((0.0, 0.0, 0.05)),
    )
    add_area_light(
        "LIGHT_DiagnosticRim",
        (1.5, 2.0, 3.2),
        520.0,
        (1.0, 0.09, 0.025),
        2.2,
        Vector((0.0, 0.25, 0.25)),
    )
    return camera


def main() -> int:
    args = parse_args()
    canonical_path = Path(args.canonical).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not canonical_path.is_file():
        raise SystemExit(f"Canonical FaceVerse diagnostic input is missing: {canonical_path}")

    data = np.load(canonical_path)
    canonical_sequence = np.asarray(data["canonical_vertices"], dtype=np.float32)
    triangles = np.asarray(data["triangles"], dtype=np.int32)
    colors = np.asarray(data["source_colors"], dtype=np.float32)
    component_boundaries = np.asarray(data["component_boundaries"], dtype=np.int32).reshape(-1)
    keyframe_index = int(np.asarray(data["keyframe_index"], dtype=np.int32).reshape(-1)[0])
    if canonical_sequence.shape[1:] != (19546, 3):
        raise RuntimeError(f"Unexpected canonical sequence shape: {canonical_sequence.shape}")
    if triangles.shape != (38792, 3):
        raise RuntimeError(f"Unexpected canonical triangle shape: {triangles.shape}")
    if component_boundaries.size < 6:
        raise RuntimeError(f"Incomplete component boundaries: {component_boundaries.tolist()}")

    clear_scene()
    camera = configure_scene()
    scene = bpy.context.scene
    canonical_keyframe = canonical_sequence[keyframe_index]
    main_end = int(component_boundaries[0])

    rendered_variants: list[dict[str, Any]] = []
    for variant_name, use_correction, use_groom, uniform_teeth in VARIANTS:
        delete_variant_objects()
        vertices = (
            correct_cranium(canonical_keyframe, main_end)
            if use_correction
            else canonical_keyframe.copy()
        )
        head, transformed = create_head_mesh(
            f"HEAD_{variant_name}",
            vertices,
            triangles,
            colors,
            component_boundaries,
            uniform_teeth,
        )
        add_context_geometry(transformed, main_end)
        groom_info = add_compact_groom(transformed, main_end) if use_groom else {
            "cap": 0,
            "top_waves": 0,
            "side_curls": 0,
        }
        scene.camera = camera
        output_path = output_dir / f"{variant_name}.png"
        scene.render.filepath = str(output_path)
        bpy.ops.render.render(write_still=True)
        if not output_path.is_file() or output_path.stat().st_size < 50000:
            raise RuntimeError(f"Diagnostic render is missing or implausibly small: {output_path}")
        main_bounds_min = np.min(transformed[:main_end], axis=0)
        main_bounds_max = np.max(transformed[:main_end], axis=0)
        rendered_variants.append(
            {
                "variant": variant_name,
                "cranium_corrected": use_correction,
                "groomed": use_groom,
                "uniform_teeth": uniform_teeth,
                "groom": groom_info,
                "render": output_path.name,
                "main_head_bounds": {
                    "min": main_bounds_min.astype(float).tolist(),
                    "max": main_bounds_max.astype(float).tolist(),
                },
                "object_count": len(bpy.data.objects),
            }
        )
        print(f"FACEVERSE_CANONICAL_BLENDER_VARIANT=PROVEN NAME={variant_name}")

    report = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "route": "FACEVERSE_V4_CANONICAL_BLENDER_HERO_DIAGNOSTIC_V6",
        "claim": (
            "Canonical-coordinate head orientation, bounded cranium correction, compact rear-biased "
            "grooming and simplified tooth materials are machine-proven; visual likeness is pending review."
        ),
        "keyframe_index": keyframe_index,
        "vertex_count": int(canonical_sequence.shape[1]),
        "triangle_count": int(triangles.shape[0]),
        "component_boundaries": component_boundaries.astype(int).tolist(),
        "coordinate_contract": {
            "blender_x": "FaceVerse X * 0.90",
            "blender_y": "FaceVerse Z * 0.90; negative is toward camera",
            "blender_z": "-FaceVerse Y * 0.90",
            "projected_image_rescaling_used": False,
        },
        "variants": rendered_variants,
        "source_frame_plane_used": False,
        "animation_rendered": False,
    }
    report_path = output_dir / "canonical_blender_diagnostic.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
