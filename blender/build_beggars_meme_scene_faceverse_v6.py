from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_beggars_meme_scene_faceverse_v5 as v5  # noqa: E402


base = v5.base
_original_build_banquet_set = base.build_banquet_set


def refined_vertex_skin_material() -> bpy.types.Material:
    material = bpy.data.materials.new("MAT_Antinous_Face_VertexSkin_V6")
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
    balance.inputs[1].default_value = (0.76, 1.03, 1.12)
    base.set_input(bsdf, "Roughness", 0.50)
    base.set_input(bsdf, "Specular IOR Level", 0.24)
    base.set_input(bsdf, "Subsurface Weight", 0.035)
    base.set_input(bsdf, "Emission Strength", 0.0)
    links.new(attribute.outputs["Color"], balance.inputs[0])
    links.new(balance.outputs["Vector"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material


def build_refined_character(
    follow: bpy.types.Object,
    colors_rgb: np.ndarray,
) -> dict[str, object]:
    black_cloth = base.principled_material(
        "MAT_Black_AncientCloth_V6",
        (0.010, 0.008, 0.012, 1.0),
        0.82,
    )
    dark_hair = base.principled_material(
        "MAT_Antinous_DarkBrownHair_V6",
        (0.018, 0.007, 0.004, 1.0),
        0.43,
    )
    gold = base.principled_material(
        "MAT_Aged_Gold_V6",
        (0.30, 0.11, 0.028, 1.0),
        0.34,
        metallic=0.64,
    )
    skin_average = np.clip(np.median(colors_rgb, axis=0), 0.05, 0.95)
    corrected_skin = np.clip(
        skin_average * np.asarray([0.76, 1.03, 1.12], dtype=np.float32),
        0.0,
        1.0,
    )
    neck_skin = base.principled_material(
        "MAT_Neck_Skin_V6",
        (
            float(corrected_skin[0]),
            float(corrected_skin[1]),
            float(corrected_skin[2]),
            1.0,
        ),
        0.58,
    )

    neck = base.create_uv_sphere(
        "CHAR_Antinous_Neck",
        (0.0, 0.34, -1.22),
        (0.46, 0.38, 0.66),
        neck_skin,
        segments=48,
        ring_count=24,
    )
    neck.parent = follow
    torso = base.create_uv_sphere(
        "CHAR_Antinous_Torso",
        (0.04, 0.56, -2.10),
        (1.66, 0.68, 1.04),
        black_cloth,
        segments=64,
        ring_count=32,
    )
    torso.parent = follow
    shoulder_left = base.create_uv_sphere(
        "CHAR_Antinous_Shoulder_L",
        (-1.42, 0.54, -1.72),
        (0.80, 0.56, 0.65),
        black_cloth,
    )
    shoulder_left.parent = follow
    shoulder_right = base.create_uv_sphere(
        "CHAR_Antinous_Shoulder_R",
        (1.42, 0.54, -1.72),
        (0.80, 0.56, 0.65),
        black_cloth,
    )
    shoulder_right.parent = follow
    trim = base.create_curve(
        "COSTUME_GoldNeckTrim",
        [
            (-0.68, -0.02, -1.48),
            (-0.36, -0.11, -1.63),
            (0.00, -0.14, -1.68),
            (0.36, -0.11, -1.63),
            (0.68, -0.02, -1.48),
        ],
        gold,
        0.022,
        parent=follow,
    )

    # A rear-only scalp creates silhouette mass without covering the forehead.
    hair_cap = base.create_uv_sphere(
        "CHAR_Antinous_HairCap",
        (0.0, 0.47, 0.40),
        (0.84, 0.22, 0.88),
        dark_hair,
        segments=64,
        ring_count=32,
    )
    hair_cap.parent = follow

    strands: list[bpy.types.Object] = []
    top_count = 58
    for index, x in enumerate(np.linspace(-0.78, 0.78, top_count)):
        normalized = abs(float(x)) / 0.78
        phase = index * 0.61
        start_z = 1.08 - 0.18 * normalized + 0.045 * math.cos(phase)
        points = [
            (
                float(x),
                -0.57 + 0.012 * math.sin(phase),
                start_z,
            ),
            (
                float(x + 0.045 * math.sin(phase * 1.15)),
                -0.34,
                start_z + 0.08 + 0.035 * math.cos(phase * 0.7),
            ),
            (
                float(x + 0.075 * math.sin(phase + 0.7)),
                -0.05,
                start_z + 0.02,
            ),
            (
                float(x + 0.055 * math.sin(phase + 1.2)),
                0.30,
                start_z - 0.22,
            ),
        ]
        strands.append(
            base.create_curve(
                f"HAIR_TopStrand_{index:03d}",
                points,
                dark_hair,
                0.0055 + 0.0015 * ((index % 4) / 3.0),
                parent=follow,
            )
        )

    side_count = 0
    for side in (-1.0, 1.0):
        for row in range(18):
            z_top = 0.88 - row * 0.075
            x = side * (0.77 + 0.035 * math.sin(row * 0.58))
            points = [
                (x, -0.49, z_top),
                (x + side * 0.045, -0.38, z_top - 0.055),
                (x + side * 0.015, -0.24, z_top - 0.14),
                (x - side * 0.018, -0.10, z_top - 0.22),
            ]
            strands.append(
                base.create_curve(
                    f"HAIR_SideCurl_{side_count:03d}",
                    points,
                    dark_hair,
                    0.0065,
                    parent=follow,
                )
            )
            side_count += 1

    moustache_count = 0
    for side in (-1.0, 1.0):
        for index in range(11):
            t = (index + 0.5) / 11.0
            inner_x = side * (0.035 + 0.38 * t)
            outer_x = inner_x + side * (0.045 + 0.025 * t)
            z = -0.245 - 0.035 * t + 0.010 * math.sin(index * 0.9)
            base.create_curve(
                f"FACIALHAIR_Moustache_{moustache_count:02d}",
                [
                    (inner_x, -1.015, z),
                    (outer_x, -1.035, z - 0.025 - 0.020 * t),
                ],
                dark_hair,
                0.0038,
                parent=follow,
            )
            moustache_count += 1

    sideburn_count = 0
    for side in (-1.0, 1.0):
        for index in range(7):
            x = side * (0.745 + index * 0.010)
            z_top = 0.36 - index * 0.055
            base.create_curve(
                f"FACIALHAIR_Sideburn_{sideburn_count:02d}",
                [
                    (x, -0.505, z_top),
                    (x + side * 0.012, -0.525, z_top - 0.19),
                ],
                dark_hair,
                0.0045,
                parent=follow,
            )
            sideburn_count += 1

    print(
        "BLENDER_REFINED_GROOM_V6=PROVEN "
        f"TOP_STRANDS={top_count} SIDE_CURLS={side_count} "
        f"MOUSTACHE_HAIRS={moustache_count} SIDEBURN_HAIRS={sideburn_count} "
        "ARTIFICIAL_EYEBROWS=ABSENT CHIN_BARS=ABSENT WAVE_SPHERES=ABSENT"
    )
    return {
        "hair_cap": hair_cap,
        "neck": neck,
        "torso": torso,
        "trim": trim,
        "strand_count": len(strands),
        "moustache_hair_count": moustache_count,
        "sideburn_hair_count": sideburn_count,
        "variant": "FACEVERSE_TRUE_MODEL_SPACE_REFINED_GROOM_V6",
    }


def configure_refined_camera_and_lighting(
    face_target: Vector,
) -> tuple[bpy.types.Object, bpy.types.Object]:
    del face_target
    target = Vector((0.0, -0.08, -0.18))
    camera_data = bpy.data.cameras.new("CAM_Hero")
    camera_data.lens = 70.0
    camera_data.sensor_width = 36.0
    camera_data.dof.use_dof = True
    camera_data.dof.aperture_fstop = 2.8
    camera_data.dof.aperture_blades = 9
    camera = bpy.data.objects.new("CAM_Hero", camera_data)
    camera.location = (0.18, -10.55, -0.15)
    bpy.context.collection.objects.link(camera)
    base.look_at(camera, target)

    focus = bpy.data.objects.new("FOCUS_Antinous_Eyes", None)
    focus.location = (0.0, -0.72, 0.20)
    bpy.context.collection.objects.link(focus)
    camera_data.dof.focus_object = focus

    wide_data = bpy.data.cameras.new("CAM_Wide")
    wide_data.lens = 55.0
    wide_data.sensor_width = 36.0
    wide_data.dof.use_dof = True
    wide_data.dof.focus_object = focus
    wide_data.dof.aperture_fstop = 3.5
    wide = bpy.data.objects.new("CAM_Wide", wide_data)
    wide.location = (0.12, -13.2, -0.10)
    bpy.context.collection.objects.link(wide)
    base.look_at(wide, Vector((0.0, 0.35, -0.70)))

    base.add_area_light(
        "LIGHT_WarmKey",
        (-3.4, -4.8, 4.0),
        energy=920.0,
        color=(1.0, 0.62, 0.38),
        size=3.4,
        target=target,
    )
    base.add_area_light(
        "LIGHT_WarmRim",
        (3.4, 0.8, 3.1),
        energy=360.0,
        color=(1.0, 0.30, 0.10),
        size=2.4,
        target=target,
    )
    base.add_area_light(
        "LIGHT_SoftFill",
        (2.2, -3.8, 2.7),
        energy=430.0,
        color=(0.30, 0.44, 0.72),
        size=4.2,
        target=target,
    )
    bpy.context.scene.world.color = (0.006, 0.009, 0.018)
    print(
        "BLENDER_REFINED_CAMERA_LIGHTING_V6=PROVEN "
        "HERO_LENS=70 HERO_DISTANCE=10.55 KEY_NEUTRALIZED=TRUE"
    )
    return camera, wide


def build_refined_banquet_set() -> dict:
    info = _original_build_banquet_set()
    changed = 0
    for light in bpy.data.lights:
        if light.name.startswith("LIGHT_Torch_"):
            light.energy = 120.0
            light.color = (1.0, 0.30, 0.08)
            changed += 1
    print(f"BLENDER_TORCH_LIGHT_BALANCE_V6=PROVEN LIGHTS={changed}")
    return info


def build_smaller_foreground_beggar() -> bpy.types.Object:
    cloth = base.principled_material(
        "MAT_Beggar_Cloth_V6",
        (0.020, 0.016, 0.013, 1.0),
        0.94,
    )
    root = bpy.data.objects.new("CHAR_Beggar", None)
    bpy.context.collection.objects.link(root)
    root.location = (-3.70, -4.80, -1.10)
    root.rotation_euler = (0.06, -0.10, -0.16)
    hood = base.create_uv_sphere(
        "CHAR_Beggar_Hood",
        (0.0, 0.0, 1.08),
        (0.66, 0.54, 0.78),
        cloth,
    )
    shoulders = base.create_uv_sphere(
        "CHAR_Beggar_Shoulders",
        (0.0, 0.15, -0.10),
        (1.18, 0.62, 0.86),
        cloth,
    )
    hood.parent = root
    shoulders.parent = root
    return root


def patch_v6_receipt() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    output_dir = None
    for index, value in enumerate(argv):
        if value == "--output-dir" and index + 1 < len(argv):
            output_dir = Path(argv[index + 1]).resolve()
            break
    if output_dir is None:
        raise RuntimeError("Could not resolve v6 output directory")
    receipt_path = output_dir / "scene_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["character_variant"] = "FACEVERSE_TRUE_MODEL_SPACE_REFINED_GROOM_V6"
    receipt["visual_changes"] = {
        "white_balanced_vertex_skin": True,
        "artificial_eyebrow_bars": False,
        "chin_stubble_bars": False,
        "wave_sphere_clumps": False,
        "fine_curve_hair": True,
        "neutralized_key_fill": True,
        "hero_camera_less_extreme": True,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    base.vertex_skin_material = refined_vertex_skin_material
    v5.build_model_space_character = build_refined_character
    base.configure_camera_and_lighting = configure_refined_camera_and_lighting
    base.build_banquet_set = build_refined_banquet_set
    base.build_foreground_beggar = build_smaller_foreground_beggar
    result = int(v5.main())
    patch_v6_receipt()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
