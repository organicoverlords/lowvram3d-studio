from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import bpy
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import build_beggars_meme_scene as base  # noqa: E402


def build_character_v2(
    follow: bpy.types.Object,
    colors_rgb: np.ndarray,
) -> dict[str, bpy.types.Object | int | str]:
    black_cloth = base.principled_material(
        "MAT_Black_AncientCloth_V2", (0.006, 0.004, 0.006, 1.0), 0.76
    )
    dark_hair = base.principled_material(
        "MAT_Antinous_DarkWavyHair_V2", (0.010, 0.004, 0.002, 1.0), 0.33
    )
    gold = base.principled_material(
        "MAT_Aged_Gold_V2", (0.32, 0.105, 0.020, 1.0), 0.30, metallic=0.70
    )
    skin_average = np.clip(np.median(colors_rgb, axis=0), 0.05, 0.95)
    neck_skin = base.principled_material(
        "MAT_Neck_Skin_V2",
        (float(skin_average[0]), float(skin_average[1]), float(skin_average[2]), 1.0),
        0.54,
    )

    # Compact rear scalp: this is the proven non-occluding placement from the prior hair audit.
    hair_cap = base.create_uv_sphere(
        "CHAR_Antinous_HairCap",
        (0.0, 0.86, 0.55),
        (0.84, 0.29, 0.96),
        dark_hair,
        segments=64,
        ring_count=32,
    )
    hair_cap.parent = follow

    neck = base.create_uv_sphere(
        "CHAR_Antinous_Neck",
        (0.02, 0.36, -1.20),
        (0.58, 0.46, 0.83),
        neck_skin,
    )
    neck.parent = follow
    torso = base.create_uv_sphere(
        "CHAR_Antinous_Torso",
        (0.08, 0.55, -2.10),
        (1.92, 0.72, 1.16),
        black_cloth,
        segments=64,
        ring_count=32,
    )
    torso.parent = follow
    shoulder_left = base.create_uv_sphere(
        "CHAR_Antinous_Shoulder_L",
        (-1.60, 0.52, -1.70),
        (0.90, 0.62, 0.72),
        black_cloth,
    )
    shoulder_left.parent = follow
    shoulder_right = base.create_uv_sphere(
        "CHAR_Antinous_Shoulder_R",
        (1.60, 0.52, -1.70),
        (0.90, 0.62, 0.72),
        black_cloth,
    )
    shoulder_right.parent = follow

    trim = base.create_curve(
        "COSTUME_GoldNeckTrim",
        [
            (-0.78, -0.08, -1.46),
            (-0.42, -0.18, -1.65),
            (0.00, -0.20, -1.72),
            (0.42, -0.18, -1.65),
            (0.78, -0.08, -1.46),
        ],
        gold,
        0.028,
        parent=follow,
    )

    # Deliberate overlapping wave masses define the silhouette without covering the eyes or cheeks.
    wave_specs = [
        ((-0.73, 0.48, 0.69), (0.34, 0.21, 0.30), -0.18),
        ((-0.56, 0.39, 0.91), (0.36, 0.20, 0.31), 0.12),
        ((-0.34, 0.34, 1.07), (0.38, 0.20, 0.30), -0.16),
        ((-0.07, 0.31, 1.17), (0.40, 0.20, 0.31), 0.08),
        ((0.22, 0.32, 1.15), (0.39, 0.20, 0.30), -0.09),
        ((0.49, 0.38, 1.02), (0.37, 0.21, 0.31), 0.16),
        ((0.70, 0.47, 0.82), (0.34, 0.22, 0.32), -0.12),
    ]
    wave_masses: list[bpy.types.Object] = []
    for index, (location, scale, roll) in enumerate(wave_specs):
        mass = base.create_uv_sphere(
            f"HAIR_WaveMass_{index:02d}",
            location,
            scale,
            dark_hair,
            segments=32,
            ring_count=16,
        )
        mass.rotation_euler[1] = roll
        mass.parent = follow
        wave_masses.append(mass)

    random.seed(31082026)
    strands: list[bpy.types.Object] = []
    # Controlled forehead waves remain near the top silhouette and bend backward.
    for index, x in enumerate(np.linspace(-0.76, 0.72, 24)):
        phase = index * 0.73
        start = (
            float(x),
            0.10 + 0.025 * math.sin(phase),
            1.09 + 0.12 * math.cos(phase * 0.72),
        )
        middle = (
            float(x + 0.08 * math.sin(phase)),
            0.24,
            0.94 + 0.08 * math.cos(phase),
        )
        end = (
            float(x + 0.12 * math.sin(phase + 0.8)),
            0.43,
            0.73 + 0.07 * math.cos(phase * 1.2),
        )
        strands.append(
            base.create_curve(
                f"HAIR_Strand_{index:03d}",
                [start, middle, end],
                dark_hair,
                0.010 + 0.002 * ((index % 3) / 2.0),
                parent=follow,
            )
        )

    # Side curls and short sideburns match the public silhouette but remain behind the facial surface.
    strand_index = len(strands)
    for side in (-1.0, 1.0):
        for row in range(9):
            z_top = 0.82 - row * 0.105
            x = side * (0.72 + 0.025 * math.sin(row * 0.8))
            points = [
                (x, 0.24, z_top),
                (x + side * 0.07, 0.30, z_top - 0.10),
                (x - side * 0.02, 0.38, z_top - 0.22),
            ]
            strands.append(
                base.create_curve(
                    f"HAIR_Strand_{strand_index:03d}",
                    points,
                    dark_hair,
                    0.011,
                    parent=follow,
                )
            )
            strand_index += 1

    moustache_left = base.create_curve(
        "FACIALHAIR_Moustache_L",
        [
            (-0.49, -0.48, -0.30),
            (-0.31, -0.55, -0.25),
            (-0.05, -0.58, -0.28),
        ],
        dark_hair,
        0.026,
        parent=follow,
    )
    moustache_right = base.create_curve(
        "FACIALHAIR_Moustache_R",
        [
            (0.05, -0.58, -0.28),
            (0.31, -0.55, -0.25),
            (0.49, -0.48, -0.30),
        ],
        dark_hair,
        0.026,
        parent=follow,
    )
    sideburn_left = base.create_curve(
        "FACIALHAIR_Sideburn_L",
        [(-0.72, -0.06, 0.34), (-0.75, -0.08, 0.08), (-0.70, -0.10, -0.16)],
        dark_hair,
        0.030,
        parent=follow,
    )
    sideburn_right = base.create_curve(
        "FACIALHAIR_Sideburn_R",
        [(0.72, -0.06, 0.34), (0.75, -0.08, 0.08), (0.70, -0.10, -0.16)],
        dark_hair,
        0.030,
        parent=follow,
    )

    return {
        "hair_cap": hair_cap,
        "neck": neck,
        "torso": torso,
        "trim": trim,
        "strand_count": len(strands),
        "wave_mass_count": len(wave_masses),
        "facial_hair_count": 4,
        "moustache_left": moustache_left,
        "moustache_right": moustache_right,
        "sideburn_left": sideburn_left,
        "sideburn_right": sideburn_right,
        "variant": "FACEVERSE_SHARED_IDENTITY_V2_GROOMED",
    }


def defer_animation_render(
    _scene: bpy.types.Scene,
    _camera: bpy.types.Object,
    path: Path,
) -> None:
    print(f"BEGGARS_ANIMATION_RENDER=DEFERRED_VISUAL_PREFLIGHT PATH={path}")


def patch_receipt() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    output_dir = None
    for index, value in enumerate(argv):
        if value == "--output-dir" and index + 1 < len(argv):
            output_dir = Path(argv[index + 1]).resolve()
            break
    if output_dir is None:
        raise RuntimeError("Could not resolve Blender v2 output directory for receipt patching")
    receipt_path = output_dir / "scene_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["classification"] = "USER_VISUAL_REVIEW_REQUIRED"
    receipt["character_variant"] = "FACEVERSE_SHARED_IDENTITY_V2_GROOMED"
    receipt["source_frame_plane_used"] = False
    receipt["animation_render"] = "DEFERRED_UNTIL_STILL_VISUAL_ACCEPTANCE"
    receipt["claim_policy"] = (
        "Blender scene construction, optimized 3D head import, grooming, still renders and save/reload are "
        "machine-proven. Likeness, grooming placement and final animation remain pending visual review."
    )
    receipt["outputs"].pop("silent_video", None)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    base.build_character = build_character_v2
    base.render_animation = defer_animation_render
    result = int(base.main())
    patch_receipt()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
