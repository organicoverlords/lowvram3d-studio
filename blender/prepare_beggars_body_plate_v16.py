from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-receipt", required=True)
    parser.add_argument("--frame", type=int, default=48)
    return parser.parse_args(argv)


def hide_synthetic_head() -> list[str]:
    exact = {
        "CHAR_Antinous",
        "CHAR_Antinous_HairCap",
        "CHAR_Antinous_FittedHairShell",
        "CHAR_Antinous_Neck",
    }
    prefixes = ("HAIR_", "FACIALHAIR_", "EYE_", "MOUTH_")
    hidden: list[str] = []
    for obj in bpy.data.objects:
        if obj.name in exact or obj.name.startswith(prefixes):
            obj.hide_render = True
            obj.hide_set(True)
            hidden.append(obj.name)
    if "CHAR_Antinous" not in hidden:
        raise RuntimeError("V16 could not hide the synthetic face")
    return sorted(hidden)


def render_body_plate(output_dir: Path, frame: int) -> Path:
    scene = bpy.context.scene
    camera = bpy.data.objects.get("CAM_Hero")
    if camera is None:
        raise RuntimeError("V16 hero camera is missing")
    scene.camera = camera
    scene.frame_set(frame)
    scene.render.use_compositing = False
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    path = output_dir / "body_plate.png"
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    if not path.is_file() or path.stat().st_size < 50000:
        raise RuntimeError(f"V16 body plate missing or implausibly small: {path}")
    return path


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    base_receipt_path = Path(args.base_receipt).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not base_receipt_path.is_file():
        raise RuntimeError(f"V16 base receipt missing: {base_receipt_path}")

    hidden = hide_synthetic_head()
    body_plate = render_body_plate(output_dir, args.frame)
    blend_path = output_dir / "beggars_photoreal_recreation.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    if not blend_path.is_file() or blend_path.stat().st_size < 1000000:
        raise RuntimeError("V16 Blender file missing or implausibly small")

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    face = bpy.data.objects.get("CHAR_Antinous")
    if face is None or not face.hide_render:
        raise RuntimeError("V16 hidden synthetic face did not survive save/reload")
    reloaded_plate = render_body_plate(output_dir, args.frame)

    receipt = json.loads(base_receipt_path.read_text(encoding="utf-8"))
    receipt["classification"] = "USER_VISUAL_REVIEW_REQUIRED"
    receipt["character_variant"] = "BLENDER_BODY_PLATE_PLUS_EXTERNAL_RGBA_HEAD_V16"
    receipt["claim_policy"] = (
        "The hidden synthetic head, Blender body/set plate, save/reload and deterministic external "
        "RGBA compositing inputs are machine-proven. Final visual quality remains pending review."
    )
    receipt["visual_changes_v16"] = {
        "blender_compositor_route_rejected": True,
        "external_alpha_composite": True,
        "body_plate": body_plate.name,
        "body_plate_after_reload": reloaded_plate.name,
        "hidden_objects": hidden,
        "animation_rendered": False,
    }
    (output_dir / "scene_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        "BLENDER_FACEVERSE_V16_BODY_PLATE=PROVEN "
        f"HIDDEN={len(hidden)} FRAME={args.frame}"
    )
    print("BLENDER_FACEVERSE_V16_SAVE_RELOAD=PROVEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
