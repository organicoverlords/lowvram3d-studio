from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import bpy


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--blend", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def set_render_visibility(objects: list[bpy.types.Object], visible: bool) -> None:
    for obj in objects:
        obj.hide_render = not visible


def render_variant(scene: bpy.types.Scene, output_dir: Path, name: str) -> Path:
    path = output_dir / f"{name}.png"
    scene.render.filepath = str(path)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    bpy.ops.render.render(write_still=True)
    if not path.is_file() or path.stat().st_size < 100_000:
        raise RuntimeError(f"Diagnostic render is missing or implausibly small: {path}")
    return path


def main() -> int:
    args = parse_args()
    blend_path = Path(args.blend).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not blend_path.is_file():
        raise SystemExit(f"Proven Blender scene is missing: {blend_path}")

    bpy.ops.wm.open_mainfile(filepath=str(blend_path))
    scene = bpy.context.scene
    scene.frame_set(max(scene.frame_start, min(scene.frame_end, 60)))
    scene.camera = bpy.data.objects.get("CAM_Hero")
    if scene.camera is None:
        raise RuntimeError("CAM_Hero is missing")
    scene.render.resolution_percentage = 75

    face = bpy.data.objects.get("CHAR_Antinous")
    hair_cap = bpy.data.objects.get("CHAR_Antinous_HairCap")
    strands = sorted(
        [obj for obj in bpy.data.objects if obj.name.startswith("HAIR_Strand_")],
        key=lambda obj: obj.name,
    )
    if face is None or hair_cap is None or not strands:
        raise RuntimeError(
            f"Required diagnostic objects missing: face={face is not None}, cap={hair_cap is not None}, strands={len(strands)}"
        )

    original_cap_location = hair_cap.location.copy()
    original_cap_scale = hair_cap.scale.copy()
    original_strand_locations = {obj.name: obj.location.copy() for obj in strands}
    outputs: list[Path] = []

    # A: reveal the reconstructed head without any procedural hair obstruction.
    hair_cap.hide_render = True
    set_render_visibility(strands, False)
    outputs.append(render_variant(scene, output_dir, "variant_a_raw_face_no_hair"))

    # B: isolate whether the curve strands alone obscure important facial regions.
    hair_cap.hide_render = True
    set_render_visibility(strands, True)
    for obj in strands:
        obj.location = original_strand_locations[obj.name]
    outputs.append(render_variant(scene, output_dir, "variant_b_raw_face_original_strands"))

    # C: retain a compact scalp volume, moved behind the reconstructed face.
    hair_cap.hide_render = False
    hair_cap.location = (0.0, 0.86, 0.55)
    hair_cap.scale = (0.82, 0.28, 0.92)
    set_render_visibility(strands, False)
    outputs.append(render_variant(scene, output_dir, "variant_c_back_cap_no_strands"))

    # D: compact rear scalp plus all strands shifted behind the facial surface.
    set_render_visibility(strands, True)
    for obj in strands:
        original = original_strand_locations[obj.name]
        obj.location = (original.x, original.y + 0.46, original.z + 0.06)
    outputs.append(render_variant(scene, output_dir, "variant_d_back_cap_rear_strands"))

    # Restore in-memory source state before exit.
    hair_cap.location = original_cap_location
    hair_cap.scale = original_cap_scale
    hair_cap.hide_render = False
    for obj in strands:
        obj.location = original_strand_locations[obj.name]
        obj.hide_render = False

    report = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "source_blend": str(blend_path),
        "source_workflow_run_id": "30863735416",
        "diagnostic_workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "frame": int(scene.frame_current),
        "resolution": [
            int(scene.render.resolution_x * scene.render.resolution_percentage / 100),
            int(scene.render.resolution_y * scene.render.resolution_percentage / 100),
        ],
        "hair_strand_count": len(strands),
        "variants": [path.name for path in outputs],
        "purpose": "Determine whether the procedural scalp cap and random hair curves are the dominant occluders of the reconstructed face.",
    }
    (output_dir / "diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
