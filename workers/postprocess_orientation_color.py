"""Bounded post-process repair for an existing textured asset.

This worker never regenerates geometry, unwraps UVs, projects source pixels, or runs LOD/xatlas.
It only evaluates four existing camera yaws, optionally applies one root-yaw correction, and then
performs one conservative base-colour recovery pass when the front render is demonstrably pale.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


YAW_NAMES = {0: "yaw_0", 90: "yaw_90", 180: "yaw_180", 270: "yaw_270"}


def foreground_mask(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        return alpha > 127
    rgb = image[:, :, :3].astype(np.float32)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background, axis=2)
    threshold = max(12.0, float(np.percentile(distance, 65)))
    mask = distance > threshold
    if mask.mean() < 0.02 or mask.mean() > 0.98:
        mask = distance > 18.0
    return mask


def normalised_patch(image: np.ndarray, mask: np.ndarray, size: int = 256) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    crop = image[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop_mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    crop_mask = cv2.resize(crop_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
    return np.where(crop_mask[..., None], crop, 0.0).astype(np.float32)


def masked_similarity(source: np.ndarray, rendered: np.ndarray) -> dict:
    source_mask = foreground_mask(source)
    render_mask = foreground_mask(rendered)
    source_patch = normalised_patch(cv2.cvtColor(source[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32), source_mask)
    render_patch = normalised_patch(cv2.cvtColor(rendered[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32), render_mask)
    if source_patch is None or render_patch is None:
        return {"score": 0.0, "silhouette_iou": 0.0, "image_correlation": 0.0}
    a = source_patch.mean(axis=2)
    b = render_patch.mean(axis=2)
    valid = (a > 0) | (b > 0)
    if valid.any() and float(a[valid].std()) > 1e-6 and float(b[valid].std()) > 1e-6:
        correlation = float(np.corrcoef(a[valid].ravel(), b[valid].ravel())[0, 1])
    else:
        correlation = 0.0
    source_small = cv2.resize(source_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST) > 0
    render_small = cv2.resize(render_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST) > 0
    iou = float((source_small & render_small).sum() / max((source_small | render_small).sum(), 1))
    score = 0.55 * iou + 0.45 * max(correlation, 0.0)
    return {"score": round(score, 6), "silhouette_iou": round(iou, 6),
            "image_correlation": round(correlation, 6)}


def choose_orientation(scores: dict[int, dict], margin: float = 0.08) -> dict:
    ordered = sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True)
    best_yaw, best = ordered[0]
    runner_up = ordered[1][1]["score"] if len(ordered) > 1 else 0.0
    clear = float(best["score"] - runner_up) >= margin
    return {"best_yaw": int(best_yaw), "best_score": best["score"],
            "runner_up_score": runner_up, "margin": round(float(best["score"] - runner_up), 6),
            "clear_margin": clear, "repair_required": bool(clear and best_yaw != 0),
            "decision": "rotate_root" if clear and best_yaw != 0 else ("keep" if clear else "undetermined")}


def colour_stats(image: np.ndarray, mask: np.ndarray) -> dict:
    rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    values = rgb[mask]
    sat = hsv[:, :, 1][mask]
    luma = (0.2126 * values[:, 0] + 0.7152 * values[:, 1] + 0.0722 * values[:, 2]) if values.size else np.array([])
    return {"saturation_mean": float(sat.mean()) if sat.size else 0.0,
            "saturation_p90": float(np.percentile(sat, 90)) if sat.size else 0.0,
            "luma_mean": float(luma.mean()) if luma.size else 0.0,
            "luma_std": float(luma.std()) if luma.size else 0.0,
            "pixel_count": int(mask.sum())}


def pale_texture_decision(source_stats: dict, render_stats: dict) -> dict:
    saturation_ratio = render_stats["saturation_mean"] / max(source_stats["saturation_mean"], 1e-6)
    contrast_ratio = render_stats["luma_std"] / max(source_stats["luma_std"], 1e-6)
    pale = saturation_ratio < 0.72 or contrast_ratio < 0.65
    return {"pale": bool(pale), "saturation_ratio": round(saturation_ratio, 6),
            "contrast_ratio": round(contrast_ratio, 6),
            "rule": "saturation<0.72_or_luma_contrast<0.65"}


def recover_basecolor(basecolor_path: Path, output_path: Path, source_stats: dict,
                      render_stats: dict, coverage_path: Path | None = None) -> dict:
    image = cv2.imread(str(basecolor_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read basecolor: {basecolor_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if coverage_path and coverage_path.exists():
        coverage = cv2.imread(str(coverage_path), cv2.IMREAD_GRAYSCALE)
        mask = coverage >= 40
    else:
        mask = rgb.max(axis=2) > 0.02
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation_gain = float(np.clip(source_stats["saturation_mean"] /
                                    max(render_stats["saturation_mean"], 1e-6), 1.0, 1.35))
    contrast_gain = float(np.clip(source_stats["luma_std"] /
                                  max(render_stats["luma_std"], 1e-6), 1.0, 1.15))
    value_scale = float(np.clip(source_stats["luma_mean"] /
                                max(render_stats["luma_mean"], 1e-6), 0.90, 1.10))
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_gain, 0.0, 1.0)
    value = hsv[:, :, 2]
    median = float(np.median(value[mask])) if mask.any() else 0.5
    value = median + (value - median) * contrast_gain
    hsv[:, :, 2] = np.clip(value * value_scale, 0.0, 1.0)
    recovered = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    recovered[~mask] = rgb[~mask]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor((recovered * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return {"saturation_gain": round(saturation_gain, 6), "contrast_gain": round(contrast_gain, 6),
            "value_scale": round(value_scale, 6), "masked_pixels": int(mask.sum()),
            "basecolor_source": str(basecolor_path), "basecolor_output": str(output_path)}


def run_blender(blender: Path, script: Path, args: list[str], repo_root: Path) -> None:
    command = [str(blender), "--background", "--python-use-system-env", "--python", str(script), "--", *args]
    env = dict(os.environ)
    env["PYTHONPATH"] = ";".join(str(repo_root / part) for part in ("blender", "workers", "src"))
    result = subprocess.run(command, cwd=str(repo_root), env=env, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stdout + result.stderr)[-3000:])


def add_required_view_aliases(directory: Path) -> None:
    for source, target in (("yaw_0.png", "front.png"), ("three_quarter.png", "three_quarter.png"),
                           ("yaw_90.png", "side.png"), ("yaw_180.png", "rear.png")):
        source_path, target_path = directory / source, directory / target
        if source_path.exists() and source_path != target_path:
            shutil.copy2(source_path, target_path)


def make_proof_sheet(before: Path, after: Path, output: Path) -> None:
    labels = ((before, "before"), (after, "after"))
    rows = []
    for directory, group in labels:
        tiles = []
        for name in ("front.png", "three_quarter.png", "side.png", "rear.png"):
            image = cv2.imread(str(directory / name), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"missing proof render: {directory / name}")
            image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_AREA)
            cv2.putText(image, f"{group}:{name[:-4]}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(image)
        rows.append(np.concatenate(tiles, axis=1))
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.concatenate(rows, axis=0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blender", default=r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
    parser.add_argument("--front-direction", choices=("+z", "-z"), default="-z")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--coverage", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output = Path(args.output_dir)
    before_dir, after_dir = output / "before", output / "after"
    output.mkdir(parents=True, exist_ok=True)
    source = cv2.imread(args.source, cv2.IMREAD_UNCHANGED)
    if source is None:
        raise RuntimeError(f"cannot read source: {args.source}")

    view_script = repo_root / "blender" / "postprocess_orientation_color_views.py"
    run_blender(Path(args.blender), view_script,
                ["--mode", "render_yaws", "--glb", args.glb, "--output-dir", str(before_dir),
                 f"--front-direction={args.front_direction}", "--resolution", str(args.resolution),
                 "--samples", str(args.samples)], repo_root)
    add_required_view_aliases(before_dir)
    scores = {}
    for yaw, name in YAW_NAMES.items():
        image = cv2.imread(str(before_dir / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        scores[yaw] = masked_similarity(source, image)
    orientation = choose_orientation(scores)

    working_glb = Path(args.glb)
    corrected_glb = output / "corrected_orientation_color.glb"
    orientation_action = "none"
    if orientation["repair_required"]:
        run_blender(Path(args.blender), view_script,
                    ["--mode", "rotate_export", "--glb", str(working_glb),
                     "--output-glb", str(corrected_glb), "--yaw-deg", str(orientation["best_yaw"])], repo_root)
        working_glb = corrected_glb
        orientation_action = "root_yaw_%d" % orientation["best_yaw"]
    else:
        shutil.copy2(working_glb, corrected_glb)

    run_blender(Path(args.blender), view_script,
                ["--mode", "render_yaws", "--glb", str(working_glb), "--output-dir", str(after_dir),
                 f"--front-direction={args.front_direction}", "--resolution", str(args.resolution),
                 "--samples", str(args.samples)], repo_root)
    add_required_view_aliases(after_dir)
    front = cv2.imread(str(after_dir / "yaw_0.png"), cv2.IMREAD_UNCHANGED)
    source_stats = colour_stats(source, foreground_mask(source))
    render_stats = colour_stats(front, foreground_mask(front))
    colour_decision = pale_texture_decision(source_stats, render_stats)
    colour_action = "none"
    recovered_path = output / "corrected_basecolor.png"
    if colour_decision["pale"]:
        colour_action = "bounded_recovery"
        recovery = recover_basecolor(Path(args.basecolor), recovered_path, source_stats, render_stats,
                                     Path(args.coverage) if args.coverage else None)
        run_blender(Path(args.blender), view_script,
                    ["--mode", "replace_basecolor_export", "--glb", str(working_glb),
                     "--basecolor", str(recovered_path), "--output-glb", str(corrected_glb)], repo_root)
        working_glb = corrected_glb
        run_blender(Path(args.blender), view_script,
                     ["--mode", "render_yaws", "--glb", str(working_glb), "--output-dir", str(after_dir),
                     f"--front-direction={args.front_direction}", "--resolution", str(args.resolution),
                     "--samples", str(args.samples)], repo_root)
        add_required_view_aliases(after_dir)
    else:
        shutil.copy2(args.basecolor, recovered_path)
        recovery = {"action": "not_needed", "basecolor_output": str(recovered_path)}

    after_scores = {}
    for yaw, name in YAW_NAMES.items():
        image = cv2.imread(str(after_dir / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        after_scores[yaw] = masked_similarity(source, image)
    after_front = cv2.imread(str(after_dir / "front.png"), cv2.IMREAD_UNCHANGED)
    after_render_stats = colour_stats(after_front, foreground_mask(after_front))
    after_colour_decision = pale_texture_decision(source_stats, after_render_stats)
    proof_sheet = output / "before_after_proof_sheet.png"
    make_proof_sheet(before_dir, after_dir, proof_sheet)
    receipt = {
        "input_glb": str(args.glb), "corrected_glb": str(working_glb),
        "orientation_qa": {"scores_before": scores, "decision": orientation,
                            "action": orientation_action, "scores_after": after_scores},
        "texture_color_qa": {"source_stats": source_stats, "front_render_stats_before": render_stats,
                              "decision": colour_decision, "action": colour_action,
                              "recovery": recovery, "front_render_stats_after": after_render_stats,
                              "decision_after": after_colour_decision},
        "geometry_regenerated": False, "projection_rerun": False, "lod_rerun": False, "xatlas_rerun": False,
        "proof_views": {"before": str(before_dir), "after": str(after_dir),
                        "before_after_proof_sheet": str(proof_sheet)},
        "classification": "PROVEN" if (orientation["decision"] != "undetermined" and not after_colour_decision["pale"])
                           else "NOT_PROVEN",
    }
    (output / "postprocess_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
