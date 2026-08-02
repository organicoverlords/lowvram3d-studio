"""Orchestrate bounded GPU source-delight and multiview texture references.

This stage produces candidate 2-D reference views only.  It never edits a mesh,
UVs, atlas, or GLB.  Every accepted image must pass structural image QA before
it is eligible for the existing CPU depth/normal/face-ID projector.

Sequence policy:
1. optional front albedo/delight job;
2. left, right and true-rear jobs, strictly one at a time;
3. one 512 attempt plus one 384 retry only when the child reports OOM;
4. reject blank, mask-mismatched, desaturated, or front-looking rear output;
5. preserve all child receipts and stop on the first rejected required view.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"IMAGE_UNREADABLE:{path}")
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise RuntimeError(f"IMAGE_CHANNELS_INVALID:{path}:{image.shape}")
    return image


def foreground_mask(image: np.ndarray) -> np.ndarray:
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        if np.count_nonzero(alpha) > 0:
            return alpha > 32
    rgb = image[:, :, :3].astype(np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background, axis=2)
    threshold = max(10.0, float(np.percentile(distance, 55)))
    mask = distance > threshold
    if mask.mean() < 0.01 or mask.mean() > 0.95:
        mask = distance > 18.0
    return mask


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
    ) > 0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        b = resize_mask(b, a.shape)
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / max(union, 1))


def crop_patch(image: np.ndarray, mask: np.ndarray, size: int = 256) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    crop = image[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1, :3]
    crop_mask = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    crop_mask = cv2.resize(
        crop_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST
    ) > 0
    return np.where(crop_mask[..., None], crop, 0.0)


def luma_correlation(a: np.ndarray, b: np.ndarray, *, mirror_b: bool = False) -> float:
    ma, mb = foreground_mask(a), foreground_mask(b)
    pa, pb = crop_patch(a, ma), crop_patch(b, mb)
    if pa is None or pb is None:
        return 1.0
    if mirror_b:
        pb = pb[:, ::-1]
    la = cv2.cvtColor(pa.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    lb = cv2.cvtColor(pb.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    valid = (la > 0) | (lb > 0)
    if np.count_nonzero(valid) < 32 or la[valid].std() < 1e-6 or lb[valid].std() < 1e-6:
        return 1.0
    return float(np.corrcoef(la[valid].ravel(), lb[valid].ravel())[0, 1])


def colour_stats(image: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    bgr = image[:, :, :3].astype(np.float32) / 255.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    values = bgr[mask]
    saturation = hsv[:, :, 1][mask]
    return {
        "foreground_fraction": round(float(mask.mean()), 6),
        "saturation_mean": round(float(saturation.mean()) if saturation.size else 0.0, 6),
        "luma_std": round(float(values.mean(axis=1).std()) if values.size else 0.0, 6),
        "foreground_pixels": int(mask.sum()),
    }


def evaluate_output(
    *,
    image_path: Path,
    target_mask_path: Path | None,
    front_reference_path: Path,
    view: str,
    minimum_mask_iou: float,
    minimum_saturation: float,
    maximum_rear_front_correlation: float,
) -> dict[str, Any]:
    image = read_image(image_path)
    mask = foreground_mask(image)
    stats = colour_stats(image, mask)
    reasons: list[str] = []
    if stats["foreground_fraction"] < 0.01 or stats["foreground_fraction"] > 0.95:
        reasons.append("FOREGROUND_MASK_INVALID")
    if float(stats["saturation_mean"]) < minimum_saturation:
        reasons.append("OUTPUT_TOO_DESATURATED")

    iou = None
    if target_mask_path is not None:
        target = read_image(target_mask_path)
        target_mask = target[:, :, 3] > 32 if target.shape[2] == 4 else foreground_mask(target)
        iou = mask_iou(mask, target_mask)
        if iou < minimum_mask_iou:
            reasons.append("SILHOUETTE_MISMATCH")

    direct = mirrored = None
    if view.lower() in {"rear", "back", "rear_yaw180"}:
        front = read_image(front_reference_path)
        direct = luma_correlation(front, image)
        mirrored = luma_correlation(front, image, mirror_b=True)
        if max(direct, mirrored) >= maximum_rear_front_correlation:
            reasons.append("REAR_LOOKS_LIKE_FRONT")

    return {
        "success": not reasons,
        "image": str(image_path),
        "sha256": sha256(image_path),
        "dimensions": [int(image.shape[1]), int(image.shape[0])],
        "view": view,
        "colour": stats,
        "target_mask_iou": None if iou is None else round(iou, 6),
        "front_correlation": None if direct is None else round(direct, 6),
        "mirrored_front_correlation": None if mirrored is None else round(mirrored, 6),
        "reasons": reasons,
    }


def selected_output(receipt: dict[str, Any]) -> Path:
    index = int(receipt.get("selected_attempt", 0)) - 1
    attempts = receipt.get("attempts", [])
    if index < 0 or index >= len(attempts):
        raise RuntimeError("GPU_JOB_SELECTED_ATTEMPT_MISSING")
    outputs = attempts[index].get("outputs", [])
    if len(outputs) != 1:
        raise RuntimeError(f"GPU_JOB_EXPECTED_ONE_OUTPUT:{len(outputs)}")
    return Path(outputs[0]["path"])


def run_job(
    *,
    worker: Path,
    job_config: Path,
    job: dict[str, Any],
    output_dir: Path,
    defaults: dict[str, Any],
) -> tuple[dict[str, Any], Path]:
    receipt_path = output_dir / "gpu_job_receipt.json"
    command = [
        sys.executable,
        str(worker),
        "--config",
        str(job_config),
        "--source",
        str(job["source"]),
        "--output-dir",
        str(output_dir),
        "--view",
        str(job["view"]),
        "--prompt",
        str(job["prompt"]),
        "--negative-prompt",
        str(job.get("negative_prompt", defaults.get("negative_prompt", ""))),
        "--seed",
        str(job.get("seed", defaults.get("seed", 1))),
        "--view-index",
        str(job.get("view_index", defaults.get("view_index", 0))),
        "--resolution",
        str(job.get("resolution", defaults.get("resolution", 512))),
        "--fallback-resolution",
        str(job.get("fallback_resolution", defaults.get("fallback_resolution", 384))),
        "--timeout",
        str(job.get("timeout_seconds", defaults.get("timeout_seconds", 300))),
        "--minimum-free-mb",
        str(job.get("minimum_free_mb", defaults.get("minimum_free_mb", 1200))),
        "--report",
        str(receipt_path),
    ]
    mesh = job.get("mesh", defaults.get("mesh", ""))
    if mesh:
        command.extend(["--mesh", str(mesh)])
    for name in ("depth", "normal", "mask"):
        value = job.get(name)
        if value:
            command.extend([f"--{name}", str(value)])
    result = subprocess.run(command, capture_output=True, text=True)
    if not receipt_path.is_file():
        raise RuntimeError(
            f"GPU_JOB_RECEIPT_MISSING:{job['view']}:rc={result.returncode}:"
            f"{(result.stdout + result.stderr)[-2000:]}"
        )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if result.returncode != 0 or not receipt.get("success"):
        raise RuntimeError(f"GPU_JOB_FAILED:{job['view']}:{json.dumps(receipt)[-2500:]}")
    return receipt, selected_output(receipt)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    output_root = Path(args.output_dir).resolve()
    report_path = Path(args.report) if args.report else output_root / "gpu_texture_sequence_receipt.json"
    worker = Path(__file__).with_name("comfyui_gpu_texture_job.py")
    job_config = Path(manifest["job_config"])
    if not job_config.is_absolute():
        job_config = (root / job_config).resolve()
    defaults = manifest.get("defaults", {})
    qa = manifest.get("qa", {})

    source_front = Path(manifest["front_reference"])
    if not source_front.is_absolute():
        source_front = (root / source_front).resolve()
    if not source_front.is_file():
        raise RuntimeError(f"FRONT_REFERENCE_MISSING:{source_front}")

    receipt: dict[str, Any] = {
        "schema": "lowvram3d_gpu_texture_repair_sequence_v1",
        "success": False,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "front_reference": str(source_front),
        "front_reference_sha256": sha256(source_front),
        "policy": {
            "gpu_jobs_serial": True,
            "atlas_write": False,
            "geometry_or_uv_mutation": False,
            "stop_on_first_rejected_required_view": True,
            "projection_deferred_to_cpu_gate": True,
        },
        "jobs": [],
    }
    output_root.mkdir(parents=True, exist_ok=True)
    completed_outputs: dict[str, Path] = {}

    for index, raw_job in enumerate(manifest.get("jobs", []), start=1):
        job = dict(raw_job)
        source_from = job.get("source_from")
        if source_from:
            if source_from not in completed_outputs:
                raise RuntimeError(f"SOURCE_DEPENDENCY_MISSING:{source_from}")
            job["source"] = str(completed_outputs[source_from])
        for key in ("source", "depth", "normal", "mask"):
            if job.get(key):
                path = Path(job[key])
                job[key] = str(path if path.is_absolute() else (root / path).resolve())
        name = str(job.get("name") or job["view"])
        job_dir = output_root / f"{index:02d}_{name}"
        try:
            child_receipt, output = run_job(
                worker=worker,
                job_config=job_config,
                job=job,
                output_dir=job_dir,
                defaults=defaults,
            )
            assessment = evaluate_output(
                image_path=output,
                target_mask_path=Path(job["mask"]) if job.get("mask") else None,
                front_reference_path=source_front,
                view=str(job["view"]),
                minimum_mask_iou=float(qa.get("minimum_mask_iou", 0.55)),
                minimum_saturation=float(qa.get("minimum_saturation", 0.08)),
                maximum_rear_front_correlation=float(
                    qa.get("maximum_rear_front_correlation", 0.82)
                ),
            )
            stable = output_root / f"{name}.png"
            if assessment["success"]:
                shutil.copy2(output, stable)
                completed_outputs[name] = stable
            receipt["jobs"].append(
                {
                    "name": name,
                    "required": bool(job.get("required", True)),
                    "gpu_receipt": str(job_dir / "gpu_job_receipt.json"),
                    "selected_output": str(output),
                    "stable_output": str(stable) if assessment["success"] else None,
                    "assessment": assessment,
                    "child_policy": child_receipt.get("policy", {}),
                }
            )
            if not assessment["success"] and bool(job.get("required", True)):
                break
        except Exception as exc:
            receipt["jobs"].append(
                {
                    "name": name,
                    "required": bool(job.get("required", True)),
                    "assessment": {"success": False, "reasons": [str(exc)]},
                }
            )
            if bool(job.get("required", True)):
                break

    required = [job for job in receipt["jobs"] if job.get("required")]
    receipt["success"] = bool(required) and all(
        job.get("assessment", {}).get("success") for job in required
    )
    receipt["classification"] = (
        "GPU_TEXTURE_REFERENCES=PROVEN" if receipt["success"] else "GPU_TEXTURE_REFERENCES=REJECTED"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
    return 0 if receipt["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
