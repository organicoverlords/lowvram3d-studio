"""Run the repository's real PNG -> generated geometry -> game-ready pipeline.

This is intentionally separate from ``highres_geometry_ladder.py``. That worker starts from an
existing GLB and is only the post-generation geometry-quality phase. This entrypoint calls
``PipelineEngine.full`` and therefore starts from the image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lowvram3d.config import PipelineConfig
from lowvram3d.pipeline import PipelineEngine


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_if_present(source: str | Path | None, destination: Path) -> str | None:
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_file():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination)
    return str(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-image-sha256", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--asset-type", default="character")
    parser.add_argument("--quality", choices=("background", "gameplay", "hero"), default="hero")
    parser.add_argument("--texture-resolution", type=int, default=2048)
    parser.add_argument("--animation-preset", default="idle")
    parser.add_argument("--no-background-removal", action="store_true")
    args = parser.parse_args()

    image = Path(args.image).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "run_result.json"

    if not image.is_file():
        raise FileNotFoundError(image)
    image_hash = sha256(image)
    expected_hash = args.expected_image_sha256.strip().lower()
    if expected_hash and image_hash.lower() != expected_hash:
        raise RuntimeError(
            f"Input image hash mismatch: expected {expected_hash}, got {image_hash}. "
            "Refusing to run a different image under the shaman benchmark name."
        )
    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    config = PipelineConfig.load(config_path)
    engine = PipelineEngine(config)
    receipt, job_dir = engine.new_job("full", {"image": str(image)})
    receipt.parameters["worker_invocation"] = {
        "entrypoint": "workers/run_full_image.py",
        "repository_head": "unknown",
        "expected_image_sha256": expected_hash or None,
        "image_sha256": image_hash,
    }
    engine._write_receipt(receipt, job_dir)

    payload: dict = {
        "success": False,
        "classification": "NOT_PROVEN",
        "image": str(image),
        "image_sha256": image_hash,
        "config": str(config_path),
        "job_id": receipt.job_id,
        "job_dir": str(job_dir),
        "asset_type": args.asset_type,
        "quality": args.quality,
        "texture_resolution": args.texture_resolution,
        "prompt": args.prompt,
    }

    try:
        final, receipt, job_dir = engine.full(
            image,
            args.prompt,
            asset_type=args.asset_type,
            quality=args.quality,
            separate_movable_parts=True,
            texture_resolution=args.texture_resolution,
            lod_enabled=True,
            remove_hidden_geometry=False,
            experimental_semantic_split=False,
            background_removal=not args.no_background_removal,
            animation_preset=args.animation_preset,
            receipt=receipt,
            job_dir=job_dir,
        )
        if not final.is_file() or final.stat().st_size <= 0:
            raise RuntimeError("Pipeline returned success without a non-empty final GLB")

        delivery = output / "deliverable"
        delivery.mkdir(parents=True, exist_ok=True)
        copied: dict[str, str] = {}
        copied["asset_glb"] = str(shutil.copy2(final, delivery / "shaman_game_ready.glb"))
        for logical_name, filename in (
            ("high_glb", "shaman_high.glb"),
            ("preview_png", "shaman_preview.png"),
            ("validation_json", "validation-report.json"),
            ("pipeline_report_json", "pipeline-report.json"),
            ("rig_report", "rig-report.json"),
        ):
            copied_path = copy_if_present(receipt.outputs.get(logical_name), delivery / filename)
            if copied_path:
                copied[logical_name] = copied_path
        copied_raw = copy_if_present(receipt.outputs.get("raw_mesh"), delivery / "generated_raw.glb")
        if copied_raw:
            copied["raw_mesh"] = copied_raw
        receipt_copy = copy_if_present(
            job_dir / "proof" / "job_receipt.json",
            delivery / "job_receipt.json",
        )
        if receipt_copy:
            copied["job_receipt_json"] = receipt_copy

        payload.update(
            {
                "success": True,
                "classification": "PIPELINE_PASSED_REQUIRES_VISUAL_REVIEW",
                "final": str(final),
                "final_sha256": sha256(final),
                "receipt_status": receipt.status,
                "selected_lane": receipt.selected_lane,
                "outputs": receipt.outputs,
                "copied_outputs": copied,
                "manual_visual_validation_required": True,
            }
        )
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            "FULL_IMAGE_TO_3D_PASSED "
            f"job_id={receipt.job_id} lane={receipt.selected_lane} final={final} result={result_path}",
            flush=True,
        )
        return 0
    except Exception as exc:
        receipt_path = job_dir / "proof" / "job_receipt.json"
        payload.update(
            {
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "receipt": str(receipt_path),
                "receipt_present": receipt_path.is_file(),
            }
        )
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            f"FULL_IMAGE_TO_3D_FAILED job_id={receipt.job_id} error={exc} result={result_path}",
            file=sys.stderr,
            flush=True,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
