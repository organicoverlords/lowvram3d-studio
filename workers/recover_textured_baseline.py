"""Recover a first textured GLB from an existing shaman pipeline job.

This intentionally stops at the validated raster-textured asset. It reuses the job's existing
UV mesh and completed MV-Adapter views, writes the metadata contract missing from older jobs,
and calls the repository's production ``run_raster_texture_route``. Geometry generation,
MV inference, map baking, splitting and rigging are not repeated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lowvram3d.config import PipelineConfig
from lowvram3d.contracts import JobReceipt, StageReceipt, now_ms
from lowvram3d.pipeline import PipelineEngine
from lowvram3d.raster_route import run_raster_texture_route

REQUIRED_VIEWS = ("front", "right", "back", "left")
OPTIONAL_VIEWS = ("top", "bottom")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing {label}: {path}")
    return path


def write_mv_metadata(views_dir: Path) -> Path:
    available: list[str] = []
    for name in (*REQUIRED_VIEWS, *OPTIONAL_VIEWS):
        image = views_dir / f"{name}.png"
        if image.is_file() and image.stat().st_size > 0:
            available.append(name)
    missing = [name for name in REQUIRED_VIEWS if name not in available]
    if missing:
        raise RuntimeError(
            "MV-Adapter view set is incomplete; missing " + ", ".join(missing)
        )

    metadata = {
        "version": 1,
        "backend": "mv_adapter_completed_view_recovery",
        "views": [
            {
                "view": name,
                "source_type": "generated",
                "confidence": 0.85,
                "image": f"{name}.png",
            }
            for name in available
        ],
        "policy": {
            "semantic_projection": ["real", "generated"],
            "barred_from_semantic_projection": ["mirrored", "synthetic"],
        },
    }
    path = views_dir / "view_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def copy_optional(source: Path, destination: Path) -> str | None:
    if not source.is_file() or source.stat().st_size <= 0:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--atlas-size", type=int, default=512)
    args = parser.parse_args()

    started = time.time()
    config = PipelineConfig.load(args.config)
    config.texture_size = min(max(int(args.atlas_size), 512), 1024)
    engine = PipelineEngine(config)

    job_dir = config.jobs_root / args.job_id
    receipt_path = require_file(job_dir / "proof" / "job_receipt.json", "job receipt")
    mesh = require_file(job_dir / "uv" / "game_ready_uv.glb", "UV mesh")
    views_dir = job_dir / "views" / "mv_adapter"
    if not views_dir.is_dir():
        raise RuntimeError(f"Missing MV-Adapter views directory: {views_dir}")
    metadata_path = write_mv_metadata(views_dir)

    receipt = JobReceipt.load(receipt_path)
    receipt.status = "running"
    receipt.error = None
    receipt.finished_at = None
    receipt.parameters["textured_baseline_recovery"] = {
        "source_job_id": args.job_id,
        "source_uv_mesh": str(mesh),
        "source_views": str(views_dir),
        "atlas_size": config.texture_size,
        "scope": "validated textured GLB only; no map bake, split, rig or final export",
    }
    engine._write_receipt(receipt, job_dir)

    output_glb, output_texture = run_raster_texture_route(
        engine,
        mesh,
        views_dir,
        receipt,
        job_dir,
    )
    require_file(output_glb, "textured GLB")
    require_file(output_texture, "base-color texture")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    delivered_glb = output_dir / "shaman_textured_baseline.glb"
    delivered_texture = output_dir / "shaman_textured_baseline_basecolor.png"
    shutil.copy2(output_glb, delivered_glb)
    shutil.copy2(output_texture, delivered_texture)

    candidate = job_dir / "textured" / "projection" / "candidate_v2"
    copied_reports: dict[str, str] = {}
    for source_name, destination_name in (
        ("geometry_cleanup_report.json", "geometry-cleanup-report.json"),
        ("raster_report.json", "raster-report.json"),
        ("raster_export_report.json", "raster-export-report.json"),
        ("geometry_validation.json", "geometry-validation.json"),
    ):
        copied = copy_optional(candidate / source_name, output_dir / destination_name)
        if copied:
            copied_reports[source_name] = copied
    copy_optional(metadata_path, output_dir / "view_metadata.json")

    finished = now_ms()
    receipt.stages.append(
        StageReceipt(
            stage="textured_baseline_recovery",
            status="passed",
            started_at=finished - int((time.time() - started) * 1000),
            finished_at=finished,
            artifacts={
                "mesh": str(output_glb),
                "basecolor": str(output_texture),
                "delivered_mesh": str(delivered_glb),
            },
            notes=[
                "Recovered from existing UV mesh and completed MV-Adapter views.",
                "Full game-ready pipeline remains incomplete; this is the requested visual baseline.",
            ],
        )
    )
    receipt.outputs["textured_baseline_glb"] = str(output_glb)
    receipt.outputs["textured_baseline_basecolor"] = str(output_texture)
    receipt.status = "running"
    receipt.error = None
    engine._write_receipt(receipt, job_dir)
    copy_optional(receipt_path, output_dir / "job_receipt.json")

    result = {
        "success": True,
        "classification": "TEXTURED_BASELINE_PASSED_FULL_PIPELINE_NOT_PROVEN",
        "job_id": args.job_id,
        "job_dir": str(job_dir),
        "source_uv_mesh": str(mesh),
        "source_uv_mesh_sha256": sha256(mesh),
        "source_views": str(views_dir),
        "view_metadata": str(metadata_path),
        "atlas_size": config.texture_size,
        "delivered_glb": str(delivered_glb),
        "delivered_glb_sha256": sha256(delivered_glb),
        "delivered_glb_bytes": delivered_glb.stat().st_size,
        "delivered_texture": str(delivered_texture),
        "reports": copied_reports,
        "duration_seconds": round(time.time() - started, 2),
    }
    result_path = output_dir / "baseline_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "TEXTURED_BASELINE_PASSED "
        f"job_id={args.job_id} glb={delivered_glb} result={result_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
