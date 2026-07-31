"""Atomic CPU raster-texture route with geometry repair and fresh-import validation."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import JobReceipt, StageReceipt, now_ms
from .runner import StageFailure, artifact_is_valid

if TYPE_CHECKING:
    from .pipeline import PipelineEngine


def run_raster_texture_route(
    engine: "PipelineEngine",
    mesh: Path,
    views: Path,
    receipt: JobReceipt,
    job_dir: Path,
) -> tuple[Path, Path]:
    """Create and validate a candidate before promoting public texture artifacts."""
    projection = job_dir / "textured" / "projection"
    candidate = projection / "candidate_v2"
    candidate.mkdir(parents=True, exist_ok=True)
    cleaned_glb = candidate / "mesh_clean.glb"
    npz = candidate / "mesh_clean.npz"
    cleanup_report = candidate / "geometry_cleanup_report.json"

    profile = receipt.parameters.get("profile", {})
    asset_type = str(
        receipt.parameters.get("asset_type") or profile.get("asset_type") or ""
    ).lower()
    separate_parts = bool(receipt.parameters.get("separate_movable_parts", True))
    preserve_continuous = bool(profile.get("preserve_continuous_body", False))
    cleanup_mode = (
        "single_subject_strict"
        if asset_type in {"avatar", "character", "creature"}
        and preserve_continuous
        and not separate_parts
        else "conservative"
    )
    engine._blender_stage(
        "geometry_repair_extract_v2",
        "raster_cleanup_extract.py",
        [
            "--input", mesh,
            "--output-glb", cleaned_glb,
            "--output-npz", npz,
            "--report", cleanup_report,
            "--cleanup-mode", cleanup_mode,
        ],
        {"mesh": cleaned_glb, "npz": npz, "report": cleanup_report},
        receipt,
        job_dir,
    )

    view_metadata = views / "view_metadata.json"
    if not artifact_is_valid(view_metadata):
        failure = StageReceipt(
            "raster_project_v2",
            "failed",
            now_ms(),
            finished_at=now_ms(),
            error="view_metadata.json missing -- projection views are incomplete",
        )
        raise StageFailure(failure.error or "raster route unavailable", failure)

    progress = candidate / "raster-progress.json"
    project_report = candidate / "raster_report.json"
    atlas = candidate / "basecolor.png"
    # The production run has now validated 2048 end-to-end.  The previous hard 1024 cap silently
    # discarded the profile's requested quality even when config.texture_size was 2048.
    atlas_size = min(max(int(engine.config.texture_size), 512), 2048)
    engine._command_stage(
        "raster_project_v2",
        [
            engine.python,
            str(engine.package_root / "workers" / "raster_project.py"),
            "--npz", str(npz),
            "--views-dir", str(views),
            "--view-metadata", str(view_metadata),
            "--output-dir", str(candidate),
            "--atlas-size", str(atlas_size),
            "--progress", str(progress),
            "--report", str(project_report),
        ],
        {"basecolor": atlas, "report": project_report},
        receipt,
        job_dir,
    )

    candidate_glb = candidate / "textured_candidate.glb"
    candidate_texture = candidate / "embedded_basecolor.png"
    export_report = candidate / "raster_export_report.json"
    engine._blender_stage(
        "raster_export_candidate_v2",
        "raster_export.py",
        [
            "--cleaned-mesh", cleaned_glb,
            "--atlas", atlas,
            "--output", candidate_glb,
            "--texture", candidate_texture,
            "--report", export_report,
        ],
        {"mesh": candidate_glb, "basecolor": candidate_texture, "report": export_report},
        receipt,
        job_dir,
    )

    validation_report = candidate / "geometry_validation.json"
    engine._blender_stage(
        "geometry_quality_validate_v2",
        "geometry_quality_validate.py",
        [
            "--input", candidate_glb,
            "--cleanup-report", cleanup_report,
            "--report", validation_report,
        ],
        {"report": validation_report},
        receipt,
        job_dir,
    )

    output = projection / "mesh.glb"
    texture = projection / "basecolor.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in ((candidate_glb, output), (atlas, texture)):
        temporary = destination.with_name(destination.name + ".promoting")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    (projection / "promotion_report.json").write_text(
        json.dumps(
            {
                "success": True,
                "candidate_glb": str(candidate_glb),
                "candidate_texture": str(atlas),
                "output_glb": str(output),
                "output_texture": str(texture),
                "atlas_size": atlas_size,
                "cleanup_mode": cleanup_mode,
                "geometry_validation": str(validation_report),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output, texture
