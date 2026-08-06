"""Atomic CPU raster-texture route with geometry repair and fresh-import validation."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import JobReceipt, StageReceipt, now_ms
from .runner import StageFailure, artifact_is_valid
from .anchor_provenance import (
    AnchorProvenanceError,
    geometry_sha256_from_glb,
    load_anchor_provenance,
)

if TYPE_CHECKING:
    from .pipeline import PipelineEngine


def verified_cleanup_geometry_hash(cleanup_report: dict, expected_hash: str) -> str:
    """Return cleanup's verified output hash or reject a mutating cleanup."""
    provenance = cleanup_report.get("provenance") or {}
    output_hash = provenance.get("output_geometry_sha256")
    if (
        not cleanup_report.get("success", False)
        or provenance.get("input_geometry_sha256") != expected_hash
        or output_hash != expected_hash
    ):
        raise ValueError("cleanup geometry hash mismatch; production promotion refused")
    return str(output_hash)


def run_raster_texture_route(
    engine: "PipelineEngine",
    mesh: Path,
    views: Path,
    receipt: JobReceipt,
    job_dir: Path,
) -> tuple[Path, Path]:
    """Create and validate a candidate before promoting public texture artifacts."""
    anchor_value = receipt.parameters.get("anchor_receipt") or receipt.outputs.get("anchor_receipt")
    if isinstance(anchor_value, dict):
        anchor_value = anchor_value.get("path")
    anchor_path = Path(anchor_value) if anchor_value else None
    expected_source = receipt.parameters.get("anchor_source_mesh_sha256")
    provenance_required = bool(
        receipt.parameters.get("require_anchor_provenance")
        or receipt.parameters.get("provenance_required")
        or receipt.parameters.get("pipeline_version") == 2
    )
    provenance_verified = False
    anchor_hash = None
    anchor_ids: list[str] = []
    if anchor_path:
        try:
            _anchor_receipt, anchor_hash, anchor_ids = load_anchor_provenance(
                anchor_path, expected_source_sha256=expected_source
            )
            provenance_verified = True
        except AnchorProvenanceError as exc:
            failure = StageReceipt("raster_provenance", "failed", now_ms(), finished_at=now_ms(),
                                   error=exc.detail, failure_class=exc.code)
            raise StageFailure(exc.detail, failure) from exc
    elif provenance_required:
        failure = StageReceipt("raster_provenance", "failed", now_ms(), finished_at=now_ms(),
                               error="anchor receipt is required for this pipeline", failure_class="ANCHOR_RECEIPT_MISSING")
        raise StageFailure(failure.error or "anchor receipt required", failure)
    geometry_provenance_required = provenance_required or provenance_verified
    accepted_geometry_hash = (
        geometry_sha256_from_glb(mesh) if geometry_provenance_required else None
    )
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
            *( ["--expected-input-geometry-sha256", accepted_geometry_hash] if accepted_geometry_hash else [] ),
            *( ["--anchor-receipt", anchor_path, "--expected-source-sha256", expected_source or ""] if provenance_verified else [] ),
            *( ["--require-anchor-provenance"] if provenance_required else [] ),
        ],
        {"mesh": cleaned_glb, "npz": npz, "report": cleanup_report},
        receipt,
        job_dir,
    )

    verified_geometry_hash = None
    if accepted_geometry_hash:
        cleanup_data = json.loads(cleanup_report.read_text(encoding="utf-8"))
        try:
            verified_geometry_hash = verified_cleanup_geometry_hash(cleanup_data, accepted_geometry_hash)
        except ValueError as exc:
            detail = str(exc)
            failure = StageReceipt(
                "raster_provenance", "failed", now_ms(), finished_at=now_ms(),
                error=detail, failure_class="GEOMETRY_MUTATION",
            )
            raise StageFailure(detail, failure)

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
            *( ["--expected-input-geometry-sha256", verified_geometry_hash] if verified_geometry_hash else [] ),
            *( ["--anchor-receipt", str(anchor_path), "--expected-source-sha256", expected_source or ""] if provenance_verified else [] ),
            *( ["--require-anchor-provenance"] if provenance_required else [] ),
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
            *( ["--expected-input-geometry-sha256", verified_geometry_hash] if verified_geometry_hash else [] ),
            *( ["--anchor-receipt", str(anchor_path), "--expected-source-sha256", expected_source or ""] if provenance_verified else [] ),
            *( ["--require-anchor-provenance"] if provenance_required else [] ),
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
                "anchor_receipt_sha256": anchor_hash,
                "anchor_ids": anchor_ids,
                "input_geometry_sha256": accepted_geometry_hash,
                "verified_geometry_sha256": verified_geometry_hash,
                "provenance_verified": provenance_verified,
                "geometry_provenance_verified": bool(verified_geometry_hash),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output, texture
