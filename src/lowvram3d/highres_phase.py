"""Production adapter for the clean-master and measured high-to-low geometry phase."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import JobReceipt, StageReceipt, now_ms
from .runner import artifact_is_valid

if TYPE_CHECKING:
    from .pipeline import PipelineEngine
    from .presets import AssetProfile


@dataclass(frozen=True, slots=True)
class HighresGeometryResult:
    clean_master: Path
    selected_mesh: Path
    lod1: Path | None
    lod2: Path | None
    report: Path
    selected_faces: int
    master_faces: int
    selection_policy: str


def _reusable(receipt: JobReceipt, stage: str, artifacts: dict[str, Path]) -> bool:
    prior = any(item.stage == stage and item.status in {"passed", "reused"} for item in receipt.stages)
    return prior and all(artifact_is_valid(path) for path in artifacts.values())


def run_highres_geometry_phase(
    engine: "PipelineEngine",
    *,
    high_glb: Path,
    profile: "AssetProfile",
    quality: str,
    prompt: str,
    source_reference: Path | None,
    receipt: JobReceipt,
    job_dir: Path,
) -> HighresGeometryResult:
    root = job_dir / "highres_geometry"
    clean_master = root / "geometry" / "clean_master.glb"
    selected_mesh = root / "geometry" / "game_ready_untextured.glb"
    lod1 = root / "geometry" / "lod1.glb"
    lod2 = root / "geometry" / "lod2.glb"
    report = root / "reports" / "highres_geometry_ladder.json"
    required = {"clean_master": clean_master, "selected_mesh": selected_mesh, "report": report}
    if profile.lod_count > 0:
        required["lod1"] = lod1
    if profile.lod_count > 1:
        required["lod2"] = lod2

    stage = "highres_geometry_ladder"
    if _reusable(receipt, stage, required):
        receipt.stages.append(
            StageReceipt(
                stage=stage,
                status="reused",
                started_at=now_ms(),
                finished_at=now_ms(),
                artifacts={name: str(path) for name, path in required.items()},
                notes=["Reused validated clean-master and measured LOD outputs."],
            )
        )
        engine._write_receipt(receipt, job_dir)
    else:
        command = [
            engine.python,
            str(engine.package_root / "workers" / "highres_geometry_ladder.py"),
            "--input", str(high_glb),
            "--output-dir", str(root),
            "--blender", str(engine.config.blender_path),
            "--asset-type", profile.asset_type.value,
            "--quality", quality,
            "--prompt", prompt,
            "--samples", "200000",
            "--silhouette-size", "384",
            "--max-candidates", "7",
        ]
        if source_reference:
            command += ["--source-image", str(source_reference)]
        engine._command_stage(stage, command, required, receipt, job_dir, timeout=28_800)

    payload = json.loads(report.read_text(encoding="utf-8"))
    if not payload.get("success"):
        raise RuntimeError("high-resolution geometry ladder did not pass")
    selected = payload.get("selected") or {}
    return HighresGeometryResult(
        clean_master=clean_master,
        selected_mesh=selected_mesh,
        lod1=lod1 if lod1.is_file() else None,
        lod2=lod2 if lod2.is_file() else None,
        report=report,
        selected_faces=int(selected.get("face_count", payload.get("master_faces", 0))),
        master_faces=int(payload.get("master_faces", 0)),
        selection_policy=str(payload.get("selection_policy", "unknown")),
    )
