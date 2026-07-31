from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import JobReceipt, StageReceipt, now_ms
from .runner import artifact_is_valid

if TYPE_CHECKING:
    from .pipeline import PipelineEngine


def preprocess_subject(
    engine: "PipelineEngine",
    image: Path,
    receipt: JobReceipt,
    job_dir: Path,
    *,
    strict_avatar: bool = False,
) -> Path:
    output = job_dir / "preprocess" / "subject.png"
    mask = job_dir / "preprocess" / "subject_mask.png"
    preview = job_dir / "preprocess" / "mask_preview.png"
    report = job_dir / "preprocess" / "avatar_report.json"
    artifacts = {"image": output, "mask": mask, "preview": preview, "report": report}
    previous_ok = any(
        stage.stage == "subject_preprocess" and stage.status in {"passed", "reused"}
        for stage in receipt.stages
    )
    if previous_ok and all(artifact_is_valid(path) for path in artifacts.values()):
        receipt.stages.append(StageReceipt(
            "subject_preprocess",
            "reused",
            now_ms(),
            finished_at=now_ms(),
            artifacts={name: str(path) for name, path in artifacts.items()},
            notes=["Reused validated high-resolution subject mask and pose analysis."],
        ))
        engine._write_receipt(receipt, job_dir)
        return output

    worker_python = Path(engine.config.mv_adapter_python)
    if not worker_python.is_file():
        worker_python = Path(engine.python)
    command = [
        str(worker_python),
        str(engine.package_root / "workers" / "avatar_preprocess.py"),
        "--input", str(image),
        "--output", str(output),
        "--mask", str(mask),
        "--preview", str(preview),
        "--report", str(report),
        "--canvas-size", "1024",
        "--max-input-size", "3072",
    ]
    if engine.config.models_offline:
        command.append("--offline")
    engine._command_stage("subject_preprocess", command, artifacts, receipt, job_dir, timeout=1800)
    data = json.loads(report.read_text(encoding="utf-8-sig"))
    framing = data.get("framing", {})
    receipt.parameters["subject_preprocess"] = {
        "backend": data.get("backend"),
        "model": data.get("model"),
        "model_revision": data.get("model_revision"),
        "framing": framing,
        "normalization": data.get("normalization", {}),
        "edge_treatment": data.get("edge_treatment", {}),
    }
    receipt.outputs.update({
        "subject_mask": str(mask),
        "subject_mask_preview": str(preview),
        "avatar_report": str(report),
    })
    if strict_avatar:
        receipt.parameters["avatar_input_quality"] = "ready" if framing.get("ready", False) else "degraded"
        receipt.parameters["avatar_rig_input_ready"] = bool(framing.get("rig_ready", False))
        if not framing.get("ready", False):
            receipt.parameters["avatar_input_warnings"] = framing.get("warnings", [])
    engine._write_receipt(receipt, job_dir)
    return output
