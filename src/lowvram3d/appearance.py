from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .contracts import JobReceipt, StageReceipt, now_ms
from .runner import StageFailure, artifact_is_valid

if TYPE_CHECKING:
    from .pipeline import PipelineEngine

VIEW_NAMES = ("front", "right", "back", "left", "top", "bottom")


def _stage_reusable(receipt: JobReceipt, stage: str, artifacts: dict[str, Path]) -> bool:
    previous = any(item.stage == stage and item.status in {"passed", "reused"} for item in receipt.stages)
    return previous and all(artifact_is_valid(path) for path in artifacts.values())


def _reuse_receipt(
    engine: "PipelineEngine",
    stage: str,
    artifacts: dict[str, Path],
    receipt: JobReceipt,
    job_dir: Path,
) -> None:
    receipt.stages.append(StageReceipt(
        stage=stage,
        status="reused",
        started_at=now_ms(),
        finished_at=now_ms(),
        artifacts={name: str(path) for name, path in artifacts.items()},
        notes=["Reused validated appearance-stage outputs from the existing job directory."],
    ))
    engine._write_receipt(receipt, job_dir)


def reusable_command(
    engine: "PipelineEngine",
    stage: str,
    command: list[str],
    artifacts: dict[str, Path],
    receipt: JobReceipt,
    job_dir: Path,
    timeout: int,
) -> None:
    if _stage_reusable(receipt, stage, artifacts):
        _reuse_receipt(engine, stage, artifacts, receipt, job_dir)
        return
    engine._command_stage(stage, command, artifacts, receipt, job_dir, timeout=timeout)


def prepare_projection_views(
    engine: "PipelineEngine",
    source_image: Path | None,
    prompt: str,
    receipt: JobReceipt,
    job_dir: Path,
) -> Path:
    views = job_dir / "views" / "projection"
    command = [
        engine.python,
        str(engine.package_root / "workers" / "make_fallback_views.py"),
        "--prompt", prompt,
        "--output-dir", str(views),
        "--size", "512",
    ]
    if source_image:
        command += ["--source-image", str(source_image)]
    artifacts = {name: views / f"{name}.png" for name in VIEW_NAMES}
    artifacts.update({
        "contact_sheet": views / "contact_sheet.png",
        "receipt": views / "worker_receipt.json",
        "view_metadata": views / "view_metadata.json",
    })
    reusable_command(engine, "prepare_projection_views", command, artifacts, receipt, job_dir, 600)
    return views


def texture_mv_adapter(
    engine: "PipelineEngine",
    mesh: Path,
    prompt: str,
    source_front: Path | None,
    receipt: JobReceipt,
    job_dir: Path,
    texture_size: int,
    padding_px: int,
) -> tuple[Path, Path]:
    controls = job_dir / "controls"
    metadata = controls / "cameras.json"
    control_artifacts = {"metadata": metadata}
    for name in VIEW_NAMES:
        control_artifacts[f"{name}_position"] = controls / f"{name}_position.png"
        control_artifacts[f"{name}_normal"] = controls / f"{name}_normal.png"
    engine._reusable_blender_stage(
        "render_controls",
        "render_controls.py",
        ["--input", mesh, "--output-dir", controls, "--metadata", metadata, "--size", 512],
        control_artifacts,
        receipt,
        job_dir,
    )

    mv_python = Path(engine.config.mv_adapter_python)
    mv_root = Path(engine.config.mv_adapter_root)
    if not mv_python.is_file() or not mv_root.is_dir():
        failure = StageReceipt(
            "mv_adapter", "failed", now_ms(), finished_at=now_ms(),
            error="MV-Adapter environment is not installed or failed readiness verification",
            failure_class="missing_dependency",
        )
        engine._record_stage_failure(receipt, job_dir, failure)
        raise StageFailure(failure.error or "MV-Adapter unavailable", failure)

    views = job_dir / "views" / "mv_adapter"
    command = [
        str(mv_python),
        str(engine.package_root / "workers" / "mv_adapter_from_controls.py"),
        "--repo", str(mv_root),
        "--controls-dir", str(controls),
        "--output-dir", str(views),
        "--prompt", prompt or "coherent realistic game asset materials",
        "--base-model", engine.config.sd21_base_model,
    ]
    if engine.config.models_offline:
        command.append("--offline")
    view_artifacts = {name: views / f"{name}.png" for name in VIEW_NAMES}
    view_artifacts.update({
        "contact_sheet": views / "contact_sheet.png",
        "receipt": views / "worker_receipt.json",
    })
    reusable_command(engine, "mv_adapter", command, view_artifacts, receipt, job_dir, 5400)

    output_dir = job_dir / "textured" / "mv_adapter"
    output = output_dir / "mesh.glb"
    texture = output_dir / "basecolor.png"
    report = output_dir / "projection_report.json"
    args: list[object] = [
        "--input", mesh,
        "--views-dir", views,
        "--output", output,
        "--texture", texture,
        "--report", report,
        "--size", texture_size,
        "--padding-px", padding_px,
    ]
    if source_front and source_front.is_file():
        args += ["--source-image", source_front]
    engine._reusable_blender_stage(
        "project_mv_views",
        "project_texture.py",
        args,
        {"mesh": output, "basecolor": texture, "report": report},
        receipt,
        job_dir,
    )
    return output, texture


def texture_projection(
    engine: "PipelineEngine",
    mesh: Path,
    views: Path,
    source_front: Path | None,
    receipt: JobReceipt,
    job_dir: Path,
    texture_size: int,
    padding_px: int,
) -> tuple[Path, Path]:
    del source_front, texture_size, padding_px  # raster route reads the prepared views and config
    if engine.config.use_raster_texture_route:
        return engine._texture_projection_raster(mesh, views, receipt, job_dir)

    output_dir = job_dir / "textured" / "projection"
    output = output_dir / "mesh.glb"
    texture = output_dir / "basecolor.png"
    report = output_dir / "projection_report.json"
    engine._reusable_blender_stage(
        "project_fallback_views_cycles_diagnostic",
        "project_texture.py",
        [
            "--input", mesh, "--views-dir", views, "--output", output,
            "--texture", texture, "--report", report,
            "--size", engine.config.texture_size, "--padding-px", 4,
        ],
        {"mesh": output, "basecolor": texture, "report": report},
        receipt,
        job_dir,
    )
    return output, texture


def resolve_source_appearance(
    engine: "PipelineEngine",
    mesh: Path,
    source_image: Path | None,
    prompt: str,
    receipt: JobReceipt,
    job_dir: Path,
    texture_size: int,
    padding_px: int,
) -> tuple[Path, Path, str]:
    views = prepare_projection_views(engine, source_image, prompt, receipt, job_dir)
    source_front = views / "front.png"
    recorded = str(receipt.parameters.get("texture_lane", "")).upper()

    if recorded == "A":
        appearance, basecolor = texture_mv_adapter(
            engine, mesh, prompt, source_front, receipt, job_dir, texture_size, padding_px,
        )
        return appearance, basecolor, "A"
    if recorded == "B":
        appearance, basecolor = texture_projection(
            engine, mesh, views, source_front, receipt, job_dir, texture_size, padding_px,
        )
        return appearance, basecolor, "B"

    last_error = "No source-appearance lane completed"
    if "A" in engine.config.lane_order:
        try:
            appearance, basecolor = texture_mv_adapter(
                engine, mesh, prompt, source_front, receipt, job_dir, texture_size, padding_px,
            )
            receipt.parameters["texture_lane"] = "A"
            receipt.outputs["appearance_mesh"] = str(appearance)
            receipt.outputs["projected_basecolor"] = str(basecolor)
            engine._write_receipt(receipt, job_dir)
            return appearance, basecolor, "A"
        except StageFailure as exc:
            last_error = exc.receipt.error or "MV-Adapter appearance lane failed"
            receipt.status = "running"
            receipt.error = None
            receipt.parameters["texture_lane_a_failure"] = last_error
            engine._write_receipt(receipt, job_dir)

    try:
        appearance, basecolor = texture_projection(
            engine, mesh, views, source_front, receipt, job_dir, texture_size, padding_px,
        )
    except StageFailure as exc:
        last_error = exc.receipt.error or "Deterministic projection lane failed"
        receipt.status = "failed"
        receipt.error = last_error
        engine._write_receipt(receipt, job_dir)
        raise

    receipt.parameters["texture_lane"] = "B"
    receipt.outputs["appearance_mesh"] = str(appearance)
    receipt.outputs["projected_basecolor"] = str(basecolor)
    receipt.parameters["texture_lane_b_limitations"] = [
        "Unseen surfaces are deterministic mirrored/palette-preserving approximations.",
        "The source-facing view remains authoritative; no semantic claim is made for hidden details.",
    ]
    engine._write_receipt(receipt, job_dir)
    return appearance, basecolor, "B"
