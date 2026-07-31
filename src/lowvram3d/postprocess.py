from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from .appearance import resolve_source_appearance
from .contracts import JobReceipt, StageReceipt, now_ms
from .highres_phase import run_highres_geometry_phase
from .presets import AssetProfile, get_profile
from .runner import artifact_is_valid
from .resume import build_contract, canonical_hash, sha256_file, store_or_validate_contract

if TYPE_CHECKING:
    from .pipeline import PipelineEngine


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _stage_reusable(receipt: JobReceipt, stage: str, artifacts: dict[str, Path]) -> bool:
    previous = any(item.stage == stage and item.status in {"passed", "reused"} for item in receipt.stages)
    return previous and all(artifact_is_valid(path) for path in artifacts.values())


def _reusable_command(
    engine: "PipelineEngine",
    stage: str,
    command: list[str],
    artifacts: dict[str, Path],
    receipt: JobReceipt,
    job_dir: Path,
    timeout: int,
) -> None:
    if _stage_reusable(receipt, stage, artifacts):
        receipt.stages.append(
            StageReceipt(
                stage=stage,
                status="reused",
                started_at=now_ms(),
                finished_at=now_ms(),
                artifacts={name: str(path) for name, path in artifacts.items()},
                notes=["Reused validated stage outputs from the existing job directory."],
            )
        )
        engine._write_receipt(receipt, job_dir)
        return
    engine._command_stage(stage, command, artifacts, receipt, job_dir, timeout=timeout)


def _blender_uv(
    engine: "PipelineEngine",
    profile: AssetProfile,
    source: Path,
    output: Path,
    report: Path,
    layout: Path,
    receipt: JobReceipt,
    job_dir: Path,
    stage: str,
) -> None:
    uv = profile.uv_options
    args: list[object] = [
        "--input", source,
        "--output", output,
        "--report", report,
        "--layout", layout,
        "--texture-size", profile.texture_size,
        "--padding-px", profile.uv_padding_px,
        "--atlas-mode", profile.atlas_mode,
        "--smart-angle-deg", float(uv.get("max_cone_deg", 66.0)),
        "--area-weight", 0.15 if profile.asset_type.value in {"character", "creature"} else 0.0,
    ]
    if profile.lightmap_uv:
        args.append("--lightmap-uv")
    engine._reusable_blender_stage(
        stage,
        "unwrap_uv.py",
        args,
        {"mesh": output, "report": report, "layout": layout},
        receipt,
        job_dir,
    )


def _copy_source_contract(
    mesh: Path,
    source_image: Path | None,
    receipt: JobReceipt,
    job_dir: Path,
    contract: dict,
) -> tuple[Path, Path | None]:
    existing_sources = sorted((job_dir / "source").glob("original.*"))
    source_original = existing_sources[0] if existing_sources else job_dir / "source" / f"original{mesh.suffix.lower()}"
    if source_original.is_file():
        if sha256_file(source_original) != contract["source"]["sha256"]:
            raise RuntimeError(
                "Resume source mismatch: the saved original mesh differs from the requested mesh. "
                "Start a new job rather than overwriting completed stage inputs."
            )
    else:
        shutil.copy2(mesh, source_original)

    source_reference: Path | None = None
    if source_image:
        if not source_image.is_file():
            raise RuntimeError(f"Source image does not exist: {source_image}")
        source_reference = job_dir / "source" / f"reference_image{source_image.suffix.lower() or '.png'}"
        if source_reference.is_file():
            if sha256_file(source_reference) != sha256_file(source_image):
                raise RuntimeError(
                    "Resume source-image mismatch: the saved reference differs from this request. "
                    "Start a new job rather than mixing appearance data from another image."
                )
        else:
            shutil.copy2(source_image, source_reference)
        receipt.input_files["source_image"] = str(source_reference)
    return source_original, source_reference


def run_postprocess(
    engine: "PipelineEngine",
    mesh: Path,
    *,
    asset_type: str = "auto",
    quality: str = "gameplay",
    separate_movable_parts: bool = True,
    texture_resolution: int | None = None,
    lod_enabled: bool = True,
    remove_hidden_geometry: bool = False,
    experimental_semantic_split: bool = False,
    prompt: str = "",
    source_image: Path | None = None,
    animation_preset: str = "dance",
    resume_job_id: str = "",
) -> tuple[Path, JobReceipt, Path]:
    profile = get_profile(
        asset_type,
        quality,
        texture_resolution,
        lod_enabled,
        prompt=prompt,
        filename=mesh.name,
    )
    if resume_job_id:
        job_dir = engine.config.jobs_root / resume_job_id
        receipt_path = job_dir / "proof" / "job_receipt.json"
        if not receipt_path.is_file():
            raise RuntimeError(f"Resume job does not exist: {resume_job_id}")
        receipt = JobReceipt.load(receipt_path)
        receipt.status = "running"
        receipt.error = None
        receipt.finished_at = None
    else:
        receipt, job_dir = engine.new_job("postprocess", {"mesh": str(mesh)})

    if source_image is None and resume_job_id:
        saved_references = sorted((job_dir / "source").glob("reference_image.*"))
        if saved_references:
            source_image = saved_references[0]

    postprocess_request = {
        "asset_type": asset_type,
        "quality": quality,
        "texture_resolution": texture_resolution,
        "lod_enabled": lod_enabled,
        "separate_movable_parts": separate_movable_parts,
        "remove_hidden_geometry": remove_hidden_geometry,
        "experimental_semantic_split": experimental_semantic_split,
        "prompt": prompt,
        "animation_preset": animation_preset,
        "geometry_policy": "clean_master_then_measured_descending_lod",
    }
    request_parameters = {
        "profile": profile.to_dict(),
        **postprocess_request,
        "source_image_contract": {
            "name": source_image.name,
            "sha256": sha256_file(source_image),
        } if source_image and source_image.is_file() else None,
    }
    contract = build_contract("postprocess", mesh, request_parameters)
    if resume_job_id and "postprocess_contract" not in receipt.parameters and "profile" in receipt.parameters:
        legacy_parameters = {
            key: receipt.parameters.get(key)
            for key in (
                "profile",
                "separate_movable_parts",
                "remove_hidden_geometry",
                "experimental_semantic_split",
                "prompt",
            )
        }
        current_legacy_shape = {key: request_parameters.get(key) for key in legacy_parameters}
        if canonical_hash(legacy_parameters) != canonical_hash(current_legacy_shape):
            raise RuntimeError(
                "Legacy resume settings mismatch. Existing completed stages were produced with different "
                "asset settings; start a new job rather than reusing incompatible outputs."
            )
    store_or_validate_contract(receipt, "postprocess_contract", contract)
    receipt.parameters.update(request_parameters)
    receipt.parameters["postprocess_request"] = postprocess_request
    receipt.input_files.setdefault("mesh", str(mesh))

    source_original, source_reference = _copy_source_contract(
        mesh, source_image, receipt, job_dir, contract
    )
    engine._write_receipt(receipt, job_dir)

    ingested_high = job_dir / "source" / "ingested_high.glb"
    ingest_report = job_dir / "reports" / "ingest.json"
    engine._reusable_blender_stage(
        "ingest_validate",
        "ingest_validate.py",
        [
            "--input", source_original,
            "--output", ingested_high,
            "--report", ingest_report,
            "--apply-transforms",
        ],
        {"high_glb": ingested_high, "report": ingest_report},
        receipt,
        job_dir,
    )

    geometry = run_highres_geometry_phase(
        engine,
        high_glb=ingested_high,
        profile=profile,
        quality=quality,
        prompt=prompt,
        source_reference=source_reference,
        receipt=receipt,
        job_dir=job_dir,
    )
    high_glb = geometry.clean_master
    selected_geometry = geometry.selected_mesh
    lod1 = geometry.lod1 or job_dir / "highres_geometry" / "geometry" / "lod1.glb"
    lod2 = geometry.lod2 or job_dir / "highres_geometry" / "geometry" / "lod2.glb"
    receipt.parameters["highres_geometry"] = {
        "master_faces": geometry.master_faces,
        "selected_faces": geometry.selected_faces,
        "selection_policy": geometry.selection_policy,
    }
    receipt.outputs.update(
        {
            "clean_highres_master": str(high_glb),
            "selected_untextured_lod0": str(selected_geometry),
            "highres_geometry_report": str(geometry.report),
        }
    )
    engine._write_receipt(receipt, job_dir)

    analysis_report = job_dir / "reports" / "analysis.json"
    engine._reusable_blender_stage(
        "analyse_clean_master",
        "analyze_asset.py",
        ["--input", high_glb, "--report", analysis_report, "--asset-type", profile.asset_type.value],
        {"report": analysis_report},
        receipt,
        job_dir,
    )

    # Splitting happens only after measured LOD0 selection. The old merge-small cleanup is disabled:
    # debris has already been classified on the high-resolution master, and face-count cleanup here
    # could delete valid accessories, branches, railings, or building parts.
    split_mesh = job_dir / "split" / "components.glb"
    parts_dir = job_dir / "split" / "parts"
    parts_manifest = job_dir / "reports" / "parts.json"
    split_args: list[object] = [
        "--input", selected_geometry,
        "--output", split_mesh,
        "--parts-dir", parts_dir,
        "--manifest", parts_manifest,
        "--asset-type", profile.asset_type.value,
        "--max-parts", profile.max_parts,
    ]
    if separate_movable_parts:
        split_args.append("--separate-movable")
    if remove_hidden_geometry:
        split_args.append("--remove-hidden")
    if experimental_semantic_split:
        split_args.append("--experimental-semantic")
    engine._reusable_blender_stage(
        "split_selected_lod0",
        "split_parts.py",
        split_args,
        {"mesh": split_mesh, "manifest": parts_manifest},
        receipt,
        job_dir,
    )
    optimized_mesh = split_mesh
    retopo_backend = "measured_highres_ladder"

    uv_mesh = job_dir / "uv" / "game_ready_uv.glb"
    uv_report = job_dir / "reports" / "uv.json"
    uv_layout = job_dir / "uv" / "uv_layout.png"
    _blender_uv(
        engine,
        profile,
        optimized_mesh,
        uv_mesh,
        uv_report,
        uv_layout,
        receipt,
        job_dir,
        "uv_after_measured_lod",
    )
    uv_backend = "blender_bounded"

    baked_mesh = job_dir / "mapped" / "game_ready_textured.glb"
    maps_dir = job_dir / "mapped" / "maps"
    bake_report = job_dir / "reports" / "bake.json"
    appearance_mesh = uv_mesh
    projected_basecolor: Path | None = None
    texture_lane = "source_materials"
    if source_reference:
        appearance_mesh, projected_basecolor, texture_lane = resolve_source_appearance(
            engine,
            uv_mesh,
            source_reference,
            prompt,
            receipt,
            job_dir,
            profile.texture_size,
            profile.uv_padding_px,
        )
        receipt.parameters["texture_lane"] = texture_lane
        receipt.outputs["appearance_mesh"] = str(appearance_mesh)
        receipt.outputs["projected_basecolor"] = str(projected_basecolor)
        engine._write_receipt(receipt, job_dir)

    if profile.texture_strategy == "preserve_existing" and not source_reference:
        engine._reusable_blender_stage(
            "preserve_materials",
            "preserve_materials.py",
            [
                "--input", uv_mesh,
                "--output", baked_mesh,
                "--maps-dir", maps_dir,
                "--report", bake_report,
            ],
            {
                "mesh": baked_mesh,
                "report": bake_report,
                "maps_manifest": maps_dir / "maps_manifest.json",
            },
            receipt,
            job_dir,
        )
    else:
        bake_args: list[object] = [
            "--high", high_glb,
            "--low", appearance_mesh,
            "--output", baked_mesh,
            "--maps-dir", maps_dir,
            "--report", bake_report,
            "--size", profile.texture_size,
            "--padding-px", profile.uv_padding_px,
        ]
        if projected_basecolor:
            bake_args += ["--basecolor-image", projected_basecolor]
        engine._reusable_blender_stage(
            "bake_from_clean_master",
            "bake_transfer.py",
            bake_args,
            {
                "mesh": baked_mesh,
                "report": bake_report,
                "basecolor": maps_dir / "basecolor.png",
                "normal": maps_dir / "normal.png",
                "roughness": maps_dir / "roughness.png",
                "metallic": maps_dir / "metallic.png",
                "ao": maps_dir / "ao.png",
            },
            receipt,
            job_dir,
            timeout=14_400,
        )

    package_input = baked_mesh
    rig_report = job_dir / "reports" / "rig.json"
    if profile.generate_rig:
        rigged_mesh = job_dir / "rigged" / "game_ready_rigged.glb"
        if profile.rigid_rig:
            rig_kind = "mechanical"
        elif profile.asset_type.value in {"avatar", "character"}:
            rig_kind = "humanoid"
        else:
            rig_kind = "creature"
        engine._reusable_blender_stage(
            "rig",
            "rig_animate.py",
            [
                "--input", baked_mesh,
                "--output", rigged_mesh,
                "--report", rig_report,
                "--kind", rig_kind,
                "--prompt", prompt,
                "--animation-preset", animation_preset,
                "--pose-report", job_dir / "preprocess" / "avatar_report.json",
            ],
            {"mesh": rigged_mesh, "report": rig_report},
            receipt,
            job_dir,
        )
        package_input = rigged_mesh

    safe_name = "".join(
        char if char.isalnum() or char in "-_" else "_" for char in mesh.stem
    ).strip("_") or "asset"
    output_root = job_dir / "output" / safe_name
    validation = output_root / "reports" / "validation-report.json"
    pipeline_report = output_root / "reports" / "pipeline-report.json"
    engine._reusable_blender_stage(
        "export_validate",
        "package_validate.py",
        [
            "--input", package_input,
            "--high", high_glb,
            "--lod1", lod1 if profile.lod_count > 0 and lod1.is_file() else "",
            "--lod2", lod2 if profile.lod_count > 1 and lod2.is_file() else "",
            "--maps-dir", maps_dir,
            "--parts-manifest", parts_manifest,
            "--uv-layout", uv_layout,
            "--stage-reports-dir", job_dir / "reports",
            "--output-root", output_root,
            "--validation", validation,
            "--pipeline-report", pipeline_report,
            "--asset-type", profile.asset_type.value,
            "--target-min", max(1, round(geometry.selected_faces * 0.95)),
            "--target-max", max(1, round(geometry.selected_faces * 1.05)),
            "--budget-mode", profile.budget_mode,
            "--collision-mode", profile.collision_mode,
            "--spatial-chunking", str(profile.spatial_chunking).lower(),
            "--cell-divisions", profile.cell_divisions,
            "--export-strategy", profile.export_strategy,
        ],
        {
            "glb": output_root / "meshes" / "game_ready.glb",
            "validation": validation,
            "pipeline_report": pipeline_report,
            "front": output_root / "previews" / "front.png",
            "side": output_root / "previews" / "side.png",
            "rear": output_root / "previews" / "rear.png",
            "perspective": output_root / "previews" / "perspective.png",
        },
        receipt,
        job_dir,
    )

    receipt.status = "passed"
    receipt.finished_at = now_ms()
    receipt.outputs.update(
        {
            "asset_glb": str(output_root / "meshes" / "game_ready.glb"),
            "high_glb": str(output_root / "meshes" / "high.glb"),
            "lod1_glb": str(output_root / "meshes" / "lod1.glb") if profile.lod_count > 0 else "",
            "lod2_glb": str(output_root / "meshes" / "lod2.glb") if profile.lod_count > 1 else "",
            "maps_dir": str(output_root / "textures"),
            "parts_manifest": str(output_root / "reports" / "parts.json"),
            "validation_json": str(validation),
            "pipeline_report_json": str(pipeline_report),
            "preview_png": str(output_root / "previews" / "perspective.png"),
            "output_root": str(output_root),
            "retopo_backend": retopo_backend,
            "uv_backend": uv_backend,
            "rig_report": str(rig_report) if profile.generate_rig else "",
            "rig_status": "template_unproven" if profile.generate_rig else "not_applicable",
            "texture_lane": texture_lane,
        }
    )
    engine._write_receipt(receipt, job_dir)
    return output_root / "meshes" / "game_ready.glb", receipt, job_dir
