"""Pipeline V2 adapters for deterministic asset-system stages.

This module layers the proven PARTS / POSE_PREP / MATERIALS policy and Blender
application workers onto the existing production pipeline without weakening any
existing gate. Optional semantic manifests degrade to explicit passthrough states;
missing evidence never invents a split, pose, material, or soft surface.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from run_asset_pipeline import REPO_ROOT, StageResult, sha256

ASSET_SYSTEM_STAGES = [
    "INGEST",
    "GENERATE",
    "GEOMETRY_QA",
    "CLEAN",
    "LOD",
    "UV",
    "BAKE",
    "TEXTURE",
    "MATERIALS",
    "TEXTURE_QA",
    "PARTS",
    "POSE_PREP",
    "RIG_READINESS",
    "RIG",
    "EXPORT",
]


def _json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    try:
        return json.loads(candidate.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def _output(pipeline, stage: str, key: str) -> Path:
    receipt = pipeline.read_receipt(stage) or {}
    entry = (receipt.get("outputs") or {}).get(key)
    if not entry or not entry.get("path"):
        raise RuntimeError(f"{stage} has no proven output {key!r}")
    return Path(entry["path"])


def _copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _blender(pipeline, script: Path, *arguments: object) -> tuple[int, str]:
    return pipeline.run([
        pipeline.blender,
        "--background",
        "--python-use-system-env",
        "--python",
        script,
        "--",
        *arguments,
    ])


def _asset_system_config(manifest: dict[str, Any]) -> dict[str, Any]:
    value = manifest.get("asset_system") or {}
    return dict(value) if isinstance(value, dict) else {}


def _optional_path(manifest: dict[str, Any], *keys: str) -> Path | None:
    config = _asset_system_config(manifest)
    for key in keys:
        value = config.get(key)
        if not value:
            value = manifest.get(key)
        if value:
            return Path(str(value))
    return None


def _safe_parts_payload() -> dict[str, Any]:
    return {
        "status": "safe_no_semantic_parts",
        "semantic_source_status": "disabled_for_stage",
        "part_count": 0,
        "parts": [],
        "decisions": [],
        "hard_split_count": 0,
        "protected_fused_count": 0,
        "unknown_count": 0,
    }


def _safe_materials_payload() -> dict[str, Any]:
    return {
        "status": "ordinary_pbr_fallback",
        "semantic_source_status": "disabled_for_stage",
        "feature_count": 0,
        "enabled_feature_count": 0,
        "features": [],
        "decisions": [],
        "material_families": ["OpaqueLit"],
        "maximum_material_slots": 4,
        "material_slot_budget_ok": True,
        "failure_codes": [],
        "soft_surface_budget": {
            "mode": "disabled_for_stage",
            "lod_card_limits": {"0": 0, "1": 0, "2": 0, "3": 0},
            "dense_groom_allowed": False,
        },
    }


def _preserve_pose_payload(*reason_codes: str) -> dict[str, Any]:
    return {
        "eligible": False,
        "action": "preserve_source_pose",
        "reason_codes": list(reason_codes or ("POSE_STAGE_DISABLED",)),
        "specification": None,
    }


def plan_variant(plan: dict[str, Any], stage: str) -> dict[str, Any]:
    """Return a stage-scoped plan so one stage cannot mutate another concern."""
    variant = json.loads(json.dumps(plan))
    stage_name = stage.upper()
    if stage_name == "MATERIALS":
        variant["parts"] = _safe_parts_payload()
        variant["pose_prep"] = _preserve_pose_payload("POSE_DISABLED_DURING_MATERIALS")
    elif stage_name == "PARTS":
        variant["materials"] = _safe_materials_payload()
        variant["pose_prep"] = _preserve_pose_payload("POSE_DISABLED_DURING_PARTS")
    elif stage_name == "POSE_PREP":
        variant["materials"] = _safe_materials_payload()
    else:
        raise ValueError(f"unsupported asset-system stage variant {stage!r}")
    return variant


def _application_outputs(
    stage: Path,
    asset_id: str,
    name: str,
) -> dict[str, Path]:
    prefix = stage / f"{asset_id}_{name}"
    return {
        "glb": prefix.with_suffix(".glb"),
        "blend": prefix.with_suffix(".blend"),
        "source_copy": stage / f"{asset_id}_{name}_source.glb",
        "report": stage / f"{name}_application_report.json",
        "plan": stage / f"{name}_plan.json",
    }


def _run_application(
    pipeline,
    *,
    input_glb: Path,
    plan: dict[str, Any],
    stage: Path,
    asset_id: str,
    name: str,
) -> tuple[int, str, dict[str, Path], dict[str, Any]]:
    outputs = _application_outputs(stage, asset_id, name)
    _write_json(outputs["plan"], plan)
    code, text = _blender(
        pipeline,
        REPO_ROOT / "blender" / "apply_asset_system_plan.py",
        "--input",
        input_glb,
        "--plan",
        outputs["plan"],
        "--output-glb",
        outputs["glb"],
        "--output-blend",
        outputs["blend"],
        "--source-pose-copy",
        outputs["source_copy"],
        "--report",
        outputs["report"],
    )
    return code, text, outputs, _json(outputs["report"])


def register_asset_system_stages(
    pipeline,
    manifest: dict[str, Any],
    stages: dict[str, Any],
) -> dict[str, Any]:
    """Add and override the asset-system stages on a production stage mapping."""
    registered = dict(stages)
    root = Path(manifest["output_root"])
    profile = pipeline.profile
    asset_id = str(manifest["asset_id"])
    config = _asset_system_config(manifest)
    parts_manifest = _optional_path(manifest, "parts_manifest", "semantic_parts_manifest")
    material_manifest = _optional_path(
        manifest,
        "material_manifest",
        "material_feature_manifest",
    )
    hero_soft_surfaces = bool(config.get("hero_soft_surfaces", False))

    def w(name: str) -> Path:
        return REPO_ROOT / "workers" / name

    def materials():
        textured = _output(pipeline, "TEXTURE", "textured_glb")
        geometry_report = _output(pipeline, "GEOMETRY_QA", "geometry_report")
        optional_inputs = [path for path in (parts_manifest, material_manifest) if path and path.is_file()]

        def runner(overrides):
            stage = pipeline.stage_dir("MATERIALS") / "candidate"
            plan_path = stage / "asset_system_plan.json"
            command: list[object] = [
                pipeline.python,
                w("build_asset_system_plan.py"),
                "--profile",
                profile.name,
                "--geometry-metrics",
                geometry_report,
                "--output",
                plan_path,
            ]
            if parts_manifest and parts_manifest.is_file():
                command.extend(["--parts-manifest", parts_manifest])
            if material_manifest and material_manifest.is_file():
                command.extend(["--material-manifest", material_manifest])
            if profile.separate_props:
                command.append("--separate-props")
            if hero_soft_surfaces:
                command.append("--hero-soft-surfaces")

            code, text = pipeline.run(command)
            full_plan = _json(plan_path)
            if code != 0 or full_plan.get("status") != "ready":
                return StageResult(
                    "failed",
                    gates={"plan": full_plan},
                    failure_codes=full_plan.get("blocking_codes", ["ASSET_SYSTEM_PLAN_FAILED"]),
                    detail=f"asset-system planner exit {code}: {text[-1200:]}",
                )

            materials_plan = full_plan.get("materials") or {}
            if materials_plan.get("status") == "ordinary_pbr_fallback":
                output = stage / f"{asset_id}_materials.glb"
                report = stage / "materials_application_report.json"
                _copy(textured, output)
                receipt = {
                    "passed": True,
                    "status": "ordinary_pbr_fallback",
                    "source_sha256": sha256(textured),
                    "output_sha256": sha256(output),
                    "material_families": ["OpaqueLit"],
                    "soft_surfaces": "disabled",
                    "fresh_import_validated": False,
                    "passthrough_byte_identical": sha256(textured) == sha256(output),
                }
                _write_json(report, receipt)
                return StageResult(
                    "passed",
                    outputs={
                        "materials_glb": output,
                        "asset_system_plan": plan_path,
                        "materials_report": report,
                    },
                    gates=receipt,
                )

            code, text, outputs, report = _run_application(
                pipeline,
                input_glb=textured,
                plan=plan_variant(full_plan, "MATERIALS"),
                stage=stage,
                asset_id=asset_id,
                name="materials",
            )
            gates = {
                "application_passed": bool(report.get("passed")),
                "fresh_import_validated": bool(report.get("fresh_import_validated")),
                "material_count": len((report.get("materials") or {}).get("created") or []),
                "soft_surface_regions": len(
                    (report.get("materials") or {}).get("soft_surface_regions") or []
                ),
                "source_hash_unchanged": (
                    report.get("input_sha256_before") == report.get("input_sha256_after")
                ),
            }
            if code != 0 or not outputs["glb"].is_file() or not report.get("passed"):
                return StageResult(
                    "failed",
                    gates=gates,
                    failure_codes=report.get("failures", ["MATERIAL_APPLICATION_FAILED"]),
                    detail=f"material application exit {code}: {text[-1200:]}",
                )
            return StageResult(
                "passed",
                outputs={
                    "materials_glb": outputs["glb"],
                    "materials_blend": outputs["blend"],
                    "materials_source": outputs["source_copy"],
                    "materials_report": outputs["report"],
                    "asset_system_plan": plan_path,
                    "materials_stage_plan": outputs["plan"],
                },
                gates=gates,
            )

        return pipeline.execute(
            "MATERIALS",
            [textured, geometry_report, *optional_inputs],
            runner,
        )

    def texture_qa():
        glb = _output(pipeline, "MATERIALS", "materials_glb")
        basecolor = _output(pipeline, "TEXTURE", "basecolor")
        orm = _output(pipeline, "TEXTURE", "orm")
        material_id = _output(pipeline, "TEXTURE", "material_id")
        coverage = _output(pipeline, "TEXTURE", "coverage")
        orientation_truth = _output(pipeline, "TEXTURE", "orientation_truth")
        region_report = _output(pipeline, "TEXTURE", "region_report")
        uv_report = _output(pipeline, "UV", "uv_report")
        geometry_report = _output(pipeline, "GEOMETRY_QA", "geometry_report")
        source_image = Path(manifest["source"]["path"])

        def runner(overrides):
            stage = pipeline.stage_dir("TEXTURE_QA") / "candidate"
            report = stage / "visual_evaluation.json"
            texture_receipt = pipeline.read_receipt("TEXTURE") or {}
            render_paths = [
                Path(value["path"])
                for key, value in (texture_receipt.get("outputs") or {}).items()
                if key.startswith("render_") and value.get("path")
            ]
            render_dir = render_paths[0].parent if render_paths else glb.parent
            bake_report = _json(_output(pipeline, "BAKE", "bake_report"))
            command = [
                pipeline.python,
                w("visual_evaluator.py"),
                "--render-dir",
                render_dir,
                "--report",
                report,
                "--profile-json",
                root / "profile.json",
                "--source-image",
                source_image,
                "--basecolor",
                basecolor,
                "--orm",
                orm,
                "--material-id",
                material_id,
                "--coverage",
                coverage,
                "--orientation-truth",
                orientation_truth,
                "--region-report",
                region_report,
                "--uv-report",
                uv_report,
                "--geometry-report",
                geometry_report,
                "--view-report",
                _output(pipeline, "TEXTURE", "view_report"),
                "--review-report",
                _output(pipeline, "TEXTURE", "review_report"),
                "--material-id-components",
                str(bake_report.get("high_component_count", 0)),
            ]
            code, text = pipeline.run(command)
            data = _json(report)
            if not data:
                return StageResult(
                    "failed",
                    detail=f"visual evaluator exit {code}: {text[-1200:]}",
                )
            gates = dict(data.get("measured") or {})
            gates["materials_glb_sha256"] = sha256(glb)
            if code != 0 or not data.get("passed"):
                return StageResult(
                    "failed",
                    gates=gates,
                    failure_codes=data.get("blocking_codes", []),
                    detail=f"advisory={data.get('advisory_codes', [])}",
                )
            return StageResult(
                "passed",
                outputs={"visual_evaluation": report},
                gates=gates,
            )

        return pipeline.execute(
            "TEXTURE_QA",
            [glb, basecolor, orm, material_id, coverage],
            runner,
        )

    def parts():
        materialized = _output(pipeline, "MATERIALS", "materials_glb")
        plan_path = _output(pipeline, "MATERIALS", "asset_system_plan")

        def runner(overrides):
            stage = pipeline.stage_dir("PARTS") / "candidate"
            full_plan = _json(plan_path)
            parts_plan = full_plan.get("parts") or {}
            if parts_plan.get("status") == "safe_no_semantic_parts":
                output = stage / f"{asset_id}_parts.glb"
                report = stage / "parts_application_report.json"
                _copy(materialized, output)
                receipt = {
                    "passed": True,
                    "status": "safe_no_semantic_parts",
                    "semantic_manifest_available": False,
                    "hard_splits": [],
                    "vertex_groups": [],
                    "protected_fused_regions": [],
                    "passthrough_byte_identical": sha256(materialized) == sha256(output),
                    "advisory_codes": ["PARTS_SEMANTIC_MANIFEST_UNAVAILABLE"],
                }
                _write_json(report, receipt)
                return StageResult(
                    "passed",
                    outputs={"parts_glb": output, "parts_report": report},
                    gates=receipt,
                )

            code, text, outputs, report = _run_application(
                pipeline,
                input_glb=materialized,
                plan=plan_variant(full_plan, "PARTS"),
                stage=stage,
                asset_id=asset_id,
                name="parts",
            )
            part_data = report.get("parts") or {}
            gates = {
                "application_passed": bool(report.get("passed")),
                "fresh_import_validated": bool(report.get("fresh_import_validated")),
                "vertex_groups": len(part_data.get("vertex_groups") or []),
                "hard_splits": len(part_data.get("hard_splits") or []),
                "preserved_regions": len(part_data.get("preserved") or []),
                "source_hash_unchanged": (
                    report.get("input_sha256_before") == report.get("input_sha256_after")
                ),
            }
            if code != 0 or not outputs["glb"].is_file() or not report.get("passed"):
                return StageResult(
                    "failed",
                    gates=gates,
                    failure_codes=report.get("failures", ["PARTS_APPLICATION_FAILED"]),
                    detail=f"parts application exit {code}: {text[-1200:]}",
                )
            return StageResult(
                "passed",
                outputs={
                    "parts_glb": outputs["glb"],
                    "parts_blend": outputs["blend"],
                    "parts_source": outputs["source_copy"],
                    "parts_report": outputs["report"],
                    "parts_stage_plan": outputs["plan"],
                },
                gates=gates,
            )

        return pipeline.execute("PARTS", [materialized, plan_path], runner)

    def pose_prep():
        parts_glb = _output(pipeline, "PARTS", "parts_glb")
        materials_glb = _output(pipeline, "MATERIALS", "materials_glb")
        plan_path = _output(pipeline, "MATERIALS", "asset_system_plan")
        parts_report_path = _output(pipeline, "PARTS", "parts_report")

        def runner(overrides):
            stage = pipeline.stage_dir("POSE_PREP") / "candidate"
            full_plan = _json(plan_path)
            pose_plan = full_plan.get("pose_prep") or {}
            action = str(pose_plan.get("action", "preserve_source_pose"))
            if action != "apply_a_pose":
                output = stage / f"{asset_id}_pose_prepared.glb"
                source = stage / f"{asset_id}_source_pose.glb"
                report = stage / "pose_prep_report.json"
                _copy(parts_glb, output)
                _copy(parts_glb, source)
                receipt = {
                    "passed": True,
                    "status": "POSE_PREP_SKIPPED_UNSAFE",
                    "action": "preserve_source_pose",
                    "reason_codes": pose_plan.get("reason_codes") or ["POSE_NOT_ELIGIBLE"],
                    "source_pose_sha256": sha256(source),
                    "output_sha256": sha256(output),
                    "passthrough_byte_identical": sha256(source) == sha256(output),
                }
                _write_json(report, receipt)
                return StageResult(
                    "passed",
                    outputs={
                        "pose_prepared_glb": output,
                        "source_pose_glb": source,
                        "pose_prep_report": report,
                    },
                    gates=receipt,
                )

            parts_report = _json(parts_report_path)
            hard_splits = (parts_report.get("parts") or {}).get("hard_splits")
            if hard_splits is None:
                hard_splits = parts_report.get("hard_splits") or []
            if hard_splits:
                return StageResult(
                    "failed",
                    gates={"hard_split_count": len(hard_splits)},
                    failure_codes=["POSE_PREP_AFTER_HARD_SPLIT_UNSUPPORTED"],
                    detail="pose prep is fail-closed after index-changing hard splits",
                )

            code, text, outputs, report = _run_application(
                pipeline,
                input_glb=materials_glb,
                plan=plan_variant(full_plan, "POSE_PREP"),
                stage=stage,
                asset_id=asset_id,
                name="pose_prepared",
            )
            pose_data = report.get("pose_prep") or {}
            gates = {
                "application_passed": bool(report.get("passed")),
                "pose_applied": bool(pose_data.get("applied")),
                "arm_angles_valid": bool(pose_data.get("arm_angles_valid")),
                "fresh_import_validated": bool(report.get("fresh_import_validated")),
                "source_hash_unchanged": (
                    report.get("input_sha256_before") == report.get("input_sha256_after")
                ),
                "face_count_preserved": (
                    (report.get("before") or {}).get("faces")
                    == (report.get("candidate") or {}).get("faces")
                ),
            }
            if (
                code != 0
                or not outputs["glb"].is_file()
                or not report.get("passed")
                or not pose_data.get("applied")
                or not pose_data.get("arm_angles_valid")
            ):
                return StageResult(
                    "failed",
                    gates=gates,
                    failure_codes=report.get("failures", ["POSE_PREP_FAILED"]),
                    detail=f"pose application exit {code}: {text[-1200:]}",
                )
            return StageResult(
                "passed",
                outputs={
                    "pose_prepared_glb": outputs["glb"],
                    "pose_prepared_blend": outputs["blend"],
                    "source_pose_glb": outputs["source_copy"],
                    "pose_prep_report": outputs["report"],
                    "pose_stage_plan": outputs["plan"],
                },
                gates=gates,
            )

        return pipeline.execute(
            "POSE_PREP",
            [parts_glb, materials_glb, plan_path, parts_report_path],
            runner,
        )

    def rig_readiness():
        mesh = _output(pipeline, "POSE_PREP", "pose_prepared_glb")

        def runner(overrides):
            stage = pipeline.stage_dir("RIG_READINESS") / "candidate"
            report = stage / "rig_readiness.json"
            code, text = pipeline.run([
                pipeline.python,
                w("pipeline_rig_readiness.py"),
                "--mesh",
                mesh,
                "--profile",
                profile.name,
                "--report",
                report,
            ])
            data = _json(report)
            if not data:
                return StageResult(
                    "failed",
                    detail=f"rig-readiness produced no report: {text[-1000:]}",
                )
            if not profile.rig_required:
                return StageResult(
                    "passed",
                    outputs={"rig_readiness": report},
                    gates={**data, "rig_not_required": True},
                )
            if code != 0 or not data.get("ready"):
                return StageResult(
                    "failed",
                    gates=data,
                    failure_codes=data.get("failure_codes", ["RIG_READINESS_FAILED"]),
                    detail="automatic rig-readiness gate failed closed",
                )
            return StageResult(
                "passed",
                outputs={"rig_readiness": report},
                gates=data,
            )

        return pipeline.execute("RIG_READINESS", [mesh], runner)

    def rig():
        mesh = _output(pipeline, "POSE_PREP", "pose_prepared_glb")
        readiness = _output(pipeline, "RIG_READINESS", "rig_readiness")

        def runner(overrides):
            stage = pipeline.stage_dir("RIG") / "candidate"
            if not profile.rig_required:
                passthrough = stage / f"{asset_id}_static.glb"
                _copy(mesh, passthrough)
                return StageResult(
                    "passed",
                    outputs={"rigged_glb": passthrough},
                    gates={"rig_required": False},
                )
            blend = stage / f"{asset_id}_rigged.blend"
            glb = stage / f"{asset_id}_rigged.glb"
            report = stage / "rig_report.json"
            code, text = _blender(
                pipeline,
                REPO_ROOT / "blender" / "pipeline_auto_rig.py",
                "--input",
                mesh,
                "--output-glb",
                glb,
                "--output-blend",
                blend,
                "--report",
                report,
                "--profile",
                profile.name,
            )
            data = _json(report)
            if code != 0 or not glb.is_file() or not data.get("passed"):
                return StageResult(
                    "failed",
                    gates=data,
                    failure_codes=["AUTO_RIG_FAILED"],
                    detail=f"auto-rig exit {code}: {text[-1200:]}",
                )
            return StageResult(
                "passed",
                outputs={
                    "rigged_glb": glb,
                    "rigged_blend": blend,
                    "rig_report": report,
                },
                gates=data,
            )

        return pipeline.execute("RIG", [mesh, readiness], runner)

    registered.update({
        "MATERIALS": materials,
        "TEXTURE_QA": texture_qa,
        "PARTS": parts,
        "POSE_PREP": pose_prep,
        "RIG_READINESS": rig_readiness,
        "RIG": rig,
    })
    return registered
