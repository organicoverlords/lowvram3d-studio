"""Repair overrides for Pipeline V2.

The base stages remain the source of truth for generation, UVs, baking and projection. This module
wraps only the failure points proven by the shaman benchmark:

* a humanoid lower stance can be fused into one floor-like bridge;
* decimation can leave microscopic detached components after CLEAN already passed;
* TEXTURE_QA used the stale pre-clean geometry report;
* source-to-mesh registration needs an explicit gate, not only atlas-orientation checks.

Every repair writes a derived candidate, promotes only after validation, and retains the original
stage output in the receipt.
"""
from __future__ import annotations

import json
from pathlib import Path

from run_asset_pipeline import REPO_ROOT, StageResult, sha256

REPAIR_SCHEMA = 2
HUMANOID_PROFILES = {"humanoid", "humanoid_complex_accessories"}


def _json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _entry(receipt: dict, key: str) -> dict:
    value = (receipt.get("outputs") or {}).get(key)
    if not value or not value.get("path"):
        raise RuntimeError(f"receipt has no output {key!r}")
    return value


def _path(pipeline, stage: str, key: str) -> Path:
    receipt = pipeline.read_receipt(stage) or {}
    return Path(_entry(receipt, key)["path"])


def _write_failed(
    pipeline, stage: str, receipt: dict, code: str, detail: str, gates: dict
) -> dict:
    failed = dict(receipt)
    failed.update({
        "status": "failed",
        "failure_codes": [code],
        "needs_human": False,
        "detail": detail,
        "gates": {**(receipt.get("gates") or {}), **gates},
    })
    pipeline.write_receipt(stage, failed)
    pipeline.log(f"[{stage}] repair FAILED {code}: {detail}")
    return failed


def apply_repair_overrides(pipeline, manifest: dict, stages: dict) -> dict:
    profile = pipeline.profile
    root = Path(manifest["output_root"])

    def w(name: str) -> Path:
        return REPO_ROOT / "workers" / name

    original_clean = stages["CLEAN"]

    def clean():
        receipt = original_clean()
        if receipt.get("status") != "passed" or profile.name not in HUMANOID_PROFILES:
            return receipt
        gates = receipt.get("gates") or {}
        current = (receipt.get("outputs") or {}).get("clean", {})
        if (
            gates.get("stance_repair_schema") == REPAIR_SCHEMA
            and current.get("path")
            and Path(current["path"]).exists()
        ):
            return receipt

        source_entry = dict(_entry(receipt, "clean"))
        source = Path(source_entry["path"])
        candidate_dir = pipeline.stage_dir("CLEAN") / "candidate"
        repaired = candidate_dir / f"{manifest['asset_id']}_stance_clean.glb"
        report = candidate_dir / "stance_report.json"
        code, output = pipeline.run([
            pipeline.python, w("pipeline_humanoid_stance.py"),
            "--input", source, "--output", repaired, "--report", report,
            "--max-centre-fraction", "0.08",
        ])
        data = _json(report)
        repair_gates = {
            "stance_repair_schema": REPAIR_SCHEMA,
            "stance_repair": data,
        }
        if code != 0 or not repaired.exists() or not data.get("passed"):
            return _write_failed(
                pipeline,
                "CLEAN",
                receipt,
                "FEET_TOO_CLOSE_FOR_RIGGING",
                f"humanoid stance repair exit {code}: {output[-1000:]}",
                repair_gates,
            )
        promoted = pipeline.promote("CLEAN", {"clean": repaired, "stance_report": report})
        updated = dict(receipt)
        updated_outputs = dict(receipt.get("outputs") or {})
        updated_outputs["pre_stance_clean"] = source_entry
        updated_outputs.update(promoted)
        updated["outputs"] = updated_outputs
        updated["gates"] = {**gates, **repair_gates}
        updated["repair_schema"] = REPAIR_SCHEMA
        pipeline.write_receipt("CLEAN", updated)
        pipeline.log(
            f"[CLEAN] stance repair promoted centre="
            f"{data.get('gap_before', {}).get('centre_fraction')}->"
            f"{data.get('gap_after', {}).get('centre_fraction')}"
        )
        return updated

    stages["CLEAN"] = clean

    original_lod = stages["LOD"]

    def lod():
        receipt = original_lod()
        if receipt.get("status") != "passed":
            return receipt
        gates = receipt.get("gates") or {}
        if gates.get("post_lod_cleanup_schema") == REPAIR_SCHEMA:
            valid = all(
                (receipt.get("outputs") or {})
                .get(f"lod{index}_geometry_report", {})
                .get("path")
                for index in range(len(profile.lod_triangle_targets))
            )
            if valid:
                return receipt

        outputs = dict(receipt.get("outputs") or {})
        post_reports = []
        for index in range(len(profile.lod_triangle_targets)):
            key = f"lod{index}"
            raw_entry = dict(_entry(receipt, key))
            raw = Path(raw_entry["path"])
            candidate_dir = pipeline.stage_dir("LOD") / "candidate"
            cleaned = candidate_dir / f"{manifest['asset_id']}_lod{index}_microclean.glb"
            cleanup_report = candidate_dir / f"lod{index}_microclean.json"
            verify_report = candidate_dir / f"lod{index}_geometry.json"
            code, output = pipeline.run([
                pipeline.python,
                w("pipeline_debris_strip.py"),
                "--input", raw,
                "--output", cleaned,
                "--report", cleanup_report,
                "--height-min", "0.66",
                "--max-triangles", "20",
                "--max-diagonal-fraction", "0.062",
            ])
            cleanup = _json(cleanup_report)
            if code != 0 or not cleaned.exists():
                return _write_failed(
                    pipeline,
                    "LOD",
                    receipt,
                    "MICRO_FLOATING_DEBRIS",
                    f"LOD{index} micro-clean exit {code}: {output[-1000:]}",
                    {"post_lod_cleanup_schema": REPAIR_SCHEMA, "lod": index},
                )
            verify_code, verify_output = pipeline.run([
                pipeline.python,
                w("pipeline_geometry_qa.py"),
                "--mesh", cleaned,
                "--report", verify_report,
                "--max-axis-ratio", str(profile.max_axis_ratio),
                "--debris-height-min", "0.66",
                "--debris-blocking",
            ])
            verify = _json(verify_report)
            loss = float(cleanup.get("triangles_removed_percent") or 0.0)
            if verify_code != 0 or verify.get("failure_codes") or loss > 0.25:
                return _write_failed(
                    pipeline,
                    "LOD",
                    receipt,
                    "MICRO_FLOATING_DEBRIS",
                    f"LOD{index} verification failed loss={loss:.4f}%: {verify_output[-1000:]}",
                    {
                        "post_lod_cleanup_schema": REPAIR_SCHEMA,
                        "lod": index,
                        "cleanup": cleanup,
                        "verify": verify,
                    },
                )
            promoted = pipeline.promote("LOD", {
                key: cleaned,
                f"lod{index}_cleanup_report": cleanup_report,
                f"lod{index}_geometry_report": verify_report,
            })
            outputs[f"raw_{key}"] = raw_entry
            outputs.update(promoted)
            post_reports.append({
                "lod": index,
                "triangles_removed": cleanup.get("triangles_removed"),
                "components_removed": cleanup.get("components_removed"),
                "triangles_removed_percent": loss,
                "remaining": (verify.get("debris") or {}).get(
                    "unsupported_components_remaining"
                ),
                "sha256": sha256(Path(promoted[key]["path"])),
            })

        updated = dict(receipt)
        updated["outputs"] = outputs
        updated["gates"] = {
            **gates,
            "post_lod_cleanup_schema": REPAIR_SCHEMA,
            "post_lod_cleanup": post_reports,
        }
        updated["repair_schema"] = REPAIR_SCHEMA
        pipeline.write_receipt("LOD", updated)
        pipeline.log(f"[LOD] post-decimation micro-clean promoted for {len(post_reports)} LODs")
        return updated

    stages["LOD"] = lod

    original_texture = stages["TEXTURE"]

    def texture():
        receipt = original_texture()
        if receipt.get("status") != "passed":
            return receipt
        outputs = dict(receipt.get("outputs") or {})
        existing = outputs.get("view_report", {}).get("path")
        if existing and Path(existing).exists():
            return receipt
        candidate = pipeline.stage_dir("TEXTURE") / "candidate" / "view_report.json"
        if not candidate.exists():
            return _write_failed(
                pipeline,
                "TEXTURE",
                receipt,
                "TEXTURE_MISREGISTRATION",
                "projection view builder wrote no registration report",
                {},
            )
        promoted = pipeline.promote("TEXTURE", {"view_report": candidate})
        report = _json(candidate)
        updated = dict(receipt)
        outputs.update(promoted)
        updated["outputs"] = outputs
        updated["gates"] = {
            **(receipt.get("gates") or {}),
            "source_mesh_registration": report.get("dense_registration"),
            "registration_gate_passed": report.get("registration_gate_passed"),
        }
        pipeline.write_receipt("TEXTURE", updated)
        return updated

    stages["TEXTURE"] = texture

    def texture_qa():
        texture_receipt = pipeline.read_receipt("TEXTURE") or {}
        texture_outputs = texture_receipt.get("outputs") or {}
        glb = Path(_entry(texture_receipt, "textured_glb")["path"])
        basecolor = Path(_entry(texture_receipt, "basecolor")["path"])
        orm = Path(_entry(texture_receipt, "orm")["path"])
        material_id = Path(_entry(texture_receipt, "material_id")["path"])
        coverage = Path(_entry(texture_receipt, "coverage")["path"])
        orientation_truth = Path(_entry(texture_receipt, "orientation_truth")["path"])
        region_report = Path(_entry(texture_receipt, "region_report")["path"])
        view_report = Path(_entry(texture_receipt, "view_report")["path"])
        uv_report = _path(pipeline, "UV", "uv_report")
        lod0_geometry = _path(pipeline, "LOD", "lod0_geometry_report")
        source_image = Path(manifest["source"]["path"])

        def runner(overrides):
            stage = pipeline.stage_dir("TEXTURE_QA") / "candidate"
            visual_report = stage / "visual_evaluation.json"
            registration_report = stage / "texture_registration.json"
            render_paths = [
                Path(value["path"])
                for key, value in texture_outputs.items()
                if key.startswith("render_") and value.get("path")
            ]
            render_dir = render_paths[0].parent if render_paths else glb.parent
            bake_report = _json(_path(pipeline, "BAKE", "bake_report"))
            visual_code, visual_output = pipeline.run([
                pipeline.python,
                w("visual_evaluator.py"),
                "--render-dir", render_dir,
                "--report", visual_report,
                "--profile-json", root / "profile.json",
                "--source-image", source_image,
                "--basecolor", basecolor,
                "--orm", orm,
                "--material-id", material_id,
                "--coverage", coverage,
                "--orientation-truth", orientation_truth,
                "--region-report", region_report,
                "--uv-report", uv_report,
                "--geometry-report", lod0_geometry,
                "--material-id-components", str(bake_report.get("high_component_count", 0)),
            ])
            registration_code, registration_output = pipeline.run([
                pipeline.python,
                w("pipeline_texture_registration_qa.py"),
                "--registration-report", view_report,
                "--output", registration_report,
            ])
            visual = _json(visual_report)
            registration = _json(registration_report)
            codes = list(visual.get("blocking_codes", []))
            if registration_code != 0 or not registration.get("passed"):
                codes.append("TEXTURE_MISREGISTRATION")
            gates = {
                **(visual.get("measured") or {}),
                "texture_registration": registration,
                "geometry_report_source": str(lod0_geometry),
            }
            if visual_code != 0 or not visual.get("passed") or codes:
                return StageResult(
                    "failed",
                    gates=gates,
                    failure_codes=sorted(set(codes)),
                    detail=(
                        f"visual={visual_output[-500:]} "
                        f"registration={registration_output[-500:]} "
                        f"advisory={visual.get('advisory_codes', [])}"
                    ),
                )
            return StageResult(
                "passed",
                outputs={
                    "visual_evaluation": visual_report,
                    "texture_registration": registration_report,
                },
                gates=gates,
            )

        inputs = [glb, basecolor, orm, material_id, coverage, view_report, lod0_geometry]
        return pipeline.execute("TEXTURE_QA", inputs, runner)

    stages["TEXTURE_QA"] = texture_qa
    return stages
