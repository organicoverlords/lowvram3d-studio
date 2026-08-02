"""Canonical static Pipeline V2 adapter.

The state machine and production workers remain authoritative in the existing
modules. This adapter only normalizes the generic manifest, constrains the
static-baseline stage order, and adds a fresh-process export QA boundary.
"""
from __future__ import annotations

import json
from pathlib import Path

from pipeline_v2_production_stages import register_production_stages
from pipeline_v2_repair_policy import apply_repair_overrides
from pipeline_v2_stages import register_stages
from run_asset_pipeline import Pipeline, StageResult, hash_inputs, sha256


CANONICAL_STAGES = (
    "INGEST",
    "GENERATE",
    "GEOMETRY_QA",
    "CLEAN",
    "LOD",
    "UV",
    "BAKE",
    "TEXTURE",
    "TEXTURE_QA",
    "EXPORT_QA",
)


def normalize_manifest(manifest: dict, *, stage_from: str = "INGEST",
                       stage_to: str = "EXPORT_QA", existing_master: str = "") -> dict:
    """Add the unified contract without replacing current manifest fields."""
    result = json.loads(json.dumps(manifest))
    source = result.get("source") or {}
    required = ("asset_id", "profile", "output_root")
    missing = [key for key in required if not result.get(key)]
    if missing:
        raise ValueError(f"UNIFIED_MANIFEST_MISSING: {', '.join(missing)}")
    if not source.get("path") or not source.get("sha256"):
        raise ValueError("UNIFIED_MANIFEST_SOURCE_HASH_REQUIRED")
    if not (Path(source["path"]).is_file() or existing_master):
        raise ValueError("UNIFIED_MANIFEST_SOURCE_PATH_MISSING")

    start, stop = stage_from.upper(), stage_to.upper()
    if start not in CANONICAL_STAGES or stop not in CANONICAL_STAGES:
        raise ValueError(f"UNIFIED_STAGE_RANGE_INVALID: {start}..{stop}")
    if CANONICAL_STAGES.index(stop) < CANONICAL_STAGES.index(start):
        raise ValueError(f"UNIFIED_STAGE_RANGE_REVERSED: {start}..{stop}")

    generator = result.get("generator")
    result["generator_settings"] = result.get("generator_settings") or {
        "backend": generator if isinstance(generator, str) else "mini_turbo",
        "preserve_current_worker": True,
    }
    result["geometry_source"] = result.get("geometry_source") or {
        "kind": "existing_master" if existing_master else "generated",
        "path": existing_master or None,
    }
    result["lod_policy"] = result.get("lod_policy") or result.get("geometry", {}).get(
        "lod_triangle_targets", []
    )
    result["uv"] = result.get("uv") or {
        "resolution": 1024,
        "padding": 4,
        "candidate_timeout_seconds": 600,
    }
    result["intent"] = {
        "static": True,
        "rigged": False,
        "animated": False,
    }
    result["stage_range"] = {"from": start, "to": stop}
    result["canonical_pipeline"] = "pipeline_v2_static_baseline"
    return result


def _receipt_output(receipt: dict, key: str) -> Path:
    value = (receipt.get("outputs") or {}).get(key)
    if not value or not value.get("path"):
        raise RuntimeError(f"TEXTURE_OUTPUT_MISSING: {key}")
    path = Path(value["path"])
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"TEXTURE_OUTPUT_INVALID: {key}={path}")
    return path


def _seed_existing_receipt(pipeline: Pipeline, stage: str, key: str,
                           source: Path, inputs: list[Path]) -> None:
    """Seed only a verified existing geometry boundary; no worker is rerun."""
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"EXISTING_MASTER_INVALID: {source}")
    outputs = pipeline.promote(stage, {key: source})
    pipeline.write_receipt(stage, {
        "stage": stage,
        "status": "passed",
        "attempts": [{"attempt": 0, "status": "passed", "detail": "seeded_existing_geometry"}],
        "input_hashes": hash_inputs(inputs),
        "outputs": outputs,
        "gates": {"existing_geometry_reused": True, "sha256": sha256(source)},
        "failure_codes": [],
        "needs_human": False,
    })


def register_unified_stages(pipeline: Pipeline, manifest: dict,
                            existing_master: str = "") -> dict:
    """Compose current core/production stages and a static EXPORT_QA boundary."""
    stages = register_stages(pipeline, manifest, existing_master=existing_master)
    stages.update(register_production_stages(pipeline, manifest))
    stages = apply_repair_overrides(pipeline, manifest, stages)

    def export_qa():
        texture = pipeline.read_receipt("TEXTURE") or {}
        textured_glb = _receipt_output(texture, "textured_glb")

        def runner(overrides):
            stage = pipeline.stage_dir("EXPORT_QA") / "candidate"
            report = stage / "fresh_import_validation.json"
            preview_dir = stage / "renders"
            command = [
                pipeline.blender,
                "--background",
                "--factory-startup",
                "--python-use-system-env",
                "--python",
                Path(__file__).resolve().parents[1] / "blender" / "validate_textured_baseline.py",
                "--",
                "--glb", textured_glb,
                "--report", report,
                "--preview-dir", preview_dir,
            ]
            code, output = pipeline.run(command)
            data = {}
            if report.is_file():
                try:
                    data = json.loads(report.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    data = {}
            checks = data.get("checks") or {}
            required = ("reimport_succeeded", "mesh_has_polygons", "mesh_has_uvs",
                        "material_present", "texture_packed", "previews_rendered")
            passed = code == 0 and bool(data.get("success")) and all(checks.get(key) for key in required)
            if not passed:
                return StageResult(
                    "failed",
                    outputs={"fresh_import_validation": report} if report.exists() else {},
                    gates={"checks": checks, "textured_glb": str(textured_glb)},
                    failure_codes=["FRESH_IMPORT_EXPORT_QA_FAILED"],
                    detail=output[-1600:],
                )
            outputs = {"fresh_import_validation": report}
            for view in ("front", "three_quarter", "side"):
                candidate = preview_dir / f"preview_{view}.png"
                if candidate.is_file():
                    outputs[f"render_{view}"] = candidate
            return StageResult("passed", outputs=outputs, gates={
                "checks": checks,
                "fresh_import": True,
                "armature_or_animation_not_required": True,
            })

        return pipeline.execute("EXPORT_QA", [textured_glb], runner)

    stages["EXPORT_QA"] = export_qa
    return stages
