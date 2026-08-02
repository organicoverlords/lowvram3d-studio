"""Pipeline V2 stage implementations.

Thin generic wrappers over workers that are already proven. Nothing here re-implements UV
unwrapping, baking, projection or overlap detection - those keep their existing modules and are
called through a uniform interface so the state machine can hash, gate, retry and resume them.

Each stage returns a StageResult carrying its gate measurements and, when it fails, the failure
codes the repair policy is keyed on.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from run_asset_pipeline import REPO_ROOT, StageResult, sha256


def read_json(path) -> dict:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def register_stages(pipeline, manifest: dict, existing_master: str = "") -> dict:
    root = Path(manifest["output_root"])
    profile = pipeline.profile
    source_image = Path(manifest["source"]["path"])
    texture_resolution = int(manifest["texture"]["resolution"])

    def w(name: str) -> Path:
        return REPO_ROOT / "workers" / name

    def b(name: str) -> Path:
        return REPO_ROOT / "blender" / name

    # ---------------------------------------------------------------- INGEST
    def ingest():
        def runner(overrides):
            stage = pipeline.stage_dir("INGEST") / "candidate"
            matte = stage / "matte.png"
            code, out = pipeline.run([
                pipeline.python, w("pipeline_matte.py"),
                "--image", source_image, "--output", matte,
                "--stats-json", stage / "matte_stats.json",
                "--mode", "hybrid", "--tolerance", "42", "--enclosed-tolerance", "32",
                "--enclosed-min-area", "5000", "--shadow-tolerance", "155",
                "--shadow-from", "0.78", "--close-radius", "2",
            ])
            if code != 0 or not matte.exists():
                return StageResult("failed", detail=f"matte worker exit {code}: {out[-800:]}")
            stats = read_json(stage / "matte_stats.json") or read_json(
                matte.with_name(matte.stem + "_stats.json"))
            gates = {"source_sha256": manifest["source"]["sha256"],
                     "matte_sha256": sha256(matte),
                     "subject_components": stats.get("subject_components"),
                     "subject_pixel_fraction": stats.get("subject_pixel_fraction")}
            fraction = stats.get("subject_pixel_fraction")
            if fraction is not None and not 0.05 <= float(fraction) <= 0.95:
                return StageResult("failed", gates=gates,
                                   detail=f"matte covers {fraction} of the frame")
            return StageResult("passed", outputs={"matte": matte}, gates=gates)
        return pipeline.execute("INGEST", [source_image], runner)

    # ---------------------------------------------------------------- GENERATE
    def generate():
        def runner(overrides):
            stage = pipeline.stage_dir("GENERATE") / "candidate"
            master = stage / "master.glb"
            if existing_master:
                shutil.copy2(existing_master, master)
                return StageResult("passed", outputs={"master": master},
                                   gates={"adopted_existing_master": str(existing_master),
                                          "sha256": sha256(master)})
            matte = pipeline.stage_dir("INGEST") / "proven" / "matte.png"
            result_json = stage / "generate_report.json"
            model_root = Path(os.environ.get(
                "LOWVRAM3D_MINI_TURBO_MODEL_ROOT",
                r"C:\AI\HY3D2\HuggingFaceHub\hunyuan3d-2mini-direct",
            ))
            code, out = pipeline.run([
                pipeline.python, w("mini_turbo_generate.py"),
                "--image", source_image,
                "--conditioning-image", matte,
                "--output", master,
                "--result-json", result_json,
                "--model-root", model_root,
                "--prompt", "weathered rustic static barn shed with separate corrugated roofs, wind-bent trees and ground vegetation; no armature, no animation, no humanoid or vehicle parts",
            ])
            if code != 0 or not master.exists():
                return StageResult("failed", detail=f"generator exit {code}: {out[-800:]}")
            return StageResult("passed", outputs={"master": master},
                               gates={"sha256": sha256(master)})
        inputs = [existing_master] if existing_master else [pipeline.stage_dir("INGEST") / "proven" / "matte.png"]
        return pipeline.execute("GENERATE", inputs, runner)

    # ---------------------------------------------------------------- GEOMETRY_QA
    def geometry_qa():
        master = pipeline.stage_dir("GENERATE") / "proven" / "master.glb"

        def runner(overrides):
            stage = pipeline.stage_dir("GEOMETRY_QA") / "candidate"
            report = stage / "geometry_report.json"
            code, out = pipeline.run([
                pipeline.python, w("pipeline_geometry_qa.py"),
                "--mesh", master, "--report", report,
                "--max-axis-ratio", str(profile.max_axis_ratio),
                "--debris-height-min", str(profile.debris_height_min),
            ])
            data = read_json(report)
            if code not in (0, 2) or not data:
                return StageResult("failed", detail=f"geometry QA exit {code}: {out[-800:]}")
            codes = data.get("failure_codes", [])
            gates = {k: data.get(k) for k in ("extent", "triangles", "components", "debris", "axis_ratio")}
            if codes:
                return StageResult("failed", gates=gates, failure_codes=codes,
                                   detail="; ".join(data.get("messages", [])))
            return StageResult("passed", outputs={"geometry_report": report}, gates=gates)
        return pipeline.execute("GEOMETRY_QA", [master], runner)

    # ---------------------------------------------------------------- CLEAN
    def clean():
        master = pipeline.stage_dir("GENERATE") / "proven" / "master.glb"

        def runner(overrides):
            stage = pipeline.stage_dir("CLEAN") / "candidate"
            cleaned = stage / "clean.glb"
            report = stage / "clean_report.json"
            height_min = float(overrides.get("debris_height_min", profile.debris_height_min))
            code, out = pipeline.run([
                pipeline.python, w("pipeline_debris_strip.py"),
                "--input", master, "--output", cleaned, "--report", report,
                "--height-min", str(height_min),
            ])
            data = read_json(report)
            if code != 0 or not cleaned.exists():
                return StageResult("failed", detail=f"debris strip exit {code}: {out[-800:]}")

            # Verify the result rather than trusting the strip: re-run the geometry gate on the
            # cleaned mesh with debris blocking. This is where a remaining shard is a real failure,
            # and it is what makes the stage self-correcting instead of merely hopeful.
            verify = stage / "verify_geometry.json"
            pipeline.run([
                pipeline.python, w("pipeline_geometry_qa.py"),
                "--mesh", cleaned, "--report", verify,
                "--max-axis-ratio", str(profile.max_axis_ratio),
                "--debris-height-min", str(height_min), "--debris-blocking",
            ])
            verified = read_json(verify)
            gates = {"triangles_before": data.get("triangles_before"),
                     "triangles_after": data.get("triangles_after"),
                     "triangles_removed_percent": data.get("triangles_removed_percent"),
                     "uv_bit_identical": data.get("uv_bit_identical"),
                     "components_removed": data.get("components_removed"),
                     "shards_remaining": (verified.get("debris") or {}).get(
                         "unsupported_components_remaining")}
            removed = float(data.get("triangles_removed_percent") or 0.0)
            # Debris removal that takes real surface with it is worse than the debris it removed.
            if removed > 2.0:
                return StageResult("failed", gates=gates,
                                   detail=f"debris strip removed {removed}% of triangles")
            codes = verified.get("failure_codes", [])
            if codes:
                return StageResult("failed", gates=gates, failure_codes=codes,
                                   detail="; ".join(verified.get("messages", [])))
            return StageResult("passed",
                               outputs={"clean": cleaned, "clean_report": report,
                                        "verify_geometry": verify},
                               gates=gates)
        return pipeline.execute("CLEAN", [master], runner)

    # ---------------------------------------------------------------- TEXTURE_QA
    def texture_qa():
        """Generic visual gate. Inputs are whatever the TEXTURE stage proved."""
        textures = manifest.get("texture_artifacts", {})

        def runner(overrides):
            stage = pipeline.stage_dir("TEXTURE_QA") / "candidate"
            report = stage / "visual_evaluation.json"
            command = [pipeline.python, w("visual_evaluator.py"),
                       "--render-dir", textures.get("render_dir", str(stage)),
                       "--report", report,
                       "--profile-json", str(root / "profile.json")]
            for flag, key in (("--source-image", "source_image"), ("--basecolor", "basecolor"),
                              ("--orm", "orm"), ("--material-id", "material_id"),
                              ("--coverage", "coverage"), ("--orientation-truth", "orientation_truth"),
                              ("--region-report", "region_report"), ("--uv-report", "uv_report"),
                              ("--geometry-report", "geometry_report")):
                if textures.get(key):
                    command += [flag, textures[key]]
            if textures.get("material_id_components") is not None:
                command += ["--material-id-components", str(textures["material_id_components"])]
            if textures.get("background_rgb"):
                command += ["--background-rgb", textures["background_rgb"]]
            code, out = pipeline.run(command)
            data = read_json(report)
            if not data:
                return StageResult("failed", detail=f"evaluator produced no report (exit {code}): {out[-800:]}")
            gates = data.get("measured", {})
            if not data.get("passed"):
                return StageResult("failed", gates=gates,
                                   failure_codes=data.get("blocking_codes", []),
                                   detail=f"advisory={data.get('advisory_codes')}")
            return StageResult("passed", outputs={"visual_evaluation": report}, gates=gates)

        inputs = [v for k, v in textures.items()
                  if isinstance(v, str) and Path(v).is_file()]
        return pipeline.execute("TEXTURE_QA", inputs, runner)

    (root / "profile.json").write_text(json.dumps(profile.to_dict(), indent=2), encoding="utf-8")

    return {
        "INGEST": ingest,
        "GENERATE": generate,
        "GEOMETRY_QA": geometry_qa,
        "CLEAN": clean,
        "TEXTURE_QA": texture_qa,
    }
