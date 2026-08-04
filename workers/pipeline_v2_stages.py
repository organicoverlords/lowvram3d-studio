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
    # The generator does not run in the pipeline's own interpreter. Mini Turbo needs the
    # standalone Python it was installed against and `hy3dgen` on PYTHONPATH; running it with the
    # shared interpreter fails on an import, not on anything about the asset. Declaring the
    # runtime in the manifest keeps that fact out of the stage code and off a wrapper script.
    generator_runtime = manifest.get("generator_runtime") or {}
    generator_python = generator_runtime.get("python") or pipeline.python
    generator_env = ({"PYTHONPATH": os.pathsep.join(str(p) for p in generator_runtime["pythonpath"])}
                     if generator_runtime.get("pythonpath") else None)
    # The ladder and step count were pinned to the bottom rung here (256:1500, one step) while the
    # proven reference run used 384:3000,320:2000,256:1500 at five steps. That is not a small
    # difference: at 256 the generator fuses a standing figure's legs into one mass, which the
    # stance repair then has to cut apart at a face-loss cost no budget will accept. The ladder
    # already degrades on its own when VRAM runs out, so starting at the top costs nothing.
    generator_settings = manifest.get("generator_settings") or {}
    octree_ladder = str(generator_settings.get("octree_ladder", "384:3000,320:2000,256:1500"))
    generator_steps = int(generator_settings.get("steps", 5))

    def w(name: str) -> Path:
        return REPO_ROOT / "workers" / name

    def b(name: str) -> Path:
        return REPO_ROOT / "blender" / name

    # ---------------------------------------------------------------- INGEST
    def ingest():
        def runner(overrides):
            stage = pipeline.stage_dir("INGEST") / "candidate"
            matte = stage / "matte.png"
            conditioning_audit = stage / "conditioning_audit.json"
            code, out = pipeline.run([
                pipeline.python, w("normalize_conditioning.py"),
                "--image", source_image, "--output", matte,
                "--audit-json", conditioning_audit,
                "--overlay", stage / "conditioning_overlay.png",
                "--original-vs-matte", stage / "original_vs_matte.png",
                "--size", "512",
            ])
            if code != 0 or not matte.exists():
                return StageResult("failed", detail=f"conditioning worker exit {code}: {out[-800:]}")
            stats = read_json(conditioning_audit)
            gates = {"source_sha256": manifest["source"]["sha256"],
                     "matte_sha256": sha256(matte),
                     "conditioning_route": stats.get("route"),
                     "source_alpha_is_useful": stats.get("source_has_useful_alpha"),
                     "foreground_coverage_percent": stats.get("selected_foreground", {}).get("foreground_coverage_percent"),
                     "transparent_margin_percent": stats.get("selected_foreground", {}).get("transparent_margin_percent"),
                     "disconnected_components": stats.get("selected_foreground", {}).get("disconnected_foreground_components"),
                     "border_contact": stats.get("selected_foreground", {}).get("border_contact"),
                     "alpha_valid": stats.get("selected_foreground", {}).get("alpha_valid")}
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
            prompt = manifest.get(
                "generation_prompt",
                f"{manifest['asset_id'].replace('_', ' ')} static non-animated asset; preserve the complete visible silhouette and thin details; no rig, no animation",
            )
            model_root = Path(os.environ.get(
                "LOWVRAM3D_MINI_TURBO_MODEL_ROOT",
                r"C:\AI\HY3D2\HuggingFaceHub\hunyuan3d-2mini-direct",
            ))
            steps = int(overrides.get("steps", generator_steps))
            code, out = pipeline.run([
                generator_python, w("mini_turbo_generate.py"),
                "--image", source_image,
                "--conditioning-image", matte,
                "--output", master,
                "--result-json", result_json,
                "--model-root", model_root,
                "--prompt", prompt,
                "--steps", str(steps),
                "--octree-ladder", octree_ladder,
                "--seed", "12345",
            ], env_extra=generator_env)
            result = read_json(result_json)
            failure_code = result.get("failure_code")
            if code != 0 or not master.exists():
                return StageResult("failed", failure_codes=[failure_code] if failure_code else [],
                                   gates={"steps": steps, "octree_ladder": octree_ladder, "seed": 12345},
                                   detail=f"generator exit {code}: {out[-800:]}")
            return StageResult("passed", outputs={"master": master},
                               gates={"sha256": sha256(master), "steps": steps, "octree_ladder": octree_ladder,
                                      "octree_resolution_used": result.get("octree_resolution"), "seed": 12345})
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
