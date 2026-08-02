"""Production stage adapters for Pipeline V2.

These are thin orchestration layers over the V1 workers proven during the shaman run.  They add
immutable-input receipts, bounded retries and generic artifact names; they do not reimplement UV,
baking or projection algorithms.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from run_asset_pipeline import REPO_ROOT, StageResult, hash_inputs, sha256


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}


def _output(pipeline, stage: str, key: str) -> Path:
    receipt = pipeline.read_receipt(stage) or {}
    entry = (receipt.get("outputs") or {}).get(key)
    if not entry or not entry.get("path"):
        raise RuntimeError(f"{stage} has no proven output {key!r}")
    return Path(entry["path"])


def _blender(pipeline, script: Path, *args: object) -> tuple[int, str]:
    return pipeline.run([
        pipeline.blender, "--background", "--python-use-system-env", "--python", script,
        "--", *args,
    ])


def _copy_named(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _find(directory: Path, *tokens: str) -> Path | None:
    matches = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if all(token.lower() in name for token in tokens):
            matches.append(path)
    return sorted(matches, key=lambda p: (len(p.name), p.name))[0] if matches else None


def register_production_stages(pipeline, manifest: dict) -> dict:
    root = Path(manifest["output_root"])
    profile = pipeline.profile
    asset_id = manifest["asset_id"]
    resolution = int(manifest["texture"]["resolution"])
    uv_resolution = int((manifest.get("uv") or {}).get("resolution", 1024))
    uv_padding = int((manifest.get("uv") or {}).get("padding", 4))
    uv_timeout = float((manifest.get("uv") or {}).get("candidate_timeout_seconds", 600))
    lod_mode = str((manifest.get("lod") or {}).get("mode", "generate")).lower()
    suffix = "4k" if resolution >= 4096 else "2k"

    def w(name: str) -> Path:
        return REPO_ROOT / "workers" / name

    def b(name: str) -> Path:
        return REPO_ROOT / "blender" / name

    # ---------------------------------------------------------------- LOD
    def lod():
        clean = _output(pipeline, "CLEAN", "clean")

        if lod_mode == "preserve_source":
            pipeline.stage_dir("LOD")
            current_hashes = hash_inputs([clean])
            previous = pipeline.read_receipt("LOD")
            if (previous and previous.get("status") == "passed"
                    and previous.get("input_hashes") == current_hashes
                    and previous.get("gates", {}).get("LOD_STAGE") == "BYPASSED_SOURCE_GEOMETRY"):
                return previous
            source_hash = sha256(clean)
            receipt = {
                "stage": "LOD", "status": "passed", "profile": profile.name,
                "asset_id": asset_id, "input_hashes": current_hashes,
                "outputs": {"lod0": {"path": str(clean), "sha256": source_hash,
                                      "bytes": clean.stat().st_size}},
                "gates": {"LOD_STAGE": "BYPASSED_SOURCE_GEOMETRY", "LOD_REQUIRED": False,
                          "SOURCE_GEOMETRY_SHA256": source_hash, "source_path": str(clean)},
                "attempts": [{"attempt": 0, "status": "passed", "detail": "source geometry preserved"}],
                "failure_codes": [], "needs_human": False,
            }
            pipeline.write_receipt("LOD", receipt)
            return receipt

        def runner(overrides):
            stage = pipeline.stage_dir("LOD") / "candidate"
            outdir = stage / "lods"
            report = stage / "lod_report.json"
            targets_list = [int(v) for v in (manifest.get("lod_policy") or profile.lod_triangle_targets)]
            targets = ",".join(str(v) for v in targets_list)
            code, out = _blender(
                pipeline, b("final_pipeline_lods.py"),
                "--input", clean, "--output-dir", outdir, "--report", report,
                "--targets", targets,
                "--prefix", asset_id,
            )
            data = _json(report)
            if code != 0 or not data.get("lods"):
                return StageResult("failed", detail=f"LOD worker exit {code}: {out[-1200:]}")

            outputs = {"lod_report": report}
            gates = {"targets": targets_list, "lods": []}
            failures = []
            for index, target in enumerate(targets_list):
                source = outdir / f"{asset_id}_lod{index}.glb"
                if not source.exists():
                    failures.append(f"LOD{index} missing")
                    continue
                destination = stage / f"{asset_id}_lod{index}.glb"
                _copy_named(source, destination)
                outputs[f"lod{index}"] = destination
                row = next((r for r in data["lods"] if int(r.get("lod", -1)) == index), {})
                achieved = int(row.get("achieved_triangles", 0))
                source_topology = row.get("source_welded_topology") or {}
                candidate_topology = row.get("candidate_welded_topology") or {}
                gates["lods"].append({"lod": index, "target": target, "triangles": achieved,
                                      "sha256": sha256(destination),
                                      "source_welded_topology": source_topology,
                                      "candidate_welded_topology": candidate_topology})
                if achieved <= 0 or achieved > max(target * 1.20, target + 5000):
                    failures.append(f"LOD{index} achieved {achieved}, target {target}")
                for metric in ("boundary_edges", "non_manifold_edges"):
                    if int(candidate_topology.get(metric, 0)) > int(source_topology.get(metric, 0)):
                        failures.append(
                            f"LOD{index} {metric} regressed "
                            f"{source_topology.get(metric)}->{candidate_topology.get(metric)}"
                        )
            if failures:
                return StageResult("failed", gates=gates, detail="; ".join(failures))
            return StageResult("passed", outputs=outputs, gates=gates)

        return pipeline.execute("LOD", [clean], runner)

    # ---------------------------------------------------------------- UV
    def uv():
        explicit = manifest.get("uv_mesh") or (manifest.get("uv") or {}).get("mesh")
        if explicit:
            explicit_path = Path(explicit)
            expected_hash = manifest.get("uv_mesh_sha256") or (manifest.get("uv") or {}).get("mesh_sha256")

            def existing_runner(_overrides):
                stage = pipeline.stage_dir("UV") / "candidate"
                report = stage / "existing_uv_validation.json"
                code, out = _blender(pipeline, b("validate_existing_uv.py"),
                                      "--input", explicit_path, "--report", report,
                                      "--require-texture")
                data = _json(report)
                actual = sha256(explicit_path) if explicit_path.is_file() else None
                gates = {"existing_uv_reused": True, "source_sha256": actual,
                         "expected_sha256": expected_hash, **(data.get("gates") or {})}
                if expected_hash and actual != expected_hash:
                    return StageResult("failed", gates=gates, failure_codes=["UV_SOURCE_HASH_MISMATCH"],
                                       detail="explicit UV source hash mismatch")
                if code != 0 or not data.get("success"):
                    return StageResult("failed", gates=gates, failure_codes=["UV_EXISTING_INVALID"],
                                       detail=f"existing UV validation exit {code}: {out[-1000:]}")
                return StageResult("passed", outputs={"uv_mesh": explicit_path, "uv_report": report}, gates=gates)

            return pipeline.execute("UV", [explicit_path], existing_runner)

        lod_receipt = pipeline.read_receipt("LOD") or {}
        if lod_receipt.get("status") == "passed":
            lod0 = _output(pipeline, "LOD", "lod0")
        elif lod_mode == "preserve_source":
            lod0 = _output(pipeline, "CLEAN", "clean")
        else:
            raise RuntimeError("UV_NO_PROVEN_GEOMETRY_INPUT")

        def runner(overrides):
            stage = pipeline.stage_dir("UV") / "candidate"
            output = stage / f"{asset_id}_lod0_uv.glb"
            report = stage / "uv_report.json"
            route = (manifest.get("uv") or {}).get("route", "fast_blender")
            if route != "xatlas":
                output = stage / f"{asset_id}_uv.glb"
                code, out = _blender(
                    pipeline, b("final_pipeline_uv.py"), "--input", lod0,
                    "--output", output, "--report", report,
                    "--resolution", str(uv_resolution), "--overlap-timeout-seconds", "180",
                )
                data = _json(report)
                if code != 0 or not output.exists() or not data.get("gate_passed"):
                    return StageResult("failed", gates=data, failure_codes=["UV_FAST_ROUTE_FAILED"],
                                       detail=f"fast Blender UV exit {code}: {out[-1000:]}")
                return StageResult("passed", outputs={"uv_mesh": output, "uv_report": report},
                                   gates={"route": "fast_blender", **data})
            code, out = pipeline.run([
                pipeline.python, w("uv_xatlas_isolated.py"),
                "--input", lod0, "--output", output, "--report", report,
                "--resolution", str(uv_resolution), "--padding", str(uv_padding),
                "--timeout", str(uv_timeout),
            ])
            data = _json(report)
            selected = data.get("selected")
            selected_row = next((row for row in data.get("presets", []) if row.get("preset") == selected), {})
            child = _json(Path(selected_row["report"])) if selected_row.get("report") else {}
            metrics = child.get("metrics") or {}
            exact = child.get("exact_overlap") or {}
            gates = {
                "gate_passed": data.get("status") == "passed",
                "selected_preset": selected,
                "chart_count": metrics.get("chart_count"),
                "profile_chart_limit": profile.uv_max_charts,
                "atlas_utilization": metrics.get("atlas_utilization"),
                "stretch_p95": metrics.get("stretch_p95"),
                "candidate_pair_count": exact.get("candidate_pair_count"),
                "tested_pair_count": exact.get("tested_pair_count"),
                "timed_out": exact.get("timed_out"),
                "positive_overlap_pairs": exact.get("positive_overlap_pair_count"),
                "positive_overlap_texels": exact.get("positive_overlap_total_texels_equivalent"),
                "degenerate_uv_triangles": exact.get("degenerate_uv_triangle_count"),
                "out_of_bounds_triangles": exact.get("out_of_bounds_triangle_count"),
            }
            codes = []
            if exact.get("timed_out") or not exact.get("success") or int(exact.get("tested_pair_count") or 0) <= 0:
                codes.append("UV_OVERLAP")
            if int(exact.get("degenerate_uv_triangle_count") or 0) > 0:
                codes.append("UV_DEGENERATE")
            if float(exact.get("positive_overlap_total_texels_equivalent") or 0.0) > 1.0:
                codes.append("UV_OVERLAP")
            if int(metrics.get("chart_count") or 0) > int(profile.uv_max_charts):
                codes.append("UV_CHART_BUDGET")
            if code != 0 or not output.exists() or data.get("status") != "passed" or codes:
                return StageResult("failed", gates=gates, failure_codes=sorted(set(codes)),
                                   detail=f"UV worker exit {code}: {out[-1000:]}")
            return StageResult("passed", outputs={"uv_mesh": output, "uv_report": report}, gates=gates)

        return pipeline.execute("UV", [lod0], runner)

    # ---------------------------------------------------------------- BAKE
    def bake():
        high = _output(pipeline, "CLEAN", "clean")
        low = _output(pipeline, "UV", "uv_mesh")

        def runner(overrides):
            stage = pipeline.stage_dir("BAKE") / "candidate"
            maps = stage / "maps"
            report = stage / "bake_report.json"
            code, out = _blender(
                pipeline, b("final_pipeline_bake.py"),
                "--high", high, "--low", low, "--output-dir", maps,
                "--report", report, "--resolution", str(resolution),
                "--samples", "48", "--suffix", suffix,
            )
            data = _json(report)
            outputs = {"bake_report": report}
            missing = []
            for key in ("normal", "ao", "cavity", "material_id"):
                source = _find(maps, key, suffix) or _find(maps, key)
                if not source:
                    missing.append(key)
                    continue
                destination = stage / f"{asset_id}_{key}_{suffix}.png"
                _copy_named(source, destination)
                outputs[key] = destination
            gates = {
                "passed": bool(data.get("passed")),
                "high_component_count": data.get("high_component_count"),
                "uv_layer_source": data.get("uv_layer_source"),
                "maps": data.get("maps"),
                "failures": data.get("failures"),
            }
            codes = []
            if int(data.get("high_component_count") or 0) > 500:
                codes.append("MATERIAL_ID_NOISE")
            if code != 0 or missing or not data.get("passed"):
                return StageResult("failed", gates=gates, failure_codes=codes,
                                   detail=f"bake exit {code}; missing={missing}; {out[-1000:]}")
            return StageResult("passed", outputs=outputs, gates=gates)

        return pipeline.execute("BAKE", [high, low], runner)

    # ---------------------------------------------------------------- TEXTURE
    def texture():
        matte = _output(pipeline, "INGEST", "matte")
        mesh = _output(pipeline, "UV", "uv_mesh")
        normal = _output(pipeline, "BAKE", "normal")
        ao = _output(pipeline, "BAKE", "ao")
        cavity = _output(pipeline, "BAKE", "cavity")
        material_id = _output(pipeline, "BAKE", "material_id")

        def runner(overrides):
            stage = pipeline.stage_dir("TEXTURE") / "candidate"
            views = stage / "views"
            projection = stage / "projection"
            npz = stage / "projection.npz"
            view_report = stage / "view_report.json"
            code, out = pipeline.run([
                pipeline.python, w("shaman_texture_views_oriented.py"),
                "--mesh", mesh, "--source", matte, "--output-npz", npz,
                "--views-dir", views, "--report", view_report,
            ])
            if code != 0 or not npz.exists():
                return StageResult("failed", detail=f"view builder exit {code}: {out[-1000:]}")

            # Everything downstream that has a notion of "front" reads it from here. The projection
            # hemisphere and the review cameras disagreeing is not a cosmetic problem: it produced a
            # correctly textured model whose renders showed the back under the name "front".
            front_direction = json.loads(Path(view_report).read_text(encoding="utf-8"))["front_direction"]

            orientation_truth = stage / "orientation_truth.json"
            code, out = pipeline.run([
                pipeline.python, w("atlas_orientation_truth.py"), "--npz", npz,
                "--view", views / "front.png", "--view-name", "front",
                "--output", orientation_truth,
            ])
            if code != 0:
                return StageResult("failed", detail=f"orientation truth exit {code}: {out[-800:]}")

            projection_report = stage / "projection_report.json"
            code, out = pipeline.run([
                pipeline.python, w("raster_project.py"), "--npz", npz,
                "--views-dir", views, "--view-metadata", views / "view_metadata.json",
                "--output-dir", projection, "--atlas-size", str(resolution),
                "--progress", projection / "raster-progress.json",
                "--report", projection_report,
            ])
            if code != 0 or not (projection / "basecolor.png").exists():
                return StageResult("failed", detail=f"projection exit {code}: {out[-1000:]}")

            base_oriented = stage / "basecolor_oriented.png"
            coverage_oriented = stage / "coverage_oriented.png"
            for source, destination in ((projection / "basecolor.png", base_oriented),
                                        (projection / "debug_coverage.png", coverage_oriented)):
                code, out = pipeline.run([
                    pipeline.python, w("atlas_to_gltf_convention.py"),
                    "--input", source, "--output", destination,
                ])
                if code != 0:
                    return StageResult("failed", failure_codes=["UV_ROW_ORIENTATION_MISMATCH"],
                                       detail=f"atlas conversion exit {code}: {out[-800:]}")

            repainted = stage / "basecolor_repainted.png"
            repaint_report = stage / "repaint_report.json"
            neighbours = int(overrides.get("donor_neighbours", 16))
            code, out = pipeline.run([
                pipeline.python, w("pipeline_prior_repaint.py"), "--mesh", mesh,
                "--basecolor", base_oriented, "--coverage", coverage_oriented,
                "--output", repainted, "--report", repaint_report,
                "--neighbours", str(neighbours),
            ])
            if code != 0:
                return StageResult("failed", failure_codes=["FLAT_NEUTRAL_ATLAS_REGIONS"],
                                   detail=f"prior repaint exit {code}: {out[-800:]}")

            basecolor = stage / f"{asset_id}_basecolor_{suffix}.png"
            detail_report = stage / "detail_report.json"
            strength = float(overrides.get("detail_strength", 0.34))
            code, out = pipeline.run([
                pipeline.python, w("atlas_detail_fill.py"), "--basecolor", repainted,
                "--coverage", coverage_oriented, "--cavity", cavity, "--ao", ao,
                "--output", basecolor, "--report", detail_report,
                "--strength", str(strength),
            ])
            if code != 0:
                return StageResult("failed", failure_codes=["UNFINISHED_SYNTHESIS"],
                                   detail=f"detail fill exit {code}: {out[-800:]}")

            orm = stage / f"{asset_id}_orm_{suffix}.png"
            class_map = stage / "material_class.png"
            orm_report = stage / "orm_report.json"
            code, out = pipeline.run([
                pipeline.python, w("pipeline_orm.py"), "--basecolor", basecolor,
                "--ao", ao, "--cavity", cavity, "--material-id", material_id,
                "--island", coverage_oriented, "--output", orm,
                "--class-map", class_map, "--report", orm_report,
            ])
            if code != 0:
                return StageResult("failed", failure_codes=["PLASTIC_ROUGHNESS"],
                                   detail=f"ORM exit {code}: {out[-800:]}")

            region_report = stage / "region_report.json"
            pipeline.run([
                pipeline.python, w("pipeline_region_diagnostic.py"), "--mesh", mesh,
                "--npz", npz, "--basecolor", basecolor, "--coverage", coverage_oriented,
                "--class-map", class_map, "--report", region_report, "--region", "all",
            ])

            glb = stage / f"{asset_id}_textured_lod0.glb"
            blend = stage / f"{asset_id}_textured_lod0.blend"
            material_manifest = stage / "material_manifest.json"
            code, out = _blender(
                pipeline, b("shaman_texture_export.py"), "--mesh", mesh,
                "--basecolor", basecolor, "--normal", normal, "--orm", orm,
                "--atlas-size", str(resolution),
                "--output-glb", glb, "--output-blend", blend,
                "--manifest", material_manifest,
            )
            if code != 0 or not glb.exists():
                return StageResult("failed", detail=f"texture export exit {code}: {out[-1000:]}")

            render_dir = stage / "renders"
            review_report = stage / "review_report.json"
            code, out = _blender(
                pipeline, b("shaman_texture_review.py"), "--glb", glb,
                "--output-dir", render_dir, "--report", review_report,
                "--resolution", "1024", "--samples", "24",
                "--front-direction", front_direction,
            )
            if code != 0:
                return StageResult("failed", detail=f"review renderer exit {code}: {out[-1000:]}")

            outputs = {
                "basecolor": basecolor, "normal": normal, "orm": orm,
                "coverage": coverage_oriented, "material_id": material_id,
                "textured_glb": glb, "textured_blend": blend,
                "projection_report": projection_report, "orientation_truth": orientation_truth,
                "region_report": region_report, "orm_report": orm_report,
                "material_manifest": material_manifest, "review_report": review_report,
                "view_report": view_report,
            }
            review = _json(review_report)
            for name, entry in (review.get("views") or {}).items():
                path = Path(entry.get("path", ""))
                if path.exists():
                    outputs[f"render_{name}"] = path
            projection_data = _json(projection_report)
            gates = {
                "observed_percent": projection_data.get("observed_semantic_coverage_percent"),
                "synthesized_percent": projection_data.get("synthesized_surface_coverage_percent"),
                "final_filled_uv_percent": projection_data.get("final_filled_uv_percent"),
                "material_slot_count": _json(material_manifest).get("material_slot_count"),
                "review_views": sorted((review.get("views") or {}).keys()),
            }
            return StageResult("passed", outputs=outputs, gates=gates)

        return pipeline.execute("TEXTURE", [matte, mesh, normal, ao, cavity, material_id], runner)

    # ---------------------------------------------------------------- TEXTURE_QA
    def texture_qa():
        glb = _output(pipeline, "TEXTURE", "textured_glb")
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
            render_paths = [Path(v["path"]) for k, v in (texture_receipt.get("outputs") or {}).items()
                            if k.startswith("render_") and v.get("path")]
            render_dir = render_paths[0].parent if render_paths else glb.parent
            bake_report = _json(_output(pipeline, "BAKE", "bake_report"))
            command = [
                pipeline.python, w("visual_evaluator.py"), "--render-dir", render_dir,
                "--report", report, "--profile-json", root / "profile.json",
                "--source-image", source_image, "--basecolor", basecolor,
                "--orm", orm, "--material-id", material_id, "--coverage", coverage,
                "--orientation-truth", orientation_truth, "--region-report", region_report,
                "--uv-report", uv_report, "--geometry-report", geometry_report,
                "--view-report", _output(pipeline, "TEXTURE", "view_report"),
                "--review-report", _output(pipeline, "TEXTURE", "review_report"),
                "--material-id-components", str(bake_report.get("high_component_count", 0)),
            ]
            code, out = pipeline.run(command)
            data = _json(report)
            if not data:
                return StageResult("failed", detail=f"visual evaluator exit {code}: {out[-1200:]}")
            gates = data.get("measured", {})
            if code != 0 or not data.get("passed"):
                return StageResult("failed", gates=gates,
                                   failure_codes=data.get("blocking_codes", []),
                                   detail=f"advisory={data.get('advisory_codes', [])}")
            return StageResult("passed", outputs={"visual_evaluation": report}, gates=gates)

        return pipeline.execute("TEXTURE_QA", [glb, basecolor, orm, material_id, coverage], runner)

    # ---------------------------------------------------------------- PARTS
    def parts():
        textured = _output(pipeline, "TEXTURE", "textured_glb")

        def runner(overrides):
            stage = pipeline.stage_dir("PARTS") / "candidate"
            report = stage / "parts_report.json"
            code, out = _blender(
                pipeline, b("final_pipeline_components.py"), "--input", textured,
                "--report", report, "--analyze-only",
            )
            data = _json(report)
            if code != 0 or not data:
                return StageResult("failed", detail=f"parts analysis exit {code}: {out[-1000:]}")
            main_fraction = float(data.get("main_component_face_fraction") or 0.0)
            gates = {"component_count": data.get("component_count"),
                     "main_component_face_fraction": main_fraction,
                     "separate_props_required": bool(profile.separate_props)}
            if profile.separate_props and main_fraction > 0.98:
                return StageResult("failed", gates=gates,
                                   failure_codes=["PARTS_SEMANTIC_CUT_REQUIRED"],
                                   detail="held prop is fused into the dominant component; loose-part split is unsafe")
            candidate = stage / f"{asset_id}_parts.glb"
            _copy_named(textured, candidate)
            return StageResult("passed", outputs={"parts_glb": candidate, "parts_report": report}, gates=gates)

        return pipeline.execute("PARTS", [textured], runner)

    # ---------------------------------------------------------------- RIG_READINESS
    def rig_readiness():
        mesh = _output(pipeline, "PARTS", "parts_glb")

        def runner(overrides):
            stage = pipeline.stage_dir("RIG_READINESS") / "candidate"
            report = stage / "rig_readiness.json"
            code, out = pipeline.run([
                pipeline.python, w("pipeline_rig_readiness.py"), "--mesh", mesh,
                "--profile", profile.name, "--report", report,
            ])
            data = _json(report)
            if not data:
                return StageResult("failed", detail=f"rig-readiness produced no report: {out[-1000:]}")
            if not profile.rig_required:
                return StageResult("passed", outputs={"rig_readiness": report},
                                   gates={**data, "rig_not_required": True})
            if code != 0 or not data.get("ready"):
                return StageResult("failed", gates=data,
                                   failure_codes=data.get("failure_codes", ["RIG_READINESS_FAILED"]),
                                   detail="automatic rig-readiness gate failed closed")
            return StageResult("passed", outputs={"rig_readiness": report}, gates=data)

        return pipeline.execute("RIG_READINESS", [mesh], runner)

    # ---------------------------------------------------------------- RIG
    def rig():
        mesh = _output(pipeline, "PARTS", "parts_glb")
        readiness = _output(pipeline, "RIG_READINESS", "rig_readiness")

        def runner(overrides):
            stage = pipeline.stage_dir("RIG") / "candidate"
            if not profile.rig_required:
                passthrough = stage / f"{asset_id}_static.glb"
                _copy_named(mesh, passthrough)
                return StageResult("passed", outputs={"rigged_glb": passthrough},
                                   gates={"rig_required": False})
            blend = stage / f"{asset_id}_rigged.blend"
            glb = stage / f"{asset_id}_rigged.glb"
            report = stage / "rig_report.json"
            code, out = _blender(
                pipeline, b("pipeline_auto_rig.py"), "--input", mesh,
                "--output-glb", glb, "--output-blend", blend,
                "--report", report, "--profile", profile.name,
            )
            data = _json(report)
            if code != 0 or not glb.exists() or not data.get("passed"):
                return StageResult("failed", gates=data,
                                   failure_codes=["AUTO_RIG_FAILED"],
                                   detail=f"auto-rig exit {code}: {out[-1200:]}")
            return StageResult("passed", outputs={"rigged_glb": glb, "rigged_blend": blend,
                                                   "rig_report": report}, gates=data)

        return pipeline.execute("RIG", [mesh, readiness], runner)

    # ---------------------------------------------------------------- EXPORT
    def export():
        rigged = _output(pipeline, "RIG", "rigged_glb")

        def runner(overrides):
            stage = pipeline.stage_dir("EXPORT") / "candidate"
            final_glb = stage / f"{asset_id}_final.glb"
            report = stage / "export_report.json"
            code, out = _blender(
                pipeline, b("pipeline_export_validate.py"), "--input", rigged,
                "--output", final_glb, "--report", report,
            )
            data = _json(report)
            if code != 0 or not final_glb.exists() or not data.get("passed"):
                return StageResult("failed", gates=data, failure_codes=["EXPORT_VALIDATION_FAILED"],
                                   detail=f"export validation exit {code}: {out[-1000:]}")
            return StageResult("passed", outputs={"final_glb": final_glb,
                                                   "export_report": report}, gates=data)

        return pipeline.execute("EXPORT", [rigged], runner)

    return {
        "LOD": lod, "UV": uv, "BAKE": bake, "TEXTURE": texture,
        "TEXTURE_QA": texture_qa, "PARTS": parts,
        "RIG_READINESS": rig_readiness, "RIG": rig, "EXPORT": export,
    }
