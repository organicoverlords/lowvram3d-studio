"""Production stage adapters for Pipeline V2.

These are thin orchestration layers over the V1 workers proven during the shaman run.  They add
immutable-input receipts, bounded retries and generic artifact names; they do not reimplement UV,
baking or projection algorithms.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from run_asset_pipeline import REPO_ROOT, StageResult, sha256
from lowvram3d.thin_feature_anchors import (
    anchor_receipt_sha256,
    discover_thin_feature_anchors,
    serialize_anchor_receipt,
)


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
    suffix = "4k" if resolution >= 4096 else "2k"

    def w(name: str) -> Path:
        return REPO_ROOT / "workers" / name

    def b(name: str) -> Path:
        return REPO_ROOT / "blender" / name

    # ---------------------------------------------------------------- LOD
    def lod():
        clean = _output(pipeline, "CLEAN", "clean")

        def runner(overrides):
            stage = pipeline.stage_dir("LOD") / "candidate"
            outdir = stage / "lods"
            report = stage / "lod_report.json"
            clean_hash_before = sha256(clean)
            anchor_receipt_path = stage / "anchor_receipt.json"
            try:
                anchor_receipt = discover_thin_feature_anchors(clean, profile=profile)
                anchor_receipt_path.write_bytes(serialize_anchor_receipt(anchor_receipt))
            except Exception as exc:
                return StageResult("failed", failure_codes=["ANCHOR_RECEIPT_INVALID"],
                                   detail=f"anchor discovery failed: {exc}")
            targets = ",".join(str(v) for v in profile.lod_triangle_targets)
            code, out = _blender(
                pipeline, b("final_pipeline_lods.py"),
                "--input", clean, "--output-dir", outdir, "--report", report,
                "--targets", targets, "--anchor-receipt", anchor_receipt_path,
            )
            data = _json(report)
            if not data.get("lods"):
                return StageResult("failed", detail=f"LOD worker exit {code}: {out[-1200:]}")

            clean_hash_after = sha256(clean)
            outputs = {"lod_report": report, "anchor_receipt": anchor_receipt_path}
            gates = {
                "targets": list(profile.lod_triangle_targets),
                "clean_master_sha256_before": clean_hash_before,
                "clean_master_sha256_after": clean_hash_after,
                "clean_master_unchanged": clean_hash_before == clean_hash_after,
                "anchor_receipt_sha256": anchor_receipt_sha256(anchor_receipt),
                "anchor_ids": [a["anchor_id"] for a in anchor_receipt.get("anchors", [])],
                "lods": [],
            }
            failures = []
            candidates = []
            if code != 0:
                failures.append(f"LOD worker exit {code}: {out[-600:]}")
            for index, target in enumerate(profile.lod_triangle_targets):
                source = outdir / f"shaman_lod{index}.glb"
                if not source.exists():
                    failures.append(f"LOD{index} missing")
                    continue
                destination = stage / f"{asset_id}_lod{index}.glb"
                row = next((r for r in data["lods"] if int(r.get("lod", -1)) == index), {})
                achieved = int(row.get("achieved_triangles", 0))
                survival = row.get("anchor_survival") or {}
                anchor_gate = {
                    "anchor_ids": row.get("anchor_ids", []),
                    "present_ids": survival.get("present_ids", []),
                    "missing_ids": survival.get("missing_ids", []),
                    "anchors": survival.get("anchors", []),
                }
                gates["lods"].append({"lod": index, "target": target, "triangles": achieved,
                                      "input_sha256": row.get("input_sha256"),
                                      "output_sha256": row.get("output_sha256"),
                                      "sha256": sha256(source), "anchors": anchor_gate,
                                      "passed": bool(row.get("passed", False)),
                                      "failure_reasons": row.get("failure_reasons", [])})
                candidates.append((index, source, destination))
                if achieved <= 0 or achieved > max(target * 1.20, target + 5000):
                    failures.append(f"LOD{index} achieved {achieved}, target {target}")
                if row.get("input_sha256") != clean_hash_before:
                    failures.append(f"LOD{index} input hash does not match clean master")
                if row.get("output_sha256") != sha256(source):
                    failures.append(f"LOD{index} output hash mismatch")
                expected_anchor_ids = gates["anchor_ids"]
                row_anchor_ids = sorted(row.get("anchor_ids", []))
                if row_anchor_ids != expected_anchor_ids:
                    failures.append(f"LOD{index} anchor ID set does not match receipt")
                records = survival.get("anchors") or []
                record_ids = sorted(
                    record.get("anchor_id")
                    for record in records
                    if isinstance(record, dict) and isinstance(record.get("anchor_id"), str)
                )
                if record_ids != expected_anchor_ids:
                    failures.append(f"LOD{index} anchor survival records are incomplete")
                if any(
                    not isinstance(record, dict)
                    or not record.get("present")
                    or record.get("under_floor_views")
                    for record in records
                ):
                    failures.append(f"LOD{index} anchor survival record is under floor")
                if not row.get("passed", False):
                    failures.append(
                        f"LOD{index} receipt failed: {row.get('failure_reasons', [])}"
                    )
                if not survival.get("all_present"):
                    failures.append(
                        f"LOD{index} anchors missing/under floor: "
                        f"{survival.get('missing_ids', [])}"
                    )
            if not gates["clean_master_unchanged"]:
                failures.append("clean master changed during LOD generation")
            if failures:
                codes = []
                if any("anchor" in failure.lower() or "receipt failed" in failure.lower() for failure in failures):
                    codes.append("LOD_ANCHOR_GATE_FAILED")
                if any("clean master" in failure.lower() for failure in failures):
                    codes.append("CLEAN_MASTER_CHANGED")
                return StageResult("failed", gates=gates, failure_codes=codes, detail="; ".join(failures))
            # Promote no individual target until every target has passed all gates.
            for index, source, destination in candidates:
                _copy_named(source, destination)
                outputs[f"lod{index}"] = destination
            return StageResult("passed", outputs=outputs, gates=gates)

        return pipeline.execute("LOD", [clean], runner)

    # ---------------------------------------------------------------- UV
    def uv():
        lod0 = _output(pipeline, "LOD", "lod0")
        lod_receipt = pipeline.read_receipt("LOD") or {}
        lod_outputs = lod_receipt.get("outputs") or {}
        anchor_entry = lod_outputs.get("anchor_receipt") or {}
        anchor_receipt = Path(anchor_entry["path"]) if anchor_entry.get("path") else None
        expected_source_sha256 = (lod_receipt.get("gates") or {}).get("clean_master_sha256_before")

        def runner(overrides):
            stage = pipeline.stage_dir("UV") / "candidate"
            output = stage / f"{asset_id}_lod0_uv.glb"
            report = stage / "uv_report.json"
            padding = 12 if overrides.get("route") == "xatlas" else 8
            code, out = pipeline.run([
                pipeline.python, w("uv_xatlas_repair.py"),
                "--input", lod0, "--output", output, "--report", report,
                "--resolution", str(resolution), "--padding", str(padding),
                "--overlap-timeout", "1200", "--max-candidate-pairs", "10000000",
                "--max-overlap-texels", "1.0",
                "--anchor-receipt", anchor_receipt or "",
                "--expected-source-sha256", expected_source_sha256 or "",
            ])
            data = _json(report)
            exact = data.get("exact_overlap") or {}
            gates = {
                "gate_passed": bool(data.get("gate_passed")),
                "chart_count": data.get("chart_count"),
                "profile_chart_limit": profile.uv_max_charts,
                "atlas_utilization": data.get("atlas_utilization"),
                "stretch_p95": data.get("stretch_p95"),
                "candidate_pair_count": exact.get("candidate_pair_count"),
                "tested_pair_count": exact.get("tested_pair_count"),
                "timed_out": exact.get("timed_out"),
                "positive_overlap_pairs": exact.get("positive_overlap_pair_count"),
                "positive_overlap_texels": exact.get("positive_overlap_total_texels_equivalent"),
                "degenerate_uv_triangles": exact.get("degenerate_uv_triangle_count"),
                "out_of_bounds_triangles": exact.get("out_of_bounds_triangle_count"),
                "provenance": data.get("provenance"),
            }
            codes = []
            codes.extend(str(code) for code in (data.get("failure_codes") or []))
            if exact.get("timed_out") or not exact.get("success") or int(exact.get("tested_pair_count") or 0) <= 0:
                codes.append("UV_OVERLAP")
            if int(exact.get("degenerate_uv_triangle_count") or 0) > 0:
                codes.append("UV_DEGENERATE")
            if float(exact.get("positive_overlap_total_texels_equivalent") or 0.0) > 1.0:
                codes.append("UV_OVERLAP")
            if int(data.get("chart_count") or 0) > int(profile.uv_max_charts):
                codes.append("UV_CHART_BUDGET")
            provenance = data.get("provenance") or {}
            if not anchor_receipt:
                codes.append("ANCHOR_RECEIPT_MISSING")
            elif not provenance.get("geometry_unchanged", False):
                codes.append("GEOMETRY_MUTATION")
            elif provenance.get("anchor_receipt_sha256") != (lod_receipt.get("gates") or {}).get("anchor_receipt_sha256"):
                codes.append("ANCHOR_RECEIPT_SOURCE_MISMATCH")
            elif sorted(provenance.get("anchor_ids") or []) != sorted((lod_receipt.get("gates") or {}).get("anchor_ids") or []):
                codes.append("ANCHOR_SET_MISMATCH")
            if code != 0 or not output.exists() or not data.get("gate_passed") or codes:
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
        lod_receipt = pipeline.read_receipt("LOD") or {}
        lod_anchor = (lod_receipt.get("outputs") or {}).get("anchor_receipt") or {}
        lod_anchor_path = Path(lod_anchor["path"]) if lod_anchor.get("path") else None
        lod_source_hash = (lod_receipt.get("gates") or {}).get("clean_master_sha256_before")
        uv_receipt = pipeline.read_receipt("UV") or {}
        uv_geometry_hash = ((uv_receipt.get("gates") or {}).get("provenance") or {}).get("output_geometry_sha256")

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
                "--anchor-receipt", lod_anchor_path or "",
                "--expected-source-sha256", lod_source_hash or "",
                "--expected-input-geometry-sha256", uv_geometry_hash or "",
                "--require-anchor-provenance",
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
                "--output-glb", glb, "--output-blend", blend,
                "--manifest", material_manifest,
                "--anchor-receipt", lod_anchor_path or "",
                "--expected-source-sha256", lod_source_hash or "",
            )
            if code != 0 or not glb.exists():
                return StageResult("failed", detail=f"texture export exit {code}: {out[-1000:]}")
            material_data = _json(material_manifest)
            export_provenance = material_data.get("provenance") or {}
            expected_gates = lod_receipt.get("gates") or {}
            export_failures = []
            if not lod_anchor_path:
                export_failures.append("ANCHOR_RECEIPT_MISSING")
            if export_provenance.get("anchor_receipt_sha256") != expected_gates.get("anchor_receipt_sha256"):
                export_failures.append("ANCHOR_RECEIPT_SOURCE_MISMATCH")
            if sorted(export_provenance.get("anchor_ids") or []) != sorted(expected_gates.get("anchor_ids") or []):
                export_failures.append("ANCHOR_SET_MISMATCH")
            if not export_provenance.get("geometry_unchanged"):
                export_failures.append("GEOMETRY_MUTATION")
            if export_failures or not material_data.get("success"):
                return StageResult("failed", gates={"provenance": export_provenance},
                                   failure_codes=sorted(set(export_failures or ["TEXTURE_EXPORT_FAILED"])),
                                   detail="textured GLB provenance gate failed")

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
                "provenance": export_provenance,
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
        lod_receipt = pipeline.read_receipt("LOD") or {}
        lod_anchor = (lod_receipt.get("outputs") or {}).get("anchor_receipt") or {}
        lod_anchor_path = Path(lod_anchor["path"]) if lod_anchor.get("path") else None
        lod_source_hash = (lod_receipt.get("gates") or {}).get("clean_master_sha256_before")

        def runner(overrides):
            stage = pipeline.stage_dir("EXPORT") / "candidate"
            final_glb = stage / f"{asset_id}_final.glb"
            report = stage / "export_report.json"
            code, out = _blender(
                pipeline, b("pipeline_export_validate.py"), "--input", rigged,
                "--output", final_glb, "--report", report,
                "--anchor-receipt", lod_anchor_path or "",
                "--expected-source-sha256", lod_source_hash or "",
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
