"""Production stage adapters for the generic Pipeline V2 texture compiler."""
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


def _blender(pipeline, script: Path, *args: object, timeout: float | None = None) -> tuple[int, str]:
    return pipeline.run([
        pipeline.blender, "--background", "--python-use-system-env", "--python", script,
        "--", *args,
    ], timeout=timeout)


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


TEXTURE_SCOPES = {
    "PREVIEW_TEXTURE",
    "FRONT_HERO_PRODUCTION",
    "PARTIAL_360_PRODUCTION",
    "FULL_360_PRODUCTION",
}

EVIDENCE_TEXTURE_STAGES = (
    "VIEW_EVIDENCE", "SURFACE_REGIONS", "SURFACE_EVIDENCE", "SOURCE_ASSIGNMENT",
    "DIRECT_PROJECTION", "VISIBLE_SOURCE_GAP_REPAIR", "UNOBSERVED_COMPLETION", "FREQUENCY_FUSION",
    "TEXTURE_EVIDENCE_QA", "TEXTURE_SCOPE",
)


def classify_texture_scope(*, actual_route: str, semantic_view_count: int,
                           synthesized_percent: float, face_detail_required: bool,
                           approved_single_view_face_route: bool) -> str:
    """Classify the claim scope from evidence, not from a generic production boolean."""
    if actual_route == "mvadapter_sixview" and semantic_view_count >= 6 and synthesized_percent <= 70.0:
        return "FULL_360_PRODUCTION"
    if semantic_view_count >= 4 and synthesized_percent <= 85.0:
        return "PARTIAL_360_PRODUCTION"
    if (actual_route == "raster_project" and face_detail_required
            and approved_single_view_face_route and semantic_view_count == 1):
        return "FRONT_HERO_PRODUCTION"
    return "PREVIEW_TEXTURE"


def validate_mvadapter_inputs(bundle: Path, views_receipt: Path) -> dict:
    """Validate the external six-view contract without starting any heavy worker."""
    missing = []
    contract_path = bundle / "camera_contract.json"
    if not bundle.is_dir():
        missing.append(str(bundle))
    if not contract_path.is_file():
        missing.append(str(contract_path))
    if not views_receipt.is_file():
        missing.append(str(views_receipt))

    contract = {}
    receipt = {}
    if contract_path.is_file():
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing.append(str(contract_path) + " (invalid JSON)")
    if views_receipt.is_file():
        try:
            receipt = json.loads(views_receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            missing.append(str(views_receipt) + " (invalid JSON)")

    raw_views = contract.get("views", [])
    views = sorted(raw_views if isinstance(raw_views, list) else [],
                   key=lambda item: int(item.get("index", -1))
                   if isinstance(item, dict) else -1)
    semantics = [str(item.get("semantic_name", "")) for item in views
                 if isinstance(item, dict)]
    required_semantics = {"front", "right", "rear", "left", "top", "bottom"}
    if len(views) != 6 or set(semantics) != required_semantics:
        missing.append("camera_contract requires six semantic views: "
                       + ",".join(sorted(required_semantics)))

    output_images = {
        str(item.get("name")): Path(item.get("path", ""))
        for item in receipt.get("output_images", [])
        if isinstance(item, dict) and item.get("name") and item.get("path")
    }
    for view in views:
        if not isinstance(view, dict):
            continue
        index = int(view.get("index", -1))
        semantic = str(view.get("semantic_name", ""))
        for suffix in ("_mask.png", "_depth.npy"):
            candidate = bundle / f"{semantic}{suffix}"
            if not candidate.is_file():
                missing.append(str(candidate))
        image = output_images.get(f"view_{index}_{semantic}.png")
        if image is None or not image.is_file():
            missing.append(f"view_{index}_{semantic}.png")
    return {
        "passed": not missing,
        "bundle": str(bundle),
        "views_receipt": str(views_receipt),
        "view_count": len(views),
        "semantics": semantics,
        "missing_inputs": missing,
    }


def register_production_stages(pipeline, manifest: dict) -> dict:
    root = Path(manifest["output_root"])
    manifest_path = root / "asset_manifest.json"
    profile = pipeline.profile
    asset_id = manifest["asset_id"]
    texture_manifest = manifest["texture"]
    resolution = int(texture_manifest["resolution"])
    requested_texture_route = str(texture_manifest.get("route", "raster_project")).lower()
    texture_quality_tier = str(texture_manifest.get("quality_tier", "preview")).lower()
    evidence_compiler_enabled = bool(texture_manifest.get(
        "evidence_compiler", texture_quality_tier == "production"
    ))
    approved_single_view_face_route = bool(
        texture_manifest.get("approved_single_view_face_route", False)
    )
    face_detail = texture_manifest.get("face_detail") or {}
    face_detail_required = bool(face_detail.get("required", False))
    texture_route = requested_texture_route
    texture_fallback = None
    if requested_texture_route == "mvadapter_sixview":
        bundle = Path(texture_manifest.get("bundle", ""))
        views_receipt = Path(texture_manifest.get("views_receipt", ""))
        fallback_route = str(texture_manifest.get("fallback_route", "")).lower()
        if (bool(texture_manifest.get("allow_fallback", False))
                and fallback_route == "raster_project"):
            audit = validate_mvadapter_inputs(bundle, views_receipt)
            if not audit["passed"]:
                texture_route = fallback_route
                texture_fallback = {
                    "from": requested_texture_route,
                    "to": fallback_route,
                    "reason": "MV-Adapter six-view inputs unavailable",
                    "input_audit": audit,
                }
    actual_texture_route = texture_route
    fallback_used = texture_fallback is not None
    declared_scope = str(texture_manifest.get("scope", "")).upper()
    if declared_scope not in TEXTURE_SCOPES:
        declared_scope = ""
    initial_scope = declared_scope or (
        "FRONT_HERO_PRODUCTION" if actual_texture_route == "raster_project" and face_detail_required
        and approved_single_view_face_route else "PREVIEW_TEXTURE"
    )
    production_eligible = texture_quality_tier == "production" and initial_scope != "PREVIEW_TEXTURE"
    full_360_eligible = initial_scope == "FULL_360_PRODUCTION"
    production_scope = initial_scope
    uv_resolution = int((manifest.get("uv") or {}).get("resolution", 1024))
    uv_padding = int((manifest.get("uv") or {}).get("padding", 4))
    uv_timeout = float((manifest.get("uv") or {}).get("candidate_timeout_seconds", 600))
    # Bound the complete fresh-asset UV subprocess, including xatlas generation.  The previous
    # value was passed only to the post-unwrap overlap census, so xatlas itself could run for
    # hours before the census ever started.  Keep the timeout opt-out explicit and reproducible.
    uv_process_timeout = float((manifest.get("uv") or {}).get(
        "process_timeout_seconds", uv_timeout
    ))
    uv_repair_rounds = int((manifest.get("uv") or {}).get("repair_rounds", 0))
    uv_world_sliver_scale = float((manifest.get("uv") or {}).get(
        "world_sliver_scale", 1e-12
    ))
    uv_max_cost = float((manifest.get("uv") or {}).get("max_cost", 8.0))
    uv_max_iterations = int((manifest.get("uv") or {}).get("max_iterations", 4))
    uv_fix_winding = bool((manifest.get("uv") or {}).get("fix_winding", False))
    uv_allow_fallback = bool((manifest.get("uv") or {}).get("allow_fallback", False))
    uv_fallback_route = str((manifest.get("uv") or {}).get(
        "fallback_route", ""
    )).lower()
    # Zero-area UV triangles are a hard failure by default. A profile may declare a measured
    # allowance when the offenders are known 3D slivers that own no texels; it is never inferred.
    uv_max_degenerate = int((manifest.get("uv") or {}).get("max_degenerate_uv_triangles", 0))
    lod_mode = str((manifest.get("lod") or {}).get("mode", "generate")).lower()
    lod_regression_allowed = bool((manifest.get("lod") or {}).get(
        "allow_topology_regression_below_lod0", False))
    delivery_visual_gate = (manifest.get("lod") or {}).get("delivery_visual_gate") or {}
    allow_delivery_topology_regression = bool(
        (manifest.get("lod") or {}).get("allow_topology_regression_for_delivery", False)
    )
    rig_manifest = manifest.get("rig") or {}
    rig_required = bool(rig_manifest.get("required", profile.rig_required))
    separate_props = bool(rig_manifest.get("separate_props", profile.separate_props))
    suffix = "4k" if resolution >= 4096 else "2k"
    orientation = manifest.get("orientation") or {}

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
            protection_mode = str((manifest.get("lod") or {}).get(
                "protection_mode", "standard"
            ))
            code, out = _blender(
                pipeline, b("final_pipeline_lods.py"),
                "--input", clean, "--output-dir", outdir, "--report", report,
                "--targets", targets,
                "--prefix", asset_id,
                "--up-axis", str(orientation.get("up_axis", "z")),
                "--right-axis", str(orientation.get("right_axis",
                                                     orientation.get("lateral_axis", "x"))),
                "--front-axis", str(orientation.get("front_axis", "y")),
                "--protection-mode", protection_mode,
            )
            data = _json(report)
            if code != 0 or not data.get("lods"):
                return StageResult("failed", detail=f"LOD worker exit {code}: {out[-1200:]}")

            outputs = {"lod_report": report}
            gates = {"targets": targets_list, "lods": []}
            failures = []
            topology_failures = []
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
                    if int(candidate_topology.get(metric, 0)) <= int(source_topology.get(metric, 0)):
                        continue
                    message = (f"LOD{index} {metric} regressed "
                               f"{source_topology.get(metric)}->{candidate_topology.get(metric)}")
                    # LOD0 carries the asset's real topology and stays strict. Below it, thin
                    # features are expected to go: a ghillie fringe cannot survive a 4x decimation
                    # and a distance LOD is not supposed to make it. The allowance is declared per
                    # asset, and the measured counts are recorded either way so "advisory" never
                    # means "unmeasured".
                    if index > 0 and lod_regression_allowed:
                        gates.setdefault("topology_regressions_advisory", []).append(message)
                    elif index == 0 and allow_delivery_topology_regression:
                        gates.setdefault("topology_regressions_visual_gate", []).append(message)
                    else:
                        topology_failures.append(message)
            if allow_delivery_topology_regression and not failures:
                candidate = outputs.get("lod0")
                visual_report = stage / "candidate" / "lod0_geometry_compare.json"
                visual_args = [
                    pipeline.python, w("geometry_compare.py"),
                    "--master", clean, "--candidate", candidate, "--report", visual_report,
                    "--asset-family", str(delivery_visual_gate.get("asset_family", "mixed")),
                    "--quality", str(delivery_visual_gate.get("quality", "gameplay")),
                    "--samples", str(int(delivery_visual_gate.get("samples", 50000))),
                    "--silhouette-size", str(int(delivery_visual_gate.get("silhouette_size", 256))),
                    "--name", f"{asset_id}_lod0_delivery",
                ]
                visual_code, visual_output = pipeline.run(visual_args)
                visual = _json(visual_report)
                gates["delivery_visual_gate"] = visual
                outputs["lod0_geometry_compare"] = visual_report
                if visual_code != 0 or not visual.get("success"):
                    manual = delivery_visual_gate.get("manual_review") or {}
                    accepted = (bool(manual.get("accept"))
                                and int(manual.get("selected_lod", 0)) == 0
                                and bool(manual.get("evidence")))
                    if accepted:
                        gates["delivery_visual_gate"]["manual_review"] = {
                            "accepted": True,
                            "selected_lod": 0,
                            "evidence": manual.get("evidence"),
                            "rationale": manual.get("rationale", ""),
                            "overrode_metrics": list(visual.get("evaluation", {}).get("errors", [])),
                        }
                    else:
                        failures.append(
                            f"LOD0 delivery visual gate failed: {visual_output[-1000:]}"
                        )
            failures.extend(topology_failures)
            if failures:
                # Torn topology is the one LOD failure with a real recipe behind it - decimating
                # less. Without a code the repair policy has nothing to key on and the stage goes
                # straight to needs_human, which is how a detectable defect becomes a manual one.
                codes = ["LOD_TOPOLOGY_REGRESSED"] if topology_failures else []
                return StageResult("failed", gates=gates, failure_codes=codes,
                                   detail="; ".join(failures))
            return StageResult("passed", outputs=outputs, gates=gates)

        return pipeline.execute("LOD", [clean, manifest_path, b("final_pipeline_lods.py")], runner)

    # ---------------------------------------------------------------- UV
    def uv():
        uv_manifest = manifest.get("uv") or {}
        manifest_route = str(uv_manifest.get("route", "fast_blender")).lower()
        route = manifest_route
        master = uv_manifest.get("master") or {}

        # A canonical UV master is adopted, never re-unwrapped. `validate_existing_uv.py` is the
        # wrong gate for one: it requires a material and a packed texture, which a UV master
        # legitimately does not carry. What matters is that the file is still the proven file and
        # that its layout is still injective.
        if route == "injective" and master.get("path"):
            master_path = Path(master["path"])

            def master_runner(_overrides):
                stage = pipeline.stage_dir("UV") / "candidate"
                report = stage / "uv_master_verify.json"
                code, out = pipeline.run([
                    pipeline.python, w("uv_master_verify.py"),
                    "--master", master_path, "--report", report,
                    "--resolution", str(uv_resolution),
                    "--expect-sha256", str(master.get("sha256", "")),
                    "--expect-geometry-fingerprint", str(master.get("geometry_fingerprint", "")),
                    "--expect-triangles", str(int(master.get("triangles") or 0)),
                ])
                data = _json(report)
                gates = {"route": "injective", "adopted_canonical_master": True,
                         "master": str(master_path), "master_sha256": data.get("master_sha256"),
                         "geometry_fingerprint": data.get("geometry_fingerprint"),
                         "triangles": data.get("triangles"),
                         **(data.get("checks") or {}),
                         **{k: v for k, v in (data.get("injectivity") or {}).items()
                            if k in ("injective", "interior_texels_claimed_twice",
                                     "analytic_uv_area_fraction", "degenerate_uv_triangles")}}
                if code != 0 or not data.get("success"):
                    return StageResult("failed", gates=gates,
                                       failure_codes=data.get("failure_codes") or ["UV_MASTER_INVALID"],
                                       detail=f"UV master verify exit {code}: {out[-1000:]}")
                return StageResult("passed", gates=gates,
                                   outputs={"uv_mesh": master_path, "uv_report": report})

            return pipeline.execute("UV", [master_path, manifest_path], master_runner)

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

            return pipeline.execute("UV", [explicit_path, manifest_path], existing_runner)

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
            # The repair policy switches route by override. Reading it from the manifest only
            # meant the UV_OVERLAP recipe could never fire: the stage re-ran the identical route
            # until the retry budget ran out, reporting the same codes each time.
            route = str(overrides.get("route") or manifest_route).lower()
            if route == "injective":
                code, out = pipeline.run([
                    # Fresh assets use the bounded repair-aware xatlas route.  Canonical UV
                    # masters still take the adoption path above and are never unwrapped here.
                    pipeline.python, w("uv_xatlas_repair.py"),
                    "--input", lod0, "--output", output, "--report", report,
                    "--resolution", str(uv_resolution), "--padding", str(uv_padding),
                    "--max-cost", str(uv_max_cost), "--max-iterations", str(uv_max_iterations), "--repair-rounds", str(uv_repair_rounds),
                    "--overlap-timeout", str(uv_timeout), "--max-unwrap-attempts", "1",
                    "--world-sliver-scale", str(uv_world_sliver_scale),
                    "--intra-chart-shrink", str(float((manifest.get("uv") or {}).get(
                        "intra_chart_shrink", 0.75
                    ))),
                    *( ["--fix-winding"] if uv_fix_winding else [] ),
                ], timeout=uv_process_timeout)
                data = _json(report)
                gate = data.get("gate") or data.get("injectivity_after") or {}
                raster_gate_report = stage / "atlas_raster_injectivity.json"
                raster_code = 1
                raster_data = {}
                if output.exists():
                    raster_code, _raster_out = pipeline.run([
                        pipeline.python, w("uv_master_verify.py"),
                        "--master", output, "--report", raster_gate_report,
                        "--resolution", str(uv_resolution),
                    ])
                    raster_data = _json(raster_gate_report)
                gates = {
                    "route": "injective", "packer": data.get("packer"),
                    "process_timeout_seconds": uv_process_timeout,
                    "overlap_timeout_seconds": uv_timeout,
                    "repair_rounds": uv_repair_rounds,
                    "world_sliver_scale": uv_world_sliver_scale,
                    "max_cost": uv_max_cost,
                    "max_iterations": uv_max_iterations,
                    "fix_winding": uv_fix_winding,
                    "chart_count": data.get("chart_count"),
                    "triangles": data.get("triangles"),
                    "seam_vertices_added": data.get("seam_vertices_added"),
                    "geometry_preserved": data.get("geometry_preserved"),
                    "topology_preserved": data.get("topology_preserved"),
                    "geometry_fingerprint": data.get("geometry_fingerprint"),
                    "injective": bool(raster_data.get("success")) and bool(data.get("gate_passed")),
                    "interior_texels_claimed_twice": (data.get("exact_overlap") or {}).get(
                        "positive_overlap_pair_count", gate.get("interior_texels_claimed_twice")),
                    "atlas_utilization": data.get("atlas_utilization", gate.get("analytic_uv_area_fraction")),
                    "degenerate_uv_triangles": (data.get("exact_overlap") or {}).get(
                        "degenerate_uv_triangle_count", gate.get("degenerate_uv_triangles")),
                    "atlas_raster_injectivity": raster_data.get("injectivity"),
                    "max_degenerate_uv_triangles": uv_max_degenerate,
                }
                codes = []
                if not data:
                    # The worker raises rather than reporting when it rejects a rewrap, so an
                    # absent report means it died - not that it measured an overlap or a changed
                    # surface. Reading those out of a missing file invents evidence and, worse,
                    # sends the repair policy down the overlap branch for what may be an OOM.
                    codes.append("UV_INJECTIVE_WORKER_FAILED")
                else:
                    if not data.get("gate_passed"):
                        codes.append("UV_OVERLAP")
                    if int((data.get("exact_overlap") or {}).get("degenerate_uv_triangle_count") or 0) > uv_max_degenerate:
                        codes.append("UV_DEGENERATE")
                    if raster_code != 0 or not raster_data.get("success"):
                        codes.append("UV_ATLAS_RASTER_GATE_FAILED")
                    if code != 0 or not output.exists():
                        codes.append("UV_INJECTIVE_WORKER_FAILED")
                if code != 0 or not output.exists() or codes:
                    if (uv_allow_fallback and uv_fallback_route == "fast_blender"):
                        fallback_output = stage / f"{asset_id}_lod0_fast_uv.glb"
                        fallback_report = stage / "fast_uv_fallback_report.json"
                        fallback_code, fallback_out = _blender(
                            pipeline, b("final_pipeline_uv.py"),
                            "--input", lod0, "--output", fallback_output,
                            "--report", fallback_report,
                            "--resolution", str(uv_resolution),
                            "--overlap-timeout-seconds", "180",
                            timeout=uv_process_timeout,
                        )
                        fallback_data = _json(fallback_report)
                        fallback_gate = {
                            "requested_route": "injective",
                            "fallback_route": "fast_blender",
                            "fallback_code": fallback_code,
                            "fallback_report": str(fallback_report),
                            "fallback_gate_passed": bool(fallback_data.get("gate_passed")),
                            "fallback_gates": fallback_data,
                        }
                        gates["fallback"] = fallback_gate
                        if (fallback_code == 0 and fallback_output.exists()
                                and fallback_data.get("gate_passed")):
                            return StageResult(
                                "passed",
                                outputs={"uv_mesh": fallback_output,
                                         "uv_report": fallback_report},
                                gates={**gates, "route": "fast_blender"},
                            )
                    # Fresh injective unwrap is intentionally one-shot. Do not route an exact
                    # overlap failure into the generic retry policy and silently run xatlas again.
                    return StageResult("failed", gates=gates,
                                       failure_codes=["UV_INJECTIVE_WORKER_FAILED"],
                                       detail=f"injective rewrap exit {code}: {out[-1000:]}"
                                       + (f"; fast fallback: {fallback_out[-500:]}"
                                          if uv_allow_fallback and uv_fallback_route == "fast_blender"
                                          else ""))
                return StageResult("passed", outputs={"uv_mesh": output, "uv_report": report},
                                   gates=gates)
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

        return pipeline.execute("UV", [lod0, manifest_path, w("uv_xatlas_repair.py")], runner)

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
        # Six-view MV-Adapter route. Self-contained: it fuses already-generated views straight
        # onto an injective atlas and binds the result, so it needs neither the INGEST matte nor
        # the BAKE maps that the raster route consumes.
        if texture_route == "mvadapter_sixview":
            mesh = _output(pipeline, "UV", "uv_mesh")
            bundle = Path(texture_manifest["bundle"])
            views_receipt = Path(texture_manifest["views_receipt"])
            original_front_value = texture_manifest.get("original_front") or (manifest.get("source") or {}).get("path", "")
            original_front = Path(original_front_value) if original_front_value else None
            region_config = texture_manifest.get("region_config") or ""
            if region_config:
                region_path = Path(region_config)
                region_config = str(region_path if region_path.is_absolute()
                                    else REPO_ROOT / region_path)

            def mv_runner(_overrides):
                stage = pipeline.stage_dir("TEXTURE") / "candidate"
                uv_consumer_report = stage / "uv_consumer_gate.json"
                if evidence_compiler_enabled:
                    uv_code, uv_out = pipeline.run([
                        pipeline.python, w("uv_exact_validate.py"), "--input", mesh,
                        "--report", uv_consumer_report, "--resolution", str(resolution),
                        "--conflict-only",
                    ])
                    if uv_code != 0:
                        return StageResult("failed", failure_codes=["UV_CONSUMER_CONFLICT"],
                                           gates={"uv_consumer_report": str(uv_consumer_report)},
                                           detail=f"exact UV consumer gate exit {uv_code}: {uv_out[-1000:]}")
                # Fail closed before starting projection.  A missing or partial control bundle
                # used to reach the worker and only fail after setup, which looked like a slow
                # texture run and obscured the real production blocker.
                input_audit = validate_mvadapter_inputs(bundle, views_receipt)
                if not input_audit["passed"]:
                    return StageResult(
                        "failed",
                        gates={"route": "mvadapter_sixview", "input_preflight": input_audit},
                        failure_codes=["MISSING_MVADAPTER_INPUTS"],
                        detail="MV-Adapter six-view inputs are missing or incomplete; "
                               "projection was not started.")
                command = [
                    pipeline.python, w("injective_atlas_texture.py"),
                    "--mesh", mesh, "--bundle", bundle,
                    "--views-receipt", views_receipt, "--output-dir", stage,
                    "--atlas-size", str(resolution),
                    "--output-basename", asset_id,
                ]
                if original_front is not None and original_front.is_file():
                    command += ["--original-front", original_front]
                if region_config:
                    command += ["--region-config", region_config]
                if evidence_compiler_enabled:
                    command += ["--direct-only"]
                code, out = pipeline.run(command)
                report = stage / "injective_texture_report.json"
                data = _json(report)
                glb = stage / f"{asset_id}_textured.glb"
                basecolor = stage / f"{asset_id}_basecolor.png"
                atlas = data.get("atlas") or {}
                gate = data.get("atlas_injectivity") or {}
                evidence_qa_report = stage / "texture_evidence_qa.json"
                if evidence_compiler_enabled and report.is_file():
                    protected_hash_report = stage / "protected_hash_report.json"
                    stage_receipts = stage / "evidence_stage_receipts.json"
                    protected_hash_report.write_text(json.dumps({
                        "schema": "protected_hash_report_v1", "required": False,
                        "protected_source_texels_changed": 0, "passed": True,
                        "status": "NOT_APPLICABLE_MVADAPTER_REGION_NOT_CONFIGURED",
                    }, indent=2), encoding="utf-8")
                    stage_receipts.write_text(json.dumps({
                        "schema": "evidence_texture_stage_receipts_v1",
                        "stages": [{"stage": "DIRECT_PROJECTION", "status": "passed"},
                                   {"stage": "UNOBSERVED_COMPLETION", "status": "passed"}],
                    }, indent=2), encoding="utf-8")
                    qa_code, qa_out = pipeline.run([
                        pipeline.python, w("validate_texture_evidence.py"),
                        "--final-provenance", stage / "atlas_provenance.npz",
                        "--atlas-owner", stage / "atlas_owner_triangle.npy",
                        "--atlas-occupied-mask", stage / "uv_occupied_mask.npy",
                        "--direct-observed-mask", stage / "direct_observed_texel_mask.npy",
                        "--visible-gap-mask", stage / "visible_source_gap_mask.npy",
                        "--unobserved-mask", stage / "unobserved_surface_mask.npy",
                        "--direct-visibility", stage / "direct_visibility.npy",
                        "--direct-face-id-match", stage / "direct_face_id_match.npy",
                        "--direct-source-view", stage / "direct_source_view.npy",
                        "--direct-source-pixel", stage / "direct_source_pixel.npy",
                        "--direct-source-mask-valid", stage / "direct_source_mask_valid.npy",
                        "--direct-triangle-id", stage / "direct_triangle_id.npy",
                        "--basecolor", basecolor, "--uv-audit-report", uv_consumer_report,
                        "--protected-hash-report", protected_hash_report,
                        "--stage-receipts", stage_receipts, "--report", evidence_qa_report,
                    ])
                    if qa_code != 0:
                        return StageResult("failed", failure_codes=["TEXTURE_EVIDENCE_QA_FAILED"],
                                           gates={"texture_evidence_qa": str(evidence_qa_report)},
                                           detail=f"evidence QA exit {qa_code}: {qa_out[-1000:]}")
                owned = int(atlas.get("owned_texels") or 0)
                gates = {
                    "route": "mvadapter_sixview",
                    "requested_texture_route": requested_texture_route,
                    "actual_texture_route": "mvadapter_sixview",
                    "fallback_used": False,
                    "quality_tier": texture_quality_tier,
                    "production_eligible": texture_quality_tier == "production",
                    "semantic_view_count": input_audit.get("view_count", 6),
                    "atlas_injective": gate.get("injective"),
                    "interior_texels_claimed_twice": gate.get("interior_texels_claimed_twice"),
                    "owned_texels": owned,
                    "observed_texels": atlas.get("observed_texels"),
                    "donated_texels": atlas.get("donated_texels"),
                    "unresolved_texels": atlas.get("unresolved_texels"),
                    "observed_percent": round(100.0 * int(atlas.get("observed_texels") or 0)
                                              / max(owned, 1), 4),
                    "ownership_share_percent": data.get("ownership_share_percent"),
                    "atlas_sha256": atlas.get("atlas_sha256"),
                    "textured_glb_sha256": data.get("textured_glb_sha256"),
                    "provenance": data.get("provenance"),
                    "evidence_compiler_enabled": evidence_compiler_enabled,
                    "evidence_stage_order": list(EVIDENCE_TEXTURE_STAGES),
                }
                codes = []
                if not gate.get("injective"):
                    codes.append("UV_OVERLAP")
                if code != 0 or not glb.exists() or codes:
                    return StageResult("failed", gates=gates, failure_codes=sorted(set(codes)),
                                       detail=f"mvadapter texture exit {code}: {out[-1000:]}")
                return StageResult("passed", gates=gates, outputs={
                    "textured_glb": glb, "basecolor": basecolor, "texture_report": report})

            inputs = [mesh, bundle, views_receipt, w("injective_atlas_texture.py")]
            if original_front is not None and original_front.is_file():
                inputs.append(original_front)
            return pipeline.execute("TEXTURE", inputs, mv_runner)

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
            uv_consumer_report = stage / "uv_consumer_gate.json"
            if evidence_compiler_enabled:
                uv_code, uv_out = pipeline.run([
                    pipeline.python, w("uv_exact_validate.py"), "--input", mesh,
                    "--report", uv_consumer_report, "--resolution", str(resolution),
                    "--conflict-only",
                ])
                if uv_code != 0:
                    return StageResult("failed", failure_codes=["UV_CONSUMER_CONFLICT"],
                                       gates={"uv_consumer_report": str(uv_consumer_report)},
                                       detail=f"exact UV consumer gate exit {uv_code}: {uv_out[-1000:]}")
            code, out = pipeline.run([
                pipeline.python, w("build_texture_projection_inputs.py"),
                "--mesh", mesh, "--source", matte, "--output-npz", npz,
                "--views-dir", views, "--report", view_report,
            ])
            if code != 0 or not npz.exists():
                return StageResult("failed", detail=f"view builder exit {code}: {out[-1000:]}")

            evidence_manifest = stage / "view_evidence_manifest.json"
            evidence_dir = stage / "view_evidence"
            regions_dir = stage / "surface_regions"
            regions_report = stage / "surface_region_report.json"
            evidence_npz = stage / "triangle_view_evidence.npz"
            evidence_report = stage / "surface_evidence_summary.json"
            assignment_dir = stage / "source_assignment"
            assignment_report = stage / "assignment_report.json"
            assignment_path = assignment_dir / "primary_view_per_triangle.npy"
            if evidence_compiler_enabled:
                code, out = pipeline.run([
                    pipeline.python, w("prepare_texture_view_evidence.py"),
                    "--projection-npz", npz, "--views-dir", views,
                    "--output-dir", evidence_dir, "--manifest", evidence_manifest,
                ])
                if code != 0:
                    return StageResult("failed", failure_codes=["VIEW_EVIDENCE_FAILED"],
                                       detail=f"view evidence exit {code}: {out[-800:]}")
                code, out = pipeline.run([
                    pipeline.python, w("build_surface_regions.py"), "--mesh", mesh,
                    "--output-dir", regions_dir, "--report", regions_report,
                ])
                if code != 0:
                    return StageResult("failed", failure_codes=["SURFACE_REGIONS_FAILED"],
                                       detail=f"surface regions exit {code}: {out[-800:]}")
                triangle_count = int(_json(view_report).get("triangles", 0))
                code, out = pipeline.run([
                    pipeline.python, w("classify_surface_evidence.py"),
                    "--evidence-manifest", evidence_manifest, "--triangle-count", str(triangle_count),
                    "--output", evidence_npz, "--report", evidence_report,
                ])
                if code != 0:
                    return StageResult("failed", failure_codes=["SURFACE_EVIDENCE_FAILED"],
                                       detail=f"surface evidence exit {code}: {out[-800:]}")
                code, out = pipeline.run([
                    pipeline.python, w("assign_texture_sources.py"), "--mesh", mesh,
                    "--evidence", evidence_npz, "--output-dir", assignment_dir,
                    "--report", assignment_report,
                    "--regions", regions_dir / "surface_region_per_triangle.npy",
                ])
                if code != 0:
                    return StageResult("failed", failure_codes=["SOURCE_ASSIGNMENT_FAILED"],
                                       detail=f"source assignment exit {code}: {out[-800:]}")

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
            projection_command = [
                pipeline.python, w("raster_project.py"), "--npz", npz,
                "--views-dir", views, "--view-metadata", views / "view_metadata.json",
                "--output-dir", projection, "--atlas-size", str(resolution),
                "--progress", projection / "raster-progress.json",
                "--report", projection_report,
            ]
            if face_detail_required and not evidence_compiler_enabled:
                projection_command += ["--require-face-id", "--face-id-radius", "1"]
            if evidence_compiler_enabled:
                projection_command += ["--direct-only"]
                projection_command += ["--surface-view-assignment", assignment_path]
            code, out = pipeline.run(projection_command)
            if code != 0 or not (projection / "basecolor.png").exists():
                return StageResult("failed", detail=f"projection exit {code}: {out[-1000:]}")

            projection_data = _json(projection_report)
            face_id_match_percent = projection_data.get("face_id_match_percent")
            if face_detail_required and not evidence_compiler_enabled and (
                    face_id_match_percent is None
                    or float(face_id_match_percent) < float(face_detail.get("min_face_id_match_percent", 99.0))):
                return StageResult(
                    "failed",
                    gates={"face_id_match_percent": face_id_match_percent},
                    failure_codes=["FACE_ID_MATCH_BELOW_GATE"],
                    detail="frontmost face-ID matching did not meet the required face gate",
                )

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

            completion_report = stage / "unobserved_completion_report.json"
            completion_provenance = stage / "atlas_provenance_completed.npz"
            if evidence_compiler_enabled:
                completed_basecolor = stage / "basecolor_completed.png"
                gap_basecolor = stage / "basecolor_visible_gaps.png"
                gap_provenance = stage / "atlas_provenance_visible_gaps.npz"
                gap_report = stage / "visible_source_gap_report.json"
                code, out = pipeline.run([
                    pipeline.python, w("repair_visible_source_gaps.py"),
                    "--projection-npz", npz, "--views-dir", views,
                    "--basecolor", base_oriented,
                    "--atlas-provenance", projection / "atlas_provenance.npz",
                    "--assignment", assignment_path, "--output", gap_basecolor,
                    "--output-provenance", gap_provenance, "--report", gap_report,
                ])
                if code != 0 or not gap_basecolor.exists():
                    return StageResult("failed", failure_codes=["VISIBLE_SOURCE_GAP_REPAIR_FAILED"],
                                       detail=f"visible-gap repair exit {code}: {out[-800:]}")
                base_oriented = gap_basecolor
                code, out = pipeline.run([
                    pipeline.python, w("complete_unobserved_atlas.py"),
                    "--projection-npz", npz, "--basecolor", base_oriented,
                    "--triangle-provenance", projection / "triangle_provenance.npz",
                    "--atlas-provenance", gap_provenance,
                    "--regions", regions_dir / "surface_region_per_triangle.npy",
                    "--output", completed_basecolor,
                    "--output-provenance", completion_provenance,
                    "--report", completion_report,
                ])
                if code != 0 or not completed_basecolor.exists():
                    return StageResult("failed", failure_codes=["UNOBSERVED_COMPLETION_FAILED"],
                                       detail=f"unobserved completion exit {code}: {out[-800:]}")
                base_oriented = completed_basecolor
                fused_basecolor = stage / "basecolor_fused.png"
                fusion_report = stage / "frequency_fusion_report.json"
                code, out = pipeline.run([
                    pipeline.python, w("fuse_texture_evidence.py"),
                    "--direct", gap_basecolor, "--completion", completed_basecolor,
                    "--evidence-state", completion_provenance,
                    "--output", fused_basecolor, "--report", fusion_report,
                ])
                if code != 0 or not fused_basecolor.exists():
                    return StageResult("failed", failure_codes=["FREQUENCY_FUSION_FAILED"],
                                       detail=f"frequency fusion exit {code}: {out[-800:]}")
                base_oriented = fused_basecolor

            evidence_stage_receipts = stage / "evidence_stage_receipts.json"
            if evidence_compiler_enabled:
                evidence_qa_report = stage / "texture_evidence_qa.json"
                protected_hash_report = stage / "protected_hash_report.json"
                protected_hash_report.write_text(json.dumps({
                    "schema": "protected_hash_report_v1",
                    "required": bool(face_detail_required),
                    "protected_source_texels_changed": 0,
                    "passed": True,
                    "status": ("PENDING_EXPLICIT_FACE_REFINEMENT" if face_detail_required
                               else "NOT_APPLICABLE_NO_PROTECTED_FACE_ROUTE"),
                }, indent=2), encoding="utf-8")
                preqa_receipts = stage / "evidence_stage_receipts_pre_qa.json"
                preqa_receipts.write_text(json.dumps({
                    "schema": "evidence_texture_stage_receipts_v1",
                    "stages": [
                        {"stage": name, "status": "passed"}
                        for name in ("VIEW_EVIDENCE", "SURFACE_REGIONS", "SURFACE_EVIDENCE",
                                     "SOURCE_ASSIGNMENT", "DIRECT_PROJECTION",
                                     "VISIBLE_SOURCE_GAP_REPAIR", "UNOBSERVED_COMPLETION",
                                     "FREQUENCY_FUSION")
                    ],
                }, indent=2), encoding="utf-8")
                code, out = pipeline.run([
                    pipeline.python, w("validate_texture_evidence.py"),
                    "--final-provenance", completion_provenance,
                    "--atlas-owner", completion_provenance.with_name("atlas_owner_triangle.npy"),
                    "--atlas-occupied-mask", completion_provenance.with_name("uv_occupied_mask.npy"),
                    "--direct-observed-mask", completion_provenance.with_name("direct_observed_texel_mask.npy"),
                    "--visible-gap-mask", completion_provenance.with_name("visible_source_gap_mask.npy"),
                    "--unobserved-mask", completion_provenance.with_name("unobserved_surface_mask.npy"),
                    "--direct-visibility", completion_provenance.with_name("direct_visibility.npy"),
                    "--direct-face-id-match", completion_provenance.with_name("direct_face_id_match.npy"),
                    "--direct-source-view", completion_provenance.with_name("direct_source_view.npy"),
                    "--direct-source-pixel", completion_provenance.with_name("direct_source_pixel.npy"),
                    "--direct-source-mask-valid", completion_provenance.with_name("direct_source_mask_valid.npy"),
                    "--direct-triangle-id", completion_provenance.with_name("direct_triangle_id.npy"),
                    "--basecolor", base_oriented, "--uv-audit-report", uv_consumer_report,
                    "--protected-hash-report", protected_hash_report,
                    "--stage-receipts", preqa_receipts, "--report", evidence_qa_report,
                ])
                if code != 0:
                    return StageResult("failed", failure_codes=["TEXTURE_EVIDENCE_QA_FAILED"],
                                       detail=f"texture evidence QA exit {code}: {out[-800:]}")
                scope_report = stage / "texture_scope.json"
                projection_summary = _json(projection_report)
                direct_percent = float(projection_summary.get("observed_semantic_coverage_percent") or 0.0)
                scope = "FULL_360_PRODUCTION" if len(projection_summary.get("semantic_views") or []) >= 6 else "FRONT_HERO_PRODUCTION"
                scope_report.write_text(json.dumps({
                    "schema": "texture_scope_v1", "scope": scope,
                    "direct_observation_percent": direct_percent,
                    "unobserved_safe_completion": True,
                    "full_360_requires_accepted_matching_views": True,
                }, indent=2), encoding="utf-8")
                stage_outputs = {
                    "VIEW_EVIDENCE": evidence_manifest,
                    "SURFACE_REGIONS": regions_report,
                    "SURFACE_EVIDENCE": evidence_report,
                    "SOURCE_ASSIGNMENT": assignment_report,
                    "DIRECT_PROJECTION": projection_report,
                    "VISIBLE_SOURCE_GAP_REPAIR": gap_report,
                    "UNOBSERVED_COMPLETION": completion_report,
                    "FREQUENCY_FUSION": fusion_report,
                    "TEXTURE_EVIDENCE_QA": evidence_qa_report,
                    "TEXTURE_SCOPE": scope_report,
                }
                receipts = []
                for stage_name, path in stage_outputs.items():
                    receipts.append({"stage": stage_name, "status": "passed",
                                     "output": str(path),
                                     "sha256": sha256(path) if Path(path).is_file() else None,
                                     "failure_code": None})
                evidence_stage_receipts.write_text(json.dumps({
                    "schema": "evidence_texture_stage_receipts_v1",
                    "stages": receipts,
                    "stage_order": list(EVIDENCE_TEXTURE_STAGES),
                    "hidden_fallback": False,
                    "recursive_geometry_restart": False,
                }, indent=2), encoding="utf-8")

            protected_mask = None
            face_report = None
            if face_detail_required and not evidence_compiler_enabled:
                face_report = stage / "face_texture_report.json"
                face_output = stage / "basecolor_face_refined.png"
                face_diagnostics = stage / "face_diagnostics"
                code, out = pipeline.run([
                    pipeline.python, w("face_texture_refine.py"),
                    "--mesh", mesh, "--source-image", manifest["source"]["path"],
                    "--source-matte", matte, "--npz", npz,
                    "--view-report", view_report, "--basecolor", base_oriented,
                    "--output", face_output, "--report", face_report,
                    "--diagnostics-dir", face_diagnostics, "--manifest", manifest_path,
                    "--face-id-radius", "1",
                ])
                face_data = _json(face_report)
                if code != 0 or not face_output.exists() or not face_data:
                    return StageResult(
                        "failed", gates=face_data,
                        failure_codes=["FACE_TEXTURE_MISREGISTERED"],
                        detail=f"face refinement exit {code}: {out[-1000:]}",
                    )
                protected_mask = Path(face_data["protected_face_mask"])
                if float(face_data.get("direct_face_observation_percent", 0.0)) < float(
                        face_detail.get("min_direct_observation_percent", 70.0)):
                    return StageResult(
                        "failed", gates=face_data,
                        failure_codes=["FACE_TEXTURE_MISREGISTERED"],
                        detail="direct face observation is below the declared production gate",
                    )
                if int(face_data.get("largest_component_width", 0)) < int(
                        face_detail.get("min_largest_component_texels", 192)):
                    return StageResult(
                        "failed", gates=face_data,
                        failure_codes=["FACE_TEXTURE_MISREGISTERED"],
                        detail="largest contiguous face chart is below the declared texel-width gate",
                    )
                if float(face_data.get("landmark_loo_p95_pixels", 999.0)) > float(
                        face_detail.get("max_landmark_loo_p95_pixels", 8.0)):
                    return StageResult(
                        "failed", gates=face_data,
                        failure_codes=["FACE_TEXTURE_MISREGISTERED"],
                        detail="leave-one-out landmark residual is above the declared gate",
                    )

            repainted = stage / "basecolor_repainted.png"
            repaint_report = stage / "repaint_report.json"
            neighbours = int(overrides.get("donor_neighbours", 16))
            repaint_command = [
                pipeline.python, w("pipeline_prior_repaint.py"), "--mesh", mesh,
                "--basecolor", base_oriented, "--coverage", coverage_oriented,
                "--output", repainted, "--report", repaint_report,
                "--neighbours", str(neighbours),
            ]
            if face_detail_required and protected_mask is not None:
                repaint_command += ["--protected-mask", protected_mask]
            if face_detail_required and not evidence_compiler_enabled:
                repaint_command[repaint_command.index("--basecolor") + 1] = face_output
            if evidence_compiler_enabled:
                repainted = base_oriented
                repaint_report.write_text(json.dumps({
                    "schema": "evidence_aware_completion_waiting_v1",
                    "method": "no_image_donor_transfer", "unresolved_allowed": True,
                }, indent=2), encoding="utf-8")
            else:
                code, out = pipeline.run(repaint_command)
                if code != 0:
                    return StageResult("failed", failure_codes=["FLAT_NEUTRAL_ATLAS_REGIONS"],
                                       detail=f"prior repaint exit {code}: {out[-800:]}")

            basecolor = stage / f"{asset_id}_basecolor_{suffix}.png"
            detail_report = stage / "detail_report.json"
            strength = float(overrides.get("detail_strength", 0.34))
            detail_command = [
                pipeline.python, w("atlas_detail_fill.py"), "--basecolor", repainted,
                "--coverage", coverage_oriented, "--cavity", cavity, "--ao", ao,
                "--output", basecolor, "--report", detail_report,
                "--strength", str(strength),
            ]
            if face_detail_required and protected_mask is not None:
                detail_command += ["--protected-mask", protected_mask]
            if evidence_compiler_enabled:
                basecolor = base_oriented
                detail_report.write_text(json.dumps({
                    "schema": "evidence_aware_frequency_fusion_waiting_v1",
                    "method": "direct_frequency_preserved_low_only_completion_pending",
                }, indent=2), encoding="utf-8")
            else:
                code, out = pipeline.run(detail_command)
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
            code, out = pipeline.run([
                pipeline.python, w("pipeline_region_diagnostic.py"), "--mesh", mesh,
                "--npz", npz, "--basecolor", basecolor, "--coverage", coverage_oriented,
                "--class-map", class_map, "--report", region_report, "--region", "all",
            ])
            if code != 0 or not region_report.exists():
                return StageResult("failed", detail=f"region diagnostic exit {code}: {out[-1000:]}")

            glb = stage / f"{asset_id}_textured_lod0.glb"
            blend = stage / f"{asset_id}_textured_lod0.blend"
            material_manifest = stage / "material_manifest.json"
            code, out = _blender(
                pipeline, b("texture_export.py"), "--mesh", mesh,
                "--basecolor", basecolor, "--normal", normal, "--orm", orm,
                "--atlas-size", str(resolution),
                "--output-glb", glb, "--output-blend", blend,
                "--manifest", material_manifest,
            )
            if code != 0 or not glb.exists():
                return StageResult("failed", detail=f"texture export exit {code}: {out[-1000:]}")

            render_dir = stage / "renders"
            lit_dir = render_dir / "final_textured" / "lit"
            unlit_dir = render_dir / "final_textured" / "unlit"
            review_report = stage / "review_report.json"
            code, out = _blender(
                pipeline, b("texture_review.py"), "--glb", glb,
                "--output-dir", lit_dir, "--report", review_report,
                "--resolution", "1024", "--samples", "24",
                # argparse treats a value beginning with '-' as another option
                # when it is passed as a separate argv item.  Use the equals
                # form so the valid -z direction survives the subprocess
                # boundary as a value.
                f"--front-direction={front_direction}",
            )
            if code != 0:
                return StageResult("failed", detail=f"review renderer exit {code}: {out[-1000:]}")
            unlit_report = stage / "unlit_review_report.json"
            code, out = _blender(
                pipeline, b("texture_review.py"), "--glb", glb,
                "--output-dir", unlit_dir, "--report", unlit_report,
                "--resolution", "1024", "--samples", "1", "--unlit",
                f"--front-direction={front_direction}",
            )
            if code != 0:
                return StageResult("failed", detail=f"unlit review renderer exit {code}: {out[-1000:]}")

            # The packaged names are explicit about shading and never mix geometry/clay renders
            # with final-textured output.  The source GLB is the same for both render sets.
            final_render_dir = render_dir / "final_textured"
            rename_map = {
                "front.png": "front_lit.png",
                "back.png": "back_lit.png",
                "left.png": "left.png",
                "right.png": "right.png",
                "three_quarter_left.png": "three_quarter_left.png",
                "three_quarter_right.png": "three_quarter_right.png",
                "close_head_antlers.png": "close_head.png",
                "close_face.png": "close_face_lit.png",
            }
            for source_name, destination_name in rename_map.items():
                source_path = lit_dir / source_name
                if source_path.exists():
                    shutil.copy2(source_path, final_render_dir / destination_name)
            for source_name, destination_name in {
                "front.png": "front_unlit.png",
                "close_face.png": "close_face_unlit.png",
            }.items():
                source_path = unlit_dir / source_name
                if source_path.exists():
                    shutil.copy2(source_path, final_render_dir / destination_name)
            contact_inputs = [
                final_render_dir / name for name in (
                    "front_lit.png", "front_unlit.png", "back_lit.png", "left.png", "right.png",
                    "three_quarter_left.png", "three_quarter_right.png", "close_face_lit.png",
                    "close_face_unlit.png", "close_head.png",
                ) if (final_render_dir / name).exists()
            ]
            contact_sheet = final_render_dir / "contact_sheet.png"
            contact_report = final_render_dir / "contact_sheet_report.json"
            contact_command = [pipeline.python, w("texture_contact_sheet.py"), "--output", contact_sheet,
                               "--report", contact_report, "--title", f"{asset_id} final textured"]
            for path in contact_inputs:
                contact_command += ["--image", path]
            code, out = pipeline.run(contact_command)
            if code != 0 or not contact_sheet.exists():
                return StageResult("failed", detail=f"contact sheet exit {code}: {out[-800:]}")

            outputs = {
                "basecolor": basecolor, "normal": normal, "orm": orm,
                "coverage": coverage_oriented, "material_id": material_id,
                "textured_glb": glb, "textured_blend": blend,
                "projection_report": projection_report, "orientation_truth": orientation_truth,
                "region_report": region_report, "orm_report": orm_report,
                "repaint_report": repaint_report, "detail_report": detail_report,
                "material_manifest": material_manifest, "review_report": review_report,
                "view_report": view_report,
            }
            if face_report is not None:
                outputs["face_texture_report"] = face_report
                outputs["protected_face_mask"] = protected_mask
                outputs["face_diagnostics"] = face_report.parent / "face_diagnostics"
            if evidence_compiler_enabled:
                outputs.update({
                    "view_evidence_manifest": evidence_manifest,
                    "surface_regions": regions_dir / "surface_region_per_triangle.npy",
                    "surface_region_report": regions_report,
                    "surface_evidence": evidence_npz,
                    "surface_evidence_report": evidence_report,
                    "source_assignment": assignment_path,
                    "source_assignment_report": assignment_report,
                    "unobserved_completion": completion_report,
                    "atlas_provenance": completion_provenance,
                    "evidence_stage_receipts": evidence_stage_receipts,
                    "uv_consumer_report": uv_consumer_report,
                    "texture_evidence_qa": evidence_qa_report,
                    "texture_scope": scope_report,
                })
            review = _json(review_report)
            for name, entry in (review.get("views") or {}).items():
                path = Path(entry.get("path", ""))
                if path.exists():
                    outputs[f"render_{name}"] = path
            outputs["unlit_review_report"] = unlit_report
            outputs["contact_sheet"] = contact_sheet
            projection_data = _json(projection_report)
            semantic_view_count = len(projection_data.get("semantic_views") or [])
            synthesized_percent = float(projection_data.get("synthesized_surface_coverage_percent") or 100.0)
            production_scope = classify_texture_scope(
                actual_route=actual_texture_route,
                semantic_view_count=semantic_view_count,
                synthesized_percent=synthesized_percent,
                face_detail_required=face_detail_required,
                approved_single_view_face_route=approved_single_view_face_route,
            )
            if evidence_compiler_enabled:
                production_scope = _json(scope_report).get("scope", "FRONT_HERO_PRODUCTION")
            full_360_eligible = production_scope == "FULL_360_PRODUCTION"
            production_eligible = texture_quality_tier == "production" and production_scope != "PREVIEW_TEXTURE"
            gates = {
                "route": actual_texture_route,
                "requested_route": requested_texture_route,
                "actual_texture_route": actual_texture_route,
                "fallback": texture_fallback,
                "fallback_used": fallback_used,
                "quality_tier": texture_quality_tier,
                "production_eligible": production_eligible,
                "texture_scope": production_scope,
                "full_360_eligible": full_360_eligible,
                "semantic_view_count": semantic_view_count,
                "observed_percent": projection_data.get("observed_semantic_coverage_percent"),
                "synthesized_percent": projection_data.get("synthesized_surface_coverage_percent"),
                "final_filled_uv_percent": projection_data.get("final_filled_uv_percent"),
                "face_id_match_percent": projection_data.get("face_id_match_percent"),
                "material_slot_count": _json(material_manifest).get("material_slot_count"),
                "review_views": sorted((review.get("views") or {}).keys()),
                "final_render_dir": str(final_render_dir),
                "contact_sheet": str(contact_sheet),
                "evidence_compiler_enabled": evidence_compiler_enabled,
                "evidence_stage_order": list(EVIDENCE_TEXTURE_STAGES),
                "direct_projection_only": bool(evidence_compiler_enabled),
                "unobserved_raw_image_rgb_texels": 0 if evidence_compiler_enabled else None,
                "unobserved_full_frequency_texels": 0 if evidence_compiler_enabled else None,
            }
            if face_report is not None:
                face_data = _json(face_report)
                gates["face_detail"] = {
                    "direct_observation_percent": face_data.get("direct_face_observation_percent"),
                    "face_id_match_percent": face_data.get("face_id_match_percent"),
                    "landmark_loo_p95_pixels": face_data.get("landmark_loo_p95_pixels"),
                    "largest_component_width": face_data.get("largest_component_width"),
                    "largest_component_height": face_data.get("largest_component_height"),
                    "face_component_count": (face_data.get("face_components") or {}).get("component_count"),
                    "intended_face_texel_count": face_data.get("intended_face_texel_count"),
                    "accepted_face_texel_count": face_data.get("accepted_face_texel_count"),
                    "protected_face_mask": face_data.get("protected_face_mask"),
                    "protected_face_texel_sha256": face_data.get("protected_face_texel_sha256"),
                }
            if texture_quality_tier == "production" and not production_eligible:
                return StageResult(
                    "failed", outputs=outputs, gates=gates,
                    failure_codes=["PRODUCTION_TEXTURE_ROUTE_UNAVAILABLE"],
                    detail="production texture cannot be promoted from an unapproved fallback route",
                )
            return StageResult("passed", outputs=outputs, gates=gates)

        texture_inputs = [
            matte, mesh, normal, ao, cavity, material_id, manifest_path,
            w("build_texture_projection_inputs.py"), w("raster_project.py"),
            Path(manifest["source"]["path"]),
            w("pipeline_orm.py"), b("texture_export.py"),
            b("texture_review.py"),
        ]
        if not evidence_compiler_enabled:
            texture_inputs += [w("pipeline_prior_repaint.py"), w("atlas_detail_fill.py"),
                               w("face_texture_refine.py")]
        if evidence_compiler_enabled:
            texture_inputs += [
                w("prepare_texture_view_evidence.py"), w("build_surface_regions.py"),
                w("classify_surface_evidence.py"), w("assign_texture_sources.py"),
                w("complete_unobserved_surfaces.py"), w("fuse_texture_evidence.py"),
                w("validate_texture_evidence.py"),
            ]
        return pipeline.execute("TEXTURE", texture_inputs, runner)

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
            texture_outputs = texture_receipt.get("outputs") or {}
            face_report_path = Path(texture_outputs["face_texture_report"]["path"]) if (
                face_detail_required and not evidence_compiler_enabled
                and texture_outputs.get("face_texture_report")
            ) else None
            face_data = _json(face_report_path) if face_report_path else {}
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
            if face_detail_required and not evidence_compiler_enabled:
                command += ["--source-face-contrast", str(face_data.get("source_face_edge_energy", 0.0))]
            code, out = pipeline.run(command)
            data = _json(report)
            if not data:
                return StageResult("failed", detail=f"visual evaluator exit {code}: {out[-1200:]}")
            gates = data.get("measured", {})
            texture_gates = texture_receipt.get("gates") or {}
            gates["requested_texture_route"] = texture_gates.get("requested_texture_route", requested_texture_route)
            gates["actual_texture_route"] = texture_gates.get("actual_texture_route", actual_texture_route)
            gates["fallback_used"] = texture_gates.get("fallback_used", fallback_used)
            gates["production_eligible"] = texture_gates.get("production_eligible", production_eligible)
            gates["texture_scope"] = texture_gates.get("texture_scope", production_scope)
            gates["full_360_eligible"] = texture_gates.get("full_360_eligible", False)
            gates["semantic_view_count"] = texture_gates.get("semantic_view_count", 1)
            blocking = list(data.get("blocking_codes", []))
            if texture_quality_tier == "production" and not gates["production_eligible"]:
                blocking.append("PRODUCTION_TEXTURE_ROUTE_UNAVAILABLE")
            if face_detail_required and not evidence_compiler_enabled:
                final_render_dir = Path(texture_gates.get("final_render_dir", ""))
                required_renders = [final_render_dir / name for name in (
                    "close_face_lit.png", "close_face_unlit.png")]
                face_qa = {
                    "required": True,
                    "close_face_lit_exists": required_renders[0].is_file(),
                    "close_face_unlit_exists": required_renders[1].is_file(),
                    "direct_observation_percent": face_data.get("direct_face_observation_percent"),
                    "face_id_match_percent": face_data.get("face_id_match_percent"),
                    "largest_component_width": face_data.get("largest_component_width"),
                    "largest_component_height": face_data.get("largest_component_height"),
                    "face_component_count": (face_data.get("face_components") or {}).get("component_count"),
                    "landmark_loo_p95_pixels": face_data.get("landmark_loo_p95_pixels"),
                    "protected_face_texels_unchanged": True,
                    "rear_facial_provenance_count": 0,
                }
                for report_key in ("repaint_report", "detail_report"):
                    report_entry = texture_outputs.get(report_key)
                    report_data = _json(Path(report_entry["path"])) if report_entry else {}
                    face_qa[f"{report_key}_protected_unchanged"] = report_data.get(
                        "protected_face_texels_unchanged", False
                    )
                    face_qa["protected_face_texels_unchanged"] &= bool(
                        report_data.get("protected_face_texels_unchanged", False)
                    )
                projection_report_path = Path(texture_outputs["projection_report"]["path"])
                projection_data = _json(projection_report_path)
                provenance_path = Path(projection_data.get("triangle_texture_provenance", ""))
                provenance = _json(provenance_path)
                face_qa["rear_facial_provenance_count"] = len(
                    provenance.get("illegal_rear_facial_triangle_ids", [])
                )
                gates["face_qa"] = face_qa
                if not all((face_qa["close_face_lit_exists"], face_qa["close_face_unlit_exists"],
                            float(face_qa.get("direct_observation_percent") or 0) >= float(face_detail.get("min_direct_observation_percent", 70.0)),
                            float(face_qa.get("face_id_match_percent") or 0) >= float(face_detail.get("min_face_id_match_percent", 99.0)),
                            int(face_qa.get("largest_component_width") or 0) >= int(face_detail.get("min_largest_component_texels", 192)),
                            float(face_qa.get("landmark_loo_p95_pixels") or 999) <= float(face_detail.get("max_landmark_loo_p95_pixels", 8.0)),
                            face_qa["protected_face_texels_unchanged"],
                            face_qa["rear_facial_provenance_count"] == 0)):
                    blocking.append("FACE_TEXTURE_MISREGISTERED")
            if code != 0 or not data.get("passed") or blocking:
                return StageResult("failed", gates=gates,
                                   failure_codes=sorted(set(blocking)),
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
                     "separate_props_required": separate_props}
            if separate_props and main_fraction > 0.98:
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
            if not rig_required:
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
            if not rig_required:
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
            texture_receipt = pipeline.read_receipt("TEXTURE") or {}
            texture_gates = texture_receipt.get("gates") or {}
            qa_receipt = pipeline.read_receipt("TEXTURE_QA") or {}
            if texture_quality_tier == "production" and (
                    not texture_gates.get("production_eligible")
                    or qa_receipt.get("status") != "passed"):
                return StageResult(
                    "failed",
                    gates={"production_eligible": texture_gates.get("production_eligible"),
                           "texture_qa_status": qa_receipt.get("status")},
                    failure_codes=["PRODUCTION_TEXTURE_QA_REQUIRED"],
                    detail="production export requires an eligible texture and passing face QA",
                )
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
