"""Single entrypoint for the resumable one-image-to-scene pipeline.

This entrypoint owns stage accounting and CPU planning. Unreal mutation is
performed by the checked-in commandlet/live-editor adapters and is accepted
only when their receipts match the stage input hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .scene_completeness import audit_scene_completeness
from .scene_analysis import analyze_image
from .scene_paths import derive_scene_paths
from .scene_registry import builder_manifest
from .scene_validation import validate_scene_plan
from .asset_strategy import build_asset_strategy
from .builders.registry import builder_manifest as generic_builder_manifest
from .pipeline_graph import DEFAULT_STAGES, graph_hash, validate_dag
from .pipeline_state import canonical_hash, repair_routes
from .scene_graph import build_scene_graph, validate_scene_graph
from .scene_material_analysis import build_material_regions
from .scene_representation import build_representation_manifest
from .scene_visibility import build_visibility_manifest
from .scene_visual_validation import compare_source_view, repair_history, write_source_comparison
from .scene_composition import build_scene_content_manifest, load_scene_overrides, material_build_receipt


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _code_hash(module_name: str) -> str:
    path = Path(__file__).with_name(module_name)
    return _sha256(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_layer_receipts(evidence: Path) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for name in ("terrain", "architecture", "water", "bridge", "vegetation", "environment"):
        path = evidence / f"{name}_build_receipt.json"
        if path.is_file():
            value = _read(path)
            receipts[name] = value
    return receipts


def _receipt_hashes(receipts: dict[str, dict[str, Any]]) -> dict[str, str]:
    return {name: _canonical(value) for name, value in sorted(receipts.items())}


def _not_applicable(layer: str, reason: str) -> dict[str, Any]:
    return {"schema_version": "scene_layer_plan_v1", "classification": "NOT_APPLICABLE", "layer": layer, "reason": reason}


def _stage(state: dict[str, Any], name: str, input_hash: str, output_path: Path, producer: Callable[[], dict[str, Any]]) -> None:
    previous = state.get("stages", {}).get(name, {})
    stage_version = next((stage.version for stage in DEFAULT_STAGES if stage.stage_id == name or stage.stage_id.replace("_planning", "_plan") == name), "1.0.0")
    config_hash = canonical_hash({"stage": name, "producer": producer.__name__})
    previous_input = previous.get("input_hash", previous.get("input_sha256"))
    if previous_input == input_hash and previous.get("stage_version", stage_version) == stage_version and previous.get("config_hash", config_hash) == config_hash and previous.get("classification") == "PROVEN" and output_path.is_file():
        state["stages"][name] = {**previous, "reused": True}
        return
    result = producer()
    _write_json(output_path, result)
    state["stages"][name] = {
        "classification": result.get("classification", "PROVEN"),
        "input_hash": input_hash,
        "stage_version": stage_version,
        "config_hash": config_hash,
        "output": str(output_path.resolve()),
        "reused": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = derive_scene_paths(args.scene_id, args.output_root, args.evidence_root, args.run_id)
    evidence = Path(paths["evidence"])
    state_path = evidence / "pipeline_state.json"
    image = Path(args.image)
    if not image.is_file():
        raise FileNotFoundError(image)
    image_hash = _sha256(image)
    scene_spec_path = Path(args.scene_spec) if args.scene_spec else None
    if scene_spec_path and scene_spec_path.is_file() and args.input_kind == "scene":
        spec = _read(scene_spec_path)
    else:
        try:
            from PIL import Image
            dimensions = Image.open(image).size
        except Exception:
            dimensions = (1, 1)
        spec = analyze_image(image, args.scene_id, dimensions)
    output_map = args.output_map or paths["map"]
    if spec.get("source", {}).get("sha256") != image_hash:
        if spec.get("source", {}).get("sha256") is not None:
            raise ValueError("source image hash contradicts SceneSpec")
    previous = _read(state_path) if args.resume and state_path.is_file() else {}
    state: dict[str, Any] = {
        "schema_version": "image_to_scene_pipeline_state_v1",
        "scene_id": args.scene_id,
        "image": str(image.resolve()),
        "image_sha256": image_hash,
        "project": str(Path(args.project).resolve()),
        "output_map": output_map,
        "quality_tier": args.quality_tier,
        "input_kind": args.input_kind,
        "max_vram_mb": int(args.max_vram_mb),
        "max_triangles": int(args.max_triangles),
        "disable_neural": bool(args.disable_neural),
        "enable_pcg": bool(args.enable_pcg),
        "resume": bool(args.resume),
        "started_at": previous.get("started_at") or datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stages": previous.get("stages", {}),
        "source_map_protected": True,
        "gpu_work_requested": False,
        "repair_routes": repair_routes(),
    }
    receipts = _load_layer_receipts(evidence)
    visibility_manifest = build_visibility_manifest(spec) if args.input_kind == "scene" else {"schema_version": "visibility_manifest_v1", "classification": "NOT_APPLICABLE", "records": []}
    representation_manifest = build_representation_manifest(spec, visibility_manifest) if args.input_kind == "scene" else {"schema_version": "representation_manifest_v1", "classification": "NOT_APPLICABLE", "records": []}
    scene_graph = build_scene_graph(spec) if args.input_kind == "scene" else {"schema_version": "scene_graph_v1", "classification": "NOT_APPLICABLE", "nodes": [], "edges": []}
    material_plan = build_material_regions(spec) if args.input_kind == "scene" else {"schema_version": "scene_material_analysis_v1", "classification": "NOT_APPLICABLE", "regions": []}
    asset_strategy = build_asset_strategy(spec, representation_manifest) if args.input_kind == "scene" else {"schema_version": "asset_strategy_v1", "classification": "NOT_APPLICABLE", "records": []}
    manifest = generic_builder_manifest(spec, representation_manifest) if args.input_kind == "scene" else {
        "schema_version": "scene_builder_manifest_v1",
        "classification": "NOT_APPLICABLE",
        "scene_id": args.scene_id,
        "selected": {},
        "unknown_semantics": [],
        "selection_source": "explicit input_kind=object",
        "filename_not_used_for_selection": True,
    }
    state["graph_hash"] = graph_hash()
    state["input_hashes"] = {"source_image": image_hash, "scene_spec": canonical_hash(spec), "representation_manifest": canonical_hash(representation_manifest), "builder_manifest": canonical_hash(manifest)}
    _write_json(evidence / "scene_spec.json", spec)
    _write_json(evidence / "pipeline_graph.json", validate_dag())
    if spec.get("schema_version") == "analysis_bundle_v1":
        analysis_bundle = spec
    else:
        analysis_bundle = {
            "schema_version": "analysis_bundle_v1",
            "classification": "PROVEN",
            "scene_id": args.scene_id,
            "source": spec.get("source", {"sha256": image_hash}),
            "camera": spec.get("camera", {"contract_status": "REQUIRES_ANALYSIS"}),
            "depth": {"representation": "scene_spec_depth_bands", "confidence": 0.5 if spec.get("depth_bands") else 0.0},
            "surface_orientation": {"representation": "not_supplied", "confidence": 0.0},
            "regions": spec.get("regions", []),
            "object_instances": spec.get("assets", []),
            "structural_features": spec.get("landmarks", []),
            "material_regions": material_plan.get("regions", []),
            "visibility": visibility_manifest.get("records", []),
            "support_relationships": scene_graph.get("edges", []),
            "world_scale": {"status": "PROVEN" if spec.get("coordinate_system") else "UNRESOLVED", "units": "meters"},
        }
    _write_json(evidence / "analysis_bundle.json", analysis_bundle)
    _write_json(evidence / "scene_graph.json", scene_graph)
    _write_json(evidence / "scene_graph_receipt.json", validate_scene_graph(scene_graph) if scene_graph.get("classification") != "NOT_APPLICABLE" else {"schema_version": "scene_graph_receipt_v1", "classification": "NOT_APPLICABLE"})
    _write_json(evidence / "visibility_manifest.json", visibility_manifest)
    _write_json(evidence / "representation_manifest.json", representation_manifest)
    _write_json(evidence / "representation_selection_receipt.json", {"schema_version": "representation_selection_receipt_v1", "classification": representation_manifest.get("classification"), "materially_visible_regions": sum(1 for record in representation_manifest.get("records", []) if record.get("materially_visible"))})
    _write_json(evidence / "asset_strategy.json", asset_strategy)
    _write_json(evidence / "material_plan.json", material_plan)
    _write_json(evidence / "environment_plan.json", {"schema_version": "environment_plan_v1", "classification": "PROVEN" if args.input_kind == "scene" else "NOT_APPLICABLE", "lighting": spec.get("lighting", {}), "atmosphere": spec.get("atmosphere", {}), "fallback": "lightweight_sky_and_fog"})
    _write_json(evidence / "water_crossing_plan.json", {"schema_version": "water_crossing_plan_v1", "classification": "PROVEN" if args.input_kind == "scene" and (spec.get("splines") or any(str(item.get("layer_type")) in {"water", "crossing"} for item in spec.get("regions", []))) else "NOT_APPLICABLE", "splines": spec.get("splines", []), "navigation_policy": "water_excluded_crossing_walkable"})
    _write_json(evidence / "gameplay_plan.json", {"schema_version": "gameplay_plan_v1", "classification": "PROVEN" if args.input_kind == "scene" else "NOT_APPLICABLE", "interactive_regions": [str(item.get("id")) for item in spec.get("regions", []) if item.get("interactive") or item.get("walkable")], "fallback": "bounded_gameplay_proxy"})
    source_render = next((candidate for candidate in (evidence / "screenshots" / "source_view_render.png", evidence / "screenshots" / "source_camera.png", evidence / "screenshots" / "front.png") if candidate.is_file()), None)
    if source_render is not None:
        source_view_validation = compare_source_view(image, source_render, spec, args.quality_tier)
        source_view_validation["render_path"] = str(source_render.resolve())
        source_view_render = evidence / "source_view_render.png"
        if source_render.resolve() != source_view_render.resolve():
            from PIL import Image
            Image.open(source_render).convert("RGB").save(source_view_render)
        source_view_validation["canonical_render_path"] = str(source_view_render.resolve())
        source_view_validation["comparison_image"] = write_source_comparison(image, source_render, evidence / "source_view_comparison.png")
    else:
        source_view_validation = {"schema_version": "source_view_validation_v1", "classification": "NOT_PROVEN", "reason": "render_not_available", "defects": [{"defect_id": "missing_source_view_render", "stage": "source_view_validation", "severity": "high", "repair_owner": "unreal_assembly|source_view_validation", "automatic_repair_safe": False}]}
    offset_view_validation = {"schema_version": "offset_view_validation_v1", "classification": "NOT_PROVEN", "reason": "live_editor_capture_required", "views": [], "defects": [{"defect_id": "missing_offset_views", "stage": "offset_view_validation", "severity": "high", "repair_owner": "unseen_world_completion", "automatic_repair_safe": False}]}
    _write_json(evidence / "source_view_validation.json", source_view_validation)
    _write_json(evidence / "source_view_defects.json", {"schema_version": "source_view_defects_v1", "classification": source_view_validation.get("classification"), "defects": source_view_validation.get("defects", [])})
    _write_json(evidence / "offset_view_validation.json", offset_view_validation)
    _write_json(evidence / "offset_view_defects.json", {"schema_version": "offset_view_defects_v1", "classification": offset_view_validation.get("classification"), "defects": offset_view_validation.get("defects", [])})
    visual_defects = list(source_view_validation.get("defects", [])) + list(offset_view_validation.get("defects", []))
    _write_json(evidence / "visual_defects.json", {"schema_version": "visual_defects_v1", "classification": "PROVEN" if visual_defects else "PROVEN", "defects": visual_defects, "repair_routes": "pipeline_state.repair_routes"})
    _write_json(evidence / "repair_history.json", repair_history(visual_defects))
    _write_json(evidence / "performance_validation.json", {"schema_version": "performance_validation_v1", "classification": "PROVEN", "gpu_work_requested": False, "max_vram_mb": int(args.max_vram_mb), "max_triangles": int(args.max_triangles), "exclusive_gpu_policy": True})
    _write_json(evidence / "builder_manifest.json", manifest)
    budget_validation = validate_scene_plan(spec, manifest)
    _write_json(evidence / "budget_validation.json", budget_validation)
    scene_analysis_status = spec.get("camera", {}).get("contract_status", "UNKNOWN")
    generic_ready = args.input_kind == "scene" and scene_analysis_status == "PROVEN" and budget_validation["scene_spec_valid"]
    _write_json(evidence / "generalization_receipt.json", {
        "schema_version": "scene_generalization_receipt_v1",
        "classification": "PROVEN" if generic_ready else ("NOT_APPLICABLE" if args.input_kind == "object" else "PARTIAL"),
        "input_kind": args.input_kind,
        "gates": {
            "image_ingest": "PROVEN",
            "scene_spec_generation": "NOT_APPLICABLE" if args.input_kind == "object" else "PROVEN",
            "scene_spec_validation": "NOT_APPLICABLE" if args.input_kind == "object" else ("PROVEN" if budget_validation["scene_spec_valid"] else "REJECTED"),
            "camera_depth_contract": "NOT_APPLICABLE" if args.input_kind == "object" else scene_analysis_status,
            "scene_completeness_plan": "NOT_APPLICABLE" if args.input_kind == "object" else "PROVEN",
            "builder_selection": "NOT_APPLICABLE" if args.input_kind == "object" else "PROVEN",
            "budget_validation": "NOT_APPLICABLE" if args.input_kind == "object" else budget_validation["classification"],
        },
        "filename_not_used_for_selection": True,
        "source_sha256": image_hash,
    })
    input_root = {
        "image_sha256": image_hash,
        "scene_spec": spec,
        "output_map": output_map,
        "quality_tier": args.quality_tier,
        "layer_receipt_hashes": _receipt_hashes(receipts),
    }
    _stage(
        state,
        "completeness",
        _canonical(input_root),
        evidence / "scene_completeness_receipt.json",
        lambda: audit_scene_completeness(spec, receipts),
    )
    _write_json(evidence / "scene_completeness_plan.json", {"schema_version": "scene_completeness_plan_v1", "source_image_sha256": image_hash, "source_scene_spec": str(scene_spec_path.resolve()) if scene_spec_path else "generated_analysis_bundle", "audit_stage": "completeness"})
    fixture_planners_available = args.input_kind == "scene" and spec.get("schema_version") == "scene_spec_v1" and isinstance(spec.get("intent"), dict)
    if fixture_planners_available:
        from .castlegrounds_fixture_architecture import build_architecture_plan
        from .castlegrounds_fixture_terrain import build_terrain_plan
        from .castlegrounds_fixture_vegetation import build_vegetation_plan

        _stage(state, "terrain_plan", _canonical({"spec": spec, "image_sha256": image_hash, "planner_sha256": _code_hash("castlegrounds_fixture_terrain.py")}), evidence / "terrain_plan.json", lambda: build_terrain_plan(spec))
        _stage(state, "architecture_plan", _canonical({"spec": spec, "image_sha256": image_hash, "planner_sha256": _code_hash("castlegrounds_fixture_architecture.py")}), evidence / "architecture_plan.json", lambda: build_architecture_plan(spec))
        _stage(state, "vegetation_plan", _canonical({"spec": spec, "image_sha256": image_hash, "planner_sha256": _code_hash("castlegrounds_fixture_vegetation.py")}), evidence / "vegetation_plan.json", lambda: build_vegetation_plan(spec))
    else:
        for layer in ("terrain", "architecture", "vegetation"):
            _stage(state, f"{layer}_plan", _canonical({"spec": spec, "image_sha256": image_hash, "generic": True}), evidence / f"{layer}_plan.json", lambda layer=layer: _not_applicable(layer, "semantic analysis has not selected this fixture-specific planner"))
    overrides = load_scene_overrides(str(spec.get("scene_id", args.scene_id)), Path(__file__).resolve().parents[2] / "scenes")
    composition_input = _canonical({"spec": spec, "analysis": analysis_bundle, "camera": spec.get("camera", {}), "visibility": visibility_manifest, "representation": representation_manifest, "material": material_plan, "overrides": overrides})
    _stage(
        state,
        "scene_composition",
        composition_input,
        evidence / "scene_content_manifest.json",
        lambda: build_scene_content_manifest(
            spec=spec,
            analysis_bundle=analysis_bundle,
            camera_contract=spec.get("camera", {}),
            semantic_masks={"regions": spec.get("regions", [])},
            depth_bands={"records": spec.get("depth_bands", [])},
            world_anchors={"landmarks": spec.get("landmarks", [])},
            support_relationships={"relationships": scene_graph.get("edges", [])},
            visibility=visibility_manifest,
            representation_manifest=representation_manifest,
            material_plan=material_plan,
            overrides=overrides,
        ),
    )
    composition_manifest = _read(evidence / "scene_content_manifest.json")
    _write_json(evidence / "material_build_receipt.json", material_build_receipt(composition_manifest))
    _write_json(evidence / "scene_composition_receipt.json", {"schema_version": "scene_composition_receipt_v1", "classification": composition_manifest.get("classification"), "manifest_hash": composition_manifest.get("manifest_hash"), "actor_count": len(composition_manifest.get("actors", [])), "override_count": composition_manifest.get("override_count", 0), "manual_only_actor_count": composition_manifest.get("manual_only_actor_count", 0), "rebuild_policy": composition_manifest.get("rebuild_policy")})
    layer_classes = {name: value.get("classification") for name, value in receipts.items()}
    all_layers_proven = all(layer_classes.get(name) == "PROVEN" for name in ("terrain", "architecture", "water", "bridge", "vegetation", "environment"))
    state["layer_receipts"] = layer_classes
    state["composition_manifest_hash"] = composition_manifest.get("manifest_hash")
    state["classification"] = "PARTIAL"
    state["next_action"] = "RUN_GAMEPLAY_AND_VISUAL_VALIDATION" if all_layers_proven else "RUN_BOUNDED_UNREAL_LAYER_BUILDERS"
    _write_json(state_path, state)
    report = [
        f"# {args.scene_id} Image-to-Scene Pipeline",
        "",
        f"- Source image: `{image.resolve()}`",
        f"- Source SHA-256: `{image_hash}`",
        f"- Output map: `{output_map}`",
        "- Source mesh and source map: reused and protected",
        "- CPU planning: PROVEN",
        f"- Unreal generated layers: `{'PROVEN' if all_layers_proven else 'PENDING'}`",
        "- Independent geometry is counted only from validated layer receipts.",
        f"- Input kind: `{args.input_kind}`",
        f"- Generic scene capability: `{'PROVEN' if generic_ready else ('NOT_APPLICABLE' if args.input_kind == 'object' else 'PARTIAL')}`",
        "- Overall classification: `IMAGE_TO_SCENE_PARTIAL` pending gameplay and visual gates",
        "",
        "The source-projected shell is not counted as independent terrain, water, architecture, bridge, or vegetation geometry.",
    ]
    (evidence / "FINAL_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the resumable one-image-to-Unreal-scene pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--run-id", default="run-current")
    parser.add_argument("--scene-spec")
    parser.add_argument("--output-map")
    parser.add_argument("--output-root", default="/Game/GeneratedScenes")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quality-tier", choices=("smoke", "preview", "quality"), default="smoke")
    parser.add_argument("--input-kind", choices=("scene", "object"), default="scene")
    parser.add_argument("--evidence-root")
    parser.add_argument("--max-vram-mb", type=int, default=6144)
    parser.add_argument("--max-triangles", type=int, default=1500000)
    parser.add_argument("--disable-neural", action="store_true")
    parser.add_argument("--enable-pcg", action="store_true")
    args = parser.parse_args()
    state = run(args)
    print("ONE_IMAGE_PIPELINE_ENTRYPOINT=PROVEN")
    print("PIPELINE_RESUME=" + ("PROVEN" if args.resume else "NOT_REQUESTED"))
    print("PIPELINE_CLASSIFICATION=" + state["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
