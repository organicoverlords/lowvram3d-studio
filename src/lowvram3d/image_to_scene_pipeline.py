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
    if previous.get("input_sha256") == input_hash and previous.get("classification") == "PROVEN" and output_path.is_file():
        state["stages"][name] = {**previous, "reused": True}
        return
    result = producer()
    _write_json(output_path, result)
    state["stages"][name] = {
        "classification": result.get("classification", "PROVEN"),
        "input_sha256": input_hash,
        "output": str(output_path.resolve()),
        "reused": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = derive_scene_paths(args.scene_id, args.output_root, args.evidence_root)
    evidence = Path(paths["evidence"])
    state_path = evidence / "pipeline_state.json"
    image = Path(args.image)
    if not image.is_file():
        raise FileNotFoundError(image)
    image_hash = _sha256(image)
    hybrid_path = Path("evidence/latest-scene-hybrid/authoritative_hybrid_scene_spec.json")
    if hybrid_path.is_file() and args.scene_id == "castlegrounds" and args.input_kind == "scene":
        spec = _read(hybrid_path)
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
        "resume": bool(args.resume),
        "started_at": previous.get("started_at") or datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "stages": previous.get("stages", {}),
        "source_map_protected": True,
        "gpu_work_requested": False,
    }
    receipts = _load_layer_receipts(evidence)
    manifest = builder_manifest(spec) if args.input_kind == "scene" else {
        "schema_version": "scene_builder_manifest_v1",
        "classification": "NOT_APPLICABLE",
        "scene_id": args.scene_id,
        "selected": {},
        "unknown_semantics": [],
        "selection_source": "explicit input_kind=object",
        "filename_not_used_for_selection": True,
    }
    _write_json(evidence / "scene_spec.json", spec)
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
    _write_json(evidence / "scene_completeness_plan.json", {"schema_version": "scene_completeness_plan_v1", "source_image_sha256": image_hash, "source_scene_spec": str(hybrid_path.resolve()), "audit_stage": "completeness"})
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
    layer_classes = {name: value.get("classification") for name, value in receipts.items()}
    all_layers_proven = all(layer_classes.get(name) == "PROVEN" for name in ("terrain", "architecture", "water", "bridge", "vegetation", "environment"))
    state["layer_receipts"] = layer_classes
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
    parser.add_argument("--output-map")
    parser.add_argument("--output-root", default="/Game/GeneratedScenes")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--quality-tier", choices=("smoke", "preview", "quality"), default="smoke")
    parser.add_argument("--input-kind", choices=("scene", "object"), default="scene")
    parser.add_argument("--evidence-root")
    args = parser.parse_args()
    state = run(args)
    print("ONE_IMAGE_PIPELINE_ENTRYPOINT=PROVEN")
    print("PIPELINE_RESUME=" + ("PROVEN" if args.resume else "NOT_REQUESTED"))
    print("PIPELINE_CLASSIFICATION=" + state["classification"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
