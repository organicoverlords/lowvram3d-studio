"""Compose a hybrid gameplay SceneSpec without duplicating camera authority."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .scene_spec import validate_scene_spec


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _add(a: list[float], b: list[float]) -> list[float]:
    return [float(x) + float(y) for x, y in zip(a, b, strict=True)]


def _scale(value: list[float], scalar: float) -> list[float]:
    return [float(item) * scalar for item in value]


def _same_vector(a: list[float], b: list[float], tolerance: float = 1e-9) -> bool:
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= tolerance for x, y in zip(a, b, strict=True))


def _camera_view(
    contract: dict[str, Any],
    view_id: str,
    purpose: str,
    translation_m: list[float],
) -> dict[str, Any]:
    origin = _add(contract["origin_m"], translation_m)
    look_at = _add(_add(contract["origin_m"], contract["forward"]), translation_m)
    return {
        "id": view_id,
        "projection": "perspective",
        "field_of_view_deg": float(contract["fov_x_deg"]),
        "position_m": origin,
        "look_at_m": look_at,
        "principal_point_norm": [
            float(contract["principal_point_normalized"][0]),
            float(contract["principal_point_normalized"][1]),
        ],
        "near_m": float(contract["near_m"]),
        "far_m": float(contract["far_m"]),
        "purpose": purpose,
    }


def compose_authoritative_hybrid_spec(
    authored_hybrid: dict[str, Any],
    camera_contract: dict[str, Any],
    scene_build_receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if camera_contract.get("schema_version") != "scene_camera_contract_v1":
        raise ValueError("unsupported camera contract")
    if camera_contract.get("classification") != "PROVEN":
        raise ValueError("camera contract is not proven")
    if scene_build_receipt.get("classification") != "SCENE_BUILD_PROVEN":
        raise ValueError("scene build receipt is not proven")

    spec = json.loads(json.dumps(authored_hybrid))
    source_view = _camera_view(camera_contract, "source", "source_match", [0.0, 0.0, 0.0])
    right = [float(item) for item in camera_contract["right"]]
    left_offset = _scale(right, -1.0)
    right_offset = _scale(right, 1.0)
    left_view = _camera_view(camera_contract, "offset_left", "offset_parallax", left_offset)
    right_view = _camera_view(camera_contract, "offset_right", "offset_parallax", right_offset)

    spec["scene_id"] = "castlegrounds_hybrid_authoritative_v1"
    spec["camera"] = {
        "source_camera": source_view,
        "required_views": [source_view, left_view, right_view],
        "envelope": {
            "max_translation_m": 2.0,
            "max_yaw_deg": 5.0,
            "max_pitch_deg": 3.0,
        },
    }
    spec["intent"]["description"] = (
        "Authoritative source camera and visual shell with bounded, independently "
        "proved gameplay and PCG layers."
    )

    source_mesh_path = scene_build_receipt.get("mesh_asset")
    map_path = scene_build_receipt.get("map")
    if not isinstance(source_mesh_path, str) or not source_mesh_path:
        raise ValueError("scene build receipt has no mesh asset")
    if not isinstance(map_path, str) or not map_path:
        raise ValueError("scene build receipt has no map")

    for asset in spec.get("assets", []):
        tags = list(asset.get("tags", []))
        if asset.get("id") == "source_mesh_v2":
            if "authoritative_camera_bound" not in tags:
                tags.append("authoritative_camera_bound")
            if "proven_unreal_import" not in tags:
                tags.append("proven_unreal_import")
        else:
            if "placement_unproven" not in tags:
                tags.append("placement_unproven")
            if "not_promoted" not in tags:
                tags.append("not_promoted")
        asset["tags"] = tags

    spec.setdefault("outputs", {})["authoritative_camera_contract"] = (
        "evidence/latest-scene-camera-local-worker/camera_contract.json"
    )
    spec["outputs"]["unreal_map"] = map_path
    spec["outputs"]["unreal_source_mesh_asset"] = source_mesh_path
    spec["outputs"]["unreal_asset_root"] = "/Game/AgentProof/ImageToSceneSmoke_20260803/"
    spec["proof"]["current_classification"] = "PARTIAL"

    validation = validate_scene_spec(spec)
    source_direction = [
        source_view["look_at_m"][index] - source_view["position_m"][index]
        for index in range(3)
    ]
    left_direction = [
        left_view["look_at_m"][index] - left_view["position_m"][index]
        for index in range(3)
    ]
    right_direction = [
        right_view["look_at_m"][index] - right_view["position_m"][index]
        for index in range(3)
    ]
    receipt = {
        "schema_version": "scene_hybrid_composition_receipt_v1",
        "classification": "PROVEN" if validation["scene_spec_valid"] else "REJECTED",
        "scene_spec_valid": validation["scene_spec_valid"],
        "validation_errors": validation["errors"],
        "authoritative_fov_x_deg": source_view["field_of_view_deg"],
        "legacy_fov_removed": source_view["field_of_view_deg"] != 48.0,
        "parallel_offset_views": _same_vector(source_direction, left_direction)
        and _same_vector(source_direction, right_direction),
        "offset_distance_m": 1.0,
        "source_mesh_unreal_asset_preserved": source_mesh_path,
        "unreal_map_preserved": map_path,
        "unbuilt_assets_marked_not_promoted": all(
            asset.get("id") == "source_mesh_v2" or "not_promoted" in asset.get("tags", [])
            for asset in spec.get("assets", [])
        ),
        "gpu_work_started": False,
        "neural_work_started": False,
        "blender_work_started": False,
        "unreal_work_started": False,
    }
    if not all(
        [
            receipt["scene_spec_valid"],
            receipt["legacy_fov_removed"],
            receipt["parallel_offset_views"],
            receipt["unbuilt_assets_marked_not_promoted"],
        ]
    ):
        receipt["classification"] = "REJECTED"
    return spec, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose authoritative hybrid SceneSpec")
    parser.add_argument("--authored-hybrid", required=True)
    parser.add_argument("--camera-contract", required=True)
    parser.add_argument("--scene-build-receipt", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()

    try:
        spec, receipt = compose_authoritative_hybrid_spec(
            _read(args.authored_hybrid),
            _read(args.camera_contract),
            _read(args.scene_build_receipt),
        )
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        receipt = {
            "schema_version": "scene_hybrid_composition_receipt_v1",
            "classification": "REJECTED",
            "scene_spec_valid": False,
            "error": str(exc),
            "gpu_work_started": False,
            "neural_work_started": False,
            "blender_work_started": False,
            "unreal_work_started": False,
        }
        Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.receipt).write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        print("SCENE_HYBRID_COMPOSITION=REJECTED")
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(f"SCENE_HYBRID_COMPOSITION={receipt['classification']}")
    return 0 if receipt["classification"] == "PROVEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
