"""Authoritative camera contracts for source-locked scene reconstruction."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vec3(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(value: list[float]) -> float:
    return math.sqrt(_dot(value, value))


def _cross(a: list[float], b: list[float]) -> list[float]:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b, strict=True)]


def _close(a: float, b: float, tolerance: float = 1e-6) -> bool:
    return abs(a - b) <= tolerance


def build_camera_contract(
    calibration: dict[str, Any],
    exact_blender_camera: dict[str, Any],
    interpretation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if calibration.get("classification") != "CAMERA_CALIBRATION_PROVEN":
        raise ValueError("camera calibration is not proven")
    if exact_blender_camera.get("classification") != "BLENDER_EXACT_SOURCE_CAMERA_RENDER_READY":
        raise ValueError("exact Blender camera is not proven")

    camera = exact_blender_camera.get("camera")
    intrinsics = calibration.get("intrinsics")
    principal = calibration.get("principal_point_normalized")
    resolution = calibration.get("input_resolution")
    interpretation_camera = interpretation.get("camera_estimate")
    if not isinstance(camera, dict):
        raise ValueError("exact Blender camera payload is missing")
    if not isinstance(interpretation_camera, dict):
        raise ValueError("legacy camera estimate is missing")
    if not isinstance(intrinsics, list) or len(intrinsics) != 3:
        raise ValueError("camera intrinsics are invalid")
    if not isinstance(principal, list) or len(principal) != 2:
        raise ValueError("principal point is invalid")
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise ValueError("input resolution is invalid")

    origin = _vec3(camera.get("origin"), "origin")
    forward = _vec3(camera.get("forward"), "forward")
    up = _vec3(camera.get("up"), "up")
    right = _vec3(camera.get("right"), "right")

    basis_norms = {"forward": _norm(forward), "up": _norm(up), "right": _norm(right)}
    if not all(_close(value, 1.0, 1e-5) for value in basis_norms.values()):
        raise ValueError(f"camera basis is not normalized: {basis_norms}")
    orthogonality = {
        "forward_up": _dot(forward, up),
        "forward_right": _dot(forward, right),
        "up_right": _dot(up, right),
    }
    if not all(abs(value) <= 1e-5 for value in orthogonality.values()):
        raise ValueError(f"camera basis is not orthogonal: {orthogonality}")
    cross_forward_up = _cross(forward, up)
    handedness_error = _norm([a - b for a, b in zip(cross_forward_up, right, strict=True)])
    if handedness_error > 1e-5:
        raise ValueError(f"camera basis handedness mismatch: {handedness_error}")

    fov_x = float(camera.get("fov_x_deg"))
    fov_y = float(camera.get("fov_y_deg"))
    if not _close(fov_x, float(calibration.get("fov_x_deg")), 1e-6):
        raise ValueError("horizontal FOV mismatch between proven receipts")
    if not _close(fov_y, float(calibration.get("fov_y_deg")), 1e-6):
        raise ValueError("vertical FOV mismatch between proven receipts")

    contract = {
        "schema_version": "scene_camera_contract_v1",
        "classification": "PROVEN",
        "camera_id": "source",
        "source_frame": calibration.get("source_frame"),
        "target_frame": "BLENDER_RIGHT_HANDED_Z_UP_CAMERA_MINUS_Z_FORWARD",
        "origin_m": origin,
        "forward": forward,
        "up": up,
        "right": right,
        "fov_x_deg": fov_x,
        "fov_y_deg": fov_y,
        "lens_mm": float(camera.get("lens_mm")),
        "principal_point_normalized": [float(principal[0]), float(principal[1])],
        "resolution_px": [int(resolution[0]), int(resolution[1])],
        "near_m": float(interpretation_camera.get("near_plane", 0.1)),
        "far_m": float(interpretation_camera.get("far_plane", 500.0)),
        "intrinsics": intrinsics,
        "basis_norms": basis_norms,
        "basis_orthogonality": orthogonality,
        "basis_handedness_error": handedness_error,
        "legacy_interpretation_fov_x_deg": float(
            interpretation_camera.get("field_of_view_deg", 0.0)
        ),
        "legacy_interpretation_superseded": not _close(
            fov_x, float(interpretation_camera.get("field_of_view_deg", 0.0)), 1e-6
        ),
    }
    source_view = {
        "id": "source",
        "projection": "perspective",
        "field_of_view_deg": fov_x,
        "position_m": origin,
        "look_at_m": _add(origin, forward),
        "principal_point_norm": contract["principal_point_normalized"],
        "near_m": contract["near_m"],
        "far_m": contract["far_m"],
        "purpose": "source_match",
    }
    receipt = {
        "schema_version": "scene_camera_contract_receipt_v1",
        "classification": "PROVEN",
        "contract_sha256": _hash(contract),
        "source_view_sha256": _hash(source_view),
        "legacy_interpretation_superseded": contract["legacy_interpretation_superseded"],
        "legacy_fov_x_deg": contract["legacy_interpretation_fov_x_deg"],
        "authoritative_fov_x_deg": fov_x,
        "authoritative_fov_y_deg": fov_y,
        "basis_normalized": True,
        "basis_orthogonal": True,
        "basis_right_handed": True,
        "gpu_work_started": False,
        "neural_work_started": False,
        "blender_render_started": False,
        "unreal_work_started": False,
    }
    return contract, {"source_view": source_view, "receipt": receipt}


def apply_camera_contract_to_scene_spec(
    spec: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    if contract.get("schema_version") != "scene_camera_contract_v1":
        raise ValueError("unsupported camera contract")
    if contract.get("classification") != "PROVEN":
        raise ValueError("camera contract is not proven")
    source_view = {
        "id": contract["camera_id"],
        "projection": "perspective",
        "field_of_view_deg": contract["fov_x_deg"],
        "position_m": contract["origin_m"],
        "look_at_m": _add(contract["origin_m"], contract["forward"]),
        "principal_point_norm": contract["principal_point_normalized"],
        "near_m": contract["near_m"],
        "far_m": contract["far_m"],
        "purpose": "source_match",
    }
    result = json.loads(json.dumps(spec))
    result["camera"]["source_camera"] = source_view
    views = result["camera"].get("required_views", [])
    replaced = False
    for index, view in enumerate(views):
        if view.get("id") == source_view["id"]:
            views[index] = dict(source_view)
            replaced = True
            break
    if not replaced:
        views.insert(0, dict(source_view))
    result["camera"]["required_views"] = views
    result.setdefault("outputs", {})["authoritative_camera_contract"] = (
        "evidence/latest-scene-camera-local-worker/camera_contract.json"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and apply an authoritative scene camera contract")
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--exact-blender-camera", required=True)
    parser.add_argument("--interpretation", required=True)
    parser.add_argument("--contract-output", required=True)
    parser.add_argument("--receipt-output", required=True)
    parser.add_argument("--scene-spec-in")
    parser.add_argument("--scene-spec-out")
    args = parser.parse_args()

    try:
        contract, bundle = build_camera_contract(
            _read(args.calibration),
            _read(args.exact_blender_camera),
            _read(args.interpretation),
        )
        contract_path = Path(args.contract_output)
        contract_path.parent.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True), encoding="utf-8")
        receipt_path = Path(args.receipt_output)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(bundle["receipt"], indent=2, sort_keys=True), encoding="utf-8"
        )
        if args.scene_spec_in or args.scene_spec_out:
            if not args.scene_spec_in or not args.scene_spec_out:
                raise ValueError("scene-spec-in and scene-spec-out must be used together")
            spec = _read(args.scene_spec_in)
            updated = apply_camera_contract_to_scene_spec(spec, contract)
            output = Path(args.scene_spec_out)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(updated, indent=2, sort_keys=True), encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"SCENE_CAMERA_CONTRACT=REJECTED: {exc}")
        return 2

    print("SCENE_CAMERA_CONTRACT=PROVEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
