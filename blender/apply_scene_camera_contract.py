"""Apply an authoritative Scene CameraContract to an existing prepared blend."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Matrix, Vector

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _vector(value: Any, name: str) -> Vector:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    vector = Vector(tuple(float(item) for item in value))
    if abs(vector.length - 1.0) > 1e-5 and name != "origin_m":
        raise ValueError(f"{name} must be normalized")
    return vector


def _camera_state(obj: bpy.types.Object) -> dict[str, Any]:
    matrix = obj.matrix_world.to_3x3()
    right = matrix @ Vector((1.0, 0.0, 0.0))
    up = matrix @ Vector((0.0, 1.0, 0.0))
    forward = matrix @ Vector((0.0, 0.0, -1.0))
    return {
        "origin_m": list(obj.matrix_world.translation),
        "right": list(right),
        "up": list(up),
        "forward": list(forward),
        "fov_x_deg": math.degrees(obj.data.angle_x) if obj.data.type == "PERSP" else None,
        "fov_y_deg": math.degrees(obj.data.angle_y) if obj.data.type == "PERSP" else None,
        "near_m": obj.data.clip_start,
        "far_m": obj.data.clip_end,
        "shift_x": obj.data.shift_x,
        "shift_y": obj.data.shift_y,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply an exact source camera contract")
    parser.add_argument("--camera-contract", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(_argv_after_double_dash())

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    before: dict[str, Any] | None = None
    after: dict[str, Any] | None = None

    try:
        contract = _read(args.camera_contract)
        if contract.get("schema_version") != "scene_camera_contract_v1":
            raise ValueError("unsupported camera contract")
        if contract.get("classification") != "PROVEN":
            raise ValueError("camera contract is not proven")

        obj = bpy.data.objects.get("SCENE_SOURCE_CAMERA")
        if obj is None or obj.type != "CAMERA":
            raise ValueError("SCENE_SOURCE_CAMERA is missing")
        before = _camera_state(obj)

        origin = Vector(tuple(float(item) for item in contract["origin_m"]))
        right = _vector(contract["right"], "right")
        up = _vector(contract["up"], "up")
        forward = _vector(contract["forward"], "forward")
        backward = -forward
        obj.matrix_world = Matrix(
            (
                (right.x, up.x, backward.x, origin.x),
                (right.y, up.y, backward.y, origin.y),
                (right.z, up.z, backward.z, origin.z),
                (0.0, 0.0, 0.0, 1.0),
            )
        )
        obj.data.type = "PERSP"
        obj.data.sensor_fit = "HORIZONTAL"
        obj.data.angle = math.radians(float(contract["fov_x_deg"]))
        obj.data.clip_start = float(contract["near_m"])
        obj.data.clip_end = float(contract["far_m"])
        principal = contract["principal_point_normalized"]
        obj.data.shift_x = 0.5 - float(principal[0])
        obj.data.shift_y = float(principal[1]) - 0.5
        obj["scene_camera_contract_sha256"] = _hash(contract)
        obj["scene_camera_contract_schema"] = contract["schema_version"]
        obj["legacy_camera_superseded"] = bool(
            contract.get("legacy_interpretation_superseded")
        )
        bpy.context.scene["scene_camera_contract_sha256"] = _hash(contract)
        bpy.context.scene["scene_camera_contract_applied"] = True
        after = _camera_state(obj)

        vector_error = {
            "origin": (Vector(after["origin_m"]) - origin).length,
            "right": (Vector(after["right"]) - right).length,
            "up": (Vector(after["up"]) - up).length,
            "forward": (Vector(after["forward"]) - forward).length,
            "fov_x_deg": abs(after["fov_x_deg"] - float(contract["fov_x_deg"])),
        }
        if any(value > 1e-5 for value in vector_error.values()):
            raise ValueError(f"applied camera differs from contract: {vector_error}")

        current_blend = Path(bpy.data.filepath)
        if not current_blend:
            raise ValueError("prepared blend has no filepath")
        bpy.ops.wm.save_as_mainfile(filepath=str(current_blend))
        classification = "PROVEN"
    except Exception as exc:
        classification = "REJECTED"
        errors.append(f"{type(exc).__name__}: {exc}")
        vector_error = {}
        contract = locals().get("contract", {})

    receipt = {
        "schema_version": "blender_scene_camera_application_receipt_v1",
        "classification": classification,
        "blend": bpy.data.filepath,
        "camera_contract": str(Path(args.camera_contract).resolve()),
        "camera_contract_sha256": _hash(contract) if contract else None,
        "before": before,
        "after": after,
        "contract_error": vector_error,
        "legacy_camera_superseded": bool(
            contract.get("legacy_interpretation_superseded")
        ) if contract else None,
        "errors": errors,
        "mesh_edit_operations": 0,
        "gpu_work_started": False,
        "neural_work_started": False,
        "unreal_work_started": False,
    }
    report_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(f"BLENDER_SCENE_CAMERA_CONTRACT={classification}")
    return 0 if classification == "PROVEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
