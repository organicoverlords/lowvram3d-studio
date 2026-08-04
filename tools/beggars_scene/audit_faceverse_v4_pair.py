from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FaceVerse v4 release schemas without reconstruction.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape(value: Any) -> list[int] | None:
    return [int(item) for item in value.shape] if hasattr(value, "shape") else None


def capture(callable_object) -> dict[str, Any]:
    try:
        return {"ok": True, "value": callable_object()}
    except Exception as error:  # diagnostic boundary
        return {
            "ok": False,
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
        }


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root).resolve()
    model_path = Path(args.model).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = np.load(model_path, allow_pickle=True).item()
    keys = set(model_data.keys())
    required = {
        "keypoints_mediapipe", "keypoints", "keypoints_68", "keypoints_name_list",
        "ver_inds", "tri_inds", "idBase", "texBase", "exBase", "meanshape",
        "face_mask", "parsing", "tri", "point_buf", "meantex",
    }
    report: dict[str, Any] = {
        "classification": "PAIR_SCHEMA_AUDIT_COMPLETE",
        "source_root": str(source_root),
        "model": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": sha256(model_path),
            "keys": sorted(str(key) for key in keys),
            "missing_official_keys": sorted(required - keys),
            "shapes": {str(key): shape(value) for key, value in sorted(model_data.items(), key=lambda item: str(item[0]))},
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
        },
    }

    def validate_model_schema() -> dict[str, Any]:
        missing = required - keys
        if missing:
            raise KeyError(f"Missing official FaceVerse v4 keys: {sorted(missing)}")
        ver_inds = np.asarray(model_data["ver_inds"]).astype(np.int64).reshape(-1)
        vertex_count = int(np.asarray(model_data["meanshape"]).reshape(-1, 3).shape[0])
        triangle_count = int(np.asarray(model_data["tri"]).reshape(-1, 3).shape[0])
        kp = np.concatenate([
            np.asarray(model_data["keypoints_mediapipe"]).flatten(),
            np.asarray(model_data["keypoints"]).flatten(),
            np.asarray(model_data["keypoints_68"]).flatten(),
            np.asarray(model_data["keypoints_mediapipe"]).flatten()[[468, 473]],
        ]).astype(np.int64)
        checks = {
            "vertex_count": vertex_count,
            "triangle_count": triangle_count,
            "ver_inds": [int(item) for item in ver_inds],
            "max_keypoint_index": int(kp.max()),
            "id_dims": int(np.asarray(model_data["idBase"]).shape[1]),
            "exp_dims": int(np.asarray(model_data["exBase"]).shape[1]),
            "tex_dims": int(np.asarray(model_data["texBase"]).shape[1]),
            "face_mask_length": int(np.asarray(model_data["face_mask"]).size),
            "skin_mask_length": int(np.asarray(model_data["parsing"]["skin"]).size),
            "point_buf_shape": shape(np.asarray(model_data["point_buf"])),
        }
        assert kp.min() >= 0 and kp.max() < vertex_count, checks
        assert int(ver_inds[-1]) == vertex_count, checks
        assert np.asarray(model_data["idBase"]).shape[0] == vertex_count * 3, checks
        assert np.asarray(model_data["exBase"]).shape[0] == vertex_count * 3, checks
        assert np.asarray(model_data["texBase"]).shape[0] == vertex_count * 3, checks
        assert np.asarray(model_data["meantex"]).size == vertex_count * 3, checks
        assert np.asarray(model_data["face_mask"]).size == vertex_count, checks
        assert np.asarray(model_data["parsing"]["skin"]).size == vertex_count, checks
        front_vertices = set(np.arange(vertex_count)[np.asarray(model_data["face_mask"]).reshape(-1) > 0].tolist())
        front_face_count = sum(
            1 for face in np.asarray(model_data["tri"]).reshape(-1, 3)
            if int(face[0]) in front_vertices and int(face[1]) in front_vertices and int(face[2]) in front_vertices
        )
        checks["front_face_count"] = int(front_face_count)
        assert front_face_count > 0, checks
        return checks

    report["official_model_schema"] = capture(validate_model_schema)

    sys.path.insert(0, str(source_root))
    from faceversev4.FaceVerse_networks import ReconNet

    checkpoint_object = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint_object.get("state_dict", checkpoint_object) if isinstance(checkpoint_object, dict) else checkpoint_object
    report["checkpoint"]["container_type"] = type(checkpoint_object).__name__
    report["checkpoint"]["top_level_keys"] = sorted(str(key) for key in checkpoint_object.keys()) if isinstance(checkpoint_object, dict) else []
    report["checkpoint"]["state_key_count"] = len(state_dict) if isinstance(state_dict, dict) else None

    expected = ReconNet().state_dict()
    expected_keys = set(expected.keys())
    actual_keys = set(state_dict.keys()) if isinstance(state_dict, dict) else set()
    report["checkpoint"]["missing_keys"] = sorted(expected_keys - actual_keys)
    report["checkpoint"]["unexpected_keys"] = sorted(actual_keys - expected_keys)
    report["checkpoint"]["shape_mismatches"] = [
        {
            "key": key,
            "expected": shape(expected[key]),
            "actual": shape(state_dict[key]),
        }
        for key in sorted(expected_keys & actual_keys)
        if tuple(expected[key].shape) != tuple(state_dict[key].shape)
    ]

    def strict_load() -> dict[str, Any]:
        result = ReconNet().load_state_dict(state_dict, strict=True)
        return {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}

    report["checkpoint_strict_load"] = capture(strict_load)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"FACEVERSE_PAIR_AUDIT={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
