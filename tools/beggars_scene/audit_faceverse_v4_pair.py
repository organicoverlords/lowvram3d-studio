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
    parser = argparse.ArgumentParser(description="Audit a FaceVerse v4 model/checkpoint pair without reconstruction.")
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


def describe(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": type(value).__name__}
    if hasattr(value, "shape"):
        result["shape"] = [int(item) for item in value.shape]
    if hasattr(value, "dtype"):
        result["dtype"] = str(value.dtype)
    if isinstance(value, dict):
        result["keys"] = sorted(str(key) for key in value.keys())
    return result


def capture(callable_object) -> dict[str, Any]:
    try:
        value = callable_object()
        return {"ok": True, "value": value}
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

    report: dict[str, Any] = {
        "classification": "PAIR_AUDIT_COMPLETE",
        "source_root": str(source_root),
        "model": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": sha256(model_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256(checkpoint_path),
        },
    }

    model_data = np.load(model_path, allow_pickle=True).item()
    required_model_keys = {
        "keypoints_mediapipe",
        "keypoints",
        "keypoints_68",
        "keypoints_name_list",
        "ver_inds",
        "tri_inds",
        "idBase",
        "texBase",
        "exBase",
        "meanshape",
        "face_mask",
        "parsing",
        "tri",
        "point_buf",
        "meantex",
    }
    report["model"]["keys"] = sorted(str(key) for key in model_data.keys())
    report["model"]["missing_official_keys"] = sorted(required_model_keys - set(model_data.keys()))
    report["model"]["schema"] = {
        str(key): describe(value)
        for key, value in sorted(model_data.items(), key=lambda item: str(item[0]))
    }

    sys.path.insert(0, str(source_root))
    from faceversev4.FaceVerseModel_torch import FaceVerseModel_torch
    from faceversev4.FaceVerse_networks import FaceVerseRecon, ReconNet

    report["model_only_initialization"] = capture(
        lambda: {
            "id_dims": int((instance := FaceVerseModel_torch(torch.device("cpu"), str(model_path), 10, 1000, 128)).id_dims),
            "exp_dims": int(instance.exp_dims),
            "tex_dims": int(instance.tex_dims),
            "vertices": int(instance.meanshape.shape[1]),
            "triangles": int(instance.tri.shape[0]),
        }
    )

    checkpoint_object = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint_object.get("state_dict", checkpoint_object) if isinstance(checkpoint_object, dict) else checkpoint_object
    report["checkpoint"]["container_type"] = type(checkpoint_object).__name__
    report["checkpoint"]["state_dict_type"] = type(state_dict).__name__
    report["checkpoint"]["state_key_count"] = len(state_dict) if isinstance(state_dict, dict) else None
    report["checkpoint"]["state_keys"] = sorted(str(key) for key in state_dict.keys()) if isinstance(state_dict, dict) else []

    expected = ReconNet().state_dict()
    expected_keys = set(expected.keys())
    actual_keys = set(state_dict.keys()) if isinstance(state_dict, dict) else set()
    report["checkpoint"]["missing_keys"] = sorted(expected_keys - actual_keys)
    report["checkpoint"]["unexpected_keys"] = sorted(actual_keys - expected_keys)
    report["checkpoint"]["shape_mismatches"] = [
        {
            "key": key,
            "expected": [int(item) for item in expected[key].shape],
            "actual": [int(item) for item in state_dict[key].shape],
        }
        for key in sorted(expected_keys & actual_keys)
        if tuple(expected[key].shape) != tuple(state_dict[key].shape)
    ]

    def load_checkpoint() -> dict[str, Any]:
        network = ReconNet()
        result = network.load_state_dict(state_dict, strict=True)
        return {"missing": list(result.missing_keys), "unexpected": list(result.unexpected_keys)}

    report["checkpoint_strict_load"] = capture(load_checkpoint)

    def load_pair() -> dict[str, Any]:
        instance = FaceVerseRecon(str(model_path), str(checkpoint_path), torch.device("cpu"))
        return {
            "id_dims": int(instance.id_dims),
            "exp_dims": int(instance.exp_dims),
            "tex_dims": int(instance.tex_dims),
            "head_channels": [int(layer.out_channels) for layer in instance.reconnet.final_layers],
        }

    report["official_pair_initialization"] = capture(load_pair)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"FACEVERSE_PAIR_AUDIT={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
