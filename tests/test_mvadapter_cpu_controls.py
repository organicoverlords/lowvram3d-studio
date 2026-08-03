from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from build_mvadapter_cpu_controls import build_camera_contract, build_controls  # noqa: E402
from mesh_io import write_glb  # noqa: E402


def _cube(path: Path) -> None:
    vertices = np.array(
        [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], np.float32
    )
    tris = np.array(
        [[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
         [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], np.uint32
    )
    uv = np.zeros((len(vertices), 2), np.float32)
    normals = vertices / np.maximum(np.linalg.norm(vertices, axis=1, keepdims=True), 1e-6)
    write_glb(path, vertices, normals, uv, tris)


def test_camera_contract_is_exact_six_view_and_opposed() -> None:
    contract = build_camera_contract()
    assert contract["schema"] == "lowvram3d_mvadapter_camera_contract_v1"
    assert contract["view_count"] == 6
    assert [view["index"] for view in contract["views"]] == list(range(6))
    assert contract["front_rear_direction_dot"] <= -0.999
    assert contract["left_right_direction_dot"] <= -0.999
    assert contract["top_bottom_direction_dot"] <= -0.999
    assert contract["handedness_proven"] is True
    assert all(view["fixture_gate_passed"] for view in contract["views"])


def test_cpu_controls_shape_encoding_and_immutable_mesh(tmp_path: Path) -> None:
    mesh = tmp_path / "fixture.glb"
    _cube(mesh)
    before = hashlib.sha256(mesh.read_bytes()).hexdigest()
    report = build_controls(mesh, tmp_path / "controls", size=32)
    tensor = np.load(tmp_path / "controls" / "control_tensor.npy")
    assert tensor.shape == (6, 6, 32, 32)
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0
    assert report["mesh_sha256_before"] == before
    assert report["mesh_sha256_after"] == before
    assert report["passed"] is True
    assert json.loads((tmp_path / "controls" / "camera_contract.json").read_text())["view_count"] == 6


def test_cpu_controls_are_deterministic(tmp_path: Path) -> None:
    mesh = tmp_path / "fixture.glb"
    _cube(mesh)
    build_controls(mesh, tmp_path / "a", size=24)
    build_controls(mesh, tmp_path / "b", size=24)
    a = np.load(tmp_path / "a" / "control_tensor.npy")
    b = np.load(tmp_path / "b" / "control_tensor.npy")
    assert np.array_equal(a, b)

