from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from workers.build_continuous_surface_manifest import bilinear, write_deterministic_npz, unresolved_front_co_depth, approved_unknown_ambiguity, classify_front_co_depth_policy, visible_edge_tie_signature, APPROVED_UNKNOWN_FRONT_GROUPS, STATE_CODE
from workers.validate_continuous_surface_manifest import semantic_digest, validate_semantics, load_manifest


def test_bilinear_fractional_coordinate_is_deterministic() -> None:
    image = np.asarray([[[0, 0, 0], [10, 20, 30]], [[30, 40, 50], [60, 80, 100]]], dtype=np.uint8)
    actual = bilinear(image, np.asarray([[0.5, 0.5]], dtype=np.float64))[0]
    np.testing.assert_allclose(actual, np.asarray([25.0, 35.0, 45.0], dtype=np.float32))


def test_deterministic_npz_replay_bytes() -> None:
    arrays = {"z": np.asarray([3, 2, 1], dtype="<i4"), "a": np.asarray([[1.0, 2.0]], dtype="<f8")}
    with tempfile.TemporaryDirectory() as directory:
        first = Path(directory) / "first.npz"
        second = Path(directory) / "second.npz"
        write_deterministic_npz(first, arrays)
        write_deterministic_npz(second, arrays)
        assert first.read_bytes() == second.read_bytes()


def test_semantic_digest_changes_when_state_changes() -> None:
    arrays = {"state": np.asarray([0, 1], dtype="<u1")}
    metadata = {"contract_version": "continuous_surface_contract_v1", "atlas_size": 256,
                "mesh_sha256": "mesh", "camera_contract_sha256": "camera", "state_names": ["VISIBLE_EXACT"]}
    original = semantic_digest(arrays, metadata)
    changed = semantic_digest({"state": np.asarray([0, 2], dtype="<u1")}, metadata)
    assert original != changed


def test_unresolved_front_co_depth_is_not_occluded() -> None:
    tolerances = {"co_depth_abs": 1e-8, "co_depth_rel": 1e-8}
    assert unresolved_front_co_depth([{"depth": 1.0, "face": 2}, {"depth": 1.0 + 1e-10, "face": 3}], tolerances)
    assert not unresolved_front_co_depth([{"depth": 1.0, "face": 2}, {"depth": 1.1, "face": 3}], tolerances)


def test_unknown_ambiguous_is_not_visible_or_arbitrary_layer() -> None:
    assert STATE_CODE["UNKNOWN_AMBIGUOUS"] != STATE_CODE["VISIBLE_EXACT"]
    assert STATE_CODE["UNKNOWN_AMBIGUOUS"] != STATE_CODE["OCCLUDED"]


def test_unknown_policy_is_closed_to_the_approved_40_identities() -> None:
    assert len(APPROVED_UNKNOWN_FRONT_GROUPS) == 40
    key, faces = next(iter(APPROVED_UNKNOWN_FRONT_GROUPS.items()))
    assert approved_unknown_ambiguity(*key, faces)
    assert not approved_unknown_ambiguity(0, 0, 0, faces)
    assert not approved_unknown_ambiguity(*key, tuple(reversed(faces)))


def test_unlisted_unresolved_front_co_depth_is_terminal_contract_error() -> None:
    tolerances = {"co_depth_abs": 1e-8, "co_depth_rel": 1e-8}
    hits = [{"face": 9001, "depth": 1.0}, {"face": 9002, "depth": 1.0}]
    assert classify_front_co_depth_policy(99, 123, 456, hits, tolerances) == "CONTRACT_ERROR"
    view = {"camera_direction": [0.0, 0.0, 1.0], "camera_position": [0.0, 0.0, 0.0]}
    signature_hits = [{"face": 9001, "depth": 1.0, "point": np.asarray([0.0, 0.0, 1.0])},
                      {"face": 9002, "depth": 1.0, "point": np.asarray([0.0, 0.0, 1.0])}]
    assert classify_front_co_depth_policy(99, 123, 456, signature_hits, tolerances, view) == "UNKNOWN_AMBIGUOUS"
    key, faces = next(iter(APPROVED_UNKNOWN_FRONT_GROUPS.items()))
    approved_hits = [{"face": face, "depth": 1.0} for face in faces]
    assert classify_front_co_depth_policy(*key, approved_hits, tolerances) == "UNKNOWN_AMBIGUOUS"


def test_visible_edge_tie_accepts_exact_shared_position_edge() -> None:
    faces = np.asarray([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, -1.0, 0.0]],
    ], dtype=np.float64)
    normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    hits = [
        {"face": 0, "depth": 1.0, "point": np.asarray([0.25, 0.0, 0.0]), "uv": np.asarray([0.25, 0.0]), "bary": np.asarray([0.75, 0.25, 0.0])},
        {"face": 1, "depth": 1.0, "point": np.asarray([0.25, 0.0, 0.0]), "uv": np.asarray([0.25, 0.0]), "bary": np.asarray([0.75, 0.25, 0.0])},
        {"face": 2, "depth": 1.1, "point": np.asarray([0.25, 0.0, 0.0])},
    ]
    view = {"camera_direction": [0.0, 0.0, 1.0], "camera_position": [0.0, 0.0, -1.0]}
    tolerances = {"co_depth_abs": 1e-8, "co_depth_rel": 1e-8, "point_abs": 1e-8,
                  "edge_point_abs": 1e-8, "edge_position_abs": 1e-8, "edge_depth_abs": 1e-8, "edge_normal_dot_min": 0.95,
                  "uv_reconstruction_abs": 1e-8, "barycentric_abs": 1e-9}
    proof = visible_edge_tie_signature(0, hits, faces, normals, tolerances, view,
                                       np.asarray([0.25, 0.0, 0.0]), np.asarray([0.25, 0.0]), np.asarray([0.75, 0.25, 0.0]))
    assert proof is not None
    assert proof["face_a"] == 0 and proof["face_b"] == 1 and proof["owner"] == 0
    assert proof["shared_position_endpoint_count"] == 2
    assert proof["depth_spread"] == 0.0


def test_visible_edge_tie_rejects_non_topological_owner_present_tie() -> None:
    faces = np.asarray([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
    ], dtype=np.float64)
    normals = np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    hits = [
        {"face": 0, "depth": 1.0, "point": np.asarray([0.25, 0.0, 0.0]), "uv": np.asarray([0.25, 0.0]), "bary": np.asarray([0.5, 0.25, 0.25])},
        {"face": 1, "depth": 1.0, "point": np.asarray([0.25, 0.0, 0.0]), "uv": np.asarray([0.25, 0.0]), "bary": np.asarray([0.5, 0.25, 0.25])},
    ]
    view = {"camera_direction": [0.0, 0.0, 1.0], "camera_position": [0.0, 0.0, -1.0]}
    tolerances = {"co_depth_abs": 1e-8, "co_depth_rel": 1e-8, "point_abs": 1e-8,
                  "edge_point_abs": 1e-8, "edge_position_abs": 1e-8, "edge_depth_abs": 1e-8, "edge_normal_dot_min": 0.95,
                  "uv_reconstruction_abs": 1e-8, "barycentric_abs": 1e-9}
    assert visible_edge_tie_signature(0, hits, faces, normals, tolerances, view,
                                      np.asarray([0.25, 0.0, 0.0]), np.asarray([0.25, 0.0]), np.asarray([0.75, 0.25, 0.0])) is None


def test_terminal_state_array_is_gate_authority_for_non_co_depth_error() -> None:
    # The full production manifest has this same state-array authority.  This
    # focused regression protects against restoring the old co-depth+missing
    # partial formula when a point/UV/layer terminal error is present.
    metadata = {
        "row_count": 6, "owned_texel_count": 1, "state_names": ["VISIBLE_EXACT", "OCCLUDED", "OUT_OF_FRAME", "BACKFACING", "CONTRACT_ERROR"],
        "mesh_sha256": "00" * 32, "triangle_index_hash": "11" * 32, "vertex_index_hash": "22" * 32, "uv_hash": "33" * 32,
        "camera_contract_sha256": "44" * 32, "evidence_hashes": {f"v{i}": "55" * 32 for i in range(6)},
        "contract_version": "continuous_surface_contract_v1", "atlas_size": 256,
        "co_depth_ambiguity_count": 0, "missing_owner_count": 0, "contract_error_count": 0,
        "per_view": [{"index": i, "semantic": f"v{i}"} for i in range(6)],
    }
    arrays = {"view_index": np.arange(6, dtype="<i2"), "sample_index": np.zeros(6, dtype="<i4"),
              "state": np.asarray([STATE_CODE["CONTRACT_ERROR"], 0, 0, 0, 0, 0], dtype="<u1"), "admitted_color": np.zeros(6, dtype=bool),
              "expected_layer": np.full(6, -1, dtype="<i4"), "layer0_face": np.full(6, -1, dtype="<i4"),
              "owner_face": np.zeros(6, dtype="<i4"), "hit_offsets": np.zeros(7, dtype="<i8"),
              "hit_face": np.zeros(0, dtype="<i4"), "hit_depth": np.zeros(0), "hit_layer": np.zeros(0, dtype="<i4"),
              "hit_co_depth_group": np.zeros(0, dtype="<i4"), "evidence_class": np.zeros(6, dtype="<u1"),
              "owner_point": np.zeros((6, 3)), "screen_xy": np.zeros((6, 2)), "evidence_rgb": np.zeros((6, 3)),
              "identity_mesh_hash": np.tile(np.frombuffer(bytes.fromhex("00" * 32), dtype=np.uint8), (6, 1)),
              "identity_triangle_index_hash": np.tile(np.frombuffer(bytes.fromhex("11" * 32), dtype=np.uint8), (6, 1)),
              "identity_vertex_index_hash": np.tile(np.frombuffer(bytes.fromhex("22" * 32), dtype=np.uint8), (6, 1)),
              "identity_uv_hash": np.tile(np.frombuffer(bytes.fromhex("33" * 32), dtype=np.uint8), (6, 1)),
              "identity_camera_hash": np.tile(np.frombuffer(bytes.fromhex("44" * 32), dtype=np.uint8), (6, 1)),
              "identity_evidence_hash": np.tile(np.frombuffer(bytes.fromhex("55" * 32), dtype=np.uint8), (6, 1))}
    result = validate_semantics(metadata, arrays)
    assert result["contract_error_count"] == 1
    assert not result["passed"]


def test_fresh_digest_tamper_cases_are_rejected() -> None:
    stage = Path(r"C:\AI\LowVRAM3D-benchmarks\production\per_texel_evidence_compiler_validation\panda_architecture_validation_20260805\forensics\continuous_surface_contract\256\run_1")
    if not (stage / "surface_manifest.npz").is_file():
        return
    metadata, source = load_manifest(stage)
    digest_metadata = {"contract_version": metadata["contract_version"], "atlas_size": metadata["atlas_size"],
                       "mesh_sha256": metadata["mesh_sha256"], "triangle_index_hash": metadata["triangle_index_hash"],
                       "vertex_index_hash": metadata["vertex_index_hash"], "uv_hash": metadata["uv_hash"],
                       "camera_contract_sha256": metadata["camera_contract_sha256"], "evidence_hashes": metadata["evidence_hashes"],
                       "state_names": metadata["state_names"]}
    cases = ("state", "owner_layer", "point_uv", "evidence")
    for case in cases:
        arrays = {name: value.copy() for name, value in source.items()}
        if case == "state":
            arrays["state"][0] = STATE_CODE["CONTRACT_ERROR"]
        elif case == "owner_layer":
            visible_row = int(np.flatnonzero(arrays["state"] == 0)[0])
            arrays["expected_layer"][visible_row] = 1
        elif case == "point_uv":
            arrays["owner_point"][0, 0] = np.nan
            arrays["owner_uv"][0, 0] += 0.25
        else:
            arrays["evidence_rgb"][0, 0] = np.nan
        tampered = dict(metadata)
        tampered["semantic_digest"] = semantic_digest(arrays, digest_metadata)
        result = validate_semantics(tampered, arrays)
        assert not result["passed"], case
