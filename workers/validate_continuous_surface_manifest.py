"""Independent semantic replay and two-run byte-replay validator for Ticket 01."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

try:
    from build_continuous_surface_manifest import (
        STATE_NAMES, canonical_json, load_evidence, sha256_array, load_contract, classify_front_co_depth_policy, visible_edge_tie_signature,
        project, rasterise, census, connected_face_components, uv_chart_components,
        query_all_hits, bilinear, unit,
    )
    from mesh_io import read_glb
except ModuleNotFoundError:  # package import from the focused test suite
    from workers.build_continuous_surface_manifest import (
        STATE_NAMES, canonical_json, load_evidence, sha256_array, load_contract, classify_front_co_depth_policy, visible_edge_tie_signature,
        project, rasterise, census, connected_face_components, uv_chart_components,
        query_all_hits, bilinear, unit,
    )
    from workers.mesh_io import read_glb


REQUIRED_ARRAYS = {
    "view_index", "sample_index", "state", "owner_face", "layer0_face", "expected_layer",
    "owner_barycentric", "owner_point", "owner_uv", "owner_normal", "owner_component", "uv_chart", "screen_xy", "source_xy", "evidence_rgb",
    "edge_tie_face_a", "edge_tie_face_b", "edge_tie_owner", "edge_tie_component_a", "edge_tie_component_b", "edge_tie_chart_a", "edge_tie_chart_b", "edge_tie_owner_component", "edge_tie_owner_chart", "edge_tie_shared_position_endpoint_count", "edge_tie_point", "edge_tie_point_error", "edge_tie_normal_dot", "edge_tie_depth_spread", "edge_tie_owner_point_error", "edge_tie_owner_uv_error", "edge_tie_owner_bary_error",
    "evidence_class", "admitted_color", "hit_offsets", "hit_face", "hit_depth",
    "hit_barycentric", "hit_point", "hit_uv", "hit_component", "hit_facing", "hit_layer", "hit_co_depth_group", "texel_xy",
    "identity_mesh_hash", "identity_triangle_index_hash", "identity_vertex_index_hash", "identity_uv_hash", "identity_camera_hash", "identity_evidence_hash",
    "texel_center_numerator",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_digest(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json(metadata))
    for name in sorted(arrays):
        digest.update(name.encode("utf-8"))
        digest.update(np.asarray(arrays[name]).tobytes(order="C"))
    return digest.hexdigest()


def load_manifest(run_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads((run_dir / "surface_manifest.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "surface_manifest.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    missing = REQUIRED_ARRAYS.difference(arrays)
    if missing:
        raise RuntimeError(f"MANIFEST_ARRAYS_MISSING:{sorted(missing)}")
    return metadata, arrays


def validate_semantics(metadata: dict[str, Any], arrays: dict[str, np.ndarray], input_identity: dict[str, Any] | None = None) -> dict[str, Any]:
    errors: list[str] = []
    rows = int(len(arrays["state"]))
    if rows != int(metadata["row_count"]):
        errors.append("ROW_COUNT_METADATA_MISMATCH")
    if rows != 6 * int(metadata["owned_texel_count"]):
        errors.append("ROW_COUNT_VIEW_PRODUCT_MISMATCH")
    states = arrays["state"].astype(np.int64, copy=False)
    if np.any((states < 0) | (states >= len(STATE_NAMES))):
        errors.append("STATE_CODE_INVALID")
    offsets = arrays["hit_offsets"].astype(np.int64, copy=False)
    if len(offsets) != rows + 1 or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        errors.append("HIT_OFFSETS_INVALID")
    if len(offsets) and offsets[-1] != len(arrays["hit_face"]):
        errors.append("HIT_OFFSET_TERMINAL_MISMATCH")
    admitted = arrays["admitted_color"].astype(bool, copy=False)
    expected_layer = arrays["expected_layer"].astype(np.int64, copy=False)
    owner_face = arrays["owner_face"].astype(np.int64, copy=False)
    layer0_face = arrays["layer0_face"].astype(np.int64, copy=False)
    visible = states == STATE_NAMES.index("VISIBLE_EXACT")
    edge_tie = states == STATE_NAMES.index("VISIBLE_EDGE_TIE")
    if np.any(admitted != (visible | edge_tie)):
        errors.append("COLOR_ADMISSION_NOT_VISIBLE_STATE")
    if np.any(admitted & ~edge_tie & (expected_layer != 0)):
        errors.append("ADMITTED_NONZERO_LAYER")
    if np.any(admitted & (layer0_face != owner_face)):
        errors.append("ADMITTED_OWNER_LAYER_FACE_MISMATCH")
    if np.any(~admitted & (arrays["evidence_class"] != 0)):
        errors.append("NONVISIBLE_EVIDENCE_CLASS_NONZERO")
    # Rejected samples may retain their projected source coordinate for
    # diagnostics; only ``admitted_color`` controls whether that coordinate
    # participates in evidence.
    if not np.isfinite(arrays["owner_point"]).all() or not np.isfinite(arrays["screen_xy"]).all():
        errors.append("OWNER_OR_SCREEN_NONFINITE")
    if not np.isfinite(arrays["evidence_rgb"]).all():
        errors.append("EVIDENCE_NONFINITE")
    if np.any(edge_tie):
        proof_fields = ("edge_tie_face_a", "edge_tie_face_b", "edge_tie_owner",
            "edge_tie_component_a", "edge_tie_component_b", "edge_tie_chart_a", "edge_tie_chart_b", "edge_tie_owner_component", "edge_tie_owner_chart",
            "edge_tie_shared_position_endpoint_count", "edge_tie_point",
            "edge_tie_point_error", "edge_tie_normal_dot", "edge_tie_depth_spread",
            "edge_tie_owner_point_error", "edge_tie_owner_uv_error", "edge_tie_owner_bary_error")
        if any(field not in arrays for field in proof_fields):
            errors.append("EDGE_TIE_PROOF_MISSING")
        else:
            if not np.all(arrays["edge_tie_owner"][edge_tie] == owner_face[edge_tie]):
                errors.append("EDGE_TIE_OWNER_MISMATCH")
            if not np.all(arrays["edge_tie_shared_position_endpoint_count"][edge_tie] == 2):
                errors.append("EDGE_TIE_TOPOLOGY_PROOF_INVALID")
            if not np.all(np.isfinite(arrays["edge_tie_point"][edge_tie])):
                errors.append("EDGE_TIE_POINT_NONFINITE")
            for field in ("edge_tie_point_error", "edge_tie_owner_point_error", "edge_tie_owner_uv_error", "edge_tie_owner_bary_error"):
                if not np.all(np.isfinite(arrays[field][edge_tie])):
                    errors.append(f"EDGE_TIE_PROOF_NONFINITE:{field}")
    identity_fields = {
        "identity_mesh_hash": metadata["mesh_sha256"],
        "identity_triangle_index_hash": metadata["triangle_index_hash"],
        "identity_vertex_index_hash": metadata["vertex_index_hash"],
        "identity_uv_hash": metadata["uv_hash"],
        "identity_camera_hash": metadata["camera_contract_sha256"],
    }
    for field, expected_hex in identity_fields.items():
        expected = np.frombuffer(bytes.fromhex(expected_hex), dtype=np.uint8)
        values = arrays[field]
        if values.shape != (rows, 32) or not np.all(values == expected[None, :]):
            errors.append(f"ROW_IDENTITY_MISMATCH:{field}")
    evidence_values = arrays["identity_evidence_hash"]
    if evidence_values.shape != (rows, 32):
        errors.append("ROW_IDENTITY_MISMATCH:identity_evidence_hash")
    else:
        for view_index, view in enumerate(metadata["per_view"]):
            expected = np.frombuffer(bytes.fromhex(metadata["evidence_hashes"][view["semantic"]]), dtype=np.uint8)
            rows_for_view = arrays["view_index"] == int(view["index"])
            if not np.all(evidence_values[rows_for_view] == expected[None, :]):
                errors.append(f"ROW_IDENTITY_MISMATCH:identity_evidence_hash:view_{view_index}")
    if input_identity is not None:
        for field, expected in input_identity.items():
            if field == "evidence_hashes":
                if metadata.get(field) != expected:
                    errors.append("STALE_INPUT:evidence_hashes")
                continue
            if field.endswith("_hash") or field.endswith("_sha256"):
                if metadata.get(field) != expected:
                    errors.append(f"STALE_INPUT:{field}")
    hit_depth = arrays["hit_depth"]
    hit_face = arrays["hit_face"]
    hit_layer = arrays["hit_layer"].astype(np.int64, copy=False)
    hit_group = arrays["hit_co_depth_group"].astype(np.int64, copy=False)
    sort_errors = 0
    for row in range(rows):
        start, end = int(offsets[row]), int(offsets[row + 1])
        if not np.array_equal(hit_layer[start:end], np.arange(end - start, dtype=np.int64)):
            errors.append(f"HIT_LAYER_RANK_INVALID:{row}")
            break
        if end > start and (hit_group[start] != 0 or np.any(hit_group[start + 1:end] < hit_group[start:end - 1])):
            errors.append(f"HIT_CO_DEPTH_GROUP_INVALID:{row}")
            break
        if end - start > 1:
            order = sorted(zip(hit_depth[start:end].tolist(), hit_face[start:end].tolist()))
            if order != list(zip(hit_depth[start:end].tolist(), hit_face[start:end].tolist())):
                sort_errors += 1
    if sort_errors:
        errors.append(f"HIT_SORT_ORDER_INVALID:{sort_errors}")
    digest_metadata = {
        "contract_version": metadata["contract_version"], "atlas_size": metadata["atlas_size"],
        "mesh_sha256": metadata["mesh_sha256"], "triangle_index_hash": metadata["triangle_index_hash"],
        "vertex_index_hash": metadata["vertex_index_hash"], "uv_hash": metadata["uv_hash"],
        "camera_contract_sha256": metadata["camera_contract_sha256"], "evidence_hashes": metadata["evidence_hashes"],
        "state_names": metadata["state_names"],
    }
    digest = semantic_digest(arrays, digest_metadata)
    if digest != metadata.get("semantic_digest"):
        errors.append("SEMANTIC_DIGEST_MISMATCH")
    terminal_contract_errors = int(np.count_nonzero(states == STATE_NAMES.index("CONTRACT_ERROR")))
    if terminal_contract_errors:
        errors.append("CONTRACT_ERROR_NONZERO")
    if int(metadata.get("contract_error_count", terminal_contract_errors)) != terminal_contract_errors:
        errors.append("CONTRACT_ERROR_COUNT_METADATA_MISMATCH")
    declared_counts = metadata.get("state_counts_total")
    if declared_counts is not None:
        actual_counts = {name: int(np.count_nonzero(states == index)) for index, name in enumerate(STATE_NAMES)}
        if declared_counts != actual_counts:
            errors.append("STATE_COUNTS_METADATA_MISMATCH")
    return {
        "passed": not errors, "errors": errors, "row_count": rows,
        "contract_error_count": terminal_contract_errors,
        "unknown_ambiguous_count": int(np.count_nonzero(states == STATE_NAMES.index("UNKNOWN_AMBIGUOUS"))),
        "visible_edge_tie_count": int(np.count_nonzero(edge_tie)),
        "admitted_evidence_count": int(np.count_nonzero(admitted)),
        "arbitrary_layer_acceptance": int(metadata.get("arbitrary_layer_acceptance", -1)),
        "implicit_pixel_rounding": int(metadata.get("implicit_pixel_rounding", -1)),
        "semantic_digest": digest,
    }


def input_identity(mesh: Path, camera_contract: Path, evidence_receipt: Path) -> dict[str, Any]:
    try:
        from mesh_io import read_glb
    except ModuleNotFoundError:
        from workers.mesh_io import read_glb
    positions, _normals, uv, triangles = read_glb(mesh)
    if uv is None:
        raise RuntimeError("STALE_INPUT_UV_MISSING")
    _images, evidence_hashes, _paths = load_evidence(evidence_receipt)
    return {
        "mesh_sha256": sha256_file(mesh),
        "triangle_index_hash": sha256_array(np.asarray(triangles, dtype="<i8")),
        "vertex_index_hash": sha256_array(np.arange(len(positions), dtype="<i8")),
        "uv_hash": sha256_array(np.asarray(uv, dtype="<f8")),
        "camera_contract_sha256": sha256_file(camera_contract),
        "evidence_receipt_sha256": sha256_file(evidence_receipt),
        "evidence_hashes": evidence_hashes,
    }


def reconstruct_semantics(
    metadata: dict[str, Any], arrays: dict[str, np.ndarray],
    mesh_path: Path, camera_path: Path, evidence_path: Path,
) -> list[str]:
    """Rebuild ownership, projection, all-hit ordering, and evidence from inputs."""
    errors: list[str] = []
    contract = load_contract(camera_path)
    images, evidence_hashes, _ = load_evidence(evidence_path)
    positions, _normals, uv, triangles = read_glb(mesh_path)
    if uv is None:
        return ["RECONSTRUCT_UV_MISSING"]
    positions = np.asarray(positions, dtype=np.float64)
    uv = np.asarray(uv, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    transform = np.asarray(contract["control_space_transform"], dtype=np.float64)
    canonical = positions @ transform.T
    canonical *= 0.5 / float(np.max(np.abs(canonical)))
    face_vertices = canonical[triangles]
    face_uv = uv[triangles]
    face_normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0])
    norms = np.linalg.norm(face_normals, axis=1)
    face_normals /= np.maximum(norms[:, None], 1e-14)
    face_components, _ = connected_face_components(triangles, len(positions))
    face_charts, _ = uv_chart_components(triangles, uv)
    n = int(metadata["atlas_size"])
    owner, weights = rasterise(uv, triangles, n)
    covered, _ = census(uv, triangles, n)
    counts = np.bincount(covered, minlength=n * n)
    owner.reshape(-1)[np.flatnonzero(counts > 1)] = -1
    owned = np.flatnonzero(owner.reshape(-1) >= 0)
    if not np.array_equal(arrays["texel_xy"], np.column_stack(np.divmod(owned, n)[::-1]).astype("<i4")):
        errors.append("RECON_TEXEL_OWNER_LATTICE_MISMATCH")
    owner_face = owner.reshape(-1)[owned].astype(np.int64)
    w = weights.reshape(-1, 2)[owned].astype(np.float64)
    # atlas_raster returns wa/wb as coefficients of face corners 1/2; replay
    # must reconstruct the canonical corner order 0/1/2 independently.
    owner_bary = np.column_stack((1.0 - w[:, 0] - w[:, 1], w[:, 0], w[:, 1]))
    owner_points = np.einsum("ni,nij->nj", owner_bary, face_vertices[owner_face])
    owner_uv = np.einsum("ni,nij->nj", owner_bary, face_uv[owner_face])
    if len(owner_face) != int(metadata["owned_texel_count"]):
        errors.append("RECON_OWNER_COUNT_MISMATCH")
    if not np.array_equal(arrays["owner_face"], np.tile(owner_face, 6)):
        errors.append("RECON_OWNER_FACE_MISMATCH")
    if not np.allclose(arrays["owner_barycentric"], np.tile(owner_bary, (6, 1)), atol=1e-10, rtol=0):
        errors.append("RECON_OWNER_BARY_MISMATCH")
    if not np.allclose(arrays["owner_point"], np.tile(owner_points, (6, 1)), atol=1e-9, rtol=0):
        errors.append("RECON_OWNER_POINT_MISMATCH")
    if not np.allclose(arrays["owner_uv"], np.tile(owner_uv, (6, 1)), atol=1e-9, rtol=0):
        errors.append("RECON_OWNER_UV_MISMATCH")
    if not np.array_equal(arrays["owner_component"], np.tile(face_components[owner_face], 6)):
        errors.append("RECON_OWNER_COMPONENT_MISMATCH")
    if not np.array_equal(arrays["uv_chart"], np.tile(face_charts[owner_face], 6)):
        errors.append("RECON_OWNER_CHART_MISMATCH")
    span = float(metadata["projection_span"])
    tolerances = {k: float(v) for k, v in metadata["numeric_tolerance_receipt"].items()}
    views = sorted(contract["views"], key=lambda item: int(item["index"]))
    offsets = arrays["hit_offsets"].astype(np.int64, copy=False)
    for view_pos, view in enumerate(views):
        start_row = view_pos * len(owner_face)
        end_row = start_row + len(owner_face)
        screen = project(owner_points, view, span)
        if not np.allclose(arrays["screen_xy"][start_row:end_row], screen, atol=1e-9, rtol=0):
            errors.append(f"RECON_SCREEN_MISMATCH:view_{view_pos}")
        image = images[str(view["proven_semantic"])]
        source = np.full_like(screen, -1.0)
        valid = np.all((screen >= -tolerances["screen_bounds"]) & (screen <= 1.0 + tolerances["screen_bounds"]), axis=1)
        source[valid] = screen[valid] * np.asarray([image.shape[1] - 1, image.shape[0] - 1], dtype=np.float64)
        if not np.allclose(arrays["source_xy"][start_row:end_row], source, atol=1e-9, rtol=0):
            errors.append(f"RECON_SOURCE_XY_MISMATCH:view_{view_pos}")
        admitted = arrays["admitted_color"][start_row:end_row].astype(bool)
        expected_rgb = np.zeros((len(owner_face), 3), dtype=np.float32)
        expected_rgb[admitted] = bilinear(image, source[admitted])
        if not np.allclose(arrays["evidence_rgb"][start_row:end_row], expected_rgb, atol=1e-4, rtol=0):
            errors.append(f"RECON_EVIDENCE_RGB_MISMATCH:view_{view_pos}")
        tri_screen = project(face_vertices.reshape(-1, 3), view, span).reshape(-1, 3, 2)
        hit_iter = query_all_hits(screen, tri_screen, canonical, face_vertices, face_uv, face_normals,
                                  face_components, view, span, n, tolerances)
        for local, hits in enumerate(hit_iter):
            row = start_row + local
            hs, he = int(offsets[row]), int(offsets[row + 1])
            if len(hits) != he - hs:
                errors.append(f"RECON_HIT_COUNT_MISMATCH:{row}")
                continue
            if hits:
                for j, hit in enumerate(hits):
                    idx = hs + j
                    if int(arrays["hit_layer"][idx]) != j:
                        errors.append(f"RECON_HIT_LAYER_MISMATCH:{row}")
                        break
                    if int(arrays["hit_face"][idx]) != int(hit["face"]):
                        errors.append(f"RECON_HIT_FACE_MISMATCH:{row}")
                        break
                    if not np.isclose(arrays["hit_depth"][idx], hit["depth"], atol=1e-8, rtol=1e-8) or not np.allclose(arrays["hit_point"][idx], hit["point"], atol=1e-8, rtol=0):
                        errors.append(f"RECON_HIT_GEOMETRY_MISMATCH:{row}")
                        break
                    if not np.allclose(arrays["hit_uv"][idx], hit["uv"], atol=1e-8, rtol=0):
                        errors.append(f"RECON_HIT_UV_MISMATCH:{row}")
                        break
                    expected_group = 0 if j == 0 else int(arrays["hit_co_depth_group"][idx - 1]) + (1 if abs(float(hit["depth"]) - float(hits[j - 1]["depth"])) > tolerances["co_depth_abs"] + tolerances["co_depth_rel"] * max(1.0, abs(float(hit["depth"]))) else 0)
                    if int(arrays["hit_co_depth_group"][idx]) != expected_group:
                        errors.append(f"RECON_HIT_GROUP_MISMATCH:{row}")
                        break
            state = int(arrays["state"][row])
            if not valid[local]:
                expected_state = STATE_NAMES.index("OUT_OF_FRAME")
            elif not hits:
                expected_state = STATE_NAMES.index("BACKFACING") if float(np.dot(face_normals[int(owner_face[local])], -unit(np.asarray(view["camera_direction"], dtype=np.float64)))) <= tolerances["facing_min"] else STATE_NAMES.index("CONTRACT_ERROR")
            else:
                first = float(hits[0]["depth"])
                group = [h for h in hits if abs(float(h["depth"]) - first) <= tolerances["co_depth_abs"] + tolerances["co_depth_rel"] * max(1.0, abs(first))]
                expected_positions = [j for j, h in enumerate(hits) if int(h["face"]) == int(owner_face[local])]
                edge_proof = visible_edge_tie_signature(
                    int(owner_face[local]), hits, face_vertices, face_normals, tolerances, view,
                    owner_points[local], owner_uv[local], arrays["owner_barycentric"][row], face_components, face_charts,
                )
                policy_state = classify_front_co_depth_policy(
                    int(view["index"]), int(local), int(owner_face[local]), hits, tolerances, view,
                )
                if edge_proof is not None:
                    expected_state = STATE_NAMES.index("VISIBLE_EDGE_TIE")
                elif policy_state is not None:
                    expected_state = STATE_NAMES.index(policy_state)
                elif not expected_positions:
                    expected_state = STATE_NAMES.index("CONTRACT_ERROR") if float(np.dot(face_normals[int(owner_face[local])], -unit(np.asarray(view["camera_direction"], dtype=np.float64)))) > tolerances["facing_min"] else STATE_NAMES.index("BACKFACING")
                elif expected_positions[0] > 0:
                    expected_state = STATE_NAMES.index("OCCLUDED")
                elif float(hits[0]["facing"]) < tolerances["facing_min"]:
                    expected_state = STATE_NAMES.index("BACKFACING")
                elif np.max(np.abs(hits[0]["point"] - owner_points[local])) > tolerances["point_abs"] or np.max(np.abs(hits[0]["uv"] - owner_uv[local])) > tolerances["uv_reconstruction_abs"]:
                    expected_state = STATE_NAMES.index("CONTRACT_ERROR")
                else:
                    expected_state = STATE_NAMES.index("VISIBLE_EXACT")
            if state != expected_state:
                errors.append(f"RECON_TERMINAL_STATE_MISMATCH:{row}")
            actual_layer = next((j for j, h in enumerate(hits) if int(h["face"]) == int(owner_face[local])), -1)
            if int(arrays["expected_layer"][row]) != actual_layer:
                errors.append(f"RECON_EXPECTED_LAYER_MISMATCH:{row}")
            if int(arrays["layer0_face"][row]) != (int(hits[0]["face"]) if hits else -1):
                errors.append(f"RECON_LAYER0_FACE_MISMATCH:{row}")
            if state == STATE_NAMES.index("VISIBLE_EDGE_TIE"):
                if edge_proof is None:
                    errors.append(f"RECON_EDGE_TIE_PROOF_MISSING:{row}")
                else:
                    checks = (
                        ("edge_tie_face_a", int(edge_proof["face_a"])),
                        ("edge_tie_face_b", int(edge_proof["face_b"])),
                        ("edge_tie_owner", int(edge_proof["owner"])),
                        ("edge_tie_component_a", int(edge_proof["component_a"])),
                        ("edge_tie_component_b", int(edge_proof["component_b"])),
                        ("edge_tie_chart_a", int(edge_proof["chart_a"])),
                        ("edge_tie_chart_b", int(edge_proof["chart_b"])),
                        ("edge_tie_owner_component", int(edge_proof["owner_component"])),
                        ("edge_tie_owner_chart", int(edge_proof["owner_chart"])),
                        ("edge_tie_shared_position_endpoint_count", 2),
                    )
                    for field, expected_value in checks:
                        if int(arrays[field][row]) != expected_value:
                            errors.append(f"RECON_EDGE_TIE_{field.upper()}_MISMATCH:{row}")
                    if not np.allclose(arrays["edge_tie_point"][row], edge_proof["point"], atol=tolerances["edge_point_abs"], rtol=0):
                        errors.append(f"RECON_EDGE_TIE_POINT_MISMATCH:{row}")
                    if not np.isclose(arrays["edge_tie_point_error"][row], edge_proof["point_error"], atol=tolerances["edge_point_abs"], rtol=0):
                        errors.append(f"RECON_EDGE_TIE_POINT_ERROR_MISMATCH:{row}")
                    if not np.isclose(arrays["edge_tie_normal_dot"][row], edge_proof["normal_dot"], atol=1e-12, rtol=0):
                        errors.append(f"RECON_EDGE_TIE_NORMAL_MISMATCH:{row}")
                    if not np.isclose(arrays["edge_tie_depth_spread"][row], edge_proof["depth_spread"], atol=tolerances["edge_depth_abs"], rtol=0):
                        errors.append(f"RECON_EDGE_TIE_DEPTH_MISMATCH:{row}")
                    for field, proof_key in (("edge_tie_owner_point_error", "owner_point_error"), ("edge_tie_owner_uv_error", "owner_uv_error"), ("edge_tie_owner_bary_error", "owner_bary_error")):
                        if not np.isclose(arrays[field][row], edge_proof[proof_key], atol=1e-12, rtol=0):
                            errors.append(f"RECON_EDGE_TIE_{field.upper()}_MISMATCH:{row}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-dir", required=True)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--camera-contract", required=True)
    parser.add_argument("--evidence-receipt", required=True)
    args = parser.parse_args()
    stage = Path(args.stage_dir)
    run1 = stage / "run_1"
    run2 = stage / "run_2"
    current_identity = input_identity(Path(args.mesh), Path(args.camera_contract), Path(args.evidence_receipt))
    metadata1, arrays1 = load_manifest(run1)
    metadata2, arrays2 = load_manifest(run2)
    semantic1 = validate_semantics(metadata1, arrays1, current_identity)
    semantic2 = validate_semantics(metadata2, arrays2, current_identity)
    reconstruction1 = reconstruct_semantics(metadata1, arrays1, Path(args.mesh), Path(args.camera_contract), Path(args.evidence_receipt))
    reconstruction2 = reconstruct_semantics(metadata2, arrays2, Path(args.mesh), Path(args.camera_contract), Path(args.evidence_receipt))
    if reconstruction1:
        semantic1["errors"].extend(reconstruction1)
        semantic1["passed"] = False
    if reconstruction2:
        semantic2["errors"].extend(reconstruction2)
        semantic2["passed"] = False
    bytes1 = sha256_file(run1 / "surface_manifest.npz")
    bytes2 = sha256_file(run2 / "surface_manifest.npz")
    byte_identical = (bytes1 == bytes2) and ((run1 / "surface_manifest.npz").read_bytes() == (run2 / "surface_manifest.npz").read_bytes())
    semantic_identical = semantic1["semantic_digest"] == semantic2["semantic_digest"]
    replay = {
        "schema": "panda_continuous_surface_replay_report_v1", "atlas_size": int(metadata1["atlas_size"]),
        "run_1": {"path": str(run1), "manifest_sha256": bytes1, "semantic": semantic1,
                  "runtime_seconds": metadata1.get("runtime_seconds"), "peak_rss_mb": metadata1.get("peak_rss_mb")},
        "run_2": {"path": str(run2), "manifest_sha256": bytes2, "semantic": semantic2,
                  "runtime_seconds": metadata2.get("runtime_seconds"), "peak_rss_mb": metadata2.get("peak_rss_mb")},
        "byte_identical_canonical_manifest": byte_identical,
        "reconstruction_classification": "INPUT_BASED_REFERENCE_REUSE",
        "independent_reference": False,
        "input_based_reconstruction": {"run_1_errors": reconstruction1, "run_2_errors": reconstruction2},
        "portable_semantic_replay": semantic_identical and semantic1["passed"] and semantic2["passed"],
        "contract_error_zero": semantic1["contract_error_count"] == 0 and semantic2["contract_error_count"] == 0,
        "arbitrary_layer_acceptance_zero": semantic1["arbitrary_layer_acceptance"] == 0 and semantic2["arbitrary_layer_acceptance"] == 0,
        "implicit_pixel_rounding_zero": semantic1["implicit_pixel_rounding"] == 0 and semantic2["implicit_pixel_rounding"] == 0,
        "passed": byte_identical and semantic_identical and semantic1["passed"] and semantic2["passed"],
    }
    write_path = stage / "replay_report.json"
    write_path.write_bytes(canonical_json(replay) + b"\n")
    # Always reconcile stage-level receipts to the exact current pair.  A
    # failed gate is explicitly recorded as blocked; it is never promoted.
    for name in ("surface_manifest.npz", "surface_manifest.json", "state_counts.json"):
        shutil.copyfile(run1 / name, stage / name)
    contract = {
            "schema": "panda_continuous_surface_contract_v1", "atlas_size": int(metadata1["atlas_size"]),
            "mesh_sha256": metadata1["mesh_sha256"], "triangle_index_hash": metadata1["triangle_index_hash"],
            "vertex_index_hash": metadata1["vertex_index_hash"], "uv_hash": metadata1["uv_hash"],
            "camera_contract_sha256": metadata1["camera_contract_sha256"], "evidence_image_hashes": metadata1["evidence_hashes"],
            "evidence_receipt_sha256": metadata1["evidence_receipt_sha256"], "evidence_hashes": metadata1["evidence_hashes"],
            "control_space_transform": metadata1["control_space_transform"],
            "projection_rule": metadata1["projection_rule"], "all_hit_rule": metadata1["all_hit_rule"],
            "layer_rule": metadata1["layer_rule"], "unknown_ambiguous_rule": metadata1.get("unknown_ambiguous_rule"),
            "visible_edge_tie_rule": metadata1.get("visible_edge_tie_rule"), "visible_edge_tie_policy_version": metadata1.get("visible_edge_tie_policy_version"),
            "unknown_ambiguous_policy_version": metadata1.get("unknown_ambiguous_policy_version"), "state_names": metadata1["state_names"],
            "legacy_rounded_id_buffers_authoritative": False,
            "arbitrary_layer_acceptance": 0, "implicit_pixel_rounding": 0,
            "validation_passed": bool(replay["passed"]),
            "contract_error_total": int(semantic1["contract_error_count"]),
            "unknown_ambiguous_total": int(semantic1["unknown_ambiguous_count"]),
            "coordinate_conventions": metadata1.get("coordinate_conventions", {}),
            "semantic_digest": metadata1["semantic_digest"], "surface_manifest_sha256": bytes1,
            "replay_report_sha256": sha256_file(write_path),
    }
    receipt_root = stage.parent if int(metadata1["atlas_size"]) == 256 else stage
    (receipt_root / "contract.json").write_bytes(canonical_json(contract) + b"\n")
    (receipt_root / "numeric_tolerances.json").write_bytes(canonical_json(metadata1["numeric_tolerance_receipt"]) + b"\n")
    print(json.dumps(replay, indent=2), flush=True)
    return 0 if replay["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
