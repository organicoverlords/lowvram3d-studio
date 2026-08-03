"""Validate and receipt the selected Castlegrounds source mesh v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh

from workers.scene_pipeline.source_mesh_repair import ROOT, PROOF, mesh_stats


V2 = ROOT / "castlegrounds_source_mesh_v2.glb"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def main() -> None:
    mesh = trimesh.load(V2, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("SOURCE_MESH_V2_MULTIPLE_MESHES")
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv = np.asarray(mesh.visual.uv, dtype=np.float32) if mesh.visual.uv is not None else np.empty((0, 2), np.float32)
    expected_faces = np.load(ROOT / "adaptive_coverage_winding_faces.npy")
    expected_points = np.load(ROOT / "points.npy").astype(np.float32)
    mask = np.load(ROOT / "mask.npy").astype(bool) & np.isfinite(expected_points).all(axis=-1) & (expected_points[..., 2] > 0)
    expected_vertices = expected_points.reshape(-1, 3)[np.flatnonzero(mask)]
    expected_uv = np.zeros((len(expected_vertices), 2), np.float32)
    height, width = mask.shape
    yy, xx = np.unravel_index(np.flatnonzero(mask.ravel()), (height, width))
    expected_uv[:, 0] = xx / max(width - 1, 1)
    expected_uv[:, 1] = 1.0 - yy / max(height - 1, 1)
    checks = {
        "finite_positions": bool(np.isfinite(vertices).all()),
        "finite_normals": bool(mesh.vertex_normals is not None and np.isfinite(mesh.vertex_normals).all()),
        "finite_uv": bool(uv.shape == expected_uv.shape and np.isfinite(uv).all()),
        "uv_in_bounds": bool(uv.shape == expected_uv.shape and np.all((uv >= -1e-6) & (uv <= 1.000001))),
        "vertex_positions_hash_equivalent": bool(vertices.shape == expected_vertices.shape and np.array_equal(vertices, expected_vertices)),
        "triangle_order_hash_equivalent": bool(faces.shape == expected_faces.shape and np.array_equal(faces, expected_faces)),
        "uv_hash_equivalent": bool(uv.shape == expected_uv.shape and np.allclose(uv, expected_uv, atol=1e-6)),
    }
    if not all(checks.values()):
        raise RuntimeError("SOURCE_MESH_V2_VALIDATION_REJECTED")
    source_receipt = json.loads((ROOT / "rejection_reason_counts.json").read_text(encoding="utf-8"))
    metrics = json.loads((ROOT / "source_mesh_v2_exact_comparison.json").read_text(encoding="utf-8")) if (ROOT / "source_mesh_v2_exact_comparison.json").is_file() else json.loads((ROOT / "adaptive_coverage_winding_exact_comparison.json").read_text(encoding="utf-8"))
    receipt = {
        "schema": "castlegrounds_source_mesh_v2_validation_v1",
        "classification": "CASTLEGROUNDS_SOURCE_VISIBLE_MESH_REPAIR_PROVEN",
        "glb": str(V2), "glb_sha256": sha256(V2), "glb_bytes": V2.stat().st_size,
        "mesh": mesh_stats(vertices, faces), "checks": checks,
        "geometry_hash": array_hash(vertices, faces), "uv_hash": array_hash(uv),
        "source_arrays_sha256": source_receipt["source_arrays_sha256"],
        "camera_contract": "M_RAW_MOGE_TO_GLB_IDENTITY_THEN_M_GLB_TO_BLENDER_AXIS_FIXTURE",
        "source_comparison": metrics,
        "selected_candidate": "adaptive_coverage_winding",
        "selected_reason": "highest exact-source coverage among bounded CPU adaptive candidates while preserving local depth boundaries; VITB640 retained as strongest fresh-model diagnostic",
    }
    (ROOT / "source_mesh_v2_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (PROOF / "source_mesh_v2_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
