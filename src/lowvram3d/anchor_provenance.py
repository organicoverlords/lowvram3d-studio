"""Small, format-neutral provenance gates for downstream geometry stages.

The anchor receipt is the identity contract; this module only carries its digest and anchor set
alongside geometry hashes.  UVs, normals, materials and texture bytes are intentionally excluded
from the geometry hash so encoding changes remain allowed while topology/position mutations fail.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .thin_feature_anchors import anchor_receipt_sha256, parse_anchor_receipt


class AnchorProvenanceError(ValueError):
    def __init__(self, code: str, detail: str):
        super().__init__(detail)
        self.code = code
        self.detail = detail


# Geometry hashes intentionally ignore a mesh's world-space translation.  The oriented raster
# view worker stores bbox-centered positions, while UV provenance is computed from the source GLB.
# Keeping the frame explicit makes those encodings comparable without weakening topology checks.
GEOMETRY_HASH_FRAME = "bbox_centered_v1"


def load_anchor_provenance(
    path: str | Path | None,
    *,
    expected_source_sha256: str | None = None,
) -> tuple[dict[str, Any], str, list[str]]:
    if not path:
        raise AnchorProvenanceError("ANCHOR_RECEIPT_MISSING", "anchor receipt is required")
    receipt_path = Path(path)
    if not receipt_path.exists():
        raise AnchorProvenanceError("ANCHOR_RECEIPT_MISSING", f"anchor receipt not found: {receipt_path}")
    try:
        receipt = parse_anchor_receipt(receipt_path.read_bytes(), expected_source_mesh_sha256=expected_source_sha256)
    except Exception as exc:
        code = "ANCHOR_RECEIPT_SOURCE_MISMATCH" if "source-hash mismatch" in str(exc) else "ANCHOR_RECEIPT_INVALID"
        raise AnchorProvenanceError(code, str(exc)) from exc
    digest = anchor_receipt_sha256(receipt)
    ids = sorted(str(item["anchor_id"]) for item in receipt.get("anchors", []))
    return receipt, digest, ids


def canonical_geometry_vertices(vertices: np.ndarray) -> np.ndarray:
    """Return vertices in the canonical translation-invariant hash frame."""
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.size == 0:
        raise ValueError("geometry hash requires vertices (N,3) and non-empty triangles")
    if not np.isfinite(points).all():
        raise ValueError("geometry hash received invalid geometry")
    return points - (points.min(axis=0) + points.max(axis=0)) * 0.5


def geometry_sha256(vertices: np.ndarray, triangles: np.ndarray, *, decimals: int = 6) -> str:
    """Hash canonical triangle coordinates, independent of translation, ordering, and UV seams."""
    points = canonical_geometry_vertices(vertices)
    faces = np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    if faces.size == 0 or (faces.min() < 0) or (faces.max() >= len(points)):
        raise ValueError("geometry hash requires vertices (N,3) and non-empty triangles")
    scale = 10 ** int(decimals)
    quantized = np.rint(points[faces] * scale).astype(np.int64, copy=False)
    # Sort the three vertices within each triangle, then sort triangles in place through a
    # structured view.  This keeps the hash independent of vertex/index ordering without
    # materialising millions of Python tuples for production meshes.
    vertex_order = np.lexsort(
        (quantized[:, :, 2], quantized[:, :, 1], quantized[:, :, 0]), axis=1
    )
    canonical = np.take_along_axis(quantized, vertex_order[:, :, None], axis=1).reshape(-1, 9)
    record_dtype = np.dtype([(f"c{index}", "<i8") for index in range(9)])
    records = np.ascontiguousarray(canonical).view(record_dtype).reshape(-1)
    records.sort(order=[f"c{index}" for index in range(9)])
    digest = hashlib.sha256()
    digest.update(json.dumps(
        {"frame": GEOMETRY_HASH_FRAME, "decimals": int(decimals), "scale": scale},
        separators=(",", ":"),
    ).encode("ascii"))
    digest.update(memoryview(records).cast("B"))
    return digest.hexdigest()


def geometry_sha256_from_npz(path: str | Path) -> str:
    data = np.load(path)
    return geometry_sha256(data["verts"], data["tris"])


def geometry_sha256_from_glb(path: str | Path) -> str:
    """Hash the first GLB primitive using the same canonical frame as raster NPZs."""
    from workers.mesh_io import read_glb

    vertices, _normals, _uv, triangles = read_glb(Path(path))
    return geometry_sha256(vertices, triangles)


def validate_stage_provenance(
    provenance: Mapping[str, Any],
    *,
    expected_receipt_sha256: str,
    expected_anchor_ids: list[str],
    expected_input_geometry_sha256: str | None = None,
) -> None:
    if provenance.get("anchor_receipt_sha256") != expected_receipt_sha256:
        raise AnchorProvenanceError("ANCHOR_RECEIPT_SOURCE_MISMATCH", "downstream receipt hash does not match accepted receipt")
    if sorted(provenance.get("anchor_ids") or []) != sorted(expected_anchor_ids):
        raise AnchorProvenanceError("ANCHOR_SET_MISMATCH", "downstream anchor ID set does not match accepted receipt")
    if expected_input_geometry_sha256 and provenance.get("input_geometry_sha256") != expected_input_geometry_sha256:
        raise AnchorProvenanceError("GEOMETRY_MUTATION", "downstream input geometry hash does not match accepted geometry")


def provenance_record(*, receipt_sha256: str, anchor_ids: list[str], input_geometry_sha256: str,
                      output_geometry_sha256: str, geometry_unchanged: bool) -> dict[str, Any]:
    return {
        "geometry_hash_frame": GEOMETRY_HASH_FRAME,
        "anchor_receipt_sha256": receipt_sha256,
        "anchor_ids": sorted(anchor_ids),
        "input_geometry_sha256": input_geometry_sha256,
        "output_geometry_sha256": output_geometry_sha256,
        "geometry_unchanged": bool(geometry_unchanged),
    }
