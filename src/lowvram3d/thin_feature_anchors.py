"""Deterministic discovery receipts for source-supported thin mesh features.

This module is deliberately independent of Blender.  Blender workers can pass a welded
``trimesh.Trimesh`` (and the SHA-256 of the source file) or load a mesh path directly.  Discovery
is conservative: small detached components and narrow attached caps are only registered when they
own pixels in at least one of the production six orthographic silhouettes.

The receipt contains geometry seeds rather than face or object indices.  Indices depend on import
and object ordering; deterministic, quantized model-normalised float positions do not.  Later
stages can therefore map the seeds back onto a freshly imported mesh without making this contract
Blender-specific.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import trimesh

from .asset_profiles import AssetProfile, PROFILES, SAFEST_PROFILE
from .geometry_compare import load_mesh


SCHEMA_VERSION = "1.0"
RECEIPT_TYPE = "thin_feature_anchor_discovery"

# Same semantic order and camera directions as blender/render_controls.py.  The values are unit
# camera-location vectors; silhouette ownership is invariant to their magnitude.
SIX_ORTHOGRAPHIC_VIEWS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("front", (0.0, -1.0, 0.0)),
    ("right", (1.0, 0.0, 0.0)),
    ("back", (0.0, 1.0, 0.0)),
    ("left", (-1.0, 0.0, 0.0)),
    ("top", (0.0, 0.0, 1.0)),
    ("bottom", (0.0, 0.0, -1.0)),
)

_REQUIRED_RECEIPT_FIELDS = (
    "schema_version",
    "receipt_type",
    "source_mesh_sha256",
    "view_set",
    "discovery",
    "anchors",
)
_REQUIRED_ANCHOR_FIELDS = (
    "anchor_id",
    "candidate_kind",
    "fingerprint",
    "seeds",
    "bounds_normalized",
    "per_view_support",
    "supported_views",
    "survival_floor",
)


class AnchorReceiptValidationError(ValueError):
    """Raised when an anchor receipt cannot be trusted."""


@dataclass(frozen=True, slots=True)
class DiscoveryConfig:
    """Asset-neutral thresholds derived from existing :class:`AssetProfile` fields."""

    render_size: int
    quantization_diag: float
    max_seed_points: int
    detached_max_area_fraction: float
    attached_max_area_fraction: float
    attached_min_aspect_ratio: float
    attached_max_transverse_diag_fraction: float
    attached_cap_start_fraction: float
    minimum_silhouette_support_ratio: float
    minimum_exclusive_pixels: int
    survival_retention_ratio: float

    @classmethod
    def from_profile(cls, profile: AssetProfile) -> "DiscoveryConfig":
        # debris_height_min already expresses how conservative the profile is around small,
        # outboard geometry.  max_axis_ratio supplies a scale-relative notion of elongation.
        detached_area = min(0.08, max(0.02, (1.0 - profile.debris_height_min) * 0.20))
        attached_area = min(0.24, max(0.08, detached_area * 3.0))
        transverse = min(0.25, max(0.08, 1.0 - profile.debris_height_min))
        return cls(
            render_size=192,
            quantization_diag=1.0e-6,
            max_seed_points=32,
            detached_max_area_fraction=round(detached_area, 8),
            attached_max_area_fraction=round(attached_area, 8),
            attached_min_aspect_ratio=round(max(3.0, profile.max_axis_ratio * 0.5), 8),
            attached_max_transverse_diag_fraction=round(transverse, 8),
            attached_cap_start_fraction=round(
                min(0.72, max(0.50, profile.debris_height_min - 0.15)), 8
            ),
            minimum_silhouette_support_ratio=round(
                min(0.20, max(0.05, 1.0 / max(profile.max_axis_ratio, 1.0))), 8
            ),
            minimum_exclusive_pixels=1,
            survival_retention_ratio=0.75 if profile.preserve_thin_features else 0.60,
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    kind: str
    face_ids: np.ndarray
    area_fraction: float
    axis_view: str | None = None


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the byte identity used to bind a receipt to its clean source mesh."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_source_hash(value: str) -> str:
    candidate = str(value).strip().lower()
    if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
        raise ValueError("source_mesh_sha256 must be a 64-character hexadecimal SHA-256")
    return candidate


def _clean_welded_copy(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("anchor discovery requires a non-empty triangular mesh")
    clean = mesh.copy()
    if clean.faces.ndim != 2 or clean.faces.shape[1] != 3:
        raise ValueError("anchor discovery requires triangular faces")
    clean.merge_vertices(merge_tex=True, merge_norm=True)
    if hasattr(clean, "nondegenerate_faces"):
        clean.update_faces(clean.nondegenerate_faces())
    elif hasattr(clean, "remove_degenerate_faces"):
        clean.remove_degenerate_faces()
    if hasattr(clean, "unique_faces"):
        clean.update_faces(clean.unique_faces())
    elif hasattr(clean, "remove_duplicate_faces"):
        clean.remove_duplicate_faces()
    clean.remove_unreferenced_vertices()
    if not len(clean.faces) or not np.isfinite(clean.vertices).all():
        raise ValueError("anchor discovery requires finite, positive-area geometry")
    return clean


def _connected_face_groups(mesh: trimesh.Trimesh, face_ids: np.ndarray | None = None) -> list[np.ndarray]:
    nodes = np.arange(len(mesh.faces), dtype=np.int64) if face_ids is None else np.asarray(face_ids, np.int64)
    if not len(nodes):
        return []
    allowed = np.zeros(len(mesh.faces), dtype=bool)
    allowed[nodes] = True
    adjacency = np.asarray(mesh.face_adjacency, np.int64)
    if len(adjacency):
        adjacency = adjacency[allowed[adjacency[:, 0]] & allowed[adjacency[:, 1]]]
    groups = trimesh.graph.connected_components(adjacency, nodes=nodes, min_len=1)
    return [np.sort(np.asarray(group, np.int64)) for group in groups]


def _quantized_geometry(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    center: np.ndarray,
    diagonal: float,
    quantum: float,
) -> list[list[list[int]]]:
    triangles = (np.asarray(mesh.triangles, np.float64)[face_ids] - center) / diagonal
    quantized = np.rint(triangles / quantum).astype(np.int64)
    canonical: list[list[list[int]]] = []
    for triangle in quantized:
        canonical.append(sorted((vertex.tolist() for vertex in triangle)))
    canonical.sort()
    return canonical


def _geometry_fingerprint(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    center: np.ndarray,
    diagonal: float,
    quantum: float,
) -> str:
    payload = json.dumps(
        _quantized_geometry(mesh, face_ids, center, diagonal, quantum),
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _group_area(mesh: trimesh.Trimesh, face_ids: np.ndarray) -> float:
    return float(np.asarray(mesh.area_faces, np.float64)[face_ids].sum())


def _detached_candidates(
    mesh: trimesh.Trimesh,
    config: DiscoveryConfig,
    center: np.ndarray,
    diagonal: float,
) -> tuple[list[_Candidate], np.ndarray]:
    groups = _connected_face_groups(mesh)
    total_area = max(float(mesh.area), 1.0e-12)
    ranked = []
    for group in groups:
        area = _group_area(mesh, group)
        fingerprint = _geometry_fingerprint(
            mesh, group, center, diagonal, config.quantization_diag
        )
        ranked.append((area, fingerprint, group))
    # A geometry-derived tie break prevents import/object ordering from choosing a different body.
    main_area, _, main = max(ranked, key=lambda item: (item[0], item[1]))
    candidates = []
    for area, _, group in ranked:
        if np.array_equal(group, main):
            continue
        fraction = area / total_area
        if fraction <= config.detached_max_area_fraction:
            candidates.append(_Candidate("detached_component", group, fraction))
    return candidates, main


def _view_basis(direction: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axis = np.asarray(direction, np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    up_hint = np.asarray((0.0, 0.0, 1.0), np.float64)
    if abs(float(np.dot(axis, up_hint))) > 0.92:
        up_hint = np.asarray((0.0, 1.0, 0.0), np.float64)
    right = np.cross(up_hint, axis)
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    up = np.cross(axis, right)
    return right, up, axis


def _attached_shape_metrics(
    points: np.ndarray,
    direction: Sequence[float],
    diagonal: float,
) -> tuple[float, float, float]:
    right, up, axis = _view_basis(direction)
    projected = np.column_stack((points @ right, points @ up, points @ axis))
    extent = np.ptp(projected, axis=0)
    axial = float(extent[2])
    transverse = float(max(extent[0], extent[1], diagonal * 1.0e-9))
    return axial, transverse, axial / transverse


def _attached_candidates(
    mesh: trimesh.Trimesh,
    main_faces: np.ndarray,
    config: DiscoveryConfig,
    total_area: float,
    diagonal: float,
) -> list[_Candidate]:
    vertices = np.asarray(mesh.vertices, np.float64)
    faces = np.asarray(mesh.faces, np.int64)
    face_centers = np.asarray(mesh.triangles_center, np.float64)
    candidates: list[_Candidate] = []

    cut_fractions = (
        config.attached_cap_start_fraction,
        (config.attached_cap_start_fraction + 0.80) * 0.5,
        0.80,
    )
    for view_name, direction_tuple in SIX_ORTHOGRAPHIC_VIEWS:
        direction = np.asarray(direction_tuple, np.float64)
        vertex_depth = vertices @ direction
        low, high = float(vertex_depth.min()), float(vertex_depth.max())
        span = high - low
        if span <= diagonal * 1.0e-9:
            continue
        center_depth = face_centers[:, :] @ direction
        for cut_fraction in cut_fractions:
            cut = low + span * cut_fraction
            selected = main_faces[center_depth[main_faces] >= cut]
            if not len(selected) or len(selected) == len(main_faces):
                continue
            accepted_at_cut: list[_Candidate] = []
            for group in _connected_face_groups(mesh, selected):
                area_fraction = _group_area(mesh, group) / total_area
                if area_fraction > config.attached_max_area_fraction:
                    continue
                points = vertices[np.unique(faces[group])]
                axial, transverse, aspect = _attached_shape_metrics(points, direction, diagonal)
                if axial <= diagonal * config.quantization_diag:
                    continue
                if transverse / diagonal > config.attached_max_transverse_diag_fraction:
                    continue
                if aspect < config.attached_min_aspect_ratio:
                    continue
                accepted_at_cut.append(
                    _Candidate("attached_protrusion", group, area_fraction, axis_view=view_name)
                )
            if accepted_at_cut:
                candidates.extend(accepted_at_cut)
                break

    # The same protrusion may be visible from neighbouring cap directions.  Keep the larger region
    # when face overlap demonstrates they describe the same geometry.
    ordered = sorted(candidates, key=lambda item: (-len(item.face_ids), item.axis_view or ""))
    unique: list[_Candidate] = []
    for candidate in ordered:
        face_set = set(candidate.face_ids.tolist())
        duplicate = False
        for previous in unique:
            previous_set = set(previous.face_ids.tolist())
            overlap = len(face_set & previous_set) / max(1, min(len(face_set), len(previous_set)))
            if overlap >= 0.65:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique


def _raster_mask(
    triangles: np.ndarray,
    direction: Sequence[float],
    center: np.ndarray,
    half_extent: float,
    size: int,
) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.uint8)
    if not len(triangles):
        return mask.astype(bool)
    right, up, _ = _view_basis(direction)
    relative = triangles - center
    x = relative @ right
    y = relative @ up
    px = np.rint((x / half_extent * 0.5 + 0.5) * (size - 1)).astype(np.int32)
    py = np.rint((1.0 - (y / half_extent * 0.5 + 0.5)) * (size - 1)).astype(np.int32)
    polygons = np.stack((px, py), axis=-1)
    for polygon in polygons:
        cv2.fillConvexPoly(mask, polygon, 1, lineType=cv2.LINE_8)
    return mask.astype(bool)


def _silhouette_support(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    center: np.ndarray,
    diagonal: float,
    config: DiscoveryConfig,
) -> dict[str, dict[str, Any]]:
    all_triangles = np.asarray(mesh.triangles, np.float64)
    selected = np.zeros(len(mesh.faces), dtype=bool)
    selected[face_ids] = True
    half_extent = diagonal * 0.55
    support: dict[str, dict[str, Any]] = {}
    for view_name, direction in SIX_ORTHOGRAPHIC_VIEWS:
        candidate_mask = _raster_mask(
            all_triangles[selected], direction, center, half_extent, config.render_size
        )
        remainder_mask = _raster_mask(
            all_triangles[~selected], direction, center, half_extent, config.render_size
        )
        candidate_pixels = int(candidate_mask.sum())
        exclusive_pixels = int((candidate_mask & ~remainder_mask).sum())
        ratio = exclusive_pixels / max(candidate_pixels, 1)
        supported = (
            exclusive_pixels >= config.minimum_exclusive_pixels
            and ratio >= config.minimum_silhouette_support_ratio
        )
        support[view_name] = {
            "candidate_pixels": candidate_pixels,
            "exclusive_pixels": exclusive_pixels,
            "support_ratio": round(float(ratio), 8),
            "supported": bool(supported),
        }
    return support


def _seed_points(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    center: np.ndarray,
    diagonal: float,
    config: DiscoveryConfig,
) -> list[list[float]]:
    vertex_ids = np.unique(np.asarray(mesh.faces, np.int64)[face_ids])
    normalized = (np.asarray(mesh.vertices, np.float64)[vertex_ids] - center) / diagonal
    # Quantize only to make the receipt deterministic; retain the normalized coordinate units so
    # downstream LOD/debris consumers can use the values directly without an implicit scale.
    quantized = np.rint(normalized / config.quantization_diag) * config.quantization_diag
    points = sorted({tuple(float(value) for value in point) for point in quantized})
    if len(points) > config.max_seed_points:
        indices = np.linspace(0, len(points) - 1, config.max_seed_points, dtype=np.int64)
        points = [points[int(index)] for index in indices]
    return [[float(value) for value in point] for point in points]


def _normalised_bounds(
    mesh: trimesh.Trimesh,
    face_ids: np.ndarray,
    center: np.ndarray,
    diagonal: float,
    quantum: float,
) -> dict[str, list[float]]:
    vertex_ids = np.unique(np.asarray(mesh.faces, np.int64)[face_ids])
    points = (np.asarray(mesh.vertices, np.float64)[vertex_ids] - center) / diagonal
    low = (np.rint(points.min(axis=0) / quantum) * quantum).tolist()
    high = (np.rint(points.max(axis=0) / quantum) * quantum).tolist()
    low = [float(value) for value in low]
    high = [float(value) for value in high]
    return {"min": low, "max": high}


def _anchor_record(
    mesh: trimesh.Trimesh,
    candidate: _Candidate,
    source_hash: str,
    center: np.ndarray,
    diagonal: float,
    config: DiscoveryConfig,
) -> dict[str, Any] | None:
    support = _silhouette_support(mesh, candidate.face_ids, center, diagonal, config)
    supported_views = [name for name, _ in SIX_ORTHOGRAPHIC_VIEWS if support[name]["supported"]]
    if not supported_views:
        return None
    fingerprint = _geometry_fingerprint(
        mesh, candidate.face_ids, center, diagonal, config.quantization_diag
    )
    identity_support = {
        name: {
            "exclusive_pixels": support[name]["exclusive_pixels"],
            "support_ratio": support[name]["support_ratio"],
        }
        for name, _ in SIX_ORTHOGRAPHIC_VIEWS
    }
    identity = {
        "source_mesh_sha256": source_hash,
        "fingerprint": fingerprint,
        "view_support": identity_support,
    }
    anchor_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    floor_pixels = {
        name: int(math.ceil(support[name]["exclusive_pixels"] * config.survival_retention_ratio))
        for name in supported_views
    }
    return {
        "anchor_id": f"tfa-{anchor_digest[:24]}",
        "candidate_kind": candidate.kind,
        "fingerprint": fingerprint,
        "seeds": _seed_points(mesh, candidate.face_ids, center, diagonal, config),
        "bounds_normalized": _normalised_bounds(
            mesh, candidate.face_ids, center, diagonal, config.quantization_diag
        ),
        "area_fraction": round(float(candidate.area_fraction), 10),
        "axis_view": candidate.axis_view,
        "per_view_support": support,
        "supported_views": supported_views,
        "survival_floor": {
            "minimum_supported_views": 1,
            "exclusive_pixel_retention_ratio": config.survival_retention_ratio,
            "per_view_exclusive_pixels": floor_pixels,
        },
    }


def discover_thin_feature_anchors(
    mesh_or_path: trimesh.Trimesh | str | Path,
    *,
    source_mesh_sha256: str | None = None,
    profile: AssetProfile | None = None,
    config: DiscoveryConfig | None = None,
) -> dict[str, Any]:
    """Discover source-silhouette-supported anchors and return a deterministic receipt.

    A path is hashed before it is read.  In-memory meshes require the caller to provide the source
    file hash explicitly so the receipt can never claim an invented source identity.
    """
    if isinstance(mesh_or_path, (str, Path)):
        path = Path(mesh_or_path)
        measured_hash = sha256_file(path)
        if source_mesh_sha256 is not None and _normalise_source_hash(source_mesh_sha256) != measured_hash:
            raise ValueError("source_mesh_sha256 does not match the mesh file bytes")
        source_hash = measured_hash
        mesh = load_mesh(path)
    else:
        if source_mesh_sha256 is None:
            raise ValueError("source_mesh_sha256 is required for in-memory mesh discovery")
        source_hash = _normalise_source_hash(source_mesh_sha256)
        mesh = mesh_or_path

    resolved_profile = profile or PROFILES[SAFEST_PROFILE]
    resolved_config = config or DiscoveryConfig.from_profile(resolved_profile)
    clean = _clean_welded_copy(mesh)
    bounds = np.asarray(clean.bounds, np.float64)
    center = bounds.mean(axis=0)
    diagonal = float(np.linalg.norm(bounds[1] - bounds[0]))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("anchor discovery requires non-zero mesh bounds")

    detached, main_faces = _detached_candidates(
        clean, resolved_config, center, diagonal
    )
    attached = _attached_candidates(
        clean,
        main_faces,
        resolved_config,
        max(float(clean.area), 1.0e-12),
        diagonal,
    )
    records = []
    for candidate in detached + attached:
        record = _anchor_record(
            clean, candidate, source_hash, center, diagonal, resolved_config
        )
        if record is not None:
            records.append(record)
    records.sort(key=lambda item: item["anchor_id"])

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "receipt_type": RECEIPT_TYPE,
        "source_mesh_sha256": source_hash,
        "view_set": [
            {"name": name, "direction": list(direction)}
            for name, direction in SIX_ORTHOGRAPHIC_VIEWS
        ],
        "discovery": {
            "algorithm": "welded_components_and_six_view_attached_caps_v1",
            "coordinate_space": "source_mesh_normalized_by_bounds_diagonal",
            "coordinate_encoding": "normalized_float_quantized_by_quantization_diag",
            # Persist the exact clean-source frame used to encode every seed and bound.  Downstream
            # stages must not derive this from a decimated mesh whose extrema may have changed.
            "normalization_frame": {
                "center": [float(value) for value in center],
                "diagonal": float(diagonal),
                "bounds_min": [float(value) for value in bounds[0]],
                "bounds_max": [float(value) for value in bounds[1]],
            },
            "profile_inputs": {
                "preserve_thin_features": resolved_profile.preserve_thin_features,
                "max_axis_ratio": resolved_profile.max_axis_ratio,
                "debris_height_min": resolved_profile.debris_height_min,
                "texture_resolution": resolved_profile.texture_resolution,
            },
            "parameters": asdict(resolved_config),
            "candidate_counts": {
                "detached": len(detached),
                "attached": len(attached),
                "registered": len(records),
            },
        },
        "anchors": records,
    }
    validate_anchor_receipt(receipt, expected_source_mesh_sha256=source_hash)
    return receipt


def serialize_anchor_receipt(receipt: Mapping[str, Any]) -> bytes:
    """Validate and encode canonical JSON bytes with no run-dependent metadata."""
    validate_anchor_receipt(receipt)
    return (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def anchor_receipt_sha256(receipt: Mapping[str, Any]) -> str:
    return hashlib.sha256(serialize_anchor_receipt(receipt)).hexdigest()


def parse_anchor_receipt(
    payload: bytes | str,
    *,
    expected_source_mesh_sha256: str | None = None,
) -> dict[str, Any]:
    try:
        parsed = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
        raise AnchorReceiptValidationError(f"malformed anchor receipt JSON: {error}") from error
    validate_anchor_receipt(parsed, expected_source_mesh_sha256=expected_source_mesh_sha256)
    return parsed


def validate_anchor_receipt(
    receipt: Mapping[str, Any] | Any,
    *,
    expected_source_mesh_sha256: str | None = None,
) -> None:
    """Validate the v1 contract and report all structural errors in one clear exception."""
    errors: list[str] = []
    if not isinstance(receipt, Mapping):
        raise AnchorReceiptValidationError("malformed anchor receipt: root must be an object")
    missing = [field for field in _REQUIRED_RECEIPT_FIELDS if field not in receipt]
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")

    version = receipt.get("schema_version")
    if not isinstance(version, str) or not version:
        errors.append("malformed schema_version: expected a non-empty version string")
    elif version.split(".", 1)[0] != SCHEMA_VERSION.split(".", 1)[0]:
        errors.append(f"unsupported schema_version {version!r}; supported major version is 1")
    if receipt.get("receipt_type") not in (None, RECEIPT_TYPE):
        errors.append(f"malformed receipt_type: expected {RECEIPT_TYPE!r}")

    source_hash = receipt.get("source_mesh_sha256")
    try:
        normalized_hash = _normalise_source_hash(source_hash) if source_hash is not None else None
    except ValueError as error:
        errors.append(str(error))
        normalized_hash = None
    if expected_source_mesh_sha256 is not None:
        try:
            expected_hash = _normalise_source_hash(expected_source_mesh_sha256)
            if normalized_hash is not None and normalized_hash != expected_hash:
                errors.append(
                    "source-hash mismatch: receipt source_mesh_sha256 "
                    f"{normalized_hash} != expected {expected_hash}"
                )
        except ValueError as error:
            errors.append(f"invalid expected source hash: {error}")

    view_set = receipt.get("view_set")
    expected_views = [name for name, _ in SIX_ORTHOGRAPHIC_VIEWS]
    if not isinstance(view_set, list):
        errors.append("malformed view_set: expected a list")
    else:
        names = [item.get("name") for item in view_set if isinstance(item, Mapping)]
        if len(names) != len(view_set) or names != expected_views:
            errors.append(
                "malformed view_set: expected ordered views " + ", ".join(expected_views)
            )
        for index, item in enumerate(view_set):
            if not isinstance(item, Mapping) or "direction" not in item:
                errors.append(f"view_set[{index}] is missing required direction")
                continue
            direction = item["direction"]
            if not isinstance(direction, list) or len(direction) != 3:
                errors.append(f"view_set[{index}].direction must contain three numbers")

    discovery = receipt.get("discovery")
    if not isinstance(discovery, Mapping):
        errors.append("malformed discovery: expected an object")
    else:
        for field in ("algorithm", "profile_inputs", "parameters", "candidate_counts"):
            if field not in discovery:
                errors.append(f"discovery is missing required field {field!r}")
        frame = discovery.get("normalization_frame")
        if not isinstance(frame, Mapping):
            errors.append("discovery.normalization_frame must be an object")
        else:
            frame_values: dict[str, list[float]] = {}
            for field in ("center", "bounds_min", "bounds_max"):
                values = frame.get(field)
                if (
                    not isinstance(values, list)
                    or len(values) != 3
                    or any(
                        type(value) is not float
                        or not math.isfinite(float(value))
                        for value in values
                    )
                ):
                    errors.append(
                        f"discovery.normalization_frame.{field} must be three finite floats"
                    )
                else:
                    frame_values[field] = [float(value) for value in values]
            diagonal = frame.get("diagonal")
            if (
                type(diagonal) is not float
                or not math.isfinite(float(diagonal))
                or float(diagonal) <= 0.0
            ):
                errors.append(
                    "discovery.normalization_frame.diagonal must be a positive finite float"
                )
            if len(frame_values) == 3:
                if any(
                    low > high
                    for low, high in zip(frame_values["bounds_min"], frame_values["bounds_max"])
                ):
                    errors.append(
                        "discovery.normalization_frame bounds_min must not exceed bounds_max"
                    )
                midpoint = [
                    (low + high) * 0.5
                    for low, high in zip(frame_values["bounds_min"], frame_values["bounds_max"])
                ]
                if any(
                    not math.isclose(center, expected, rel_tol=1.0e-7, abs_tol=1.0e-7)
                    for center, expected in zip(frame_values["center"], midpoint)
                ):
                    errors.append(
                        "discovery.normalization_frame.center must be the bounds midpoint"
                    )
                if type(diagonal) is float and math.isfinite(float(diagonal)):
                    measured = math.sqrt(
                        sum(
                            (high - low) ** 2
                            for low, high in zip(
                                frame_values["bounds_min"], frame_values["bounds_max"]
                            )
                        )
                    )
                    if not math.isclose(
                        float(diagonal), measured, rel_tol=1.0e-7, abs_tol=1.0e-7
                    ):
                        errors.append(
                            "discovery.normalization_frame.diagonal must match bounds"
                        )

    anchors = receipt.get("anchors")
    if not isinstance(anchors, list):
        errors.append("malformed anchors: expected a list")
    else:
        anchor_ids: list[str] = []
        for index, anchor in enumerate(anchors):
            if not isinstance(anchor, Mapping):
                errors.append(f"anchors[{index}] must be an object")
                continue
            anchor_missing = [field for field in _REQUIRED_ANCHOR_FIELDS if field not in anchor]
            if anchor_missing:
                errors.append(
                    f"anchors[{index}] missing required fields: {', '.join(anchor_missing)}"
                )
            anchor_id = anchor.get("anchor_id")
            if not isinstance(anchor_id, str) or not anchor_id.startswith("tfa-"):
                errors.append(f"anchors[{index}].anchor_id is malformed")
            else:
                anchor_ids.append(anchor_id)
            if not isinstance(anchor.get("per_view_support"), Mapping):
                errors.append(f"anchors[{index}].per_view_support must be an object")
            elif (
                list(anchor["per_view_support"].keys())
                and set(anchor["per_view_support"].keys()) != set(expected_views)
            ) or len(anchor["per_view_support"]) != len(expected_views):
                errors.append(
                    f"anchors[{index}].per_view_support must contain the six-view set"
                )
            if not isinstance(anchor.get("survival_floor"), Mapping):
                errors.append(f"anchors[{index}].survival_floor must be an object")
            seeds = anchor.get("seeds")
            if not isinstance(seeds, list) or not seeds:
                errors.append(f"anchors[{index}].seeds must be a non-empty list")
            else:
                for seed_index, seed in enumerate(seeds):
                    if (
                        not isinstance(seed, list)
                        or len(seed) != 3
                        or any(
                            type(value) is not float
                            or not math.isfinite(float(value))
                            for value in seed
                        )
                    ):
                        errors.append(
                            f"anchors[{index}].seeds[{seed_index}] must be three finite normalized floats"
                        )
            bounds = anchor.get("bounds_normalized")
            if not isinstance(bounds, Mapping):
                errors.append(f"anchors[{index}].bounds_normalized must be an object")
            else:
                bound_values: dict[str, list[float]] = {}
                for bound_name in ("min", "max"):
                    values = bounds.get(bound_name)
                    if (
                        not isinstance(values, list)
                        or len(values) != 3
                        or any(
                            type(value) is not float
                            or not math.isfinite(float(value))
                            for value in values
                        )
                    ):
                        errors.append(
                            f"anchors[{index}].bounds_normalized.{bound_name} must be three finite normalized floats"
                        )
                    else:
                        bound_values[bound_name] = [float(value) for value in values]
                if len(bound_values) == 2 and any(
                    low > high
                    for low, high in zip(bound_values["min"], bound_values["max"])
                ):
                    errors.append(f"anchors[{index}].bounds_normalized min exceeds max")
        if len(anchor_ids) != len(set(anchor_ids)):
            errors.append("anchor_id values must be unique")
        if anchor_ids != sorted(anchor_ids):
            errors.append("anchors must be ordered by anchor_id for deterministic serialization")

    if errors:
        raise AnchorReceiptValidationError("invalid anchor receipt: " + "; ".join(errors))
