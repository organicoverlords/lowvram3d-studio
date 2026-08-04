"""Compact, transitive texture provenance for projection and synthesis stages."""
from __future__ import annotations

from enum import IntEnum, IntFlag
from typing import Iterable

import numpy as np


class SourceClass(IntEnum):
    UNKNOWN = 0
    ORIGINAL_FACE = 1
    ORIGINAL_NONFACE = 2
    GENERATED_FRONT = 3
    GENERATED_SIDE = 4
    GENERATED_REAR = 5
    SAFE_DONOR = 6
    COMPONENT_PRIOR = 7
    GLOBAL_PRIOR = 8
    FACE_REFINEMENT = 9


class Lineage(IntFlag):
    UNKNOWN = 0
    ORIGINAL_FACE = 1 << 0
    ORIGINAL_NONFACE = 1 << 1
    GENERATED_FRONT = 1 << 2
    GENERATED_SIDE = 1 << 3
    GENERATED_REAR = 1 << 4
    DONOR_TRANSFER = 1 << 5
    COMPONENT_PRIOR = 1 << 6
    GLOBAL_PRIOR = 1 << 7
    FACE_REFINEMENT = 1 << 8


LINEAGE_FIELDS = tuple(item.name for item in Lineage if item is not Lineage.UNKNOWN)


def _container(shape: tuple[int, ...]) -> dict[str, np.ndarray]:
    return {name.lower(): np.zeros(shape, dtype=np.uint16) for name in ("lineage",)}


def create_empty_triangle_provenance(triangle_count: int) -> dict[str, np.ndarray]:
    if triangle_count < 0:
        raise ValueError("triangle_count must be non-negative")
    return {
        "lineage": np.zeros(triangle_count, dtype=np.uint16),
        "source_class": np.full(triangle_count, SourceClass.UNKNOWN, dtype=np.uint8),
        "primary_view": np.full(triangle_count, -1, dtype=np.int16),
        "confidence": np.zeros(triangle_count, dtype=np.float32),
    }


def create_empty_atlas_provenance(height: int, width: int | None = None) -> dict[str, np.ndarray]:
    if width is None:
        width = height
    if height < 0 or width < 0:
        raise ValueError("atlas dimensions must be non-negative")
    return {
        "lineage": np.zeros((height, width), dtype=np.uint16),
        "source_class": np.full((height, width), SourceClass.UNKNOWN, dtype=np.uint8),
        "source_view": np.full((height, width), -1, dtype=np.int16),
        "source_pixel": np.full((height, width, 2), -1, dtype=np.int32),
        "confidence": np.zeros((height, width), dtype=np.float32),
        "triangle_id": np.full((height, width), -1, dtype=np.int32),
    }


def _as_lineage(value: int | SourceClass | Lineage) -> np.uint16:
    if isinstance(value, SourceClass):
        return np.uint16({
            SourceClass.ORIGINAL_FACE: Lineage.ORIGINAL_FACE,
            SourceClass.ORIGINAL_NONFACE: Lineage.ORIGINAL_NONFACE,
            SourceClass.GENERATED_FRONT: Lineage.GENERATED_FRONT,
            SourceClass.GENERATED_SIDE: Lineage.GENERATED_SIDE,
            SourceClass.GENERATED_REAR: Lineage.GENERATED_REAR,
            SourceClass.SAFE_DONOR: Lineage.DONOR_TRANSFER,
            SourceClass.COMPONENT_PRIOR: Lineage.COMPONENT_PRIOR,
            SourceClass.GLOBAL_PRIOR: Lineage.GLOBAL_PRIOR,
            SourceClass.FACE_REFINEMENT: Lineage.ORIGINAL_FACE | Lineage.FACE_REFINEMENT,
        }.get(value, Lineage.UNKNOWN))
    return np.uint16(int(value))


def direct_projection_lineage(triangle_provenance: dict[str, np.ndarray], triangle_ids: Iterable[int], source_class: SourceClass, *, view_ids=None, confidence=None) -> dict[str, np.ndarray]:
    ids = np.asarray(list(triangle_ids), dtype=np.int64)
    valid = (ids >= 0) & (ids < len(triangle_provenance["lineage"]))
    ids = ids[valid]
    bits = _as_lineage(source_class)
    triangle_provenance["lineage"][ids] |= bits
    triangle_provenance["source_class"][ids] = np.uint8(source_class)
    if view_ids is not None:
        triangle_provenance["primary_view"][ids] = np.asarray(view_ids, dtype=np.int16)[valid]
    if confidence is not None:
        triangle_provenance["confidence"][ids] = np.asarray(confidence, dtype=np.float32)[valid]
    return triangle_provenance


def propagate_donor_lineage(target: dict[str, np.ndarray], target_ids, donor: dict[str, np.ndarray], donor_ids) -> dict[str, np.ndarray]:
    targets = np.asarray(target_ids, dtype=np.int64)
    donors = np.asarray(donor_ids, dtype=np.int64)
    if targets.shape != donors.shape:
        raise ValueError("target_ids and donor_ids must have the same shape")
    valid = ((targets >= 0) & (targets < len(target["lineage"])) &
             (donors >= 0) & (donors < len(donor["lineage"])))
    target["lineage"][targets[valid]] |= donor["lineage"][donors[valid]] | np.uint16(Lineage.DONOR_TRANSFER)
    target["source_class"][targets[valid]] = np.uint8(SourceClass.SAFE_DONOR)
    target["confidence"][targets[valid]] = donor["confidence"][donors[valid]]
    return target


def merge_face_refinement_lineage(provenance: dict[str, np.ndarray], ids=None) -> dict[str, np.ndarray]:
    if ids is None:
        ids = np.arange(len(provenance["lineage"]), dtype=np.int64)
    ids = np.asarray(ids, dtype=np.int64)
    valid = (ids >= 0) & (ids < len(provenance["lineage"]))
    provenance["lineage"][ids[valid]] |= np.uint16(Lineage.ORIGINAL_FACE | Lineage.FACE_REFINEMENT)
    provenance["source_class"][ids[valid]] = np.uint8(SourceClass.FACE_REFINEMENT)
    return provenance


def rasterize_triangle_lineage_to_atlas(triangle_provenance: dict[str, np.ndarray], owner: np.ndarray) -> dict[str, np.ndarray]:
    owner = np.asarray(owner, dtype=np.int64)
    result = create_empty_atlas_provenance(*owner.shape)
    valid = (owner >= 0) & (owner < len(triangle_provenance["lineage"]))
    result["triangle_id"][valid] = owner[valid]
    for name in ("lineage", "source_class", "confidence"):
        result[name][valid] = triangle_provenance[name][owner[valid]]
    if "primary_view" in triangle_provenance:
        result["source_view"][valid] = triangle_provenance["primary_view"][owner[valid]]
    return result


def summarize_provenance(provenance: dict[str, np.ndarray]) -> dict:
    lineage = np.asarray(provenance["lineage"])
    summary = {"count": int(lineage.size), "lineage_counts": {}}
    for item in Lineage:
        if item is Lineage.UNKNOWN:
            continue
        summary["lineage_counts"][item.name] = int(np.count_nonzero(lineage & np.uint16(item)))
    if "source_class" in provenance:
        values, counts = np.unique(provenance["source_class"], return_counts=True)
        summary["source_class_counts"] = {SourceClass(int(v)).name if int(v) in SourceClass._value2member_map_ else str(int(v)): int(c) for v, c in zip(values, counts)}
    return summary


def validate_no_forbidden_lineage(provenance: dict[str, np.ndarray], forbidden: int | Lineage, ids=None) -> dict:
    lineage = np.asarray(provenance["lineage"])
    mask = np.ones(lineage.shape, dtype=bool) if ids is None else np.isin(np.arange(len(lineage)), np.asarray(ids, dtype=np.int64))
    violations = mask & ((lineage & np.uint16(forbidden)) != 0)
    return {"passed": not bool(violations.any()), "violation_count": int(violations.sum()), "violation_ids": np.flatnonzero(violations).astype(int).tolist()}


def save_npz(path, provenance: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **provenance)


def load_npz(path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {name: data[name] for name in data.files}
