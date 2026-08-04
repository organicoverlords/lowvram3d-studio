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


class EvidenceState(IntEnum):
    """How a triangle/texel received its current texture authority."""

    UNKNOWN = 0
    DIRECT_OBSERVED = 1
    ADJACENT_OBSERVED = 2
    GENERATED_OBSERVED = 3
    UNOBSERVED = 4
    PROCEDURAL_COMPLETION = 5
    MATERIAL_PRIOR = 6
    UNRESOLVED = 7


class FrequencyAuthority(IntEnum):
    """Maximum spatial frequency a provenance class may contribute."""

    NONE = 0
    LOW_ONLY = 1
    LOW_AND_MEDIUM = 2
    FULL = 3


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
        "triangle_id": np.arange(triangle_count, dtype=np.int32),
        "lineage": np.zeros(triangle_count, dtype=np.uint16),
        "lineage_bits": np.zeros(triangle_count, dtype=np.uint16),
        "source_class": np.full(triangle_count, SourceClass.UNKNOWN, dtype=np.uint8),
        "evidence_state": np.full(triangle_count, EvidenceState.UNKNOWN, dtype=np.uint8),
        "source_view": np.full(triangle_count, -1, dtype=np.int16),
        "primary_view": np.full(triangle_count, -1, dtype=np.int16),
        "confidence": np.zeros(triangle_count, dtype=np.float32),
        "frequency_authority": np.full(triangle_count, FrequencyAuthority.NONE, dtype=np.uint8),
        "completion_method": np.full(triangle_count, "unresolved", dtype="U32"),
        "primary_surface_region": np.full(triangle_count, -1, dtype=np.int32),
    }


def create_empty_atlas_provenance(height: int, width: int | None = None) -> dict[str, np.ndarray]:
    if width is None:
        width = height
    if height < 0 or width < 0:
        raise ValueError("atlas dimensions must be non-negative")
    return {
        "triangle_id": np.full((height, width), -1, dtype=np.int32),
        "lineage": np.zeros((height, width), dtype=np.uint16),
        "lineage_bits": np.zeros((height, width), dtype=np.uint16),
        "source_class": np.full((height, width), SourceClass.UNKNOWN, dtype=np.uint8),
        "evidence_state": np.full((height, width), EvidenceState.UNKNOWN, dtype=np.uint8),
        "source_view": np.full((height, width), -1, dtype=np.int16),
        "primary_view": np.full((height, width), -1, dtype=np.int16),
        "source_pixel": np.full((height, width, 2), -1, dtype=np.int32),
        "confidence": np.zeros((height, width), dtype=np.float32),
        "frequency_authority": np.full((height, width), FrequencyAuthority.NONE, dtype=np.uint8),
        "completion_method": np.full((height, width), "unresolved", dtype="U32"),
        "primary_surface_region": np.full((height, width), -1, dtype=np.int32),
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
    if "lineage_bits" in triangle_provenance:
        triangle_provenance["lineage_bits"][ids] |= bits
    triangle_provenance["source_class"][ids] = np.uint8(source_class)
    if "evidence_state" in triangle_provenance:
        triangle_provenance["evidence_state"][ids] = np.uint8(
            EvidenceState.GENERATED_OBSERVED if source_class in {
                SourceClass.GENERATED_FRONT, SourceClass.GENERATED_SIDE, SourceClass.GENERATED_REAR
            } else EvidenceState.DIRECT_OBSERVED
        )
    if "frequency_authority" in triangle_provenance:
        triangle_provenance["frequency_authority"][ids] = np.uint8(FrequencyAuthority.FULL)
    if "completion_method" in triangle_provenance:
        triangle_provenance["completion_method"][ids] = "direct_projection"
    if view_ids is not None:
        values = np.asarray(view_ids, dtype=np.int16)[valid]
        triangle_provenance["primary_view"][ids] = values
        if "source_view" in triangle_provenance:
            triangle_provenance["source_view"][ids] = values
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
    if "lineage_bits" in target:
        target["lineage_bits"][targets[valid]] |= donor.get("lineage_bits", donor["lineage"])[donors[valid]] | np.uint16(Lineage.DONOR_TRANSFER)
    target["source_class"][targets[valid]] = np.uint8(SourceClass.SAFE_DONOR)
    if "evidence_state" in target:
        target["evidence_state"][targets[valid]] = np.uint8(EvidenceState.ADJACENT_OBSERVED)
    if "frequency_authority" in target:
        target["frequency_authority"][targets[valid]] = np.uint8(FrequencyAuthority.LOW_ONLY)
    if "completion_method" in target:
        target["completion_method"][targets[valid]] = "legacy_donor_lineage"
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
    for name in ("lineage", "lineage_bits", "source_class", "evidence_state", "confidence", "frequency_authority", "completion_method", "primary_surface_region"):
        if name in triangle_provenance:
            result[name][valid] = triangle_provenance[name][owner[valid]]
    if "primary_view" in triangle_provenance:
        result["source_view"][valid] = triangle_provenance["primary_view"][owner[valid]]
        result["primary_view"][valid] = triangle_provenance["primary_view"][owner[valid]]
    return result


def raw_rgb_allowed(*, evidence_state, frequency_authority, visible, facing, face_id_match,
                    source_mask_valid, coherent_assignment) -> np.ndarray:
    """Return the fail-closed gate for direct source RGB.

    This is intentionally a pure array function so every projector and test uses the same
    invariant.  ``coherent_assignment`` is a per-sample boolean, not a semantic guess.
    """
    state = np.asarray(evidence_state)
    authority = np.asarray(frequency_authority)
    return np.isin(state, (EvidenceState.DIRECT_OBSERVED, EvidenceState.GENERATED_OBSERVED)) \
        & (authority == FrequencyAuthority.FULL) \
        & np.asarray(visible, dtype=bool) & (np.asarray(facing, dtype=np.float32) > 0.0) \
        & np.asarray(face_id_match, dtype=bool) & np.asarray(source_mask_valid, dtype=bool) \
        & np.asarray(coherent_assignment, dtype=bool)


def validate_evidence_invariants(provenance: dict[str, np.ndarray]) -> dict:
    """Validate that raw/high-frequency authority is never attached to unobserved data."""
    state = np.asarray(provenance["evidence_state"])
    authority = np.asarray(provenance["frequency_authority"])
    lineage = np.asarray(provenance["lineage"])
    if "lineage_bits" in provenance:
        lineage = lineage | np.asarray(provenance["lineage_bits"])
    observed_states = np.isin(state, (EvidenceState.DIRECT_OBSERVED, EvidenceState.GENERATED_OBSERVED))
    unobserved_raw = (~observed_states) & (lineage & np.uint16(
        Lineage.ORIGINAL_FACE | Lineage.ORIGINAL_NONFACE | Lineage.GENERATED_FRONT |
        Lineage.GENERATED_SIDE | Lineage.GENERATED_REAR
    ) != 0)
    unobserved_full = (~observed_states) & (authority == FrequencyAuthority.FULL)
    return {
        "passed": not bool(unobserved_raw.any() or unobserved_full.any()),
        "unobserved_raw_image_rgb_texels": int(unobserved_raw.sum()),
        "unobserved_full_frequency_texels": int(unobserved_full.sum()),
    }


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
