"""Pure geometry-cleanup policy shared by Blender stages and unit tests.

The policy is intentionally conservative. UV seams and split normals can make a glTF mesh look
fragmented until coincident vertices are welded, so component decisions must only be made after
that topology reconstruction. Large detached components are never silently deleted in
``conservative`` mode. ``single_subject_strict`` is reserved for profiles that explicitly require
one continuous fused subject and do not request movable/separate parts.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComponentMetrics:
    face_count: int
    face_fraction: float
    area_fraction: float
    extent_fraction: float
    contact_ratio: float
    nearest_distance_fraction: float
    is_main: bool = False


@dataclass(frozen=True, slots=True)
class CleanupDecision:
    action: str
    reason: str
    removable: bool


TINY_MAX_FACES = 32
TINY_MAX_FACE_FRACTION = 0.001
TINY_MAX_AREA_FRACTION = 0.0005
TINY_MAX_EXTENT_FRACTION = 0.025
STRICT_PROTECT_FACE_FRACTION = 0.08
STRICT_PROTECT_EXTENT_FRACTION = 0.50
STRICT_SPARSE_ARTIFACT_MAX_FACES = 32
STRICT_SPARSE_ARTIFACT_MAX_AREA_FRACTION = 0.001
STRICT_STRETCHED_ARTIFACT_MAX_FACES = 2500
STRICT_STRETCHED_ARTIFACT_MAX_FACE_FRACTION = 0.005
STRICT_STRETCHED_ARTIFACT_MAX_AREA_FRACTION = 0.01
STRICT_STRETCHED_ARTIFACT_MIN_EXTENT_FRACTION = 0.40


def decide_component(metrics: ComponentMetrics, mode: str) -> CleanupDecision:
    if metrics.is_main:
        return CleanupDecision("KEEP_MAIN", "largest welded surface", False)

    if mode not in {"conservative", "single_subject_strict"}:
        raise ValueError(f"Unsupported cleanup mode: {mode}")

    if mode == "single_subject_strict":
        # A malformed generated surface can contain a handful of triangles stretched across a
        # large distance.  Extent alone would incorrectly protect that line/bulb-shaped debris
        # as a "major part".  Remove only the unambiguously sparse, detached, negligible-area
        # case; meaningful detached parts remain protected below.
        if (
            metrics.contact_ratio <= 0.0
            and metrics.face_count <= STRICT_SPARSE_ARTIFACT_MAX_FACES
            and metrics.area_fraction <= STRICT_SPARSE_ARTIFACT_MAX_AREA_FRACTION
        ):
            return CleanupDecision(
                "REMOVE_SPARSE_DETACHED_ARTIFACT",
                "detached component is too sparse and negligible in area to be a subject part",
                True,
            )
        if (
            metrics.contact_ratio <= 0.0
            and metrics.face_count <= STRICT_STRETCHED_ARTIFACT_MAX_FACES
            and metrics.face_fraction <= STRICT_STRETCHED_ARTIFACT_MAX_FACE_FRACTION
            and metrics.area_fraction <= STRICT_STRETCHED_ARTIFACT_MAX_AREA_FRACTION
            and metrics.extent_fraction >= STRICT_STRETCHED_ARTIFACT_MIN_EXTENT_FRACTION
        ):
            return CleanupDecision(
                "REMOVE_STRETCHED_DETACHED_ARTIFACT",
                "detached component is sparse, low-area, and stretched across the subject bounds",
                True,
            )
        protected = (
            metrics.face_fraction >= STRICT_PROTECT_FACE_FRACTION
            or metrics.extent_fraction >= STRICT_PROTECT_EXTENT_FRACTION
        )
        if protected:
            return CleanupDecision(
                "KEEP_PROTECTED_MAJOR_PART",
                "detached component is too large to delete automatically",
                False,
            )
        return CleanupDecision(
            "REMOVE_DETACHED_SINGLE_SUBJECT",
            "profile requires one fused subject and no separate movable parts",
            True,
        )

    clearly_tiny = (
        metrics.face_count <= TINY_MAX_FACES
        and metrics.face_fraction <= TINY_MAX_FACE_FRACTION
        and metrics.area_fraction <= TINY_MAX_AREA_FRACTION
        and metrics.extent_fraction <= TINY_MAX_EXTENT_FRACTION
    )
    detached = metrics.contact_ratio <= 0.0
    if clearly_tiny and detached:
        return CleanupDecision(
            "REMOVE_TINY_DEBRIS",
            "detached component is tiny by faces, area, and extent",
            True,
        )
    if metrics.contact_ratio > 0.0:
        return CleanupDecision("KEEP_ATTACHED", "surface contact with main component", False)
    return CleanupDecision(
        "KEEP_AMBIGUOUS_DETACHED",
        "not enough evidence for destructive automatic removal",
        False,
    )


def topology_gate(
    *,
    faces_before: int,
    faces_after: int,
    boundary_before: int,
    boundary_after: int,
    mode: str,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if faces_before <= 0 or faces_after <= 0:
        errors.append("mesh contains no faces")
        return False, errors

    removed_fraction = (faces_before - faces_after) / faces_before
    allowed_removed = 0.30 if mode == "single_subject_strict" else 0.05
    if removed_fraction > allowed_removed:
        errors.append(
            f"removed face fraction {removed_fraction:.4f} exceeds {allowed_removed:.4f}"
        )

    allowed_boundary_growth = max(8, int(boundary_before * 0.02))
    if boundary_after > boundary_before + allowed_boundary_growth:
        errors.append(
            f"boundary edges grew from {boundary_before} to {boundary_after}"
        )
    return not errors, errors
