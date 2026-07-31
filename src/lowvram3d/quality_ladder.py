"""Adaptive high-resolution-master policy for geometry reduction.

The pipeline must never choose a game mesh from triangle count alone.  It preserves a cleaned
high-resolution master, generates a bounded descending ladder, measures every candidate against the
master, and selects the lowest face count that still passes explicit quality gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AssetFamily(str, Enum):
    ORGANIC = "organic"
    HARD_SURFACE = "hard_surface"
    ARCHITECTURAL = "architectural"
    NATURAL = "natural"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class SimilarityThresholds:
    silhouette_iou_min: float
    surface_distance_p95_diag: float
    surface_distance_p99_diag: float
    reverse_distance_p95_diag: float
    normal_deviation_p95_deg: float
    thin_feature_recall_min: float
    meaningful_component_recall_min: float = 1.0
    max_boundary_growth_fraction: float = 0.02
    max_boundary_growth_absolute: int = 8
    max_non_manifold_growth: int = 0


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    name: str
    target_faces: int
    source_faces: int
    ratio: float


@dataclass(slots=True)
class CandidateEvaluation:
    name: str
    face_count: int
    silhouette_iou_min: float
    surface_distance_p95_diag: float
    surface_distance_p99_diag: float
    reverse_distance_p95_diag: float
    normal_deviation_p95_deg: float
    thin_feature_recall: float
    meaningful_component_recall: float
    boundary_edges_before: int
    boundary_edges_after: int
    non_manifold_before: int
    non_manifold_after: int
    valid: bool = False
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "face_count": self.face_count,
            "silhouette_iou_min": self.silhouette_iou_min,
            "surface_distance_p95_diag": self.surface_distance_p95_diag,
            "surface_distance_p99_diag": self.surface_distance_p99_diag,
            "reverse_distance_p95_diag": self.reverse_distance_p95_diag,
            "normal_deviation_p95_deg": self.normal_deviation_p95_deg,
            "thin_feature_recall": self.thin_feature_recall,
            "meaningful_component_recall": self.meaningful_component_recall,
            "boundary_edges_before": self.boundary_edges_before,
            "boundary_edges_after": self.boundary_edges_after,
            "non_manifold_before": self.non_manifold_before,
            "non_manifold_after": self.non_manifold_after,
            "valid": self.valid,
            "errors": list(self.errors),
        }


_ASSET_FAMILIES = {
    "avatar": AssetFamily.ORGANIC,
    "character": AssetFamily.ORGANIC,
    "creature": AssetFamily.ORGANIC,
    "vehicle": AssetFamily.HARD_SURFACE,
    "prop": AssetFamily.MIXED,
    "building": AssetFamily.ARCHITECTURAL,
    "room": AssetFamily.ARCHITECTURAL,
    "scene": AssetFamily.MIXED,
    "level": AssetFamily.MIXED,
    "natural": AssetFamily.NATURAL,
    "vegetation": AssetFamily.NATURAL,
}

# Generated in descending quality order.  For a 1.8M-face Turbo master the hero ladder is roughly
# 1.44M, 1.21M, 900k, 648k, 450k, 288k, 180k.  The selector keeps the last passing candidate,
# rather than forcing the model to 45k before quality is measured.
_RATIOS = {
    "hero": (0.80, 0.67, 0.50, 0.36, 0.25, 0.16, 0.10),
    "gameplay": (0.67, 0.50, 0.36, 0.25, 0.16, 0.10, 0.06),
    "background": (0.36, 0.25, 0.16, 0.10, 0.06, 0.035),
}

_MINIMUM_FACES = {
    AssetFamily.ORGANIC: 30_000,
    AssetFamily.HARD_SURFACE: 20_000,
    AssetFamily.ARCHITECTURAL: 25_000,
    AssetFamily.NATURAL: 20_000,
    AssetFamily.MIXED: 30_000,
    AssetFamily.UNKNOWN: 30_000,
}


def family_for_asset_type(asset_type: str) -> AssetFamily:
    return _ASSET_FAMILIES.get(str(asset_type).lower(), AssetFamily.UNKNOWN)


def thresholds_for(asset_family: AssetFamily, quality: str) -> SimilarityThresholds:
    quality = str(quality).lower()
    if quality == "hero":
        distance_p95, distance_p99, reverse_p95 = 0.0025, 0.0075, 0.0030
        normal_p95, thin_recall = 18.0, 0.97
    elif quality == "background":
        distance_p95, distance_p99, reverse_p95 = 0.0080, 0.0250, 0.0100
        normal_p95, thin_recall = 35.0, 0.85
    else:
        distance_p95, distance_p99, reverse_p95 = 0.0040, 0.0120, 0.0050
        normal_p95, thin_recall = 26.0, 0.93

    silhouette = {
        AssetFamily.ARCHITECTURAL: 0.997,
        AssetFamily.HARD_SURFACE: 0.995,
        AssetFamily.MIXED: 0.992,
        AssetFamily.ORGANIC: 0.990,
        AssetFamily.NATURAL: 0.988,
        AssetFamily.UNKNOWN: 0.990,
    }[asset_family]
    if quality == "background":
        silhouette -= 0.010
    elif quality == "hero":
        silhouette += 0.002

    return SimilarityThresholds(
        silhouette_iou_min=min(silhouette, 0.999),
        surface_distance_p95_diag=distance_p95,
        surface_distance_p99_diag=distance_p99,
        reverse_distance_p95_diag=reverse_p95,
        normal_deviation_p95_deg=normal_p95,
        thin_feature_recall_min=thin_recall,
    )


def candidate_ladder(
    source_faces: int,
    quality: str,
    asset_family: AssetFamily,
    *,
    max_candidates: int = 7,
) -> tuple[CandidatePlan, ...]:
    if source_faces <= 0:
        raise ValueError("source_faces must be positive")
    quality_key = str(quality).lower()
    if quality_key not in _RATIOS:
        raise ValueError(f"unsupported quality: {quality}")
    floor = min(source_faces, _MINIMUM_FACES[asset_family])
    budgets: list[int] = []
    for ratio in _RATIOS[quality_key]:
        budget = max(floor, min(source_faces, round(source_faces * ratio)))
        if budget < source_faces and budget not in budgets:
            budgets.append(budget)
        if len(budgets) >= max_candidates:
            break
    return tuple(
        CandidatePlan(
            name=f"candidate_{index:02d}_{budget}",
            target_faces=budget,
            source_faces=source_faces,
            ratio=budget / source_faces,
        )
        for index, budget in enumerate(budgets, start=1)
    )


def evaluate_candidate(
    evaluation: CandidateEvaluation,
    thresholds: SimilarityThresholds,
) -> CandidateEvaluation:
    errors: list[str] = []
    if evaluation.face_count <= 0:
        errors.append("candidate contains no faces")
    if evaluation.silhouette_iou_min < thresholds.silhouette_iou_min:
        errors.append(
            f"silhouette IoU {evaluation.silhouette_iou_min:.6f} below "
            f"{thresholds.silhouette_iou_min:.6f}"
        )
    if evaluation.surface_distance_p95_diag > thresholds.surface_distance_p95_diag:
        errors.append("source-to-candidate p95 surface distance exceeded")
    if evaluation.surface_distance_p99_diag > thresholds.surface_distance_p99_diag:
        errors.append("source-to-candidate p99 surface distance exceeded")
    if evaluation.reverse_distance_p95_diag > thresholds.reverse_distance_p95_diag:
        errors.append("candidate-to-source p95 surface distance exceeded")
    if evaluation.normal_deviation_p95_deg > thresholds.normal_deviation_p95_deg:
        errors.append("p95 normal deviation exceeded")
    if evaluation.thin_feature_recall < thresholds.thin_feature_recall_min:
        errors.append("thin-feature recall below threshold")
    if evaluation.meaningful_component_recall < thresholds.meaningful_component_recall_min:
        errors.append("one or more meaningful components disappeared")

    boundary_allowance = max(
        thresholds.max_boundary_growth_absolute,
        round(evaluation.boundary_edges_before * thresholds.max_boundary_growth_fraction),
    )
    if evaluation.boundary_edges_after > evaluation.boundary_edges_before + boundary_allowance:
        errors.append("boundary-edge count regressed")
    if (
        evaluation.non_manifold_after - evaluation.non_manifold_before
        > thresholds.max_non_manifold_growth
    ):
        errors.append("non-manifold-edge count regressed")

    evaluation.errors = errors
    evaluation.valid = not errors
    return evaluation


def select_lowest_passing(
    evaluations: list[CandidateEvaluation],
) -> CandidateEvaluation | None:
    """Choose the smallest valid candidate; master preservation is handled by the caller."""
    valid = [candidate for candidate in evaluations if candidate.valid]
    if not valid:
        return None
    return min(valid, key=lambda candidate: candidate.face_count)


def should_stop_descending(
    evaluations_in_generation_order: list[CandidateEvaluation],
    *,
    consecutive_failures: int = 2,
) -> bool:
    """Save runtime after quality has clearly crossed the failure boundary.

    Candidates are generated high-to-low.  Once at least one candidate passed, two consecutive
    failures stop the ladder.  This tolerates one noisy decimation result without exploring every
    lower budget.
    """
    if consecutive_failures < 1:
        raise ValueError("consecutive_failures must be positive")
    if not any(item.valid for item in evaluations_in_generation_order):
        return False
    tail = evaluations_in_generation_order[-consecutive_failures:]
    return len(tail) == consecutive_failures and all(not item.valid for item in tail)
