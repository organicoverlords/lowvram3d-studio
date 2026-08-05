"""Deterministic thresholds for the hybrid visual gate.

The tiny VLM could not tell an oversized generic repair from a faithful one, so the authority
sits here instead: measurable image evidence with fixed thresholds. This module is pure, so the
same decision can be unit-tested without loading an image.

No feature is hardcoded. The caller supplies the region of interest and, optionally, feature
masks; the thresholds below are generic ratios, not staff pixel coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass

VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH = "VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH"
VISUAL_REPAIR_EXCEEDS_SOURCE_BOUNDARY = "VISUAL_REPAIR_EXCEEDS_SOURCE_BOUNDARY"
VISUAL_LOCAL_STRUCTURE_LOST = "VISUAL_LOCAL_STRUCTURE_LOST"
VISUAL_COLLATERAL_CHANGE = "VISUAL_COLLATERAL_CHANGE"
VISUAL_NO_CHANGE_DETECTED = "VISUAL_NO_CHANGE_DETECTED"
VISUAL_ALIGNMENT_UNRELIABLE = "VISUAL_ALIGNMENT_UNRELIABLE"
VISUAL_DELTA_OK = "VISUAL_DELTA_OK"


@dataclass(frozen=True)
class DeltaThresholds:
    """Generic ratios. Tighten per-feature via the manifest, never by editing this file."""

    # Fraction of pixels outside the repair ROI that may differ at all.
    max_outside_region_change: float = 0.02
    # Candidate silhouette area relative to the baseline, inside the crop.
    min_silhouette_area_ratio: float = 0.90
    max_silhouette_area_ratio: float = 1.10
    # Candidate feature opening relative to the SOURCE feature opening.
    min_feature_scale_ratio: float = 0.75
    max_feature_scale_ratio: float = 1.15
    # Structural agreement between source and candidate silhouette edges.
    min_edge_similarity: float = 0.25
    # Below this, the crops are not the same subject at all.
    min_alignment_confidence: float = 0.30
    # Both edge metrics are REPORTED but not gated by default. The source crop is concept art and
    # the candidate is a clay render at a different framing, scale and rotation; without an actual
    # registration step their edge correlation measures ~0 even for a correct repair, so gating on
    # it rejects everything. Enable only once crops are registered.
    gate_edge_similarity: bool = False
    gate_alignment: bool = False
    # A repair that changes nothing has not repaired anything.
    min_before_candidate_distance: float = 0.001


REQUIRED_METRICS = (
    "outside_region_change",
    "silhouette_area_ratio",
    "feature_scale_ratio",
    "edge_similarity",
    "source_candidate_distance",
    "before_candidate_distance",
)


def decide(metrics: dict, thresholds: DeltaThresholds | None = None,
           require_change: bool = True,
           alignment_confidence: float | None = None) -> dict:
    """Grade a candidate from measured image metrics.

    `require_change` is False for a no-change control, where an unchanged candidate is the
    expected, correct outcome rather than a failed repair.
    """
    thresholds = thresholds or DeltaThresholds()
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise ValueError(f"missing metrics: {', '.join(sorted(missing))}")

    codes: list[str] = []

    if (
        thresholds.gate_alignment
        and alignment_confidence is not None
        and alignment_confidence < thresholds.min_alignment_confidence
    ):
        codes.append(VISUAL_ALIGNMENT_UNRELIABLE)

    if metrics["outside_region_change"] > thresholds.max_outside_region_change:
        codes.append(VISUAL_COLLATERAL_CHANGE)

    scale = metrics["feature_scale_ratio"]
    if scale > thresholds.max_feature_scale_ratio:
        codes.append(VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH)
        codes.append(VISUAL_REPAIR_EXCEEDS_SOURCE_BOUNDARY)
    elif scale < thresholds.min_feature_scale_ratio:
        codes.append(VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH)

    ratio = metrics["silhouette_area_ratio"]
    if ratio < thresholds.min_silhouette_area_ratio or ratio > thresholds.max_silhouette_area_ratio:
        codes.append(VISUAL_LOCAL_STRUCTURE_LOST)

    if thresholds.gate_edge_similarity and metrics["edge_similarity"] < thresholds.min_edge_similarity:
        codes.append(VISUAL_LOCAL_STRUCTURE_LOST)

    if require_change and metrics["before_candidate_distance"] < thresholds.min_before_candidate_distance:
        codes.append(VISUAL_NO_CHANGE_DETECTED)

    codes = list(dict.fromkeys(codes))
    passed = not codes
    return {
        "passed": passed,
        "metrics": {key: round(float(metrics[key]), 6) for key in REQUIRED_METRICS},
        "reason_codes": codes or [VISUAL_DELTA_OK],
    }


def combine(deterministic: dict, visual_advisory: dict | None,
            hard_gates_passed: bool = True) -> dict:
    """Final promotion decision.

    Promotion needs the hard gates AND the deterministic visual gate. The tiny VLM may add
    corroboration or veto, but its uncertainty carries no authority and it is never the only
    positive evidence.
    """
    reasons: list[str] = []
    if not hard_gates_passed:
        reasons.append("hard gates failed")
    if not deterministic.get("passed"):
        reasons.extend(deterministic.get("reason_codes", []))

    advisory_rejected = bool(
        visual_advisory and visual_advisory.get("decision") == "reject"
    )
    if advisory_rejected:
        reasons.extend(visual_advisory.get("reason_codes", []))

    promote = hard_gates_passed and bool(deterministic.get("passed")) and not advisory_rejected
    return {
        "promote": promote,
        "preserve_baseline": not promote,
        "reason_codes": list(dict.fromkeys(reasons)),
        "advisory_agreement": bool(
            visual_advisory and visual_advisory.get("decision") == "accept"
        ),
        "advisory_authority": "none",
    }
