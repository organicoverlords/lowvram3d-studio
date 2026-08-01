import pytest

from lowvram3d.visual_delta_policy import (
    VISUAL_ALIGNMENT_UNRELIABLE,
    VISUAL_COLLATERAL_CHANGE,
    VISUAL_DELTA_OK,
    VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH,
    VISUAL_LOCAL_STRUCTURE_LOST,
    VISUAL_NO_CHANGE_DETECTED,
    VISUAL_REPAIR_EXCEEDS_SOURCE_BOUNDARY,
    DeltaThresholds,
    combine,
    decide,
)


def metrics(**overrides):
    values = {
        "outside_region_change": 0.0,
        "silhouette_area_ratio": 1.0,
        "feature_scale_ratio": 1.0,
        "edge_similarity": 0.5,
        "source_candidate_distance": 0.2,
        "before_candidate_distance": 0.05,
    }
    values.update(overrides)
    return values


def test_clean_candidate_passes():
    verdict = decide(metrics())
    assert verdict["passed"] is True
    assert verdict["reason_codes"] == [VISUAL_DELTA_OK]


def test_missing_metric_is_rejected():
    broken = metrics()
    del broken["feature_scale_ratio"]
    with pytest.raises(ValueError):
        decide(broken)


# ------------------------------------------------------------------ scale


def test_oversized_feature_is_rejected_with_both_codes():
    """The measured staff regression: candidate opening 1.35x the source proportion."""
    verdict = decide(metrics(feature_scale_ratio=1.35053))
    assert verdict["passed"] is False
    assert VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH in verdict["reason_codes"]
    assert VISUAL_REPAIR_EXCEEDS_SOURCE_BOUNDARY in verdict["reason_codes"]


def test_undersized_feature_is_a_mismatch_but_not_a_boundary_violation():
    verdict = decide(metrics(feature_scale_ratio=0.62))
    assert VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH in verdict["reason_codes"]
    assert VISUAL_REPAIR_EXCEEDS_SOURCE_BOUNDARY not in verdict["reason_codes"]


def test_measured_good_repair_ratio_passes():
    assert decide(metrics(feature_scale_ratio=0.980573))["passed"] is True


@pytest.mark.parametrize("ratio", [1.15, 0.75])
def test_scale_bounds_are_inclusive(ratio):
    assert decide(metrics(feature_scale_ratio=ratio))["passed"] is True


# ------------------------------------------------------------------ collateral


def test_outside_region_change_is_collateral_damage():
    verdict = decide(metrics(outside_region_change=0.102396))
    assert verdict["passed"] is False
    assert VISUAL_COLLATERAL_CHANGE in verdict["reason_codes"]


def test_identical_candidate_is_not_reported_as_collateral_damage():
    """The no-change control must never be accused of damaging anything."""
    verdict = decide(
        metrics(outside_region_change=0.0, before_candidate_distance=0.0),
        require_change=False,
    )
    assert VISUAL_COLLATERAL_CHANGE not in verdict["reason_codes"]


def test_collateral_is_independent_of_a_correct_scale():
    """The collateral fixture had a good ratio (0.997) and must still fail."""
    verdict = decide(metrics(feature_scale_ratio=0.997242, outside_region_change=0.10))
    assert verdict["passed"] is False
    assert verdict["reason_codes"] == [VISUAL_COLLATERAL_CHANGE]


# ------------------------------------------------------------------ change / structure


def test_no_change_flags_when_a_repair_was_required():
    verdict = decide(metrics(before_candidate_distance=0.0), require_change=True)
    assert VISUAL_NO_CHANGE_DETECTED in verdict["reason_codes"]


def test_no_change_is_allowed_when_not_required():
    verdict = decide(metrics(before_candidate_distance=0.0), require_change=False)
    assert VISUAL_NO_CHANGE_DETECTED not in verdict["reason_codes"]


def test_silhouette_collapse_is_structure_loss():
    assert VISUAL_LOCAL_STRUCTURE_LOST in decide(
        metrics(silhouette_area_ratio=0.5)
    )["reason_codes"]


def test_edge_similarity_is_not_gated_by_default():
    """Unregistered concept-art vs clay-render edges correlate near zero even when correct."""
    assert decide(metrics(edge_similarity=-0.02))["passed"] is True


def test_edge_similarity_gates_when_explicitly_enabled():
    thresholds = DeltaThresholds(gate_edge_similarity=True)
    assert decide(metrics(edge_similarity=-0.02), thresholds)["passed"] is False


def test_alignment_is_not_gated_by_default():
    assert decide(metrics(), alignment_confidence=-0.002)["passed"] is True


def test_alignment_gates_when_explicitly_enabled():
    thresholds = DeltaThresholds(gate_alignment=True)
    verdict = decide(metrics(), thresholds, alignment_confidence=-0.002)
    assert VISUAL_ALIGNMENT_UNRELIABLE in verdict["reason_codes"]


def test_reason_codes_are_deduplicated():
    verdict = decide(metrics(feature_scale_ratio=0.2, silhouette_area_ratio=0.2))
    assert len(verdict["reason_codes"]) == len(set(verdict["reason_codes"]))


# ------------------------------------------------------------------ promotion


def test_promotion_requires_deterministic_pass():
    assert combine(decide(metrics()), None)["promote"] is True
    assert combine(decide(metrics(feature_scale_ratio=1.4)), None)["promote"] is False


def test_hard_gates_veto_promotion():
    outcome = combine(decide(metrics()), None, hard_gates_passed=False)
    assert outcome["promote"] is False
    assert outcome["preserve_baseline"] is True


def test_vlm_acceptance_alone_cannot_promote_a_failing_candidate():
    """The tiny model has no authority: it cannot rescue a deterministic failure."""
    outcome = combine(
        decide(metrics(feature_scale_ratio=1.4)),
        {"decision": "accept", "confidence": 0.99},
    )
    assert outcome["promote"] is False
    assert outcome["advisory_authority"] == "none"


def test_vlm_rejection_can_veto_a_passing_candidate():
    outcome = combine(
        decide(metrics()),
        {"decision": "reject", "reason_codes": ["VISUAL_GENERIC_REPAIR"]},
    )
    assert outcome["promote"] is False
    assert "VISUAL_GENERIC_REPAIR" in outcome["reason_codes"]


def test_vlm_uncertainty_has_no_authority():
    outcome = combine(decide(metrics()), {"decision": "uncertain", "confidence": 0.8})
    assert outcome["promote"] is True
    assert outcome["advisory_agreement"] is False


def test_failure_always_preserves_the_baseline():
    outcome = combine(decide(metrics(outside_region_change=0.9)), None)
    assert outcome["preserve_baseline"] is True
