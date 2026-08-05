import pytest

from lowvram3d.visual_qa_policy import (
    DECISION_ACCEPT,
    DECISION_REJECT,
    DECISION_UNCERTAIN,
    MODE_AUTO,
    MODE_OFF,
    MODE_REQUIRED,
    PROMOTION_CONFIDENCE,
    REQUIRED_CHECK_KEYS,
    STATUS_PASSED,
    STATUS_REJECTED,
    STATUS_UNAVAILABLE,
    STATUS_UNCERTAIN,
    VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH,
    VISUAL_GENERIC_REPAIR,
    VISUAL_INSUFFICIENT_EVIDENCE,
    VISUAL_LOW_CONFIDENCE,
    VISUAL_MALFORMED_OUTPUT,
    VISUAL_TIMEOUT,
    ManifestError,
    build_result,
    gate_outcome,
    parse_choice,
    unavailable_result,
    validate_manifest,
)

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


def manifest(**overrides):
    values = {
        "source_crop": "source.png",
        "before_crop": "before.png",
        "candidate_crop": "candidate.png",
        "feature_name": "staff ring through-hole",
        "expected_description": "small opening inside the original recess",
        "constraints": ["preserve the inner lip"],
    }
    values.update(overrides)
    return values


# ---------------------------------------------------------------- manifest


def test_valid_manifest_is_accepted():
    assert validate_manifest(manifest())["feature_name"] == "staff ring through-hole"


@pytest.mark.parametrize("key", ["source_crop", "before_crop", "candidate_crop", "feature_name"])
def test_missing_manifest_key_is_rejected(key):
    broken = manifest()
    del broken[key]
    with pytest.raises(ManifestError):
        validate_manifest(broken)


@pytest.mark.parametrize("key", ["source_crop", "before_crop", "candidate_crop", "feature_name"])
def test_blank_manifest_value_is_rejected(key):
    with pytest.raises(ManifestError):
        validate_manifest(manifest(**{key: "   "}))


def test_constraints_must_be_a_list():
    with pytest.raises(ManifestError):
        validate_manifest(manifest(constraints="preserve the lip"))


def test_non_object_manifest_is_rejected():
    with pytest.raises(ManifestError):
        validate_manifest(["not", "an", "object"])


# ---------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A", "A"),
        ("ANSWER: B", "B"),
        ("answer: c", "C"),
        ("(B)", "B"),
        ("B - the hole is far too large", "B"),
        ("The answer is C because the crops are dark", "C"),
        ("ANSWER:A", "A"),
    ],
)
def test_parse_choice_extracts_letter(text, expected):
    assert parse_choice(text) == expected


@pytest.mark.parametrize("text", ["", "   ", None, "no verdict here", "DEFG"])
def test_parse_choice_returns_none_when_malformed(text):
    assert parse_choice(text) is None


def test_parse_choice_prefers_labelled_answer_over_stray_letter():
    assert parse_choice("Considering A and B, ANSWER: B") == "B"


# ---------------------------------------------------------------- results


def test_high_confidence_a_passes_and_may_promote():
    result = build_result("A", 0.93, MODEL)
    assert result.status == STATUS_PASSED
    assert result.decision == DECISION_ACCEPT
    assert result.checks["feature_matches_source"] is True
    assert result.checks["candidate_looks_generic_or_oversized"] is False
    assert gate_outcome(result, MODE_AUTO)["promote"] is True


def test_low_confidence_a_is_not_promotable():
    result = build_result("A", PROMOTION_CONFIDENCE - 0.01, MODEL)
    assert result.status == STATUS_UNCERTAIN
    assert result.decision == DECISION_UNCERTAIN
    assert VISUAL_LOW_CONFIDENCE in result.reason_codes
    assert gate_outcome(result, MODE_AUTO)["promote"] is False


def test_confidence_exactly_at_threshold_promotes():
    result = build_result("A", PROMOTION_CONFIDENCE, MODEL)
    assert result.status == STATUS_PASSED
    assert gate_outcome(result, MODE_AUTO)["promote"] is True


def test_b_rejects_with_required_reason_code():
    result = build_result("B", 0.91, MODEL, raw_response="B")
    assert result.status == STATUS_REJECTED
    assert result.decision == DECISION_REJECT
    assert VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH in result.reason_codes
    assert result.checks["candidate_looks_generic_or_oversized"] is True


def test_b_mentioning_generic_adds_generic_reason_code():
    result = build_result("B", 0.9, MODEL, raw_response="B - it looks like a generic donut")
    assert VISUAL_GENERIC_REPAIR in result.reason_codes


def test_b_wording_can_flag_collateral_damage():
    result = build_result("B", 0.9, MODEL, raw_response="B - the inner lip is missing")
    assert result.checks["collateral_damage_visible"] is True


def test_b_without_damage_wording_reports_no_collateral_damage():
    result = build_result("B", 0.9, MODEL, raw_response="B")
    assert result.checks["collateral_damage_visible"] is False


def test_c_is_uncertain():
    result = build_result("C", 0.7, MODEL)
    assert result.status == STATUS_UNCERTAIN
    assert VISUAL_INSUFFICIENT_EVIDENCE in result.reason_codes


def test_malformed_output_is_uncertain_not_accepted():
    result = build_result(None, 0.99, MODEL, raw_response="banana")
    assert result.status == STATUS_UNCERTAIN
    assert result.decision == DECISION_UNCERTAIN
    assert VISUAL_MALFORMED_OUTPUT in result.reason_codes
    assert result.confidence == 0.0


def test_contract_shape_is_stable():
    contract = build_result("A", 0.9, MODEL, device="cpu",
                            load_seconds=1.2, inference_seconds=0.3).to_contract()
    assert set(contract) == {
        "status", "decision", "confidence", "checks", "reason_codes",
        "model", "device", "load_seconds", "inference_seconds",
    }
    assert set(contract["checks"]) == set(REQUIRED_CHECK_KEYS)
    assert contract["load_seconds"] == 1.2
    assert contract["device"] == "cpu"


def test_device_defaults_to_unknown():
    assert build_result("A", 0.9, MODEL).to_contract()["device"] == "unknown"
    assert unavailable_result(MODEL).to_contract()["device"] == "unknown"


def test_confidence_is_clamped():
    assert build_result("A", 4.2, MODEL).confidence == 1.0
    assert build_result("A", -3.0, MODEL).confidence == 0.0


# ---------------------------------------------------------------- gating


def test_visual_acceptance_never_overrides_failed_hard_gates():
    result = build_result("A", 0.99, MODEL)
    outcome = gate_outcome(result, MODE_AUTO, hard_gates_passed=False)
    assert outcome["promote"] is False
    assert outcome["blocking"] is True


def test_auto_mode_continues_when_model_unavailable():
    outcome = gate_outcome(unavailable_result(MODEL), MODE_AUTO)
    assert outcome["promote"] is False
    assert outcome["blocking"] is False


def test_auto_mode_continues_on_timeout():
    result = unavailable_result(MODEL, VISUAL_TIMEOUT)
    assert result.status == STATUS_UNAVAILABLE
    assert gate_outcome(result, MODE_AUTO)["blocking"] is False


def test_required_mode_blocks_when_model_unavailable():
    assert gate_outcome(unavailable_result(MODEL), MODE_REQUIRED)["blocking"] is True


def test_required_mode_blocks_on_uncertain():
    assert gate_outcome(build_result("C", 0.5, MODEL), MODE_REQUIRED)["blocking"] is True


def test_rejection_blocks_in_every_active_mode():
    result = build_result("B", 0.9, MODEL)
    for mode in (MODE_AUTO, MODE_REQUIRED):
        assert gate_outcome(result, mode)["blocking"] is True


def test_off_mode_skips_the_judge_entirely():
    outcome = gate_outcome(build_result("B", 0.99, MODEL), MODE_OFF)
    assert outcome["promote"] is True
    assert outcome["blocking"] is False


def test_off_mode_still_respects_hard_gates():
    assert gate_outcome(None, MODE_OFF, hard_gates_passed=False)["promote"] is False


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        gate_outcome(build_result("A", 0.9, MODEL), "sometimes")


# ------------------------------------------------- staff-hole regression shape


def test_rejected_staff_hole_fixture_shape():
    """The oversized staff hole must reject with one of the two required reason codes."""
    result = build_result("B", 0.88, MODEL, raw_response="B - the opening is too large and generic")
    contract = result.to_contract()
    assert contract["status"] == "rejected"
    assert contract["decision"] == "reject"
    assert {VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH, VISUAL_GENERIC_REPAIR} & set(
        contract["reason_codes"]
    )
    assert gate_outcome(result, MODE_AUTO)["promote"] is False


def test_no_change_control_reports_no_collateral_damage():
    """Identical before/candidate crops: accept or uncertain, but never collateral damage."""
    for choice in ("A", "C"):
        result = build_result(choice, 0.85, MODEL, raw_response=choice)
        assert result.checks["collateral_damage_visible"] is False
        assert result.decision in (DECISION_ACCEPT, DECISION_UNCERTAIN)
