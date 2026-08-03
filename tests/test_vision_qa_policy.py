from lowvram3d.vision_qa.contracts import ModelDecision, VisionQaPacket
from lowvram3d.vision_qa.policy import OutcomeStatus, evaluate_decisions


def packet(status="PROVEN", attempts=None, manual_rejection=None):
    return VisionQaPacket.from_dict({
        "schema": "vision_qa_packet_v1",
        "packet_id": "p1",
        "stage": "geometry",
        "artifacts": [{"artifact_id": "render", "kind": "unlit", "path": "render.png"}],
        "hard_gates": [{"name": "SOURCE_CAMERA", "status": status, "evidence_ids": ["render"]}],
        "retry_state": {"attempts_by_action": attempts or {}},
        "manual_rejection": manual_rejection,
    })


def decision(verdict="pass", confidence=0.95, action=None, model="primary"):
    return ModelDecision.from_dict({
        "schema": "vision_qa_decision_v1",
        "model_id": model,
        "packet_id": "p1",
        "stage": "geometry",
        "verdict": verdict,
        "confidence": confidence,
        "summary": verdict,
        "findings": [],
        "proposed_action": action,
        "uncertainties": [],
    })


def test_model_cannot_pass_rejected_hard_gate():
    outcome = evaluate_decisions(packet("REJECTED"), decision("pass"))
    assert outcome.status is OutcomeStatus.REJECTED


def test_bounded_repair_can_be_selected_for_rejected_gate():
    action = {"name": "repair_mesh_coverage_from_saved_arrays", "parameters": {"max_gap_pixels": 2}}
    outcome = evaluate_decisions(packet("REJECTED"), decision("retry", 0.92, action))
    assert outcome.status is OutcomeStatus.RETRY
    assert outcome.selected_action == action["name"]


def test_arbitrary_action_is_invalid():
    action = {"name": "run_shell_command", "parameters": {"command": "rm -rf"}}
    outcome = evaluate_decisions(packet("REJECTED"), decision("retry", 0.99, action))
    assert outcome.status is OutcomeStatus.INVALID_DECISION


def test_retry_budget_exhaustion_requires_review():
    name = "repair_mesh_coverage_from_saved_arrays"
    action = {"name": name, "parameters": {}}
    outcome = evaluate_decisions(packet("REJECTED", {name: 2}), decision("retry", 0.99, action))
    assert outcome.status is OutcomeStatus.USER_REVIEW


def test_medium_confidence_requests_second_opinion():
    outcome = evaluate_decisions(packet(), decision("pass", 0.80))
    assert outcome.status is OutcomeStatus.SECOND_OPINION


def test_disagreement_requires_user_review():
    primary = decision("pass", 0.92, model="primary")
    secondary = decision("quarantine", 0.91, model="secondary")
    outcome = evaluate_decisions(packet(), primary, secondary)
    assert outcome.status is OutcomeStatus.USER_REVIEW


def test_manual_rejection_overrides_models_and_metrics():
    outcome = evaluate_decisions(packet(manual_rejection="user says face is missing"), decision("pass", 0.99))
    assert outcome.status is OutcomeStatus.REJECTED
    assert "authoritative" in outcome.reason
