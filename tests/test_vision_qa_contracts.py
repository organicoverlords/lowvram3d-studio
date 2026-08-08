import pytest

from lowvram3d.vision_qa.contracts import ContractError, ModelDecision, VisionQaPacket


def packet_dict():
    return {
        "schema": "vision_qa_packet_v1",
        "packet_id": "castlegrounds-geometry-001",
        "stage": "geometry",
        "artifacts": [
            {"artifact_id": "source", "kind": "source", "path": "source.png"},
            {"artifact_id": "metrics", "kind": "metrics", "path": "metrics.json"},
        ],
        "hard_gates": [{"name": "RAW_REPROJECTION", "status": "PROVEN", "evidence_ids": ["metrics"]}],
    }


def decision_dict():
    return {
        "schema": "vision_qa_decision_v1",
        "model_id": "test-model",
        "packet_id": "castlegrounds-geometry-001",
        "stage": "geometry",
        "verdict": "pass",
        "confidence": 0.95,
        "summary": "No visible defect in supplied evidence.",
        "findings": [],
        "proposed_action": None,
        "uncertainties": [],
    }


def test_packet_round_trip_uses_public_enum_values():
    packet = VisionQaPacket.from_dict(packet_dict())
    assert packet.to_dict()["hard_gates"][0]["status"] == "PROVEN"


def test_unknown_evidence_reference_fails_closed():
    data = packet_dict()
    data["hard_gates"][0]["evidence_ids"] = ["missing"]
    with pytest.raises(ContractError, match="unknown evidence"):
        VisionQaPacket.from_dict(data)


def test_duplicate_artifact_id_fails_closed():
    data = packet_dict()
    data["artifacts"].append(dict(data["artifacts"][0]))
    with pytest.raises(ContractError, match="unique"):
        VisionQaPacket.from_dict(data)


def test_retry_requires_action():
    data = decision_dict()
    data["verdict"] = "retry"
    with pytest.raises(ContractError, match="requires proposed_action"):
        ModelDecision.from_dict(data)


def test_non_retry_forbids_action():
    data = decision_dict()
    data["proposed_action"] = {"name": "rerender_exact_source_camera", "parameters": {}}
    with pytest.raises(ContractError, match="only retry"):
        ModelDecision.from_dict(data)


def test_decision_must_match_packet_identity():
    packet = VisionQaPacket.from_dict(packet_dict())
    data = decision_dict()
    data["packet_id"] = "wrong"
    decision = ModelDecision.from_dict(data)
    with pytest.raises(ContractError, match="does not match"):
        decision.validate_against(packet)
