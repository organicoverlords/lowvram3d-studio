"""Deterministic prompt construction for local vision supervisors."""
from __future__ import annotations

import json
from typing import Iterable

from .contracts import EvidenceArtifact, VisionQaPacket
from .policy import DEFAULT_ACTIONS


SYSTEM_PROMPT = """You are a visual QA classifier inside a fail-closed 3D asset pipeline.
You are not the controller and cannot run arbitrary tools. Hard gates and user visual rejection
are authoritative. Use only supplied evidence. Never claim that an unseen image, hidden file,
metric, geometry property, or engine import is proven. Return exactly one JSON object matching
vision_qa_decision_v1. A retry may choose only one action from allowed_actions. Use quarantine or
user_review when evidence is insufficient. Do not put prose outside JSON."""


def build_user_prompt(packet: VisionQaPacket, *, max_artifacts: int = 16) -> str:
    artifacts = list(packet.artifacts)[:max_artifacts]
    compact = {
        "packet_id": packet.packet_id,
        "stage": packet.stage,
        "hard_gates": [
            {"name": gate.name, "status": gate.status.value, "detail": gate.detail, "evidence_ids": list(gate.evidence_ids)}
            for gate in packet.hard_gates
        ],
        "metrics": [
            {"name": metric.name, "value": metric.value, "status": metric.status.value, "threshold": metric.threshold, "evidence_ids": list(metric.evidence_ids)}
            for metric in packet.metrics
        ],
        "evidence": [_artifact_prompt_item(item) for item in artifacts],
        "allowed_actions": sorted(DEFAULT_ACTIONS.get(packet.stage, frozenset())),
        "retry_state": dict(packet.retry_state.attempts_by_action),
        "manual_rejection": packet.manual_rejection,
        "context": dict(packet.context),
    }
    response_contract = {
        "schema": "vision_qa_decision_v1",
        "model_id": "<runtime model id>",
        "packet_id": packet.packet_id,
        "stage": packet.stage,
        "verdict": "pass|retry|quarantine|user_review|request_second_opinion",
        "confidence": "0.0..1.0",
        "summary": "brief evidence-grounded conclusion",
        "findings": [{
            "code": "UPPER_SNAKE_CASE",
            "severity": "info|low|medium|high|critical",
            "confidence": "0.0..1.0",
            "evidence_ids": ["artifact id"],
            "detail": "visible observation only",
        }],
        "proposed_action": "null unless verdict=retry; otherwise {name, parameters}",
        "uncertainties": ["what cannot be established"],
    }
    return "Assess this evidence packet.\nPACKET:\n" + json.dumps(compact, indent=2, sort_keys=True) + (
        "\nRESPONSE CONTRACT:\n" + json.dumps(response_contract, indent=2, sort_keys=True)
    )


def select_image_artifacts(
    artifacts: Iterable[EvidenceArtifact],
    *,
    max_images: int = 8,
) -> tuple[EvidenceArtifact, ...]:
    preferred = {"source", "unlit", "albedo", "silhouette", "wireframe", "depth", "normal", "beauty", "mask"}
    selected = [item for item in artifacts if item.kind.value in preferred]
    selected.sort(key=lambda item: (item.kind.value != "source", not item.required, item.artifact_id))
    return tuple(selected[:max_images])


def _artifact_prompt_item(artifact: EvidenceArtifact) -> dict:
    return {
        "artifact_id": artifact.artifact_id,
        "kind": artifact.kind.value,
        "view": artifact.view,
        "sha256": artifact.sha256,
        "description": artifact.description,
        "required": artifact.required,
        "metadata": dict(artifact.metadata),
    }
