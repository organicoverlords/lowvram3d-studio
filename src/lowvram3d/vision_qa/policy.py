"""Fail-closed controller policy for visual QA decisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from .contracts import (
    ContractError,
    GateStatus,
    ModelDecision,
    Verdict,
    VisionQaPacket,
)


class OutcomeStatus(str, Enum):
    PASS = "PASS"
    RETRY = "RETRY"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    QUARANTINED = "QUARANTINED"
    USER_REVIEW = "USER_REVIEW"
    SECOND_OPINION = "SECOND_OPINION_REQUIRED"
    INVALID_DECISION = "INVALID_DECISION"


DEFAULT_ACTIONS: dict[str, frozenset[str]] = {
    "geometry": frozenset({
        "repair_mesh_coverage_from_saved_arrays",
        "build_bounded_edge_threshold_candidate",
        "build_bounded_winding_candidate",
        "rerender_exact_source_camera",
    }),
    "uv": frozenset({"rerun_uv_with_locked_geometry", "repack_uv_without_geometry_change"}),
    "texture": frozenset({
        "rerun_projection_for_region",
        "increase_region_projection_priority",
        "rerender_texture_evidence",
    }),
    "rig": frozenset({"rerun_weights_for_region", "rerender_pose_evidence"}),
    "export": frozenset({"fresh_reimport", "rerender_engine_evidence"}),
}

DEFAULT_RETRY_LIMITS: dict[str, int] = {
    "repair_mesh_coverage_from_saved_arrays": 2,
    "build_bounded_edge_threshold_candidate": 3,
    "build_bounded_winding_candidate": 2,
    "rerender_exact_source_camera": 2,
    "rerun_uv_with_locked_geometry": 2,
    "repack_uv_without_geometry_change": 2,
    "rerun_projection_for_region": 3,
    "increase_region_projection_priority": 2,
    "rerender_texture_evidence": 2,
    "rerun_weights_for_region": 2,
    "rerender_pose_evidence": 2,
    "fresh_reimport": 2,
    "rerender_engine_evidence": 2,
}


@dataclass(frozen=True)
class PolicyConfig:
    primary_accept_confidence: float = 0.90
    second_opinion_floor: float = 0.65
    retry_confidence_floor: float = 0.70
    allowed_actions: Mapping[str, frozenset[str]] = field(default_factory=lambda: DEFAULT_ACTIONS)
    retry_limits: Mapping[str, int] = field(default_factory=lambda: DEFAULT_RETRY_LIMITS)


@dataclass(frozen=True)
class ControlOutcome:
    status: OutcomeStatus
    reason: str
    selected_action: str | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    model_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


def evaluate_decisions(
    packet: VisionQaPacket,
    primary: ModelDecision,
    secondary: ModelDecision | None = None,
    *,
    config: PolicyConfig | None = None,
) -> ControlOutcome:
    config = config or PolicyConfig()
    try:
        primary.validate_against(packet)
        if secondary is not None:
            secondary.validate_against(packet)
    except ContractError as exc:
        return ControlOutcome(OutcomeStatus.INVALID_DECISION, str(exc))

    if packet.manual_rejection:
        return ControlOutcome(
            OutcomeStatus.REJECTED,
            f"manual visual rejection is authoritative: {packet.manual_rejection}",
            model_ids=_model_ids(primary, secondary),
        )

    gate_statuses = {gate.status for gate in packet.hard_gates}
    if GateStatus.REJECTED in gate_statuses:
        return _route_rejected_gate(packet, primary, config, secondary)
    if GateStatus.BLOCKED in gate_statuses:
        return ControlOutcome(
            OutcomeStatus.BLOCKED,
            "one or more hard gates are blocked; a model opinion cannot promote the stage",
            model_ids=_model_ids(primary, secondary),
        )
    if GateStatus.NOT_PROVEN in gate_statuses:
        return ControlOutcome(
            OutcomeStatus.QUARANTINED,
            "one or more hard gates are not proven",
            model_ids=_model_ids(primary, secondary),
        )

    if primary.confidence < config.second_opinion_floor:
        return ControlOutcome(
            OutcomeStatus.QUARANTINED,
            "primary model confidence is below the second-opinion floor",
            model_ids=(primary.model_id,),
        )
    if primary.confidence < config.primary_accept_confidence and secondary is None:
        return ControlOutcome(
            OutcomeStatus.SECOND_OPINION,
            "primary decision requires an independent second opinion",
            model_ids=(primary.model_id,),
        )
    if secondary is not None and _decisions_conflict(primary, secondary):
        return ControlOutcome(
            OutcomeStatus.USER_REVIEW,
            "independent visual judges disagree",
            model_ids=_model_ids(primary, secondary),
        )

    selected = _higher_confidence(primary, secondary)
    if selected.verdict is Verdict.PASS:
        return ControlOutcome(
            OutcomeStatus.PASS,
            "all hard gates are proven and visual decision passed",
            model_ids=_model_ids(primary, secondary),
        )
    if selected.verdict is Verdict.RETRY:
        return _evaluate_retry(packet, selected, config, secondary)
    if selected.verdict is Verdict.USER_REVIEW:
        return ControlOutcome(OutcomeStatus.USER_REVIEW, selected.summary, model_ids=_model_ids(primary, secondary))
    if selected.verdict is Verdict.SECOND_OPINION:
        return ControlOutcome(OutcomeStatus.SECOND_OPINION, selected.summary, model_ids=_model_ids(primary, secondary))
    return ControlOutcome(OutcomeStatus.QUARANTINED, selected.summary, model_ids=_model_ids(primary, secondary))


def _route_rejected_gate(
    packet: VisionQaPacket,
    decision: ModelDecision,
    config: PolicyConfig,
    secondary: ModelDecision | None,
) -> ControlOutcome:
    if decision.verdict is Verdict.RETRY:
        return _evaluate_retry(packet, decision, config, secondary, hard_gate_rejected=True)
    return ControlOutcome(
        OutcomeStatus.REJECTED,
        "a hard gate is rejected and the model did not provide a permitted bounded repair",
        model_ids=_model_ids(decision, secondary),
    )


def _evaluate_retry(
    packet: VisionQaPacket,
    decision: ModelDecision,
    config: PolicyConfig,
    secondary: ModelDecision | None,
    *,
    hard_gate_rejected: bool = False,
) -> ControlOutcome:
    if decision.confidence < config.retry_confidence_floor:
        return ControlOutcome(
            OutcomeStatus.USER_REVIEW,
            "retry recommendation confidence is below policy threshold",
            model_ids=_model_ids(decision, secondary),
        )
    action = decision.proposed_action
    if action is None:
        return ControlOutcome(OutcomeStatus.INVALID_DECISION, "retry is missing proposed_action")
    allowed = config.allowed_actions.get(packet.stage, frozenset())
    if action.name not in allowed:
        return ControlOutcome(
            OutcomeStatus.INVALID_DECISION,
            f"action {action.name!r} is not allowed for stage {packet.stage!r}",
            model_ids=_model_ids(decision, secondary),
        )
    limit = config.retry_limits.get(action.name, 0)
    used = packet.retry_state.attempts_by_action.get(action.name, 0)
    if used >= limit:
        return ControlOutcome(
            OutcomeStatus.USER_REVIEW,
            f"retry budget exhausted for {action.name}: {used}/{limit}",
            model_ids=_model_ids(decision, secondary),
        )
    prefix = "bounded repair selected for rejected hard gate" if hard_gate_rejected else "bounded retry selected"
    return ControlOutcome(
        OutcomeStatus.RETRY,
        f"{prefix}; attempt {used + 1}/{limit}",
        selected_action=action.name,
        parameters=dict(action.parameters),
        model_ids=_model_ids(decision, secondary),
    )


def _decisions_conflict(a: ModelDecision, b: ModelDecision) -> bool:
    if a.verdict is b.verdict:
        if a.verdict is Verdict.RETRY:
            return a.proposed_action != b.proposed_action
        return False
    compatible = {Verdict.QUARANTINE, Verdict.USER_REVIEW, Verdict.SECOND_OPINION}
    return not ({a.verdict, b.verdict} <= compatible)


def _higher_confidence(primary: ModelDecision, secondary: ModelDecision | None) -> ModelDecision:
    if secondary is None or primary.confidence >= secondary.confidence:
        return primary
    return secondary


def _model_ids(primary: ModelDecision, secondary: ModelDecision | None) -> tuple[str, ...]:
    return (primary.model_id,) if secondary is None else (primary.model_id, secondary.model_id)
