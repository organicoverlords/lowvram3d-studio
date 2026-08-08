"""Dependency-free contracts for fail-closed visual pipeline supervision.

The contracts intentionally separate measurable pipeline evidence from model opinion.
A vision model can describe or route a failure, but cannot promote a result whose hard
gates are rejected, blocked, or not proven.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping


class ContractError(ValueError):
    """Raised when a packet or model decision violates the public contract."""


class GateStatus(str, Enum):
    PROVEN = "PROVEN"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    NOT_PROVEN = "NOT_PROVEN"


class Verdict(str, Enum):
    PASS = "pass"
    RETRY = "retry"
    QUARANTINE = "quarantine"
    USER_REVIEW = "user_review"
    SECOND_OPINION = "request_second_opinion"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceKind(str, Enum):
    SOURCE = "source"
    BEAUTY = "beauty"
    UNLIT = "unlit"
    ALBEDO = "albedo"
    SILHOUETTE = "silhouette"
    WIREFRAME = "wireframe"
    DEPTH = "depth"
    NORMAL = "normal"
    MATERIAL_ID = "material_id"
    UV_ATLAS = "uv_atlas"
    UV_SEAMS = "uv_seams"
    MASK = "mask"
    METRICS = "metrics"
    RECEIPT = "receipt"
    LOG = "log"


@dataclass(frozen=True)
class EvidenceArtifact:
    artifact_id: str
    kind: EvidenceKind
    path: str
    sha256: str | None = None
    view: str | None = None
    required: bool = True
    description: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvidenceArtifact":
        return cls(
            artifact_id=str(data["artifact_id"]),
            kind=EvidenceKind(data["kind"]),
            path=str(data["path"]),
            sha256=data.get("sha256"),
            view=data.get("view"),
            required=bool(data.get("required", True)),
            description=str(data.get("description", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True)
class HardGate:
    name: str
    status: GateStatus
    evidence_ids: tuple[str, ...] = ()
    detail: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HardGate":
        return cls(
            name=str(data["name"]),
            status=GateStatus(data["status"]),
            evidence_ids=tuple(str(item) for item in data.get("evidence_ids", [])),
            detail=str(data.get("detail", "")),
        )


@dataclass(frozen=True)
class Metric:
    name: str
    value: float | int | str | bool | None
    status: GateStatus
    threshold: str = ""
    evidence_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Metric":
        return cls(
            name=str(data["name"]),
            value=data.get("value"),
            status=GateStatus(data["status"]),
            threshold=str(data.get("threshold", "")),
            evidence_ids=tuple(str(item) for item in data.get("evidence_ids", [])),
        )


@dataclass(frozen=True)
class RetryState:
    attempts_by_action: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "RetryState":
        data = data or {}
        raw = data.get("attempts_by_action", data)
        return cls({str(key): int(value) for key, value in raw.items()})


@dataclass(frozen=True)
class VisionQaPacket:
    schema: str
    packet_id: str
    stage: str
    artifacts: tuple[EvidenceArtifact, ...]
    hard_gates: tuple[HardGate, ...]
    metrics: tuple[Metric, ...] = ()
    retry_state: RetryState = field(default_factory=RetryState)
    context: Mapping[str, Any] = field(default_factory=dict)
    manual_rejection: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisionQaPacket":
        packet = cls(
            schema=str(data.get("schema", "")),
            packet_id=str(data["packet_id"]),
            stage=str(data["stage"]),
            artifacts=tuple(EvidenceArtifact.from_dict(x) for x in data.get("artifacts", [])),
            hard_gates=tuple(HardGate.from_dict(x) for x in data.get("hard_gates", [])),
            metrics=tuple(Metric.from_dict(x) for x in data.get("metrics", [])),
            retry_state=RetryState.from_dict(data.get("retry_state")),
            context=dict(data.get("context", {})),
            manual_rejection=data.get("manual_rejection"),
        )
        packet.validate()
        return packet

    def validate(self) -> None:
        if self.schema != "vision_qa_packet_v1":
            raise ContractError(f"unsupported packet schema: {self.schema!r}")
        if not self.packet_id.strip() or not self.stage.strip():
            raise ContractError("packet_id and stage are required")
        ids = [artifact.artifact_id for artifact in self.artifacts]
        if len(ids) != len(set(ids)):
            raise ContractError("artifact_id values must be unique")
        known = set(ids)
        for item in (*self.hard_gates, *self.metrics):
            missing = set(item.evidence_ids) - known
            if missing:
                raise ContractError(f"unknown evidence ids referenced by {item.name}: {sorted(missing)}")
        for action, attempts in self.retry_state.attempts_by_action.items():
            if not action or attempts < 0:
                raise ContractError("retry counts require non-empty action names and non-negative values")

    def to_dict(self) -> dict[str, Any]:
        return _enum_values(asdict(self))


@dataclass(frozen=True)
class Finding:
    code: str
    severity: Severity
    confidence: float
    evidence_ids: tuple[str, ...]
    detail: str

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Finding":
        finding = cls(
            code=str(data["code"]),
            severity=Severity(data["severity"]),
            confidence=float(data["confidence"]),
            evidence_ids=tuple(str(x) for x in data.get("evidence_ids", [])),
            detail=str(data.get("detail", "")),
        )
        if not 0.0 <= finding.confidence <= 1.0:
            raise ContractError("finding confidence must be within [0, 1]")
        return finding


@dataclass(frozen=True)
class ProposedAction:
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "ProposedAction | None":
        if data is None:
            return None
        return cls(name=str(data["name"]), parameters=dict(data.get("parameters", {})))


@dataclass(frozen=True)
class ModelDecision:
    schema: str
    model_id: str
    packet_id: str
    stage: str
    verdict: Verdict
    confidence: float
    summary: str
    findings: tuple[Finding, ...] = ()
    proposed_action: ProposedAction | None = None
    uncertainties: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelDecision":
        decision = cls(
            schema=str(data.get("schema", "")),
            model_id=str(data["model_id"]),
            packet_id=str(data["packet_id"]),
            stage=str(data["stage"]),
            verdict=Verdict(data["verdict"]),
            confidence=float(data["confidence"]),
            summary=str(data.get("summary", "")),
            findings=tuple(Finding.from_dict(x) for x in data.get("findings", [])),
            proposed_action=ProposedAction.from_dict(data.get("proposed_action")),
            uncertainties=tuple(str(x) for x in data.get("uncertainties", [])),
        )
        decision.validate()
        return decision

    def validate(self) -> None:
        if self.schema != "vision_qa_decision_v1":
            raise ContractError(f"unsupported decision schema: {self.schema!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ContractError("decision confidence must be within [0, 1]")
        if self.verdict is Verdict.RETRY and self.proposed_action is None:
            raise ContractError("retry verdict requires proposed_action")
        if self.verdict is not Verdict.RETRY and self.proposed_action is not None:
            raise ContractError("only retry verdict may include proposed_action")

    def validate_against(self, packet: VisionQaPacket) -> None:
        if self.packet_id != packet.packet_id or self.stage != packet.stage:
            raise ContractError("decision packet_id/stage does not match packet")
        known = {artifact.artifact_id for artifact in packet.artifacts}
        for finding in self.findings:
            missing = set(finding.evidence_ids) - known
            if missing:
                raise ContractError(f"finding {finding.code} references unknown evidence: {sorted(missing)}")

    def to_dict(self) -> dict[str, Any]:
        return _enum_values(asdict(self))


def _enum_values(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _enum_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_enum_values(item) for item in value]
    return value


def statuses(items: Iterable[HardGate | Metric]) -> set[GateStatus]:
    return {item.status for item in items}
