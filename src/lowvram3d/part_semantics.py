"""Deterministic semantic-part contracts and fail-safe handling policy.

The vision layer may suggest labels and masks, but this module owns the production
decision.  It never invents a part, never assumes that a fused region is safely
separable, and degrades uncertain regions to vertex groups or protected fused
regions instead of destructive cuts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

PART_LABELS = frozenset({
    "body", "torso", "head", "face",
    "left_arm", "right_arm", "left_hand", "right_hand",
    "left_leg", "right_leg", "left_foot", "right_foot",
    "tail", "wing", "ear", "antler_or_horn",
    "staff", "weapon", "shield", "held_prop",
    "hanging_ornament", "rope_or_chain", "cloth_strip",
    "hair", "fur", "feather", "backpack", "rigid_accessory",
    "unknown",
})

HANDLING_MODES = frozenset({
    "hard_split", "vertex_group_only", "protected_fused_region",
    "secondary_motion_region", "material_region", "leave_unknown",
})

BODY_LABELS = frozenset({
    "body", "torso", "head", "face",
    "left_arm", "right_arm", "left_hand", "right_hand",
    "left_leg", "right_leg", "left_foot", "right_foot",
    "ear", "antler_or_horn",
})

HELD_PROP_LABELS = frozenset({"staff", "weapon", "shield", "held_prop"})
RIGID_PROP_LABELS = HELD_PROP_LABELS | frozenset({"backpack", "rigid_accessory"})
SECONDARY_MOTION_LABELS = frozenset({
    "tail", "wing", "hanging_ornament", "rope_or_chain",
    "cloth_strip", "hair", "feather",
})
MATERIAL_ONLY_LABELS = frozenset({"fur"})


class PartManifestError(ValueError):
    """Raised when a semantic part manifest is structurally unsafe."""


@dataclass(frozen=True)
class ViewEvidence:
    name: str
    confidence: float
    mask: str | None = None
    bbox: tuple[float, float, float, float] | None = None
    source_type: str = "real"

    @classmethod
    def from_raw(cls, name: str, raw: dict[str, Any]) -> "ViewEvidence":
        confidence = _unit(raw.get("confidence", 0.0), f"views.{name}.confidence")
        bbox_raw = raw.get("bbox")
        bbox = None
        if bbox_raw is not None:
            if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
                raise PartManifestError(f"views.{name}.bbox must have four numbers")
            bbox = tuple(float(value) for value in bbox_raw)
            if bbox[2] < bbox[0] or bbox[3] < bbox[1]:
                raise PartManifestError(f"views.{name}.bbox is inverted")
        source_type = str(raw.get("source_type", "real")).strip().lower()
        if source_type not in {"real", "render", "mirrored", "synthetic"}:
            raise PartManifestError(f"unsupported source_type {source_type!r}")
        return cls(
            name=name,
            confidence=confidence,
            mask=str(raw["mask"]) if raw.get("mask") else None,
            bbox=bbox,
            source_type=source_type,
        )


@dataclass
class PartRegion:
    id: str
    label: str
    category: str
    confidence: float
    mesh_state: str = "uncertain"
    requested_handling: str | None = None
    views: dict[str, ViewEvidence] = field(default_factory=dict)
    vertex_indices: tuple[int, ...] = ()
    face_indices: tuple[int, ...] = ()
    metrics: dict[str, float | int | bool | str | None] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "PartRegion":
        if not isinstance(raw, dict):
            raise PartManifestError("each part must be an object")
        identifier = str(raw.get("id", "")).strip()
        if not identifier:
            raise PartManifestError("part id is required")
        label = str(raw.get("label", "unknown")).strip().lower()
        if label not in PART_LABELS:
            raise PartManifestError(f"part {identifier!r} has unsupported label {label!r}")
        category = str(raw.get("category", label)).strip().lower() or label
        confidence = _unit(raw.get("confidence", 0.0), f"part {identifier}.confidence")
        mesh_state = str(raw.get("mesh_state", "uncertain")).strip().lower()
        if mesh_state not in {"separate", "fused", "uncertain"}:
            raise PartManifestError(f"part {identifier!r} has invalid mesh_state {mesh_state!r}")
        requested = raw.get("handling") or raw.get("requested_handling")
        requested_handling = str(requested).strip().lower() if requested else None
        if requested_handling and requested_handling not in HANDLING_MODES:
            raise PartManifestError(
                f"part {identifier!r} has invalid handling {requested_handling!r}"
            )
        views_raw = raw.get("views") or {}
        if not isinstance(views_raw, dict):
            raise PartManifestError(f"part {identifier!r}.views must be an object")
        views = {
            str(name): ViewEvidence.from_raw(str(name), value or {})
            for name, value in views_raw.items()
        }
        return cls(
            id=identifier,
            label=label,
            category=category,
            confidence=confidence,
            mesh_state=mesh_state,
            requested_handling=requested_handling,
            views=views,
            vertex_indices=_indices(raw.get("vertex_indices"), f"part {identifier}.vertex_indices"),
            face_indices=_indices(raw.get("face_indices"), f"part {identifier}.face_indices"),
            metrics=dict(raw.get("metrics") or {}),
            metadata=dict(raw.get("metadata") or {}),
        )

    @property
    def independent_real_views(self) -> int:
        return sum(
            1
            for evidence in self.views.values()
            if evidence.source_type in {"real", "render"} and evidence.confidence >= 0.70
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["vertex_indices"] = list(self.vertex_indices)
        result["face_indices"] = list(self.face_indices)
        return result


@dataclass(frozen=True)
class HandlingDecision:
    part_id: str
    label: str
    handling: str
    allowed: bool
    confidence: float
    reason_codes: tuple[str, ...]
    gates: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "label": self.label,
            "handling": self.handling,
            "allowed": self.allowed,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "gates": self.gates,
        }


def _unit(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise PartManifestError(f"{field_name} must be numeric") from exc
    if not 0.0 <= number <= 1.0:
        raise PartManifestError(f"{field_name} must be in [0, 1]")
    return number


def _indices(value: Any, field_name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise PartManifestError(f"{field_name} must be a list")
    result: set[int] = set()
    for item in value:
        try:
            index = int(item)
        except (TypeError, ValueError) as exc:
            raise PartManifestError(f"{field_name} contains a non-integer") from exc
        if index < 0:
            raise PartManifestError(f"{field_name} contains a negative index")
        result.add(index)
    return tuple(sorted(result))


def load_parts_manifest(source: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    """Load and validate a semantic proposal manifest.

    Missing input is a valid safe state: the returned manifest contains no parts
    and explicitly records that semantic evidence was unavailable.
    """
    if source is None or source == "":
        return {
            "schema_version": SCHEMA_VERSION,
            "parts": [],
            "status": "missing_optional_manifest",
        }
    if isinstance(source, dict):
        raw = source
    else:
        path = Path(source)
        if not path.is_file():
            return {
                "schema_version": SCHEMA_VERSION,
                "parts": [],
                "status": "missing_optional_manifest",
                "missing_path": str(path),
            }
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise PartManifestError(f"invalid JSON in {path}: {exc}") from exc
    if int(raw.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise PartManifestError("unsupported semantic-parts schema version")
    parts_raw = raw.get("parts") or []
    if not isinstance(parts_raw, list):
        raise PartManifestError("parts must be a list")
    parts = [PartRegion.from_raw(item) for item in parts_raw]
    identifiers = [part.id for part in parts]
    if len(identifiers) != len(set(identifiers)):
        raise PartManifestError("part ids must be unique")
    return {
        "schema_version": SCHEMA_VERSION,
        "parts": parts,
        "status": str(raw.get("status", "provided")),
        "source_sha256": raw.get("source_sha256"),
        "metadata": dict(raw.get("metadata") or {}),
    }


def hard_split_gates(part: PartRegion) -> tuple[bool, dict[str, Any], tuple[str, ...]]:
    """Apply the destructive split gate without making any geometry changes."""
    metrics = part.metrics
    boundary_confidence = float(metrics.get("boundary_confidence", 0.0) or 0.0)
    body_leakage = float(metrics.get("body_label_leakage", 1.0) or 0.0)
    part_loss = float(metrics.get("part_label_loss", 1.0) or 0.0)
    attachment_ratio = float(metrics.get("attachment_boundary_ratio", 1.0) or 0.0)
    topology_delta = int(metrics.get("topology_regression_edges", 1) or 0)
    fresh_import = bool(metrics.get("fresh_import_validated", False))
    protected_capture = bool(metrics.get("protected_neighbour_captured", True))

    gates = {
        "mesh_state_separate": part.mesh_state == "separate",
        "independent_views": part.independent_real_views,
        "mean_confidence": part.confidence,
        "boundary_confidence": boundary_confidence,
        "body_label_leakage": body_leakage,
        "part_label_loss": part_loss,
        "attachment_boundary_ratio": attachment_ratio,
        "topology_regression_edges": topology_delta,
        "fresh_import_validated": fresh_import,
        "protected_neighbour_captured": protected_capture,
    }
    reasons: list[str] = []
    if part.mesh_state != "separate":
        reasons.append("PART_NOT_GEOMETRICALLY_SEPARATE")
    if part.independent_real_views < 2:
        reasons.append("PART_MULTIVIEW_EVIDENCE_INSUFFICIENT")
    if part.confidence < 0.90:
        reasons.append("PART_CONFIDENCE_TOO_LOW_FOR_SPLIT")
    if boundary_confidence < 0.85:
        reasons.append("PART_BOUNDARY_CONFIDENCE_TOO_LOW")
    if body_leakage > 0.02:
        reasons.append("PART_BODY_LABEL_LEAKAGE")
    if part_loss > 0.05:
        reasons.append("PART_LABEL_LOSS")
    if attachment_ratio > 0.15:
        reasons.append("PART_ATTACHMENT_NOT_NARROW")
    if topology_delta != 0:
        reasons.append("PART_TOPOLOGY_REGRESSION")
    if protected_capture:
        reasons.append("PART_PROTECTED_NEIGHBOUR_CAPTURED")
    if not fresh_import:
        reasons.append("PART_FRESH_IMPORT_NOT_VALIDATED")
    return not reasons, gates, tuple(reasons)


def choose_handling(part: PartRegion, *, separate_props: bool = False) -> HandlingDecision:
    """Choose a non-destructive handling mode unless every split gate passes."""
    reasons: list[str] = []
    gates: dict[str, Any] = {
        "semantic_confidence": part.confidence,
        "mesh_state": part.mesh_state,
        "independent_views": part.independent_real_views,
    }

    if part.confidence < 0.70 or part.label == "unknown":
        return HandlingDecision(
            part.id, part.label, "leave_unknown", True, part.confidence,
            ("PART_UNCERTAIN_PRESERVED",), gates,
        )

    if part.label in MATERIAL_ONLY_LABELS:
        return HandlingDecision(
            part.id, part.label, "material_region", True, part.confidence,
            ("PART_MATERIAL_ONLY",), gates,
        )

    if part.label in SECONDARY_MOTION_LABELS:
        handling = "secondary_motion_region" if part.confidence >= 0.85 else "vertex_group_only"
        return HandlingDecision(
            part.id, part.label, handling, True, part.confidence,
            ("PART_SECONDARY_MOTION_CANDIDATE",), gates,
        )

    if part.label in RIGID_PROP_LABELS:
        wants_split = separate_props or part.requested_handling == "hard_split"
        if wants_split:
            allowed, split_gates, split_reasons = hard_split_gates(part)
            gates.update(split_gates)
            if allowed:
                return HandlingDecision(
                    part.id, part.label, "hard_split", True, part.confidence,
                    ("PART_HARD_SPLIT_GATES_PASSED",), gates,
                )
            reasons.extend(split_reasons)
        fallback = "protected_fused_region" if part.mesh_state != "separate" else "vertex_group_only"
        if not reasons:
            reasons.append("PART_PROP_SPLIT_NOT_REQUESTED")
        return HandlingDecision(
            part.id, part.label, fallback, True, part.confidence,
            tuple(reasons), gates,
        )

    if part.label in BODY_LABELS:
        return HandlingDecision(
            part.id, part.label, "vertex_group_only", True, part.confidence,
            ("PART_DEFORM_REGION",), gates,
        )

    return HandlingDecision(
        part.id, part.label, "vertex_group_only", True, part.confidence,
        ("PART_SAFE_VERTEX_GROUP_DEFAULT",), gates,
    )


def build_parts_plan(
    source: str | Path | dict[str, Any] | None,
    *,
    separate_props: bool = False,
) -> dict[str, Any]:
    manifest = load_parts_manifest(source)
    parts: list[PartRegion] = manifest["parts"]
    decisions = [choose_handling(part, separate_props=separate_props) for part in parts]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "planned" if parts else "safe_no_semantic_parts",
        "semantic_source_status": manifest["status"],
        "part_count": len(parts),
        "parts": [part.to_dict() for part in parts],
        "decisions": [decision.to_dict() for decision in decisions],
        "hard_split_count": sum(decision.handling == "hard_split" for decision in decisions),
        "protected_fused_count": sum(
            decision.handling == "protected_fused_region" for decision in decisions
        ),
        "unknown_count": sum(decision.handling == "leave_unknown" for decision in decisions),
    }


def labels_present(parts: Iterable[PartRegion]) -> set[str]:
    return {part.label for part in parts if part.confidence >= 0.70}
