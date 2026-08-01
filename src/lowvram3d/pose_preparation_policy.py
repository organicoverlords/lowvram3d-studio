"""Fail-safe pose-preparation eligibility and promotion policy.

This module decides whether a mesh may be posed. It does not deform geometry.
The Blender worker consumes the returned plan only when every eligibility gate
passes; otherwise the source pose remains the production output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from lowvram3d.part_semantics import PartRegion, labels_present, load_parts_manifest

SUPPORTED_HUMANOID_PROFILES = frozenset({"humanoid", "humanoid_complex_accessories"})
REQUIRED_LABELS = frozenset({"torso", "left_arm", "right_arm", "left_leg", "right_leg"})
PROTECTED_PROP_LABELS = frozenset({"staff", "weapon", "shield", "held_prop"})


@dataclass(frozen=True)
class APoseTarget:
    upper_arm_below_horizontal_degrees: float = 40.0
    elbow_bend_degrees: float = 8.0
    shoulder_elevation_degrees: float = 0.0
    root_translation_allowed: bool = False
    preserve_feet: bool = True
    preserve_head: bool = True
    preserve_protected_props: bool = True


@dataclass(frozen=True)
class PoseEligibility:
    eligible: bool
    action: str
    reason_codes: tuple[str, ...]
    gates: dict[str, Any]
    target: APoseTarget

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "action": self.action,
            "reason_codes": list(self.reason_codes),
            "gates": self.gates,
            "target": asdict(self.target),
        }


def _part_map(parts: list[PartRegion]) -> dict[str, list[PartRegion]]:
    result: dict[str, list[PartRegion]] = {}
    for part in parts:
        result.setdefault(part.label, []).append(part)
    return result


def _best_confidence(mapping: dict[str, list[PartRegion]], label: str) -> float:
    return max((part.confidence for part in mapping.get(label, [])), default=0.0)


def evaluate_a_pose_eligibility(
    profile_name: str,
    parts_source,
    *,
    geometry_metrics: dict[str, Any] | None = None,
    minimum_limb_confidence: float = 0.85,
) -> PoseEligibility:
    """Return a production-safe A-pose decision.

    A missing or ambiguous semantic manifest is not an error: the correct action
    is to preserve the source pose and continue. The function therefore returns
    a structured skip rather than raising for normal uncertainty.
    """
    metrics = dict(geometry_metrics or {})
    target = APoseTarget()
    gates: dict[str, Any] = {
        "profile": profile_name,
        "minimum_limb_confidence": minimum_limb_confidence,
    }
    reasons: list[str] = []

    if profile_name not in SUPPORTED_HUMANOID_PROFILES:
        reasons.append("POSE_PROFILE_NOT_HUMANOID")

    manifest = load_parts_manifest(parts_source)
    parts: list[PartRegion] = manifest["parts"]
    mapping = _part_map(parts)
    present = labels_present(parts)
    gates["semantic_source_status"] = manifest["status"]
    gates["labels_present"] = sorted(present)

    missing = sorted(REQUIRED_LABELS - present)
    gates["missing_required_labels"] = missing
    if missing:
        reasons.append("POSE_REQUIRED_PARTS_MISSING")

    limb_confidence = {
        label: _best_confidence(mapping, label)
        for label in sorted(REQUIRED_LABELS)
    }
    gates["limb_confidence"] = limb_confidence
    if any(value < minimum_limb_confidence for value in limb_confidence.values()):
        reasons.append("POSE_LIMB_CONFIDENCE_TOO_LOW")

    unstable_shoulders = bool(metrics.get("shoulder_landmarks_unstable", False))
    unstable_hips = bool(metrics.get("hip_landmarks_unstable", False))
    hands_fused = bool(metrics.get("hands_inseparably_fused", False))
    arms_fused_to_torso = bool(metrics.get("arms_inseparably_fused_to_torso", False))
    depth_ratio = float(metrics.get("depth_to_height_ratio", 1.0) or 0.0)
    minimum_depth_ratio = float(metrics.get("minimum_pose_depth_ratio", 0.10) or 0.10)

    gates.update({
        "shoulder_landmarks_stable": not unstable_shoulders,
        "hip_landmarks_stable": not unstable_hips,
        "hands_inseparably_fused": hands_fused,
        "arms_inseparably_fused_to_torso": arms_fused_to_torso,
        "depth_to_height_ratio": depth_ratio,
        "minimum_pose_depth_ratio": minimum_depth_ratio,
    })

    if unstable_shoulders or unstable_hips:
        reasons.append("POSE_LANDMARKS_UNSTABLE")
    if hands_fused:
        reasons.append("POSE_HANDS_INSEPARABLY_FUSED")
    if arms_fused_to_torso:
        reasons.append("POSE_ARMS_INSEPARABLY_FUSED")
    if depth_ratio < minimum_depth_ratio:
        reasons.append("POSE_GEOMETRY_TOO_SHALLOW")

    protected_props = [
        part for part in parts
        if part.label in PROTECTED_PROP_LABELS and part.confidence >= 0.70
    ]
    fused_protected_props = [part.id for part in protected_props if part.mesh_state != "separate"]
    gates["protected_props"] = [part.id for part in protected_props]
    gates["fused_protected_props"] = fused_protected_props

    if fused_protected_props and not bool(metrics.get("protected_props_can_remain_fixed", False)):
        reasons.append("POSE_FUSED_PROTECTED_PROP")

    source_has_skeleton = bool(metrics.get("source_has_skeleton", False))
    valid_existing_weights = bool(metrics.get("valid_existing_weights", False))
    gates["source_has_skeleton"] = source_has_skeleton
    gates["valid_existing_weights"] = valid_existing_weights

    unique_reasons = tuple(dict.fromkeys(reasons))
    eligible = not unique_reasons
    return PoseEligibility(
        eligible=eligible,
        action="apply_a_pose" if eligible else "preserve_source_pose",
        reason_codes=unique_reasons or ("POSE_ELIGIBILITY_PASSED",),
        gates=gates,
        target=target,
    )


def validate_pose_result(metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply post-deformation gates and return a promotion verdict."""
    failures: list[str] = []

    source_hash_unchanged = bool(metrics.get("source_hash_unchanged", False))
    finite = bool(metrics.get("finite", False))
    fresh_import = bool(metrics.get("fresh_import_validated", False))
    component_regression = int(metrics.get("component_count_regression", 0) or 0)
    topology_regression = int(metrics.get("topology_regression_edges", 0) or 0)
    root_displacement = float(metrics.get("root_displacement", 1.0) or 0.0)
    foot_displacement = float(metrics.get("max_planted_foot_displacement", 1.0) or 0.0)
    torso_volume_delta = abs(float(metrics.get("torso_volume_delta_fraction", 1.0) or 0.0))
    protected_prop_displacement = float(metrics.get("protected_prop_displacement", 1.0) or 0.0)
    self_intersection_delta = int(metrics.get("self_intersection_delta", 1) or 0)
    arm_clearance_improved = bool(metrics.get("arm_clearance_improved", False))
    arm_angles_valid = bool(metrics.get("arm_angles_valid", False))

    if not source_hash_unchanged:
        failures.append("POSE_SOURCE_HASH_CHANGED")
    if not finite:
        failures.append("POSE_NONFINITE_GEOMETRY")
    if not fresh_import:
        failures.append("POSE_FRESH_IMPORT_FAILED")
    if component_regression != 0:
        failures.append("POSE_COMPONENT_REGRESSION")
    if topology_regression != 0:
        failures.append("POSE_TOPOLOGY_REGRESSION")
    if root_displacement > 1e-5:
        failures.append("POSE_ROOT_MOVED")
    if foot_displacement > 1e-4:
        failures.append("POSE_PLANTED_FEET_MOVED")
    if torso_volume_delta > 0.02:
        failures.append("POSE_TORSO_VOLUME_CHANGED")
    if protected_prop_displacement > 1e-4:
        failures.append("POSE_PROTECTED_PROP_MOVED")
    if self_intersection_delta > 0:
        failures.append("POSE_SELF_INTERSECTION_REGRESSION")
    if not arm_clearance_improved:
        failures.append("POSE_ARM_CLEARANCE_NOT_IMPROVED")
    if not arm_angles_valid:
        failures.append("POSE_TARGET_ANGLES_INVALID")

    return {
        "passed": not failures,
        "failure_codes": failures,
        "measured": {
            "source_hash_unchanged": source_hash_unchanged,
            "finite": finite,
            "fresh_import_validated": fresh_import,
            "component_count_regression": component_regression,
            "topology_regression_edges": topology_regression,
            "root_displacement": root_displacement,
            "max_planted_foot_displacement": foot_displacement,
            "torso_volume_delta_fraction": torso_volume_delta,
            "protected_prop_displacement": protected_prop_displacement,
            "self_intersection_delta": self_intersection_delta,
            "arm_clearance_improved": arm_clearance_improved,
            "arm_angles_valid": arm_angles_valid,
        },
    }
