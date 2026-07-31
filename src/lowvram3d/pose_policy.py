"""Deterministic source-, bind-, and retarget-pose contracts.

Geometry comparison always occurs in the source pose. Pose normalization is a later rigging step
and must preserve a byte-identifiable source-pose artifact beside the normalized bind-pose export.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class PoseMode(str, Enum):
    AUTO = "auto"
    PRESERVE_SOURCE = "preserve_source"
    A_POSE = "a_pose"
    T_POSE = "t_pose"
    NEUTRAL_BIPED = "neutral_biped"
    NEUTRAL_QUADRUPED = "neutral_quadruped"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class APoseSpecification:
    arm_from_torso_degrees: float = 40.0
    elbow_bend_degrees: float = 8.0
    foot_outward_degrees: float = 5.0
    feet_shoulder_width_fraction: float = 1.0
    minimum_hand_torso_clearance_fraction: float = 0.03
    maximum_floor_error_fraction: float = 0.01


@dataclass(frozen=True, slots=True)
class PoseContract:
    source_pose: PoseMode
    bind_pose: PoseMode
    retarget_pose: str
    alternate_pose: PoseMode | None
    compare_before_normalization: bool
    preserve_source_artifact: bool
    source_output_name: str
    bind_output_name: str
    specification: APoseSpecification | None = None

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["source_pose"] = self.source_pose.value
        payload["bind_pose"] = self.bind_pose.value
        payload["alternate_pose"] = self.alternate_pose.value if self.alternate_pose else None
        return payload


@dataclass(frozen=True, slots=True)
class PoseValidation:
    valid: bool
    errors: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"valid": self.valid, "errors": list(self.errors)}


_BIPED_TYPES = {"avatar", "human", "humanoid", "character", "anthropomorphic"}
_QUADRUPED_TYPES = {"quadruped", "animal", "four_legged"}
_PRESERVE_TYPES = {"vehicle", "building", "room", "scene", "level", "prop", "static"}


def resolve_pose_contract(
    asset_type: str,
    *,
    generate_rig: bool,
    requested: str | PoseMode = PoseMode.AUTO,
    benchmark_compare: bool = False,
    unreal_compatible: bool = True,
) -> PoseContract:
    """Resolve a fail-closed pose contract without changing benchmark geometry semantics."""
    kind = str(asset_type).strip().lower()
    mode = requested if isinstance(requested, PoseMode) else PoseMode(str(requested).lower())

    if benchmark_compare:
        bind = PoseMode.PRESERVE_SOURCE
    elif mode is not PoseMode.AUTO:
        bind = mode
    elif not generate_rig or kind in _PRESERVE_TYPES:
        bind = PoseMode.PRESERVE_SOURCE
    elif kind in _QUADRUPED_TYPES:
        bind = PoseMode.NEUTRAL_QUADRUPED
    elif kind in _BIPED_TYPES:
        bind = PoseMode.A_POSE
    else:
        bind = PoseMode.PRESERVE_SOURCE

    specification = APoseSpecification() if bind is PoseMode.A_POSE else None
    retarget = "ue5_mannequin_a_pose" if unreal_compatible and bind is PoseMode.A_POSE else "source_skeleton"
    alternate = PoseMode.T_POSE if bind is PoseMode.A_POSE else None
    return PoseContract(
        source_pose=PoseMode.PRESERVE_SOURCE,
        bind_pose=bind,
        retarget_pose=retarget,
        alternate_pose=alternate,
        compare_before_normalization=True,
        preserve_source_artifact=True,
        source_output_name="geometry/source_pose.glb",
        bind_output_name=("rigged/bind_a_pose.glb" if bind is PoseMode.A_POSE else "rigged/bind_pose.glb"),
        specification=specification,
    )


def validate_pose_measurements(measurements: dict, contract: PoseContract) -> PoseValidation:
    """Validate normalized-pose measurements emitted by Blender.

    Values are normalized by character height where the key ends in ``_fraction``. Missing
    measurements fail closed rather than being treated as zero.
    """
    errors: list[str] = []
    required_true = (
        "expected_limbs_present",
        "left_right_bone_pairs_complete",
        "skeleton_hierarchy_valid",
        "skin_weight_coverage_complete",
        "no_visible_unweighted_vertices",
        "no_new_self_intersections",
        "equipment_assignment_valid",
        "source_pose_artifact_present",
    )
    for key in required_true:
        if measurements.get(key) is not True:
            errors.append(key)

    if contract.bind_pose is PoseMode.A_POSE:
        spec = contract.specification or APoseSpecification()
        arm_angles = measurements.get("arm_from_torso_degrees")
        if not isinstance(arm_angles, (list, tuple)) or len(arm_angles) != 2:
            errors.append("arm_from_torso_degrees")
        elif any(abs(float(value) - spec.arm_from_torso_degrees) > 7.5 for value in arm_angles):
            errors.append("arm_angle_out_of_range")

        clearance = measurements.get("minimum_hand_torso_clearance_fraction")
        if clearance is None or float(clearance) < spec.minimum_hand_torso_clearance_fraction:
            errors.append("hand_torso_clearance")

        floor_error = measurements.get("maximum_foot_floor_error_fraction")
        if floor_error is None or float(floor_error) > spec.maximum_floor_error_fraction:
            errors.append("foot_floor_contact")

    deformation = measurements.get("maximum_vertex_deformation_fraction")
    if deformation is None:
        errors.append("maximum_vertex_deformation_fraction")
    elif float(deformation) > 0.35:
        errors.append("geometry_collapse_or_excessive_deformation")

    return PoseValidation(valid=not errors, errors=tuple(errors))
