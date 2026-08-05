from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "workers"))

import pytest

from build_asset_system_plan import build_plan
from lowvram3d.material_feature_policy import (
    MaterialManifestError,
    build_material_plan,
    decide_feature,
    load_material_manifest,
    soft_surface_budget,
)
from lowvram3d.part_semantics import (
    PartManifestError,
    PartRegion,
    build_parts_plan,
    choose_handling,
    load_parts_manifest,
)
from lowvram3d.pose_preparation_policy import (
    evaluate_a_pose_eligibility,
    validate_pose_result,
)


def part(
    identifier: str,
    label: str,
    confidence: float = 0.95,
    *,
    mesh_state: str = "fused",
    metrics: dict | None = None,
    views: dict | None = None,
) -> dict:
    return {
        "id": identifier,
        "label": label,
        "category": label,
        "confidence": confidence,
        "mesh_state": mesh_state,
        "metrics": metrics or {},
        "views": views or {},
    }


def humanoid_parts(*, include_staff: bool = False) -> dict:
    parts = [
        part("torso", "torso"),
        part("arm_l", "left_arm"),
        part("arm_r", "right_arm"),
        part("leg_l", "left_leg"),
        part("leg_r", "right_leg"),
    ]
    if include_staff:
        parts.append(part("staff", "staff", mesh_state="fused"))
    return {"schema_version": 1, "parts": parts}


def test_missing_parts_manifest_is_safe() -> None:
    plan = build_parts_plan(None, separate_props=True)
    assert plan["status"] == "safe_no_semantic_parts"
    assert plan["part_count"] == 0
    assert plan["hard_split_count"] == 0


def test_unknown_label_is_rejected() -> None:
    with pytest.raises(PartManifestError):
        load_parts_manifest({"schema_version": 1, "parts": [part("x", "spaceship_engine")]})


def test_fused_staff_is_protected_not_split() -> None:
    manifest = load_parts_manifest({"schema_version": 1, "parts": [part("staff", "staff")]})
    decision = choose_handling(manifest["parts"][0], separate_props=True)
    assert decision.handling == "protected_fused_region"
    assert "PART_NOT_GEOMETRICALLY_SEPARATE" in decision.reason_codes


def test_hard_split_requires_every_gate() -> None:
    metrics = {
        "boundary_confidence": 0.93,
        "body_label_leakage": 0.01,
        "part_label_loss": 0.02,
        "attachment_boundary_ratio": 0.08,
        "topology_regression_edges": 0,
        "fresh_import_validated": True,
        "protected_neighbour_captured": False,
    }
    views = {
        "front": {"confidence": 0.95, "source_type": "real"},
        "side": {"confidence": 0.91, "source_type": "render"},
    }
    manifest = load_parts_manifest({
        "schema_version": 1,
        "parts": [part("sword", "weapon", mesh_state="separate", metrics=metrics, views=views)],
    })
    decision = choose_handling(manifest["parts"][0], separate_props=True)
    assert decision.handling == "hard_split"
    assert decision.allowed is True


def test_mirrored_view_does_not_count_as_independent_evidence() -> None:
    raw = part(
        "shield",
        "shield",
        mesh_state="separate",
        metrics={
            "boundary_confidence": 1.0,
            "body_label_leakage": 0.0,
            "part_label_loss": 0.0,
            "attachment_boundary_ratio": 0.0,
            "topology_regression_edges": 0,
            "fresh_import_validated": True,
            "protected_neighbour_captured": False,
        },
        views={
            "front": {"confidence": 0.95, "source_type": "real"},
            "rear_mirror": {"confidence": 0.99, "source_type": "mirrored"},
        },
    )
    region = PartRegion.from_raw(raw)
    decision = choose_handling(region, separate_props=True)
    assert decision.handling != "hard_split"
    assert "PART_MULTIVIEW_EVIDENCE_INSUFFICIENT" in decision.reason_codes


def test_hair_becomes_secondary_motion_region() -> None:
    region = PartRegion.from_raw(part("hair", "hair", 0.91))
    assert choose_handling(region).handling == "secondary_motion_region"


def test_low_confidence_part_is_preserved_unknown() -> None:
    region = PartRegion.from_raw(part("maybe", "rigid_accessory", 0.55))
    decision = choose_handling(region, separate_props=True)
    assert decision.handling == "leave_unknown"


def test_clean_humanoid_is_a_pose_eligible() -> None:
    result = evaluate_a_pose_eligibility(
        "humanoid",
        humanoid_parts(),
        geometry_metrics={"depth_to_height_ratio": 0.3},
    )
    assert result.eligible is True
    assert result.action == "apply_a_pose"


def test_fused_staff_blocks_automatic_a_pose() -> None:
    result = evaluate_a_pose_eligibility(
        "humanoid_complex_accessories",
        humanoid_parts(include_staff=True),
        geometry_metrics={"depth_to_height_ratio": 0.3},
    )
    assert result.eligible is False
    assert "POSE_FUSED_PROTECTED_PROP" in result.reason_codes


def test_fused_staff_can_remain_fixed_when_proven() -> None:
    result = evaluate_a_pose_eligibility(
        "humanoid_complex_accessories",
        humanoid_parts(include_staff=True),
        geometry_metrics={
            "depth_to_height_ratio": 0.3,
            "protected_props_can_remain_fixed": True,
        },
    )
    assert result.eligible is True


def test_missing_arm_skips_pose() -> None:
    manifest = humanoid_parts()
    manifest["parts"] = [row for row in manifest["parts"] if row["label"] != "right_arm"]
    result = evaluate_a_pose_eligibility(
        "humanoid",
        manifest,
        geometry_metrics={"depth_to_height_ratio": 0.3},
    )
    assert result.eligible is False
    assert "POSE_REQUIRED_PARTS_MISSING" in result.reason_codes


def test_non_humanoid_never_gets_a_pose() -> None:
    result = evaluate_a_pose_eligibility(
        "quadruped",
        humanoid_parts(),
        geometry_metrics={"depth_to_height_ratio": 0.3},
    )
    assert result.eligible is False
    assert "POSE_PROFILE_NOT_HUMANOID" in result.reason_codes


def test_pose_validation_accepts_only_zero_regression() -> None:
    good = {
        "source_hash_unchanged": True,
        "finite": True,
        "fresh_import_validated": True,
        "component_count_regression": 0,
        "topology_regression_edges": 0,
        "root_displacement": 0.0,
        "max_planted_foot_displacement": 0.0,
        "torso_volume_delta_fraction": 0.01,
        "protected_prop_displacement": 0.0,
        "self_intersection_delta": 0,
        "arm_clearance_improved": True,
        "arm_angles_valid": True,
    }
    assert validate_pose_result(good)["passed"] is True
    bad = dict(good, topology_regression_edges=1)
    receipt = validate_pose_result(bad)
    assert receipt["passed"] is False
    assert "POSE_TOPOLOGY_REGRESSION" in receipt["failure_codes"]


def feature(
    identifier: str,
    category: str,
    confidence: float = 0.95,
    *,
    evidence_types: list[str] | None = None,
    auto_enable: bool = True,
    uv_mask: str = "mask.png",
) -> dict:
    return {
        "id": identifier,
        "category": category,
        "subtype": category,
        "confidence": confidence,
        "uv_mask": uv_mask,
        "evidence_types": evidence_types or ["semantic", "mask"],
        "auto_enable": auto_enable,
    }


def test_missing_material_manifest_uses_ordinary_pbr() -> None:
    plan = build_material_plan(None, profile_name="humanoid")
    assert plan["status"] == "ordinary_pbr_fallback"
    assert plan["material_families"] == ["OpaqueLit"]
    assert plan["enabled_feature_count"] == 0


def test_emission_is_not_enabled_from_brightness_only() -> None:
    manifest = load_material_manifest({
        "schema_version": 1,
        "features": [feature("eyes", "emissive", evidence_types=["brightness", "colour"])],
    })
    decision = decide_feature(manifest["features"][0])
    assert decision.enabled is False
    assert "MATERIAL_COLOUR_ONLY_EVIDENCE_REJECTED" in decision.reason_codes


def test_emission_with_independent_evidence_is_enabled() -> None:
    manifest = load_material_manifest({
        "schema_version": 1,
        "features": [feature("rune", "emissive", evidence_types=["semantic", "halo"])],
    })
    decision = decide_feature(manifest["features"][0])
    assert decision.enabled is True
    assert decision.material_family == "OpaqueLit"


def test_medium_confidence_material_is_proposal_only() -> None:
    manifest = load_material_manifest({
        "schema_version": 1,
        "features": [feature("fur", "fur", 0.8)],
    })
    decision = decide_feature(manifest["features"][0])
    assert decision.status == "proposal_only"
    assert decision.enabled is False


def test_soft_surface_uses_masked_family() -> None:
    manifest = load_material_manifest({
        "schema_version": 1,
        "features": [feature("hair", "hair")],
    })
    decision = decide_feature(manifest["features"][0])
    assert decision.material_family == "MaskedSoft"


def test_glass_uses_translucent_family() -> None:
    manifest = load_material_manifest({
        "schema_version": 1,
        "features": [feature("wing", "glass", evidence_types=["semantic", "transparency"])],
    })
    decision = decide_feature(manifest["features"][0])
    assert decision.enabled is True
    assert decision.material_family == "TranslucentSpecial"


def test_duplicate_material_ids_are_rejected() -> None:
    with pytest.raises(MaterialManifestError):
        load_material_manifest({
            "schema_version": 1,
            "features": [feature("x", "cloth"), feature("x", "metal")],
        })


def test_low_vram_card_budgets_are_bounded() -> None:
    budget = soft_surface_budget("humanoid")
    assert budget["lod_card_limits"] == {"0": 2000, "1": 900, "2": 300, "3": 0}
    assert budget["dense_groom_allowed"] is False


def test_integrated_plan_preserves_all_fallbacks_without_manifests() -> None:
    plan = build_plan(profile="humanoid_complex_accessories")
    assert plan["status"] == "ready"
    assert plan["parts"]["status"] == "safe_no_semantic_parts"
    assert plan["pose_prep"]["action"] == "preserve_source_pose"
    assert plan["materials"]["status"] == "ordinary_pbr_fallback"
    assert plan["safe_fallbacks"]["soft_surfaces"] == "disabled"


def test_integrated_plan_accepts_clean_semantic_inputs() -> None:
    materials = {
        "schema_version": 1,
        "features": [
            feature("cloth", "cloth"),
            feature("hair", "hair"),
        ],
    }
    plan = build_plan(
        profile="humanoid",
        parts_manifest=humanoid_parts(),
        material_manifest=materials,
        geometry_metrics={"depth_to_height_ratio": 0.3},
    )
    assert plan["status"] == "ready"
    assert plan["pose_prep"]["eligible"] is True
    assert plan["materials"]["enabled_feature_count"] == 2
