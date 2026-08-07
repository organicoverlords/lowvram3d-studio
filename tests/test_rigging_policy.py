from lowvram3d.rigging_policy import (
    build_rigging_plan,
    evaluate_rig_promotion,
    needs_segmentation_recovery,
    pipeline_stage_order,
)


def _passing_report(plan):
    return {
        "armature_present": True,
        "skin_weights_present": True,
        "materials_preserved": True,
        "peak_vram_mb": 5100,
        "deformation_poses": {pose: {"passed": True} for pose in plan.required_deformation_poses},
    }


def test_humanoid_routes_to_mia_without_pre_segmentation():
    plan = build_rigging_plan("avatar", rig_kind="humanoid")
    assert plan.backend == "mia"
    assert plan.dtype == "fp16"
    assert plan.segmentation_before_rig is False
    assert plan.preserve_textured_lod0 is True
    assert plan.generate_lods_after_rig is True
    assert plan.fallback_backends == ("unirig",)


def test_general_creature_routes_to_puppeteer_sdpa():
    plan = build_rigging_plan("creature")
    assert plan.backend == "puppeteer"
    assert plan.attention_backend == "sdpa"
    assert "skeletal_lod_generation" == pipeline_stage_order(plan)[-1]
    assert pipeline_stage_order(plan).index("deformation_qa") < pipeline_stage_order(plan).index("skeletal_lod_generation")


def test_ambiguous_character_is_not_silently_assumed_humanoid():
    general = build_rigging_plan("character")
    humanoid = build_rigging_plan("character", rig_kind="humanoid")
    assert general.backend == "puppeteer"
    assert humanoid.backend == "mia"


def test_mechanical_assets_keep_rigid_route():
    plan = build_rigging_plan("vehicle", rig_kind="mechanical")
    assert plan.backend == "legacy_rigid"
    assert plan.required_deformation_poses == ()


def test_static_assets_do_not_get_armature():
    plan = build_rigging_plan("building")
    assert plan.backend == "none"
    assert pipeline_stage_order(plan) == ("preserve_textured_lod0", "engine_export")


def test_promotion_fails_closed_on_missing_deformation_pose():
    plan = build_rigging_plan("avatar", rig_kind="humanoid")
    report = _passing_report(plan)
    report["deformation_poses"]["elbow_bend"] = {"passed": False}
    passed, failures = evaluate_rig_promotion(report, plan)
    assert passed is False
    assert "deformation_pose_failed:elbow_bend" in failures


def test_promotion_rejects_vram_ceiling_violation():
    plan = build_rigging_plan("creature", vram_ceiling_mb=5600)
    report = _passing_report(plan)
    report["peak_vram_mb"] = 5601
    passed, failures = evaluate_rig_promotion(report, plan)
    assert passed is False
    assert "vram_ceiling_exceeded" in failures


def test_segmentation_is_only_a_recovery_action():
    assert needs_segmentation_recovery({}) is False
    assert needs_segmentation_recovery({"weight_bleed_detected": True}) is True
    assert needs_segmentation_recovery({"rigid_accessory_requires_isolation": True}) is True
