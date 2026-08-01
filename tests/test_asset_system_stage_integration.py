from __future__ import annotations

from types import SimpleNamespace

from workers.pipeline_v2_asset_system_stages import (
    ASSET_SYSTEM_STAGES,
    plan_variant,
    register_asset_system_stages,
)


def base_plan() -> dict:
    return {
        "status": "ready",
        "parts": {
            "status": "planned",
            "part_count": 1,
            "parts": [{"id": "staff"}],
            "decisions": [{"part_id": "staff", "handling": "protected_fused_region"}],
        },
        "pose_prep": {
            "eligible": True,
            "action": "apply_a_pose",
            "reason_codes": [],
            "specification": {"arm_from_torso_degrees": 40.0},
        },
        "materials": {
            "status": "planned",
            "feature_count": 1,
            "features": [{"id": "hair"}],
            "decisions": [{"feature_id": "hair", "enabled": True}],
        },
    }


def test_asset_system_stage_order_is_explicit() -> None:
    assert ASSET_SYSTEM_STAGES == [
        "INGEST",
        "GENERATE",
        "GEOMETRY_QA",
        "CLEAN",
        "LOD",
        "UV",
        "BAKE",
        "TEXTURE",
        "MATERIALS",
        "TEXTURE_QA",
        "PARTS",
        "POSE_PREP",
        "RIG_READINESS",
        "RIG",
        "EXPORT",
    ]


def test_material_variant_cannot_apply_parts_or_pose() -> None:
    variant = plan_variant(base_plan(), "MATERIALS")
    assert variant["materials"]["status"] == "planned"
    assert variant["parts"]["status"] == "safe_no_semantic_parts"
    assert variant["parts"]["decisions"] == []
    assert variant["pose_prep"]["action"] == "preserve_source_pose"


def test_parts_variant_cannot_apply_materials_or_pose() -> None:
    variant = plan_variant(base_plan(), "PARTS")
    assert variant["parts"]["status"] == "planned"
    assert variant["materials"]["status"] == "ordinary_pbr_fallback"
    assert variant["materials"]["decisions"] == []
    assert variant["pose_prep"]["action"] == "preserve_source_pose"


def test_pose_variant_retains_parts_but_disables_materials() -> None:
    variant = plan_variant(base_plan(), "POSE_PREP")
    assert variant["parts"]["status"] == "planned"
    assert variant["pose_prep"]["action"] == "apply_a_pose"
    assert variant["materials"]["status"] == "ordinary_pbr_fallback"
    assert variant["materials"]["decisions"] == []


def test_registration_adds_and_overrides_required_stages(tmp_path) -> None:
    profile = SimpleNamespace(
        name="humanoid_complex_accessories",
        separate_props=True,
        rig_required=True,
    )
    pipeline = SimpleNamespace(profile=profile)
    manifest = {
        "output_root": str(tmp_path / "asset"),
        "asset_id": "fixture",
        "source": {"path": str(tmp_path / "source.png")},
    }
    sentinel = lambda: None
    existing = {
        "TEXTURE_QA": sentinel,
        "PARTS": sentinel,
        "RIG_READINESS": sentinel,
        "RIG": sentinel,
        "EXPORT": sentinel,
    }
    registered = register_asset_system_stages(pipeline, manifest, existing)
    for name in ("MATERIALS", "TEXTURE_QA", "PARTS", "POSE_PREP", "RIG_READINESS", "RIG"):
        assert name in registered
        assert callable(registered[name])
    assert registered["EXPORT"] is sentinel
    assert registered["TEXTURE_QA"] is not sentinel
    assert registered["PARTS"] is not sentinel
