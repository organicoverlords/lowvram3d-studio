"""Repository-owned regression suite definition.

Fixtures describe failure modes that the general pipeline must survive.  They do not introduce
asset-specific cleanup or texture rules; production decisions still come from generic geometry,
visibility, topology, UV and appearance metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

from .quality_ladder import AssetFamily


@dataclass(frozen=True, slots=True)
class BenchmarkFixture:
    fixture_id: str
    title: str
    family: AssetFamily
    source_image_name: str
    prompt_tags: tuple[str, ...]
    stressors: tuple[str, ...]
    required_checks: tuple[str, ...]
    quality: str = "hero"
    master_reference_name: str = ""

    def as_dict(self) -> dict:
        return {
            "fixture_id": self.fixture_id,
            "title": self.title,
            "family": self.family.value,
            "source_image_name": self.source_image_name,
            "master_reference_name": self.master_reference_name or None,
            "prompt_tags": list(self.prompt_tags),
            "stressors": list(self.stressors),
            "required_checks": list(self.required_checks),
            "quality": self.quality,
        }


COMMON_CHECKS = (
    "clean_master_preserved",
    "verified_debris_remaining_zero",
    "meaningful_component_recall",
    "multi_view_silhouette_iou",
    "bidirectional_surface_distance",
    "normal_deviation",
    "boundary_and_non_manifold_regression",
    "uv_positive_area_overlap",
    "uv_utilization_and_fragmentation",
    "observed_and_synthesized_coverage_separate",
    "fresh_glb_reimport",
)


FIXTURES = (
    BenchmarkFixture(
        fixture_id="turbo_bird_high_detail",
        title="Turbo bird high-detail anchor",
        family=AssetFamily.ORGANIC,
        source_image_name="turbo_bird.png",
        master_reference_name="turbo_bird_master.glb",
        prompt_tags=("bird", "feathers", "high detail", "organic"),
        stressors=(
            "approximately 1.8M-face master",
            "dense feather relief",
            "thin beak, wing edges and talons",
            "large visible quality loss under aggressive decimation",
        ),
        required_checks=COMMON_CHECKS + (
            "thin_feature_recall",
            "high_frequency_normal_retention",
            "master_to_lod_detail_comparison",
        ),
    ),
    BenchmarkFixture(
        fixture_id="red_panda_character",
        title="Red-panda character with equipment",
        family=AssetFamily.MIXED,
        source_image_name="red_panda_character.png",
        master_reference_name="red_panda_master.glb",
        prompt_tags=("animal character", "fabric", "equipment", "weapon"),
        stressors=(
            "floating reconstruction debris",
            "legitimate detached accessories",
            "front-heavy source detail",
            "low-confidence rear appearance",
        ),
        required_checks=COMMON_CHECKS + (
            "depth_separated_debris_detection",
            "source_front_detail_retention",
            "unseen_surface_provenance",
        ),
    ),
    BenchmarkFixture(
        fixture_id="humanoid_thin_props",
        title="Humanoid character with thin props and layered clothing",
        family=AssetFamily.MIXED,
        source_image_name="humanoid_thin_props.png",
        prompt_tags=("humanoid", "hair", "layered clothing", "thin weapon"),
        stressors=(
            "face and hair detail",
            "thin blade or tool",
            "fingers, straps and cloth edges",
            "overlapping layered garments",
        ),
        required_checks=COMMON_CHECKS + (
            "thin_feature_recall",
            "face_region_detail_retention",
            "layered_component_preservation",
        ),
    ),
    BenchmarkFixture(
        fixture_id="articulated_land_train",
        title="Multi-car hard-surface land train",
        family=AssetFamily.HARD_SURFACE,
        source_image_name="articulated_land_train.png",
        prompt_tags=("vehicle", "land train", "wheels", "cabins"),
        stressors=(
            "many repeated wheels",
            "long articulated structure",
            "thin railings and pipes",
            "hard edges and planar panels",
        ),
        required_checks=COMMON_CHECKS + (
            "repeated_part_count_retention",
            "hard_edge_retention",
            "longitudinal_continuity",
        ),
    ),
    BenchmarkFixture(
        fixture_id="creature_rider_equipment",
        title="Large creature carrying rider and equipment",
        family=AssetFamily.MIXED,
        source_image_name="creature_rider_equipment.png",
        prompt_tags=("large creature", "rider", "gear", "long tail"),
        stressors=(
            "organic body plus separate rider",
            "long snout, tail and limbs",
            "small gear attached to large body",
            "valid detached components resembling debris",
        ),
        required_checks=COMMON_CHECKS + (
            "hierarchical_component_preservation",
            "long_thin_feature_recall",
            "rider_and_gear_retention",
        ),
    ),
    BenchmarkFixture(
        fixture_id="biomechanical_mount",
        title="Mixed organic-mechanical armored mount",
        family=AssetFamily.MIXED,
        source_image_name="biomechanical_mount.png",
        prompt_tags=("organic mechanical", "armored creature", "saddle", "chains"),
        stressors=(
            "organic shell beside metal parts",
            "thin legs, chains and fixtures",
            "material-boundary preservation",
            "saddle and shell relief",
        ),
        required_checks=COMMON_CHECKS + (
            "material_boundary_retention",
            "thin_feature_recall",
            "mixed_surface_normal_retention",
        ),
    ),
    BenchmarkFixture(
        fixture_id="fortress_vessel",
        title="Architectural fortress vessel",
        family=AssetFamily.ARCHITECTURAL,
        source_image_name="fortress_vessel.png",
        prompt_tags=("architecture", "ship", "fortress", "towers"),
        stressors=(
            "large planar surfaces",
            "windows, arches and openings",
            "towers and roof silhouettes",
            "railings and repeated architectural detail",
        ),
        required_checks=COMMON_CHECKS + (
            "opening_recall",
            "planar_surface_error",
            "architectural_edge_retention",
        ),
    ),
)


def fixture_by_id(fixture_id: str) -> BenchmarkFixture:
    for fixture in FIXTURES:
        if fixture.fixture_id == fixture_id:
            return fixture
    raise KeyError(f"unknown benchmark fixture: {fixture_id}")


def manifest() -> dict:
    return {
        "schema_version": 1,
        "policy": {
            "production_rules_may_not_reference_fixture_ids": True,
            "clean_high_resolution_master_required": True,
            "selection": "lowest candidate passing generic master-similarity gates",
            "manual_review_required": False,
        },
        "fixtures": [fixture.as_dict() for fixture in FIXTURES],
    }
