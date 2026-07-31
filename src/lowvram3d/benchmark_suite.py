"""Repository-owned regression-suite definitions.

Fixtures describe failure modes the general pipeline must survive. They never introduce
asset-specific cleanup, UV, LOD, or texture rules. Production decisions remain generic and are
based on geometry, visibility, topology, UV quality, material evidence, and master similarity.

The source images and large GLB masters live in a local benchmark pack and are intentionally not
committed to the repository. ``source_image_name`` and ``master_reference_name`` are stable lookup
names used by the benchmark runner.
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
    group: str = "general"
    source_status: str = "local_fixture_required"

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
            "group": self.group,
            "source_status": self.source_status,
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

CHARACTER_CHECKS = COMMON_CHECKS + (
    "face_region_detail_retention",
    "thin_feature_recall",
    "layered_component_preservation",
    "source_front_detail_retention",
    "unseen_surface_provenance",
)

ARCHITECTURE_CHECKS = COMMON_CHECKS + (
    "opening_recall",
    "planar_surface_error",
    "architectural_edge_retention",
    "interior_access_preservation",
)

NATURAL_CHECKS = COMMON_CHECKS + (
    "high_frequency_normal_retention",
    "thin_feature_recall",
    "branch_or_foliage_recall",
)


FIXTURES = (
    BenchmarkFixture(
        fixture_id="turbo_bird_high_detail",
        title="Turbo bird high-detail anchor",
        family=AssetFamily.ORGANIC,
        source_image_name="turbo_bird.png",
        master_reference_name="turbo_bird_master.glb",
        group="quality_anchors",
        source_status="known_local_reference",
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
        group="quality_anchors",
        source_status="known_local_reference",
        prompt_tags=("animal character", "fabric", "equipment", "weapon"),
        stressors=(
            "floating reconstruction debris",
            "legitimate detached accessories",
            "front-heavy source detail",
            "low-confidence rear appearance",
        ),
        required_checks=CHARACTER_CHECKS + (
            "depth_separated_debris_detection",
        ),
    ),
    BenchmarkFixture(
        fixture_id="tactical_snow_leopard",
        title="Snow-leopard tactical character",
        family=AssetFamily.MIXED,
        source_image_name="tactical_snow_leopard.png",
        group="generated_character_examples",
        prompt_tags=("snow leopard", "tactical clothing", "rifle", "backpack"),
        stressors=(
            "white patterned fur beside pale fabric",
            "face and ear silhouette",
            "thin rifle, straps and buckles",
            "large tail crossing behind legs",
        ),
        required_checks=CHARACTER_CHECKS + (
            "material_boundary_retention",
            "tail_silhouette_recall",
        ),
    ),
    BenchmarkFixture(
        fixture_id="tactical_badger_ghillie",
        title="Badger character in layered ghillie equipment",
        family=AssetFamily.MIXED,
        source_image_name="tactical_badger_ghillie.png",
        group="generated_character_examples",
        prompt_tags=("badger", "ghillie", "layered fabric", "rifle"),
        stressors=(
            "dense ragged layers",
            "dark face surrounded by similar dark material",
            "many legitimate thin detached-looking strips",
            "weapon and body overlap",
        ),
        required_checks=CHARACTER_CHECKS + (
            "ragged_edge_recall",
            "false_debris_rejection",
        ),
    ),
    BenchmarkFixture(
        fixture_id="tactical_arctic_fox",
        title="Arctic-fox tactical character",
        family=AssetFamily.MIXED,
        source_image_name="tactical_arctic_fox.png",
        group="generated_character_examples",
        prompt_tags=("arctic fox", "black tactical clothing", "weapon", "white tail"),
        stressors=(
            "very bright fur beside near-black equipment",
            "clean smooth clothing surfaces",
            "thin ears and rifle barrel",
            "large soft tail with weak geometric edge",
        ),
        required_checks=CHARACTER_CHECKS + (
            "high_contrast_material_separation",
            "tail_silhouette_recall",
        ),
    ),
    BenchmarkFixture(
        fixture_id="tactical_wolf_ghillie",
        title="Wolf tactical character with long rifle",
        family=AssetFamily.MIXED,
        source_image_name="tactical_wolf_ghillie.png",
        group="generated_character_examples",
        prompt_tags=("wolf", "ghillie", "long rifle", "backpack"),
        stressors=(
            "long rifle and suppressor",
            "layered clothing and fur boundaries",
            "tail partly occluded by body",
            "small pouches and straps",
        ),
        required_checks=CHARACTER_CHECKS + (
            "long_thin_feature_recall",
            "occluded_component_preservation",
        ),
    ),
    BenchmarkFixture(
        fixture_id="tactical_boar_heavy",
        title="Heavy boar tactical character",
        family=AssetFamily.MIXED,
        source_image_name="tactical_boar_heavy.png",
        group="generated_character_examples",
        prompt_tags=("boar", "heavy tactical gear", "tusks", "drum magazine"),
        stressors=(
            "tusks and ears as thin protrusions",
            "large body under many pouches",
            "circular magazine and sling",
            "dark low-frequency material palette",
        ),
        required_checks=CHARACTER_CHECKS + (
            "tusk_and_ear_recall",
            "circular_hard_surface_retention",
        ),
    ),
    BenchmarkFixture(
        fixture_id="articulated_land_train",
        title="Multi-car hard-surface land train",
        family=AssetFamily.HARD_SURFACE,
        source_image_name="articulated_land_train.png",
        group="generated_vehicle_examples",
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
        group="generated_creature_examples",
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
        fixture_id="steampunk_snapping_turtle",
        title="Steampunk snapping turtle with saddle and shell machinery",
        family=AssetFamily.MIXED,
        source_image_name="steampunk_snapping_turtle.png",
        group="generated_creature_examples",
        prompt_tags=("snapping turtle", "steampunk", "saddle", "rusted machinery"),
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
        fixture_id="mountain_demigod",
        title="Gigantic mountain demigod creature",
        family=AssetFamily.ORGANIC,
        source_image_name="mountain_demigod.png",
        group="generated_creature_examples",
        prompt_tags=("mountain creature", "long legs", "rock", "barnacles"),
        stressors=(
            "extreme scale and long limbs",
            "rock-like high-frequency surface",
            "thin silhouette gaps between legs",
            "irregular attached growths resembling debris",
        ),
        required_checks=COMMON_CHECKS + (
            "long_limb_recall",
            "silhouette_gap_preservation",
            "rock_surface_normal_retention",
        ),
    ),
    BenchmarkFixture(
        fixture_id="eternal_great_tree",
        title="Eternal great tree",
        family=AssetFamily.ORGANIC,
        source_image_name="eternal_great_tree.png",
        group="generated_natural_examples",
        prompt_tags=("ancient tree", "branches", "roots", "foliage"),
        stressors=(
            "branching topology",
            "many thin twigs and root tips",
            "large trunk beside tiny features",
            "foliage masses that may be disconnected legitimately",
        ),
        required_checks=NATURAL_CHECKS + (
            "root_recall",
            "legitimate_foliage_component_preservation",
        ),
    ),
    BenchmarkFixture(
        fixture_id="open_plan_building",
        title="Large open-plan enterable building",
        family=AssetFamily.ARCHITECTURAL,
        source_image_name="open_plan_building.png",
        group="generated_building_examples",
        prompt_tags=("open plan building", "enterable", "rooms", "large openings"),
        stressors=(
            "exterior and interior surfaces visible together",
            "large wall and floor planes",
            "door, window and stair openings",
            "thin railings and structural supports",
        ),
        required_checks=ARCHITECTURE_CHECKS + (
            "room_connectivity",
            "wall_thickness_preservation",
            "stair_and_railing_recall",
        ),
    ),
    BenchmarkFixture(
        fixture_id="lighthouse_archipelago_fortress",
        title="Lighthouse archipelago fortress",
        family=AssetFamily.ARCHITECTURAL,
        source_image_name="lighthouse_archipelago_fortress.png",
        group="generated_building_examples",
        prompt_tags=("lighthouse", "coastal fortress", "islands", "bridges"),
        stressors=(
            "several legitimate disconnected islands",
            "towers, roofs, bridges and stairs",
            "rock and architecture material boundaries",
            "large water-adjacent negative spaces",
        ),
        required_checks=ARCHITECTURE_CHECKS + (
            "multi_island_component_preservation",
            "bridge_recall",
            "tower_silhouette_retention",
        ),
    ),
    BenchmarkFixture(
        fixture_id="river_casino_vessel",
        title="Enterable river-casino vessel",
        family=AssetFamily.ARCHITECTURAL,
        source_image_name="river_casino_vessel.png",
        group="generated_building_examples",
        prompt_tags=("river casino", "wooden ship", "enterable building", "decks"),
        stressors=(
            "ship hull plus multi-level building",
            "open decks, doors and windows",
            "railings, chimneys and repeated trim",
            "interior access and long hull continuity",
        ),
        required_checks=ARCHITECTURE_CHECKS + (
            "hull_continuity",
            "deck_opening_recall",
            "railing_and_trim_recall",
        ),
    ),
    BenchmarkFixture(
        fixture_id="fortress_vessel",
        title="Architectural fortress vessel",
        family=AssetFamily.ARCHITECTURAL,
        source_image_name="fortress_vessel.png",
        group="generated_building_examples",
        prompt_tags=("architecture", "ship", "fortress", "towers"),
        stressors=(
            "large planar surfaces",
            "windows, arches and openings",
            "towers and roof silhouettes",
            "railings and repeated architectural detail",
        ),
        required_checks=ARCHITECTURE_CHECKS,
    ),
)


def fixture_by_id(fixture_id: str) -> BenchmarkFixture:
    for fixture in FIXTURES:
        if fixture.fixture_id == fixture_id:
            return fixture
    raise KeyError(f"unknown benchmark fixture: {fixture_id}")


def manifest() -> dict:
    groups = sorted({fixture.group for fixture in FIXTURES})
    return {
        "schema_version": 2,
        "policy": {
            "production_rules_may_not_reference_fixture_ids": True,
            "clean_high_resolution_master_required": True,
            "selection": "lowest candidate passing generic master-similarity gates",
            "manual_review_required": False,
            "large_binary_fixtures_committed": False,
            "fixture_images_resolved_from_local_benchmark_pack": True,
        },
        "groups": groups,
        "fixture_count": len(FIXTURES),
        "fixtures": [fixture.as_dict() for fixture in FIXTURES],
    }
