"""Repository-owned benchmark definitions and mandatory anchor ordering.

Large canonical images and GLB masters live in the local benchmark pack. Small repository
previews are review aids only and must never be selected as generation inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

from .quality_ladder import AssetFamily


PRIMARY_ANCHOR_ID = "antlered_bird_shaman_anchor"
ANCHOR_IDS = (
    PRIMARY_ANCHOR_ID,
    "turbo_bird_high_detail",
    "red_panda_character",
)


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
    priority: int = 100
    required_first: bool = False

    @property
    def anchor(self) -> bool:
        return self.fixture_id in ANCHOR_IDS

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
            "priority": self.priority,
            "required_first": self.required_first,
            "anchor": self.anchor,
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


def _fixture(
    fixture_id: str,
    title: str,
    family: AssetFamily,
    source_image_name: str,
    *,
    group: str,
    prompt_tags: tuple[str, ...],
    stressors: tuple[str, ...],
    required_checks: tuple[str, ...] = COMMON_CHECKS,
    master_reference_name: str = "",
    source_status: str = "local_fixture_required",
    priority: int = 100,
    required_first: bool = False,
) -> BenchmarkFixture:
    return BenchmarkFixture(
        fixture_id=fixture_id,
        title=title,
        family=family,
        source_image_name=source_image_name,
        master_reference_name=master_reference_name,
        group=group,
        source_status=source_status,
        priority=priority,
        required_first=required_first,
        prompt_tags=prompt_tags,
        stressors=stressors,
        required_checks=required_checks,
    )


FIXTURES = (
    _fixture(
        PRIMARY_ANCHOR_ID,
        "Antlered bird-shaman ornament anchor",
        AssetFamily.MIXED,
        "antlered_bird_shaman_anchor.png",
        master_reference_name="antlered_bird_shaman_anchor.glb",
        group="quality_anchors",
        source_status="known_local_reference",
        priority=0,
        required_first=True,
        prompt_tags=("bird mask", "antlers", "staff", "robes", "hanging ornaments"),
        stressors=(
            "large horizontal branch-antler silhouette",
            "many thin cords, hooks, rings and hanging ornaments",
            "staff, fingers, layered cloth and bird-like feet",
            "legitimate detached detail must not be removed as debris",
        ),
        required_checks=CHARACTER_CHECKS + (
            "critical_component_recall_100_percent",
            "staff_and_antler_continuity",
            "ornament_and_cord_recall",
            "source_pose_preserved_for_master_comparison",
            "manual_visual_completeness_gate",
        ),
    ),
    _fixture(
        "turbo_bird_high_detail",
        "Turbo bird high-detail anchor",
        AssetFamily.ORGANIC,
        "turbo_bird.png",
        master_reference_name="turbo_bird_master.glb",
        group="quality_anchors",
        source_status="known_local_reference",
        priority=1,
        prompt_tags=("bird", "feathers", "high detail", "organic"),
        stressors=("approximately 1.8M-face master", "dense feather relief", "thin beak, wing edges and talons"),
        required_checks=COMMON_CHECKS + ("thin_feature_recall", "high_frequency_normal_retention", "master_to_lod_detail_comparison"),
    ),
    _fixture(
        "red_panda_character",
        "Red-panda character with equipment",
        AssetFamily.MIXED,
        "red_panda_character.png",
        master_reference_name="red_panda_master.glb",
        group="quality_anchors",
        source_status="known_local_reference",
        priority=2,
        prompt_tags=("animal character", "fabric", "equipment", "weapon"),
        stressors=("floating reconstruction debris", "legitimate detached accessories", "front-heavy source detail"),
        required_checks=CHARACTER_CHECKS + ("depth_separated_debris_detection",),
    ),
    _fixture("tactical_snow_leopard", "Snow-leopard tactical character", AssetFamily.MIXED, "tactical_snow_leopard.png", group="generated_character_examples", prompt_tags=("snow leopard", "tactical clothing", "rifle", "backpack"), stressors=("white fur beside pale fabric", "thin rifle and straps", "large tail"), required_checks=CHARACTER_CHECKS + ("material_boundary_retention", "tail_silhouette_recall")),
    _fixture("tactical_badger_ghillie", "Badger character in layered ghillie equipment", AssetFamily.MIXED, "tactical_badger_ghillie.png", group="generated_character_examples", prompt_tags=("badger", "ghillie", "layered fabric", "rifle"), stressors=("dense ragged layers", "thin detached-looking strips", "weapon overlap"), required_checks=CHARACTER_CHECKS + ("ragged_edge_recall", "false_debris_rejection")),
    _fixture("tactical_arctic_fox", "Arctic-fox tactical character", AssetFamily.MIXED, "tactical_arctic_fox.png", group="generated_character_examples", prompt_tags=("arctic fox", "black tactical clothing", "weapon", "white tail"), stressors=("high-contrast materials", "thin ears and barrel", "large soft tail"), required_checks=CHARACTER_CHECKS + ("high_contrast_material_separation", "tail_silhouette_recall")),
    _fixture("tactical_wolf_ghillie", "Wolf tactical character with long rifle", AssetFamily.MIXED, "tactical_wolf_ghillie.png", group="generated_character_examples", prompt_tags=("wolf", "ghillie", "long rifle", "backpack"), stressors=("long rifle and suppressor", "layered fur and clothing", "occluded tail"), required_checks=CHARACTER_CHECKS + ("long_thin_feature_recall", "occluded_component_preservation")),
    _fixture("tactical_boar_heavy", "Heavy boar tactical character", AssetFamily.MIXED, "tactical_boar_heavy.png", group="generated_character_examples", prompt_tags=("boar", "heavy tactical gear", "tusks", "drum magazine"), stressors=("thin tusks and ears", "many pouches", "circular magazine"), required_checks=CHARACTER_CHECKS + ("tusk_and_ear_recall", "circular_hard_surface_retention")),
    _fixture("articulated_land_train", "Multi-car hard-surface land train", AssetFamily.HARD_SURFACE, "articulated_land_train.png", group="generated_vehicle_examples", prompt_tags=("vehicle", "land train", "wheels", "cabins"), stressors=("repeated wheels", "long articulated structure", "railings and pipes"), required_checks=COMMON_CHECKS + ("repeated_part_count_retention", "hard_edge_retention", "longitudinal_continuity")),
    _fixture("creature_rider_equipment", "Large creature carrying rider and equipment", AssetFamily.MIXED, "creature_rider_equipment.png", group="generated_creature_examples", prompt_tags=("large creature", "rider", "gear", "long tail"), stressors=("organic body and separate rider", "long limbs and tail", "valid detached gear"), required_checks=COMMON_CHECKS + ("hierarchical_component_preservation", "long_thin_feature_recall", "rider_and_gear_retention")),
    _fixture("steampunk_snapping_turtle", "Steampunk snapping turtle with saddle and shell machinery", AssetFamily.MIXED, "steampunk_snapping_turtle.png", group="generated_creature_examples", prompt_tags=("snapping turtle", "steampunk", "saddle", "rusted machinery"), stressors=("organic and metal boundaries", "thin legs and fixtures", "shell relief"), required_checks=COMMON_CHECKS + ("material_boundary_retention", "thin_feature_recall", "mixed_surface_normal_retention")),
    _fixture("mountain_demigod", "Gigantic mountain demigod creature", AssetFamily.ORGANIC, "mountain_demigod.png", group="generated_creature_examples", prompt_tags=("mountain creature", "long legs", "rock", "barnacles"), stressors=("extreme scale", "long limbs", "irregular attached growths"), required_checks=COMMON_CHECKS + ("long_limb_recall", "silhouette_gap_preservation", "rock_surface_normal_retention")),
    _fixture("eternal_great_tree", "Eternal great tree", AssetFamily.ORGANIC, "eternal_great_tree.png", group="generated_natural_examples", prompt_tags=("ancient tree", "branches", "roots", "foliage"), stressors=("branching topology", "thin twigs and roots", "legitimate foliage components"), required_checks=NATURAL_CHECKS + ("root_recall", "legitimate_foliage_component_preservation")),
    _fixture("open_plan_building", "Large open-plan enterable building", AssetFamily.ARCHITECTURAL, "open_plan_building.png", group="generated_building_examples", prompt_tags=("open plan building", "enterable", "rooms", "large openings"), stressors=("interior and exterior surfaces", "large planes", "stairs and railings"), required_checks=ARCHITECTURE_CHECKS + ("room_connectivity", "wall_thickness_preservation", "stair_and_railing_recall")),
    _fixture("lighthouse_archipelago_fortress", "Lighthouse archipelago fortress", AssetFamily.ARCHITECTURAL, "lighthouse_archipelago_fortress.png", group="generated_building_examples", prompt_tags=("lighthouse", "coastal fortress", "islands", "bridges"), stressors=("disconnected islands", "towers and bridges", "rock and architecture boundaries"), required_checks=ARCHITECTURE_CHECKS + ("multi_island_component_preservation", "bridge_recall", "tower_silhouette_retention")),
    _fixture("river_casino_vessel", "Enterable river-casino vessel", AssetFamily.ARCHITECTURAL, "river_casino_vessel.png", group="generated_building_examples", prompt_tags=("river casino", "wooden ship", "enterable building", "decks"), stressors=("long hull", "open decks", "railings and repeated trim"), required_checks=ARCHITECTURE_CHECKS + ("hull_continuity", "deck_opening_recall", "railing_and_trim_recall")),
    _fixture("fortress_vessel", "Architectural fortress vessel", AssetFamily.ARCHITECTURAL, "fortress_vessel.png", group="generated_building_examples", prompt_tags=("architecture", "ship", "fortress", "towers"), stressors=("planar surfaces", "windows and arches", "tower silhouettes"), required_checks=ARCHITECTURE_CHECKS),
)


def ordered_fixtures() -> tuple[BenchmarkFixture, ...]:
    """Return deterministic priority order; tuple position is not the gate."""
    return tuple(sorted(FIXTURES, key=lambda fixture: (fixture.priority, fixture.fixture_id)))


def fixture_by_id(fixture_id: str) -> BenchmarkFixture:
    for fixture in FIXTURES:
        if fixture.fixture_id == fixture_id:
            return fixture
    raise KeyError(f"unknown benchmark fixture: {fixture_id}")


def manifest() -> dict:
    fixtures = ordered_fixtures()
    return {
        "schema_version": 3,
        "policy": {
            "production_rules_may_not_reference_fixture_ids": True,
            "clean_high_resolution_master_required": True,
            "selection": "lowest candidate passing generic master-similarity gates",
            "manual_review_required": True,
            "large_binary_fixtures_committed": False,
            "fixture_images_resolved_from_local_benchmark_pack": True,
            "primary_anchor": PRIMARY_ANCHOR_ID,
            "primary_anchor_must_be_proven_before_later_fixtures": True,
        },
        "groups": sorted({fixture.group for fixture in fixtures}),
        "fixture_count": len(fixtures),
        "fixtures": [fixture.as_dict() for fixture in fixtures],
    }
