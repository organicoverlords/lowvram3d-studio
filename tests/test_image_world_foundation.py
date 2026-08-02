import json
from pathlib import Path

import pytest

from lowvram3d.image_world.classifier import InputSignals, classify_input
from lowvram3d.image_world.contracts import (
    BackendReceipt,
    CameraObservation,
    ContractError,
    ImageWorldRoute,
    ObservationPackage,
    RouteDecision,
    SourceImageIdentity,
)
from lowvram3d.image_world.profiles import QUALITY_PROFILES, gtx1660_super_6gb
from lowvram3d.image_world.stage_graph import build_plan


HASH = "a" * 64


def make_package():
    return ObservationPackage(
        source=SourceImageIdentity("vista.png", HASH, 1448, 1086, False),
        route=RouteDecision(
            selected=ImageWorldRoute.PERSPECTIVE_VISTA,
            confidence=0.7,
            alternatives={
                ImageWorldRoute.PERSPECTIVE_VISTA: 0.7,
                ImageWorldRoute.DIORAMA_MAP: 0.2,
                ImageWorldRoute.ISOLATED_ASSET: 0.05,
                ImageWorldRoute.COMPOSITE_SCENE: 0.05,
            },
            reasons=("perspective landscape evidence",),
        ),
        geometry={"point_map": "geometry/points.exr"},
        masks={"terrain": "masks/terrain.png"},
        camera=CameraObservation(
            intrinsics=((1.0, 0.0, 0.5), (0.0, 1.0, 0.5), (0.0, 0.0, 1.0)),
            world_up=(0.0, 0.0, 1.0),
            estimated_fov_degrees=56.0,
            confidence=0.8,
        ),
        world={"width_m": 4096, "water_level_m": 0.0},
        provenance=(BackendReceipt("moge", "2", model="vits-normal", precision="fp16"),),
        observed_fraction=0.37,
        completed_fraction=0.63,
        seed=42,
    )


def test_observation_package_round_trips_json():
    original = make_package()
    assert ObservationPackage.from_json(original.to_json()) == original


def test_artifact_paths_must_be_package_relative():
    package = make_package()
    invalid = ObservationPackage(
        source=package.source,
        route=package.route,
        geometry={"point_map": "../escape.exr"},
    )
    with pytest.raises(ContractError, match="package-relative"):
        invalid.validate()


def test_observed_and_completed_fractions_cannot_exceed_one():
    package = make_package()
    invalid = ObservationPackage(
        source=package.source,
        route=package.route,
        observed_fraction=0.8,
        completed_fraction=0.3,
    )
    with pytest.raises(ContractError, match="cannot sum above one"):
        invalid.validate()


def test_selected_route_must_be_highest_score():
    decision = RouteDecision(
        selected=ImageWorldRoute.ISOLATED_ASSET,
        confidence=0.4,
        alternatives={
            ImageWorldRoute.ISOLATED_ASSET: 0.4,
            ImageWorldRoute.DIORAMA_MAP: 0.6,
        },
    )
    with pytest.raises(ContractError, match="highest"):
        decision.validate()


def test_transparent_tree_routes_to_isolated_asset():
    result = classify_input(InputSignals(
        has_alpha=True,
        transparent_border_fraction=0.72,
        foreground_edge_touch_fraction=0.02,
        semantic_group_count=1,
    ))
    assert result.selected is ImageWorldRoute.ISOLATED_ASSET


def test_planar_board_routes_to_diorama():
    result = classify_input(InputSignals(
        board_plane_confidence=0.95,
        top_down_confidence=0.88,
        horizon_confidence=0.05,
        layout_coverage=0.9,
        semantic_group_count=3,
    ))
    assert result.selected is ImageWorldRoute.DIORAMA_MAP


def test_archipelago_routes_to_perspective_vista():
    result = classify_input(InputSignals(
        horizon_confidence=0.92,
        perspective_strength=0.9,
        sky_fraction=0.25,
        water_fraction=0.35,
        semantic_group_count=3,
    ))
    assert result.selected is ImageWorldRoute.PERSPECTIVE_VISTA


def test_occluded_tree_and_shack_routes_to_composite_scene():
    result = classify_input(InputSignals(
        horizon_confidence=0.75,
        perspective_strength=0.82,
        semantic_group_count=5,
        occlusion_fraction=0.72,
        sky_fraction=0.2,
    ))
    assert result.selected is ImageWorldRoute.COMPOSITE_SCENE


def test_ambiguous_signal_requires_review():
    assert classify_input(InputSignals(), review_threshold=0.6).manual_review_required


def test_every_route_builds_a_valid_plan():
    for route in ImageWorldRoute:
        plan = build_plan(route)
        plan.validate()
        assert plan.stages[0].stage_id == "source_ingest"


def test_perspective_route_separates_landmarks_before_reprojection():
    ids = [stage.stage_id for stage in build_plan(ImageWorldRoute.PERSPECTIVE_VISTA).stages]
    assert ids.index("landmark_extraction") < ids.index("visible_terrain_reprojection")
    assert ids.index("world_validation") < ids.index("unreal_world_package")


def test_composite_route_stops_at_review_package():
    ids = {stage.stage_id for stage in build_plan(ImageWorldRoute.COMPOSITE_SCENE).stages}
    assert "composite_review_package" in ids
    assert "unreal_world_package" not in ids


def test_experimental_completion_is_opt_in():
    stable = build_plan(ImageWorldRoute.DIORAMA_MAP)
    experimental = build_plan(ImageWorldRoute.DIORAMA_MAP, include_experimental=True)
    assert "experimental_terrain_completion" not in {stage.stage_id for stage in stable.stages}
    assert experimental.stage("experimental_terrain_completion").optional


def test_target_hardware_keeps_one_heavy_gpu_worker():
    profile = gtx1660_super_6gb()
    assert profile.max_gpu_memory_mb == 5600
    assert profile.heavy_gpu_workers == 1
    assert QUALITY_PROFILES["preview"].terrain_resolution == 513
    assert QUALITY_PROFILES["production"].terrain_resolution == 1009


def test_image_world_fixture_manifest_has_unique_verified_identities():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "benchmarks/image_world/manifest.json").read_text(encoding="utf-8")
    )
    fixtures = manifest["fixtures"]
    ids = [fixture["fixture_id"] for fixture in fixtures]
    hashes = [fixture["sha256"] for fixture in fixtures]
    assert len(fixtures) == 9
    assert len(ids) == len(set(ids))
    assert len(hashes) == len(set(hashes))
    assert all(len(value) == 64 for value in hashes)
    assert {fixture["expected_route"] for fixture in fixtures} == {
        route.value for route in ImageWorldRoute
    }
    assert all(fixture["required_proofs"] for fixture in fixtures)
