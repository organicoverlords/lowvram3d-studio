"""Semantic six-view orientation must be proven, never declared."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from build_mvadapter_cpu_controls import (  # noqa: E402
    FRAMING_OCCUPANCY_MAX,
    FRAMING_OCCUPANCY_MIN,
    PROJECTION_HALF_SPAN,
    PROJECTION_SPAN,
    _prove_semantic_orientation,
    build_camera_contract,
)
from mvadapter_orientation_fixture import (  # noqa: E402
    AXIS_OCCLUDED_COMPONENT,
    AXIS_SIGNATURE_COMPONENT,
    COMPONENT_NAMES,
    _is_asymmetric,
    build_fixture,
    normalise,
)


def test_fixture_has_six_distinct_named_components() -> None:
    fixture = build_fixture()
    for axis, component in AXIS_SIGNATURE_COMPONENT.items():
        assert component in COMPONENT_NAMES, axis
    assert fixture["triangle_count"] > 0
    assert len(fixture["component_of_triangle"]) == fixture["triangle_count"]
    assert len(set(fixture["component_of_triangle"].tolist())) == len(COMPONENT_NAMES)


def test_fixture_is_not_rotationally_or_reflectionally_symmetric() -> None:
    fixture = build_fixture()
    assert _is_asymmetric(fixture["vertices"]) is True
    # A symmetric point cloud must be rejected by the same check.
    cube = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=np.float64,
    )
    assert _is_asymmetric(cube) is False


def test_fixture_normalises_into_the_official_half_span() -> None:
    vertices = normalise(build_fixture()["vertices"])
    assert pytest.approx(0.5, abs=1e-9) == float(np.max(np.abs(vertices)))
    assert PROJECTION_HALF_SPAN == 0.55
    assert PROJECTION_SPAN == pytest.approx(1.10)


def test_semantic_index_mapping_is_read_out_of_rendered_geometry() -> None:
    contract = build_camera_contract()
    assert contract["index_semantics"] == {
        "0": "front",
        "1": "right",
        "2": "rear",
        "3": "left",
        "4": "top",
        "5": "bottom",
    }
    assert contract["index_semantics"]["4"] == "top"
    assert contract["index_semantics"]["5"] == "bottom"


def test_every_view_proves_its_signature_component_and_occlusion() -> None:
    contract = build_camera_contract()
    evidence = contract["fixture_evidence"]["evidence"]
    assert len(evidence) == 6
    for record in evidence:
        axis = record["axis_label"]
        assert record["measured_closest_component"] == AXIS_SIGNATURE_COMPONENT[axis]
        assert AXIS_OCCLUDED_COMPONENT[axis] not in record["measured_visible_components"]
        assert record["signature_passed"] is True
        assert record["occlusion_passed"] is True
        assert record["passed"] is True
        assert record["component_pixels"], axis


def test_handedness_and_top_bottom_rotation_are_proven() -> None:
    contract = build_camera_contract()
    assert contract["handedness_proven"] is True
    assert contract["top_rotation_proven"] is True
    assert contract["bottom_rotation_proven"] is True
    assert contract["top_bottom_rotation_proven"] is True
    assert contract["fixture_evidence"]["handedness_checks"] >= 4


def test_image_side_placements_are_measured_not_assumed() -> None:
    contract = build_camera_contract()
    placements = [
        placement
        for record in contract["fixture_evidence"]["evidence"]
        for placement in record["image_side_placements"]
    ]
    assert placements, "no image-side placement evidence was recorded"
    checked_x = [p for p in placements if "expected_image_side_x" in p]
    checked_y = [p for p in placements if "expected_image_side_y" in p]
    assert checked_x and checked_y
    for placement in checked_x:
        assert placement["measured_image_side_x"] == placement["expected_image_side_x"]
    for placement in checked_y:
        assert placement["measured_image_side_y"] == placement["expected_image_side_y"]


def test_top_and_bottom_views_use_the_official_near_pole_horizontal_basis() -> None:
    """The near-pole camera utility determines the horizontal basis."""
    contract = build_camera_contract()
    evidence = {record["index"]: record for record in contract["fixture_evidence"]["evidence"]}
    sides = {}
    for index in (4, 5):
        for placement in evidence[index]["image_side_placements"]:
            if placement["component"] == "right_fin":
                sides[index] = placement["measured_image_side_x"]
    assert set(sides) == {4, 5}
    assert sides[4] == "image_left"
    assert sides[5] == "image_left"


def test_proof_flags_are_not_unconditional() -> None:
    """Corrupting the declared semantics must fail the gate, not pass it."""
    contract = build_camera_contract()
    views = copy.deepcopy(contract["views"])
    # Claim that index 0 shows the rear - the render says otherwise.
    views[0]["axis_label"] = "rear"
    views[2]["axis_label"] = "front"
    with pytest.raises(RuntimeError, match="CAMERA_CONTRACT_ASYMMETRIC_FIXTURE_FAILED"):
        _prove_semantic_orientation(views)


def test_flipped_camera_up_fails_the_rotation_gate() -> None:
    contract = build_camera_contract()
    views = copy.deepcopy(contract["views"])
    views[4]["camera_up"] = [-value for value in views[4]["camera_up"]]
    with pytest.raises(RuntimeError, match="CAMERA_CONTRACT_ASYMMETRIC_FIXTURE_FAILED"):
        _prove_semantic_orientation(views)


def test_framing_gate_constants_match_the_official_contract() -> None:
    assert FRAMING_OCCUPANCY_MIN == 0.89
    assert FRAMING_OCCUPANCY_MAX == 0.93
    assert PROJECTION_SPAN == pytest.approx(1.10)
    assert PROJECTION_HALF_SPAN == pytest.approx(0.55)
