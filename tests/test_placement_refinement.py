"""The placement objective, and the limits it is held to.

Every term here corresponds to a measurement the pipeline already took and
previously only reported. The tests that matter most are the ones asserting the
optimiser is *not* trusted: that it cannot teleport an actor, and that a result
which fails to improve the objective is discarded rather than applied.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from lowvram3d.placement_refinement import (  # noqa: E402
    MAX_SHIFT_PER_AXIS_M, ground_height_m, project_box, refine)

CAMERA = {"fov_x_deg": 92.878, "fov_y_deg": 76.523}
# The real fitted plane from the barn scene, residual included. Leaving the
# residual out makes the ground term falsely certain, which is exactly the
# failure this fixture exists to catch.
PLANE = {"slope_forward": -0.17564, "slope_right": 0.02765,
         "height_at_camera_m": -1.137, "residual_p95_m": 1.186}


def actor(kind, location_m, size_m, bbox=None):
    spec = {"kind": kind, "region_id": f"r_{kind}",
            "location_cm": [v * 100.0 for v in location_m],
            "size_m": list(size_m)}
    if bbox is not None:
        spec["source_bbox_norm_xyxy"] = list(bbox)
    return spec


def _photographed_at(centre, size):
    """The source bbox an actor at this pose would have produced, in CAMERA."""
    import math
    return project_box(
        np.array(centre, dtype=float), np.array(size, dtype=float),
        math.tan(math.radians(CAMERA["fov_x_deg"]) / 2),
        math.tan(math.radians(CAMERA["fov_y_deg"]) / 2))


def test_a_box_straight_ahead_projects_to_the_image_centre():
    box = project_box(np.array([10.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]),
                      1.0, 0.75)
    assert (box[0] + box[2]) / 2 == pytest.approx(0.5)
    assert (box[1] + box[3]) / 2 == pytest.approx(0.5)


def test_projection_puts_right_of_camera_right_of_centre():
    box = project_box(np.array([10.0, 3.0, 0.0]), np.array([1.0, 1.0, 1.0]),
                      1.0, 0.75)
    assert (box[0] + box[2]) / 2 > 0.5


def test_projection_puts_above_camera_above_centre():
    """Image v grows downward, so 'up' must decrease it."""
    box = project_box(np.array([10.0, 0.0, 3.0]), np.array([1.0, 1.0, 1.0]),
                      1.0, 0.75)
    assert (box[1] + box[3]) / 2 < 0.5


def test_a_box_behind_the_camera_does_not_produce_nan():
    box = project_box(np.array([-5.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]),
                      1.0, 0.75)
    assert np.isfinite(box).all()


def test_ground_height_follows_the_fitted_slope():
    assert ground_height_m(PLANE, 10.0, 0.0) < ground_height_m(PLANE, 2.0, 0.0)
    assert ground_height_m(None, 10.0, 0.0) is None


def test_an_actor_floating_above_the_plane_is_brought_down():
    placement = {"actors": [actor("structure", (10.0, 0.0, 6.0), (4.0, 4.0, 4.0))]}
    receipt = refine(placement, PLANE, CAMERA)
    assert receipt["applied"] is True
    settled = placement["actors"][0]["location_cm"][2] / 100.0
    assert settled < 6.0                       # it came down
    assert settled - 2.0 > ground_height_m(PLANE, 10.0, 0.0) - 1.0


def test_two_interpenetrating_solids_are_pushed_apart():
    first = actor("structure", (10.0, 0.0, 0.0), (4.0, 4.0, 4.0))
    second = actor("structure", (10.0, 0.5, 0.0), (4.0, 4.0, 4.0))
    second["region_id"] = "a_different_region"
    placement = {"actors": [first, second]}
    before = _overlap(placement)
    receipt = refine(placement, None, CAMERA)
    assert receipt["applied"] is True
    assert _overlap(placement) < before


def _overlap(placement):
    from lowvram3d.placement_refinement import _overlap_volume
    a, b = placement["actors"]
    return _overlap_volume(
        np.array(a["location_cm"]) / 100.0, np.array(a["size_m"]),
        np.array(b["location_cm"]) / 100.0, np.array(b["size_m"]))


def test_the_ground_is_never_pushed_out_from_under_the_scene():
    """Terrain overlaps everything by design; penalising that lifts the world."""
    placement = {"actors": [
        actor("ground_plane", (10.0, 0.0, -2.0), (40.0, 40.0, 0.5)),
        actor("structure", (10.0, 0.0, 0.0), (4.0, 4.0, 4.0)),
    ]}
    refine(placement, PLANE, CAMERA)
    ground = placement["actors"][0]["location_cm"]
    assert ground[2] / 100.0 == pytest.approx(-2.0, abs=0.5)


def test_no_actor_is_moved_further_than_the_cap():
    """A metre of correction is a fix; ten is a fabrication."""
    placement = {"actors": [
        actor("structure", (10.0, 0.0, 40.0), (4.0, 4.0, 4.0))]}
    receipt = refine(placement, PLANE, CAMERA)
    assert receipt["max_shift_per_axis_m"] <= MAX_SHIFT_PER_AXIS_M + 1e-6
    assert receipt["at_cap_count"] >= 1


def test_reprojection_holds_an_actor_where_it_was_photographed():
    """Ground contact alone would drag an actor down; the pixels say otherwise."""
    high = actor("structure", (10.0, 0.0, 5.0), (2.0, 2.0, 2.0),
                 bbox=list(_photographed_at((10.0, 0.0, 5.0), (2.0, 2.0, 2.0))))
    placement = {"actors": [high]}
    refine(placement, PLANE, CAMERA)
    # The plane at 10 m is near -2.9 m, so ground contact alone wants this down
    # by nearly seven metres. The pixels win: an object can sit on a hill, on a
    # roof, or where the plane's extrapolation is simply wrong at that distance.
    assert placement["actors"][0]["location_cm"][2] / 100.0 > 4.0


def test_a_confident_plane_pulls_harder_than_a_vague_one():
    """The ground term's authority is the fit's own residual, not a constant."""
    def settle(residual):
        placement = {"actors": [actor("structure", (10.0, 0.0, 5.0), (2.0, 2.0, 2.0),
                                      bbox=list(_photographed_at(
                                          (10.0, 0.0, 5.0), (2.0, 2.0, 2.0))))]}
        refine(placement, {**PLANE, "residual_p95_m": residual}, CAMERA)
        return placement["actors"][0]["location_cm"][2] / 100.0

    assert settle(0.1) < settle(3.0)


def test_an_empty_scene_is_not_an_error():
    receipt = refine({"actors": []}, PLANE, CAMERA)
    assert receipt["classification"] == "NOT_APPLICABLE"


def test_the_receipt_reports_the_objective_it_achieved():
    placement = {"actors": [actor("structure", (10.0, 0.0, 6.0), (4.0, 4.0, 4.0))]}
    receipt = refine(placement, PLANE, CAMERA)
    assert receipt["cost_after"] < receipt["cost_before"]
    assert receipt["ground_plane_used"] is True
    assert set(receipt["tolerances"]) == {
        "reprojection_norm", "ground_m", "penetration_fraction",
        "measured_anchor_m"}
