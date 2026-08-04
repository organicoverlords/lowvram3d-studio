"""What gets generated, and from which pixels.

These cover the decisions the generation stage makes before any GPU is touched,
because those are the ones that quietly produce a wrong-looking scene: a crop
taken from the wrong window, twelve generations where one would do, or a surface
class handed to a generator that can only make objects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lowvram3d.asset_generation import (  # noqa: E402
    MIN_CROP_ASPECT, _widen_to_aspect, plan)
from lowvram3d.region_placement import place  # noqa: E402


def segmentation(*regions):
    return {"image_dimensions": [1200, 900],
            "camera": {"fov_x_deg": 90.0},
            "regions": list(regions)}


def region(region_id, layer, label, bbox, depth=10.0):
    return {"id": region_id, "layer_type": layer, "semantic_label": label,
            "bbox_norm_xyxy": list(bbox),
            "depth_m": {"near": depth - 1, "median": depth, "far": depth + 1},
            "confidence": 0.8, "observed_fraction": 0.9}


def test_surfaces_are_not_generated():
    """Terrain and water are measured extents, not objects with a silhouette."""
    placement = place(segmentation(
        region("terrain_grass_001", "terrain", "grass", (0.0, 0.7, 1.0, 1.0)),
        region("water_lake_002", "water", "water", (0.0, 0.5, 0.5, 0.7)),
        region("sky_sky_003", "sky", "sky", (0.0, 0.0, 1.0, 0.5)),
    ))
    assert plan(placement) == []


def test_architecture_is_generated_from_its_own_bbox():
    bbox = (0.3, 0.5, 0.8, 0.75)
    placement = place(segmentation(
        region("architecture_house_004", "architecture", "house", bbox)))
    jobs = plan(placement)
    assert len(jobs) == 1
    assert jobs[0]["asset_id"] == "architecture_house_004"
    assert jobs[0]["crop_bbox_norm_xyxy"] == pytest.approx(list(bbox))
    assert jobs[0]["shared_across_instances"] is False


def test_scatter_region_generates_once_for_every_instance():
    """A tree line is one generation reused, not one generation per tree."""
    placement = place(segmentation(
        region("vegetation_tree_005", "vegetation", "tree", (0.0, 0.1, 0.9, 0.7))))
    instances = [a for a in placement["actors"] if a["kind"] == "scatter_instance"]
    assert len(instances) > 1

    jobs = plan(placement)
    assert len(jobs) == 1
    assert jobs[0]["shared_across_instances"] is True
    assert jobs[0]["instance_count"] == len(instances)
    assert len(jobs[0]["actor_indices"]) == len(instances)


def test_scatter_crop_is_a_subject_not_the_whole_line():
    """The crop must be narrower than the region but wide enough to read."""
    region_bbox = (0.0, 0.1, 0.9, 0.7)
    placement = place(segmentation(
        region("vegetation_tree_005", "vegetation", "tree", region_bbox)))
    crop = plan(placement)[0]["crop_bbox_norm_xyxy"]

    assert crop[2] - crop[0] < (region_bbox[2] - region_bbox[0])
    assert crop[0] >= region_bbox[0] and crop[2] <= region_bbox[2]
    aspect = (crop[2] - crop[0]) / (crop[3] - crop[1])
    assert aspect >= MIN_CROP_ASPECT - 1e-9


def test_widening_keeps_the_window_centred_and_inside_bounds():
    widened = _widen_to_aspect([0.46, 0.05, 0.54, 0.72], [0.0, 0.05, 0.92, 0.72])
    assert widened[0] < 0.46 and widened[2] > 0.54
    assert (widened[0] + widened[2]) / 2 == pytest.approx(0.5)
    assert widened[1] == 0.05 and widened[3] == 0.72


def test_widening_never_escapes_the_region():
    """Clamping on one side must not push the window past the other."""
    bounds = [0.0, 0.0, 0.10, 0.9]
    widened = _widen_to_aspect([0.0, 0.0, 0.02, 0.9], bounds)
    assert widened[0] >= bounds[0] and widened[2] <= bounds[2]


def test_widening_leaves_a_healthy_window_alone():
    bbox = [0.3, 0.5, 0.8, 0.75]
    assert _widen_to_aspect(bbox, [0.0, 0.0, 1.0, 1.0]) == bbox


def test_every_generated_actor_is_accounted_for():
    """Each job's actor indices must address real actors of that region."""
    placement = place(segmentation(
        region("architecture_house_004", "architecture", "house", (0.3, 0.5, 0.8, 0.75)),
        region("vegetation_tree_005", "vegetation", "tree", (0.0, 0.1, 0.9, 0.7)),
        region("terrain_grass_001", "terrain", "grass", (0.0, 0.7, 1.0, 1.0)),
    ))
    actors = placement["actors"]
    for job in plan(placement):
        assert job["actor_indices"]
        for index in job["actor_indices"]:
            assert actors[index]["region_id"] == job["region_id"]
