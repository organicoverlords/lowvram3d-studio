"""What a semantic mask can and cannot tell you about instances.

The point of these is the boundary, not the capability. Connected components
separates objects that have background between them, for free, with no model.
It cannot separate objects that touch, and the barn photograph turns out to be
entirely the second case -- so these tests also record why the design notes rank
a real instance model first rather than treating it as a refinement.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from lowvram3d.scene_segmentation import (  # noqa: E402
    MIN_CLUSTER_POINTS, cluster_region_points, connected_components)


def mask(shape, *boxes):
    grid = np.zeros(shape, dtype=bool)
    for y0, x0, y1, x1 in boxes:
        grid[y0:y1, x0:x1] = True
    return grid


def depth_map(shape, forward=10.0):
    """A MoGe-shaped point map: X right, Y down, Z forward."""
    height, width = shape
    xs = np.linspace(-5.0, 5.0, width)[None, :].repeat(height, axis=0)
    ys = np.linspace(-3.0, 3.0, height)[:, None].repeat(width, axis=1)
    return np.stack([xs, ys, np.full(shape, forward)], axis=-1)


def test_objects_with_a_gap_between_them_are_separate():
    """Two trees with sky between them need no model to tell apart."""
    pieces = connected_components(mask((60, 120), (10, 5, 50, 40),
                                       (10, 70, 50, 110)))
    assert len(pieces) == 2


def test_objects_that_touch_are_one_component():
    """The honest limit: a hedge is one mass however many trees are in it."""
    pieces = connected_components(mask((60, 120), (10, 5, 50, 60),
                                       (10, 58, 50, 110)))
    assert len(pieces) == 1


def test_specks_are_not_instances():
    pieces = connected_components(mask((60, 120), (10, 5, 50, 40), (0, 0, 2, 2)))
    assert len(pieces) == 1


def test_separate_objects_are_marked_separable():
    shape = (60, 120)
    observed = mask(shape, (10, 5, 50, 40), (10, 70, 50, 110))
    clusters = cluster_region_points(depth_map(shape), None, observed,
                                     shape[1], shape[0])
    assert len(clusters) == 2
    assert all(c["separable"] for c in clusters)
    assert {c["component_id"] for c in clusters} == {0, 1}


def test_one_mass_is_one_object_however_wide_it_is():
    """The barn photograph is a single tree arching across the frame.

    Subdividing it produced twelve canopy slices, none of which was a subject,
    and every downstream defect followed: a slab from a crop with no silhouette,
    instanced twelve times, then reported as the scene's worst overlap.
    """
    shape = (60, 400)
    observed = mask(shape, (5, 5, 55, 395))
    clusters = cluster_region_points(depth_map(shape), None, observed,
                                     shape[1], shape[0])
    assert len(clusters) == 1
    assert clusters[0]["separable"] is True
    assert clusters[0]["aspect_lateral_over_height"] > 1.0


def test_each_mass_gets_exactly_one_instance():
    shape = (60, 400)
    observed = mask(shape, (5, 5, 55, 190), (5, 210, 55, 395))
    clusters = cluster_region_points(depth_map(shape), None, observed,
                                     shape[1], shape[0], max_clusters=6)
    assert len(clusters) == 2
    assert {c["component_id"] for c in clusters} == {0, 1}


def test_an_empty_mask_yields_nothing():
    shape = (60, 120)
    assert cluster_region_points(depth_map(shape), None,
                                 np.zeros(shape, dtype=bool),
                                 shape[1], shape[0]) == []


def test_a_mask_below_the_minimum_is_not_an_instance():
    assert connected_components(
        np.ones((4, 4), dtype=bool), MIN_CLUSTER_POINTS) == []
