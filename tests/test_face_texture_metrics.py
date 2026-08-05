import inspect

import numpy as np

from face_texture_refine import (
    _component_stats,
    _direct_observation_percent,
    _leave_one_out,
    _piecewise_affine,
)
from pipeline_v2_production_stages import classify_texture_scope


def test_leave_one_out_residual_is_not_self_fulfilling_zero():
    source = np.asarray([[0, 0], [10, 0], [0, 10], [10, 10], [5, 5]], dtype=np.float32)
    target = np.asarray([[0, 0], [10, 0], [0, 10], [10, 10], [8, 8]], dtype=np.float32)
    errors = _leave_one_out(source, target)
    assert errors.shape == (5,)
    assert float(errors.max()) > 0.0


def test_disconnected_charts_do_not_inflate_contiguous_width():
    mask = np.zeros((32, 32), dtype=bool)
    mask[4:8, 2:6] = True
    mask[20:24, 25:29] = True
    stats, _labels = _component_stats(mask)
    assert stats["component_count"] == 2
    assert stats["largest_component_width"] == 4
    assert stats["largest_component_height"] == 4


def test_direct_observation_uses_intended_denominator():
    intended = np.ones((4, 4), dtype=bool)
    accepted = intended.copy()
    accepted[:2] = False
    assert _direct_observation_percent(accepted, intended) == 50.0


def test_local_warp_declares_filtered_sampling_not_nearest_neighbour():
    source = inspect.getsource(_piecewise_affine)
    assert "INTER_LINEAR" in source
    assert "INTER_AREA" in source
    assert "np.rint" not in source


def test_single_view_cannot_claim_full_360_scope():
    assert classify_texture_scope(
        actual_route="raster_project", semantic_view_count=1,
        synthesized_percent=88.49, face_detail_required=True,
        approved_single_view_face_route=True,
    ) == "FRONT_HERO_PRODUCTION"
    assert classify_texture_scope(
        actual_route="mvadapter_sixview", semantic_view_count=1,
        synthesized_percent=88.49, face_detail_required=True,
        approved_single_view_face_route=True,
    ) != "FULL_360_PRODUCTION"


def test_six_semantic_views_are_required_for_full_360_scope():
    assert classify_texture_scope(
        actual_route="mvadapter_sixview", semantic_view_count=6,
        synthesized_percent=60.0, face_detail_required=False,
        approved_single_view_face_route=False,
    ) == "FULL_360_PRODUCTION"
