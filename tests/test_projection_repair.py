import numpy as np
import pytest

from workers.projection_repair import (
    facial_source_region,
    gated_sample_mask,
    rear_face_provenance_violations,
)


def test_sample_gate_requires_every_condition():
    result = gated_sample_mask(
        depth_visible=True,
        facing_score=0.8,
        face_id_match=np.array([True, True, False, True]),
        source_mask_valid=np.array([True, False, True, True]),
        confidence=np.array([0.9, 0.9, 0.9, np.nan], dtype=np.float32),
    )
    assert result.tolist() == [True, False, False, False]


@pytest.mark.parametrize("depth,facing", [(False, 0.8), (True, 0.1), (True, np.nan)])
def test_triangle_gate_fails_closed(depth, facing):
    result = gated_sample_mask(
        depth_visible=depth,
        facing_score=facing,
        face_id_match=np.array([True]),
        source_mask_valid=np.array([True]),
        confidence=np.array([1.0], dtype=np.float32),
    )
    assert not result.any()


def test_facial_region_is_inside_foreground_and_central():
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:10, 2:10] = True
    region = facial_source_region(mask)
    assert region.any()
    assert not region[9].any()
    assert not region[:, 2].any()
    assert np.all(~region | mask)


def test_rear_facial_provenance_is_rejected():
    violations = rear_face_provenance_violations(
        np.array([True, True, False, True]),
        np.array([0, -1, 0, 2]),
        np.array([True, False, True, False]),
    )
    assert violations.tolist() == [True, False, False, False]


def test_provenance_shape_mismatch_fails_closed():
    with pytest.raises(ValueError):
        rear_face_provenance_violations(
            np.array([True]), np.array([-1, -1]), np.array([False])
        )
