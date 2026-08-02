import numpy as np
import pytest

from lowvram3d.image_world.contracts import ContractError
from lowvram3d.image_world.semantic_masks import (
    SEMANTIC_CLASSES,
    build_semantic_mask_set,
    mask_report,
)


def probabilities(shape=(4, 5)):
    values = {name: np.zeros(shape, dtype=np.float32) for name in SEMANTIC_CLASSES}
    values["terrain"][:] = 0.9
    return values


def test_clear_terrain_is_accepted():
    result = build_semantic_mask_set(probabilities())
    assert result.terrain_candidate.all()
    assert not result.unresolved.any()
    assert np.all(result.class_index == 0)


def test_water_excludes_high_terrain_score():
    values = probabilities()
    values["water"][1, 2] = 0.8
    result = build_semantic_mask_set(values)
    assert not result.terrain_candidate[1, 2]
    assert result.unresolved[1, 2]


def test_structure_and_vegetation_never_become_terrain_by_default():
    values = probabilities()
    values["structure"][0, 0] = 0.65
    values["vegetation"][0, 1] = 0.7
    result = build_semantic_mask_set(values)
    assert not result.terrain_candidate[0, 0]
    assert not result.terrain_candidate[0, 1]


def test_low_margin_remains_unresolved():
    values = probabilities()
    values["terrain"][2, 3] = 0.61
    values["residual"][2, 3] = 0.55
    result = build_semantic_mask_set(values)
    assert result.unresolved[2, 3]


def test_invalid_pixels_are_not_promoted():
    valid = np.ones((4, 5), dtype=bool)
    valid[3, 4] = False
    result = build_semantic_mask_set(probabilities(), valid_mask=valid)
    assert not result.terrain_candidate[3, 4]
    assert not result.unresolved[3, 4]


def test_mismatched_or_nonfinite_maps_fail_closed():
    values = probabilities()
    values["water"] = np.zeros((3, 5), dtype=np.float32)
    with pytest.raises(ContractError, match="mismatched"):
        build_semantic_mask_set(values)

    values = probabilities()
    values["sky"][0, 0] = np.nan
    with pytest.raises(ContractError, match="non-finite"):
        build_semantic_mask_set(values)


def test_report_does_not_claim_model_quality_or_promotion():
    report = mask_report(build_semantic_mask_set(probabilities()))
    assert report["classification"] == "SEMANTIC_SEPARATION_NOT_MODEL_QUALITY_PROOF"
    assert report["promotion_allowed"] is False
    assert report["terrain_candidate_fraction"] == 1.0
