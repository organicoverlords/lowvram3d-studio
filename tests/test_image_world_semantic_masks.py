import numpy as np
import pytest

from lowvram3d.image_world.contracts import ContractError
from lowvram3d.image_world.semantic_masks import build_semantic_mask_set, mask_report


def probabilities(shape=(6, 8)):
    values = {
        name: np.full(shape, 0.02, dtype=np.float32)
        for name in ("terrain", "water", "sky", "vegetation", "structure", "residual")
    }
    values["terrain"][:] = 0.90
    return values


def test_clear_terrain_is_accepted():
    result = build_semantic_mask_set(probabilities())
    assert result.terrain_candidate.all()
    assert not result.unresolved.any()
    assert mask_report(result)["promotion_allowed"] is False


@pytest.mark.parametrize("name", ["water", "sky", "vegetation", "structure", "residual"])
def test_nonterrain_classes_are_excluded(name):
    values = probabilities()
    values["terrain"][0, 0] = 0.20
    values[name][0, 0] = 0.90
    result = build_semantic_mask_set(values)
    assert not result.terrain_candidate[0, 0]
    assert result.unresolved[0, 0]


def test_ambiguous_pixel_remains_unresolved():
    values = probabilities()
    values["terrain"][1, 1] = 0.60
    values["vegetation"][1, 1] = 0.55
    result = build_semantic_mask_set(values, exclusion_threshold=0.60)
    assert not result.terrain_candidate[1, 1]
    assert result.unresolved[1, 1]


def test_invalid_source_pixel_is_not_terrain_or_unresolved():
    valid = np.ones((6, 8), dtype=bool)
    valid[2, 3] = False
    result = build_semantic_mask_set(probabilities(), valid_mask=valid)
    assert not result.terrain_candidate[2, 3]
    assert not result.unresolved[2, 3]


def test_missing_and_malformed_maps_fail_closed():
    values = probabilities()
    del values["sky"]
    with pytest.raises(ContractError, match="missing semantic class"):
        build_semantic_mask_set(values)

    values = probabilities()
    values["water"] = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ContractError, match="mismatched shapes"):
        build_semantic_mask_set(values)


def test_nonfinite_and_out_of_range_probabilities_fail_closed():
    values = probabilities()
    values["terrain"][0, 0] = np.nan
    with pytest.raises(ContractError, match="non-finite"):
        build_semantic_mask_set(values)

    values = probabilities()
    values["terrain"][0, 0] = 1.2
    with pytest.raises(ContractError, match="\[0, 1\]"):
        build_semantic_mask_set(values)
