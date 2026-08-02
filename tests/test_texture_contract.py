import pytest

from workers.texture_contract import (
    AtlasResolutionContractError,
    assert_atlas_dimensions,
    validate_requested_atlas_size,
)
from lowvram3d.uv_quality import MIN_ATLAS_UTILIZATION


def test_1024_contract_rejects_silent_512_output():
    with pytest.raises(AtlasResolutionContractError, match="ATLAS_RESOLUTION_CONTRACT_MISMATCH"):
        assert_atlas_dimensions((512, 512), 1024, "test stage")


def test_1024_contract_accepts_matching_square_output():
    assert validate_requested_atlas_size(1024) == 1024
    assert_atlas_dimensions((1024, 1024), 1024, "test stage")


def test_xatlas_panda_utilization_gate_is_at_least_55_percent():
    assert MIN_ATLAS_UTILIZATION >= 0.55


@pytest.mark.parametrize("size", [0, 513, 5000])
def test_invalid_requested_size_fails_closed(size):
    with pytest.raises(AtlasResolutionContractError):
        validate_requested_atlas_size(size)
