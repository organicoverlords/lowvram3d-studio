import pytest

from lowvram3d.vision_qa.contracts import ContractError
from lowvram3d.vision_qa.model_registry import MODELS, eligible_models, get_model


def test_registry_ids_are_unique():
    ids = [item.model_id for item in MODELS]
    assert len(ids) == len(set(ids))


def test_watchlist_is_not_eligible():
    assert "moge3-watch" not in {item.model_id for item in eligible_models()}


def test_moge_is_marked_non_independent():
    assert not get_model("moge2-vits-normal-baseline").independent_of_moge


def test_unknown_model_fails_closed():
    with pytest.raises(ContractError):
        get_model("imaginary-model")
