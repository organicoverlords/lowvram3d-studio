import argparse

import pytest

from workers.image_world.project_moge_surface import _load_semantic_masks


def test_surface_worker_requires_both_semantic_paths():
    args = argparse.Namespace(semantic_arrays="arrays", semantic_probabilities=None)
    with pytest.raises(ValueError, match="must be supplied together"):
        _load_semantic_masks(args, (4, 4))


def test_surface_worker_allows_unclassified_mode():
    args = argparse.Namespace(semantic_arrays=None, semantic_probabilities=None)
    assert _load_semantic_masks(args, (4, 4)) is None
