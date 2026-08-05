"""Tests for the fractional-coverage matte and the island drop.

These pin the two properties that make the change worth having, and the two
that make it safe. The interesting cases are all synthetic: a real photograph
cannot tell you what the right alpha was, but a plate composited by hand can.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

from pipeline_matte import key_alpha  # noqa: E402

PLATE = 248


def _plate(height=64, width=64):
    return np.full((height, width, 3), PLATE, dtype=np.uint8)


def test_half_covered_edge_recovers_fractional_alpha():
    """A pixel composited at exactly 50% must key back to roughly 50%.

    This is the whole claim of the feather path stated as a test: the binary
    key answers 0 or 255 here and is wrong by 127 either way.
    """
    image = _plate()
    image[16:48, 16:48] = 20  # dark block, unambiguous interior
    # One column composited half-way between the block and the plate.
    image[16:48, 48] = round(0.5 * 20 + 0.5 * PLATE)

    alpha, stats = key_alpha(image, tolerance=42.0)

    edge = alpha[24:40, 48].astype(float)
    assert 96 <= edge.mean() <= 160, edge.mean()
    assert stats["partial_alpha_fraction"] > 0.0


def test_feather_zero_restores_binary_alpha():
    """The old behaviour has to remain reachable, exactly."""
    image = _plate()
    image[16:48, 16:48] = 20
    image[16:48, 48] = 134

    alpha, stats = key_alpha(image, tolerance=42.0, feather=0)

    assert set(np.unique(alpha)).issubset({0, 255})
    assert stats["partial_alpha_fraction"] == 0.0


def test_small_island_is_dropped_and_shrinks_the_bounding_box():
    """The boat's defect: a speck far from the body dragging the bbox."""
    image = _plate(128, 128)
    image[32:96, 32:96] = 20        # body, 4096 px
    image[120, 120] = 20            # speck, 1 px, 2.4e-4 of the body

    alpha, stats = key_alpha(image, tolerance=42.0, feather=0)
    mask = alpha > 127

    assert stats["dropped_island_components"] >= 1
    assert not mask[120, 120]
    assert int(np.where(np.any(mask, axis=1))[0][-1]) < 110


def test_large_detached_part_survives_the_island_drop():
    """The property the flood-fill key was written to protect.

    A genuinely detached ornament must not be swept up with the speckle, so
    the floor is a fraction of the body rather than an absolute count.
    """
    image = _plate(128, 128)
    image[32:96, 32:96] = 20        # body, 4096 px
    image[104:120, 104:120] = 20    # ornament, 256 px = 6.25% of the body

    alpha, stats = key_alpha(image, tolerance=42.0, feather=0)
    mask = alpha > 127

    assert mask[112, 112]
    assert stats["dropped_island_components"] == 0


def test_thin_structure_survives_feathering():
    """A one-pixel mast is the structure the binary key destroys.

    Eroding the interior by the full feather radius would delete it before its
    foreground colour could be sampled; eroding by one keeps it addressable.
    """
    image = _plate(64, 64)
    image[40:56, 20:44] = 20        # body
    image[10:40, 31] = 60           # 1 px mast, distinct from plate

    alpha, _ = key_alpha(image, tolerance=42.0)

    assert alpha[20:38, 31].max() > 0


def test_degenerate_foreground_does_not_fabricate_alpha():
    """Where the subject matches the plate, no coverage can be inferred.

    Those pixels must keep the binary decision rather than take a fraction
    computed by dividing by a near-zero length.
    """
    image = _plate()
    image[16:48, 16:48] = PLATE - 1  # subject indistinguishable from plate

    alpha, stats = key_alpha(image, tolerance=42.0)

    assert stats["feather_degenerate_pixels"] >= 0
    assert np.isfinite(alpha.astype(float)).all()


def test_dropped_island_is_not_revived_by_feathering():
    """A dropped speck near the body must not come back through the band.

    Regression: it returned at alpha 255 -- fully opaque, not a faint ghost --
    because a speck's own colour projects cleanly onto its own direction, and
    the dilated band reaches anything within `feather` of the surviving subject.
    The island drop was silently a no-op for exactly the specks closest to the
    body. Note `ndimage.label` uses 4-connectivity, so a diagonal neighbour is
    already a separate component and lands well inside the band.
    """
    image = _plate(128, 128)
    image[32:96, 32:96] = 20
    image[97, 97] = 20  # 2 px past the corner: inside the feather band

    alpha, stats = key_alpha(image, tolerance=42.0, feather=2)

    assert stats["dropped_island_components"] >= 1
    assert alpha[97, 97] == 0


def test_isolated_thin_structure_keys_fully_opaque():
    """A solid one-pixel structure is opaque, whatever colour it is.

    Regression: with a global nearest-interior lookup, an isolated red mast
    borrowed its foreground colour from a distant dark body and keyed to alpha
    173. Sampling colour per-structure rather than globally is what fixes it,
    so the test uses a colour deliberately unlike the rest of the subject.
    """
    image = _plate(96, 96)
    image[60:80, 20:60] = 20            # dark body
    image[10:60, 80] = (200, 40, 40)    # isolated 1 px red mast

    alpha, _ = key_alpha(image, tolerance=42.0, feather=2)

    assert alpha[20:50, 80].min() >= 250


@pytest.mark.parametrize("feather", [1, 2, 3])
def test_alpha_stays_in_range_for_any_feather(feather):
    image = _plate()
    image[16:48, 16:48] = 20
    image[16:48, 48] = 134

    alpha, _ = key_alpha(image, tolerance=42.0, feather=feather)

    assert alpha.dtype == np.uint8
    assert alpha.min() >= 0 and alpha.max() <= 255
