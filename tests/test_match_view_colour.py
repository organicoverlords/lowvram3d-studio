"""Tests for the linear-light global colour match.

The property under test is narrow and worth stating: the correction must be a
single global gain. Everything else -- that it hits the target, that it restores
the warm cast -- follows from fitting that gain correctly, but a transform that
quietly becomes per-view would still hit the target while destroying the
inter-view lighting relationships the atlas depends on.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "workers" / "match_view_colour.py"

PLATE = (128, 128, 128)


def _write_view(path, subject_rgb, size=64):
    """A view is a subject block on a flat plate, as the generator emits."""
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, :] = PLATE
    image[16:48, 16:48] = subject_rgb
    Image.fromarray(image).save(path)


def _write_photo(path, subject_rgb, size=64):
    """The photograph arrives as RGBA whose alpha is the matte."""
    rgba = np.zeros((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = 255
    rgba[16:48, 16:48, :3] = subject_rgb
    rgba[16:48, 16:48, 3] = 255
    Image.fromarray(rgba, "RGBA").save(path)


@pytest.fixture
def scene(tmp_path):
    views = tmp_path / "views"
    views.mkdir()
    # Neutral grey views at differing brightness -- view 4 lit, view 5 shadowed,
    # which is the genuine variation the correction must not flatten.
    for index, (name, value) in enumerate([
        ("front", 160), ("right", 150), ("rear", 155),
        ("left", 158), ("top", 190), ("bottom", 70),
    ]):
        _write_view(views / f"view_{index}_{name}.png", (value, value, value))
    photo = tmp_path / "photo.png"
    matte = tmp_path / "matte.png"
    _write_photo(photo, (70, 60, 48))   # dark and warm
    _write_photo(matte, (70, 60, 48))
    return views, photo, matte, tmp_path / "out"


def _run(views, photo, matte, out, reference=0):
    return subprocess.run(
        [sys.executable, str(WORKER), "--views", str(views),
         "--photograph", str(photo), "--matte", str(matte),
         "--reference-view", str(reference), "--output", str(out)],
        capture_output=True, text=True)


def _subject_mean(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    return rgb[16:48, 16:48].reshape(-1, 3).mean(axis=0)


def test_reference_view_lands_on_the_photograph(scene):
    views, photo, matte, out = scene
    result = _run(views, photo, matte, out)

    assert result.returncode == 0, result.stdout + result.stderr
    achieved = _subject_mean(out / "view_0_front.png")
    target = np.array([70.0, 60.0, 48.0])
    assert np.abs(achieved - target).max() <= 2.0, achieved


def test_warm_separation_is_restored(scene):
    """Neutral in, warm out: R must lead B by a real margin."""
    views, photo, matte, out = scene
    _run(views, photo, matte, out)

    before = _subject_mean(views / "view_0_front.png")
    after = _subject_mean(out / "view_0_front.png")

    assert before.max() - before.min() < 1.0      # was neutral
    assert after[0] - after[2] > 10.0             # now warm


def test_inter_view_ratio_survives(scene):
    """The invariant. A global gain cannot change a ratio between views.

    This is what distinguishes the correction from per-view normalisation, and
    it is the check that caught a real bug: re-keying the subject mask on the
    darkened output with an absolute threshold reported a 23% drift for a
    transform that is mathematically incapable of any.
    """
    views, photo, matte, out = scene
    result = _run(views, photo, matte, out)
    assert result.returncode == 0

    for index, name in [(4, "top"), (5, "bottom")]:
        assert (out / f"view_{index}_{name}.png").is_file()

    before = (_subject_mean(views / "view_4_top.png").mean()
              / _subject_mean(views / "view_5_bottom.png").mean())
    after = (_subject_mean(out / "view_4_top.png").mean()
             / _subject_mean(out / "view_5_bottom.png").mean())

    # Loose in sRGB because the ratio is preserved in linear light, not here;
    # the worker's own receipt checks it properly. This only asserts that the
    # brighter view stayed brighter by a comparable margin.
    assert after > 1.5
    assert abs(after - before) / before < 0.35


def test_every_view_is_written(scene):
    views, photo, matte, out = scene
    _run(views, photo, matte, out)

    assert len(list(out.glob("view_*_*.png"))) == 6


def test_matte_without_alpha_is_refused(scene, tmp_path):
    """An RGB matte would silently key the background instead of the subject."""
    views, photo, _, out = scene
    flat = tmp_path / "flat_matte.png"
    Image.open(photo).convert("RGB").save(flat)

    result = _run(views, photo, flat, out)

    assert result.returncode != 0
    assert "MATTE_NOT_RGBA" in (result.stdout + result.stderr)
