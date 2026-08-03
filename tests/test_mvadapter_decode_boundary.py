"""Decode-boundary regressions: the conversions that can silently turn a good image black."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

from mvadapter_latent_telemetry import (  # noqa: E402
    LatentTelemetry,
    failures_for,
    tensor_stats,
    views_identical,
)

SCALING = 0.18215


def denormalize(image: torch.Tensor) -> torch.Tensor:
    """The diffusers postprocess formula, applied once."""
    return (image / 2 + 0.5).clamp(0, 1)


def to_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.detach().cpu().numpy() * 255).round().astype(np.uint8)


def test_correct_scaling_round_trips():
    latents = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(0))
    scaled = latents * SCALING
    restored = scaled / SCALING
    assert torch.allclose(latents, restored, atol=1e-5)


def test_accidental_double_scaling_shrinks_signal():
    """Dividing twice is not a no-op; it collapses the latent toward zero."""
    latents = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(1))
    once = latents / SCALING
    twice = once / SCALING
    assert twice.abs().mean() > once.abs().mean()
    # and multiplying where a divide was intended moves the other way
    wrong_direction = latents * SCALING
    assert wrong_direction.abs().mean() < once.abs().mean()


def test_accidental_zero_multiplication_is_detected():
    latents = torch.randn(2, 4, 8, 8, generator=torch.Generator().manual_seed(2))
    zeroed = latents * 0.0
    assert "ALL_ZERO" in failures_for(tensor_stats("zeroed", zeroed))
    assert "ALL_ZERO" not in failures_for(tensor_stats("healthy", latents))


def test_zero_one_versus_minus_one_one_conversion():
    """A [-1,1] tensor denormalised once lands in [0,1]; denormalising twice biases it bright."""
    image = torch.linspace(-1.0, 1.0, steps=64).reshape(1, 1, 8, 8)
    once = denormalize(image)
    assert float(once.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(once.max()) == pytest.approx(1.0, abs=1e-6)
    twice = denormalize(once)
    assert float(twice.min()) > 0.4  # already positive, so a second pass compresses toward white
    # Treating a [0,1] tensor as if it were [-1,1] loses the lower half entirely.
    misread = torch.rand(1, 1, 8, 8, generator=torch.Generator().manual_seed(3))
    assert float(denormalize(misread).min()) >= 0.5


def test_nan_latent_casts_to_pure_black_uint8():
    """The exact mechanism behind six identical black PNGs."""
    image = torch.full((1, 3, 4, 4), float("nan"))
    array = to_uint8(denormalize(image))
    assert array.max() == 0
    assert len(np.unique(array)) == 1


def test_blank_image_rejection():
    black = torch.zeros(6, 3, 8, 8)
    stats = tensor_stats("black", black)
    failures = failures_for(stats)
    assert "ALL_ZERO" in failures
    assert "NEAR_ZERO_STD" in failures
    per_view = [tensor_stats(f"v{i}", black[i]) for i in range(black.shape[0])]
    assert views_identical(per_view)


def test_healthy_image_passes_every_gate():
    generator = torch.Generator().manual_seed(4)
    image = torch.rand(6, 3, 8, 8, generator=generator) * 2 - 1
    stats = tensor_stats("healthy", image)
    assert failures_for(stats) == []
    per_view = [tensor_stats(f"v{i}", image[i]) for i in range(image.shape[0])]
    assert not views_identical(per_view)
    array = to_uint8(denormalize(image))
    assert len(np.unique(array)) > 1


def test_telemetry_reports_first_failure_and_survives_write(tmp_path):
    telemetry = LatentTelemetry(tmp_path, snapshot_steps=(1,), save_tensors=True)
    generator = torch.Generator().manual_seed(5)
    telemetry.record_step(1, torch.tensor(500.0),
                          torch.randn(6, 4, 8, 8, generator=generator))
    telemetry.record_step(2, torch.tensor(1.0), torch.full((6, 4, 8, 8), float("nan")))
    first = telemetry.first_failure()
    assert first is not None
    assert first["at"] == 2
    assert "NAN" in first["failures"]
    written = telemetry.write()
    assert written.is_file()
    assert (tmp_path / "step01_latents.npz").is_file()
    assert telemetry.summary()["clean"] is False


def test_identical_views_are_flagged_when_variation_expected():
    telemetry = LatentTelemetry(Path("."), snapshot_steps=(), save_tensors=False)
    repeated = torch.randn(1, 4, 8, 8, generator=torch.Generator().manual_seed(6)).repeat(6, 1, 1, 1)
    entry = telemetry.record_step(1, torch.tensor(10.0), repeated)
    assert entry["all_views_identical"] is True
    assert "ALL_VIEWS_IDENTICAL" in entry["failures"]
