"""Telemetry classification regressions.

Every test here corresponds to a misclassification that actually sent a diagnosis to the wrong
stage: a scalar integer timestep reported as a collapsed-to-zero tensor, an all-NaN tensor reported
as all-zero, the final image reported as the first failure when the latents had died at step 1, and
six all-NaN views reported as "identical" because their bytes matched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

from mvadapter_latent_telemetry import (  # noqa: E402
    LatentTelemetry,
    failures_for,
    tensor_stats,
    views_identical,
)


def test_scalar_integer_timestep_is_not_an_all_zero_failure():
    timestep = torch.zeros((), dtype=torch.long)
    assert failures_for(tensor_stats("timestep", timestep)) == []


def test_integer_step_counter_vector_is_not_an_all_zero_failure():
    counters = torch.zeros((4,), dtype=torch.int64)
    assert failures_for(tensor_stats("step_counters", counters)) == []


def test_scalar_float_zero_is_not_an_all_zero_failure():
    """A zero-dimensional float carries no spatial extent to collapse."""
    assert failures_for(tensor_stats("sigma", torch.tensor(0.0))) == []


def test_genuinely_collapsed_float_latents_are_still_reported():
    failures = failures_for(tensor_stats("latents", torch.zeros((6, 4, 32, 32))))
    assert "ALL_ZERO" in failures
    assert "NEAR_ZERO_STD" in failures


def test_nan_tensor_is_not_also_reported_as_all_zero():
    stats = tensor_stats("latents", torch.full((6, 4, 8, 8), float("nan")))
    failures = failures_for(stats)

    assert failures == ["NAN", "NON_FINITE_PRESENT"]
    assert "ALL_ZERO" not in failures
    assert "NEAR_ZERO_STD" not in failures
    assert stats["zero_fraction"] is None


def test_partially_nan_tensor_reports_only_the_non_finite_defect():
    latents = torch.randn((6, 4, 8, 8))
    latents[0, 0, 0, 0] = float("inf")
    failures = failures_for(tensor_stats("latents", latents))

    assert failures == ["INF", "NON_FINITE_PRESENT"]


def test_identical_all_nan_views_are_not_reported_as_identical():
    """Identical bytes across all-NaN views say nothing about the content the model produced."""
    nan_views = [tensor_stats(f"view{index}", torch.full((4, 8, 8), float("nan")))
                 for index in range(6)]
    assert views_identical(nan_views) is None


def test_identical_finite_views_are_reported_as_identical():
    finite_views = [tensor_stats(f"view{index}", torch.ones((4, 8, 8))) for index in range(6)]
    assert views_identical(finite_views) is True


def test_distinct_finite_views_are_not_reported_as_identical():
    views = [tensor_stats(f"view{index}", torch.full((4, 8, 8), float(index))) for index in range(6)]
    assert views_identical(views) is False


def test_first_failure_is_the_earliest_in_real_time_not_the_earliest_checkpoint(tmp_path):
    telemetry = LatentTelemetry(tmp_path, snapshot_steps=(), save_tensors=False)

    telemetry.record("initial_latents", torch.randn((6, 4, 8, 8)))
    telemetry.record_step(1, 999, torch.full((6, 4, 8, 8), float("nan")))
    telemetry.record("final_postprocessed_image_tensor", torch.zeros((6, 8, 8, 3)))

    first = telemetry.first_failure()
    assert first["kind"] == "step"
    assert first["at"] == 1
    assert "NAN" in first["failures"]
    # The blank final image is a real violation too -- it just is not the first one.
    assert len(telemetry.summary()["violations"]) == 2


def test_first_failure_can_be_a_checkpoint_when_the_checkpoint_came_first(tmp_path):
    telemetry = LatentTelemetry(tmp_path, snapshot_steps=(), save_tensors=False)

    telemetry.record("reference_unet_output", torch.full((1, 4, 8, 8), float("nan")))
    telemetry.record_step(1, 999, torch.full((6, 4, 8, 8), float("nan")))

    first = telemetry.first_failure()
    assert first["kind"] == "checkpoint"
    assert first["at"] == "reference_unet_output"


def test_integer_timestep_recorded_on_a_step_does_not_fail_the_step(tmp_path):
    telemetry = LatentTelemetry(tmp_path, snapshot_steps=(), save_tensors=False)

    entry = telemetry.record_step(1, torch.zeros((), dtype=torch.long), torch.randn((6, 4, 8, 8)))

    assert entry["timestep"] == 0.0
    assert entry["failures"] == []
    assert telemetry.summary()["clean"] is True


def test_step_hashes_are_reported_as_distinct_when_latents_actually_change(tmp_path):
    telemetry = LatentTelemetry(tmp_path, snapshot_steps=(), save_tensors=False)

    telemetry.record_step(1, 900, torch.randn((6, 4, 8, 8)))
    telemetry.record_step(2, 800, torch.randn((6, 4, 8, 8)))

    assert telemetry.summary()["step_hashes_all_distinct"] is True


def test_repeated_identical_step_latents_are_reported_as_not_distinct(tmp_path):
    telemetry = LatentTelemetry(tmp_path, snapshot_steps=(), save_tensors=False)

    latents = torch.randn((6, 4, 8, 8))
    telemetry.record_step(1, 900, latents)
    telemetry.record_step(2, 800, latents)

    assert telemetry.summary()["step_hashes_all_distinct"] is False
