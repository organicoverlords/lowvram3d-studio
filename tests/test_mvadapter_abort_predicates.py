from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from workers.mvadapter_latent_telemetry import LatentTelemetry, tensor_stats, failures_for
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))
from workers.mvadapter_step_split_probe import StepSplitProbe


VIEWS = 6
SHAPE = (VIEWS, 4, 8, 8)


class Scheduler:
    def __init__(self, zero_output: bool = False) -> None:
        self.zero_output = zero_output

    def scale_model_input(self, sample, timestep=None):
        return sample

    def step(self, model_output, timestep, sample, **_kwargs):
        previous = torch.zeros_like(sample) if self.zero_output else sample - model_output * 0.1
        return SimpleNamespace(prev_sample=previous, pred_original_sample=sample)


class ReferenceCacheUNet:
    def __init__(self, expected, *, cache_mode="valid", denoise_outputs=None,
                 reference_output=None):
        self.expected = expected
        self.cache_mode = cache_mode
        self.denoise_outputs = list(denoise_outputs or [torch.randn(SHAPE)])
        self.reference_output = reference_output
        self.calls = 0

    def forward(self, sample, timestep=None, encoder_hidden_states=None, **kwargs):
        cross = kwargs.get("cross_attention_kwargs") or {}
        if "cache_hidden_states" in cross:
            sink = cross["cache_hidden_states"]
            if self.cache_mode == "valid":
                sink.update({key: value.clone() for key, value in self.expected.items()})
            elif self.cache_mode == "nonfinite":
                sink.update({key: torch.full_like(value, float("nan"))
                             for key, value in self.expected.items()})
            if self.reference_output is not None:
                return (self.reference_output,)
            return (torch.zeros_like(sample),)
        value = self.denoise_outputs[min(self.calls, len(self.denoise_outputs) - 1)]
        self.calls += 1
        return (value,)


def _run(pipe, probe, *, steps=1, reference_kwargs=None):
    probe.install(pipe)
    pipe.unet.forward(torch.randn((1, 4, 8, 8)), 0, None,
                      cross_attention_kwargs=reference_kwargs or {"cache_hidden_states": {}})
    latents = torch.randn(SHAPE)
    for index in range(steps):
        timestep = 900 - index * 100
        scaled = pipe.scheduler.scale_model_input(latents, timestep)
        noise = pipe.unet.forward(scaled, timestep, None)[0]
        latents = pipe.scheduler.step(noise, timestep, latents).prev_sample
    probe.uninstall()


def _expected_cache():
    return {"rowcol.processor": torch.randn((1, 4, 8), dtype=torch.float32)}


def _pipe(expected, **kwargs):
    return SimpleNamespace(
        scheduler=Scheduler(kwargs.pop("scheduler_zero", False)),
        unet=ReferenceCacheUNet(expected, **kwargs),
    )


def _probe(tmp_path, expected, **kwargs):
    return StepSplitProbe(
        tmp_path, target_step=None, save_scheduler_inputs=False,
        abort_on_first_failure=False, expected_reference_cache=expected,
        **kwargs,
    )


def test_valid_reference_zero_sentinel_and_exact_cache_are_accepted(tmp_path):
    expected = _expected_cache()
    probe = _probe(tmp_path, expected)
    _run(_pipe(expected), probe)

    assert probe.first_nonfinite is None
    assert probe.first_unexpected_flat is None
    assert probe.first_failure is None
    assert probe.reference_output_zero_sentinel["accepted"] is True
    assert probe.reference_cache_contract["valid"] is True
    assert probe.classification() == "MVADAPTER_ALL_RECORDED_CHECKPOINTS_FINITE"


def test_zero_reference_sentinel_missing_cache_is_a_contract_failure(tmp_path):
    expected = _expected_cache()
    probe = _probe(tmp_path, expected)
    _run(_pipe(expected, cache_mode="missing"), probe)

    assert probe.classification() == "MVADAPTER_REFERENCE_CACHE_CONTRACT_FAILED"
    assert probe.reference_cache_contract["missing_keys"] == ["rowcol.processor"]


def test_zero_reference_sentinel_nonfinite_cache_is_nonfinite_failure(tmp_path):
    expected = _expected_cache()
    probe = _probe(tmp_path, expected)
    _run(_pipe(expected, cache_mode="nonfinite"), probe)

    assert probe.classification() == "MVADAPTER_RUNTIME_REJECTED_NONFINITE"
    assert probe.first_nonfinite is not None
    assert probe.reference_cache_contract["all_finite"] is False


def test_nan_reference_return_is_nonfinite(tmp_path):
    expected = _expected_cache()
    probe = _probe(tmp_path, expected)
    pipe = _pipe(expected, reference_output=torch.full((1, 4, 8, 8), float("nan")))
    _run(pipe, probe)

    assert probe.classification() == "MVADAPTER_RUNTIME_REJECTED_NONFINITE"
    assert probe.reference_output_zero_sentinel["finite"] is False


def test_zero_denoising_unet_output_is_unexpected_flat(tmp_path):
    expected = _expected_cache()
    probe = _probe(tmp_path, expected)
    _run(_pipe(expected, denoise_outputs=[torch.zeros(SHAPE)]), probe)

    assert probe.classification() == "MVADAPTER_RUNTIME_REJECTED_FLAT_TENSOR"
    assert probe.first_unexpected_flat["checkpoint"] == "step01_09_unet_raw_output"


def test_zero_scheduler_output_is_unexpected_flat(tmp_path):
    expected = _expected_cache()
    probe = _probe(tmp_path, expected)
    _run(_pipe(expected, scheduler_zero=True), probe)

    assert probe.classification() == "MVADAPTER_RUNTIME_REJECTED_FLAT_TENSOR"
    assert probe.first_unexpected_flat["checkpoint"] == "step01_14_prev_sample"


def test_zero_denoising_latent_is_unexpected_flat(tmp_path):
    expected = _expected_cache()
    probe = _probe(tmp_path, expected)
    _run(_pipe(expected, scheduler_zero=True), probe)

    assert probe.first_unexpected_flat is not None
    assert "ALL_ZERO" in probe.first_unexpected_flat["failures"]


def test_nan_tensor_is_never_all_zero():
    stats = tensor_stats("nan", torch.full((6, 4, 8, 8), float("nan")))
    failures = failures_for(stats)
    assert "NAN" in failures
    assert "ALL_ZERO" not in failures


def test_integer_timestep_zero_is_not_flat():
    assert failures_for(tensor_stats("timestep", torch.zeros((), dtype=torch.long))) == []


def test_first_nonfinite_is_earliest_checkpoint(tmp_path):
    expected = _expected_cache()
    outputs = [torch.randn(SHAPE), torch.full(SHAPE, float("nan"))]
    probe = _probe(tmp_path, expected)
    _run(_pipe(expected, denoise_outputs=outputs), probe, steps=2)

    assert probe.first_nonfinite["checkpoint"] == "step02_09_unet_raw_output"


def test_first_unexpected_flat_is_earliest_flat_checkpoint(tmp_path):
    expected = _expected_cache()
    outputs = [torch.zeros(SHAPE), torch.randn(SHAPE)]
    probe = _probe(tmp_path, expected)
    _run(_pipe(expected, denoise_outputs=outputs), probe, steps=2)

    assert probe.first_unexpected_flat["checkpoint"] == "step01_09_unet_raw_output"


def test_first_failure_merges_probe_and_telemetry_chronologically(tmp_path):
    expected = _expected_cache()
    telemetry = LatentTelemetry(tmp_path / "telemetry", snapshot_steps=(), save_tensors=False)
    telemetry.record("initial_latent", torch.randn(SHAPE))
    probe = _probe(tmp_path / "probe", expected)
    _run(_pipe(expected, denoise_outputs=[torch.zeros(SHAPE)]), probe)
    telemetry.attach_probe(probe.summary(), probe.records)
    telemetry.record_step(1, 900, torch.full(SHAPE, float("nan")))

    assert telemetry.first_failure()["kind"] == "probe"
    assert telemetry.chronological_records()


def test_reference_and_denoising_passes_have_distinct_stage_labels(tmp_path):
    expected = _expected_cache()
    probe = _probe(tmp_path, expected)
    _run(_pipe(expected), probe)
    stages = {entry["checkpoint"]: entry["stage"] for entry in probe.records if entry.get("tensor")}
    assert stages["ref01_unet_raw_output"] == "reference_unet"
    assert stages["step01_09_unet_raw_output"] == "unet_output"


def test_identical_view_detection_is_skipped_for_nonfinite_views():
    # Use the public helper through a normal telemetry record to keep this CPU-only.
    from workers.mvadapter_latent_telemetry import views_identical
    records = [tensor_stats(f"view{index}", torch.full((4, 8, 8), float("nan"))) for index in range(6)]
    assert views_identical(records) is None


def test_valid_reference_zero_sentinel_does_not_abort_before_step_one(tmp_path):
    expected = _expected_cache()
    probe = StepSplitProbe(
        tmp_path, target_step=None, save_scheduler_inputs=False,
        abort_on_first_failure=True, expected_reference_cache=expected,
    )
    pipe = _pipe(expected)
    _run(pipe, probe)
    assert probe.current_step == 1
    assert probe.first_failure is None
