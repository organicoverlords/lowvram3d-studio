"""Whole-run finite gate built on the step-split probe.

The single-step diagnostic mode answers "where does it break". This mode answers "is it still
broken anywhere", which is the question a repair proof has to answer, and it has to answer it per
stage rather than as one aggregate -- a run where only the reference pass is finite and a run where
everything is finite must not produce the same verdict.

The pipeline is faked here on purpose: these are checks on the recorder's bookkeeping, not on the
UNet, and they must run without a GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "workers"))

from mvadapter_step_split_probe import StepSplitProbe  # noqa: E402

VIEWS = 6
LATENT_SHAPE = (VIEWS, 4, 8, 8)


class FakeScheduler:
    """Just enough scheduler to be instrumented: scale, then step."""

    def scale_model_input(self, sample, timestep=None):
        return sample * 1.0

    def step(self, model_output, timestep, sample, **_kwargs):
        return SimpleNamespace(prev_sample=sample - model_output * 0.1,
                               pred_original_sample=sample)


class FakeUNet:
    def __init__(self, outputs=None):
        self.outputs = outputs
        self.calls = 0

    def forward(self, sample, timestep=None, encoder_hidden_states=None, **_kwargs):
        self.calls += 1
        if self.outputs is not None:
            return (self.outputs[self.calls - 1],)
        return (torch.randn(sample.shape) * 0.01,)


def _run(pipe, steps: int, latents=None, reference_pass: bool = True):
    """One fake denoising run: an optional batch-1 reference forward, then N steps."""
    if reference_pass:
        pipe.unet.forward(torch.randn((1, 4, 8, 8)), 0, None)
    latents = torch.randn(LATENT_SHAPE) if latents is None else latents
    for index in range(steps):
        timestep = 900 - index * 100
        scaled = pipe.scheduler.scale_model_input(latents, timestep)
        noise = pipe.unet.forward(scaled, timestep, None)[0]
        latents = pipe.scheduler.step(noise, timestep, latents).prev_sample
    return latents


def _probe(tmp_path, **kwargs):
    probe = StepSplitProbe(tmp_path, target_step=None, save_scheduler_inputs=False,
                           abort_on_first_failure=False, **kwargs)
    return probe


def test_clean_two_step_run_passes_every_named_check(tmp_path):
    pipe = SimpleNamespace(scheduler=FakeScheduler(), unet=FakeUNet())
    probe = _probe(tmp_path)
    probe.install(pipe)

    _run(pipe, steps=2)
    probe.uninstall()
    gate = probe.finite_gate(expected_steps=2)

    assert gate["passed"] is True
    assert gate["missing_checks"] == []
    assert gate["failed_checks"] == []
    assert gate["checks"]["reference_unet_output_finite"] is True
    assert gate["checks"]["step1_raw_unet_output_finite"] is True
    assert gate["checks"]["step2_raw_unet_output_finite"] is True
    assert gate["checks"]["final_latent_finite"] is True
    assert gate["checks"]["latent_hashes_change_between_steps"] is True
    assert gate["checks"]["views_not_all_identical"] is True


def test_reference_pass_is_recorded_separately_from_step_one(tmp_path):
    """A batch-1 timestep-0 forward is not the first denoising forward."""
    pipe = SimpleNamespace(scheduler=FakeScheduler(), unet=FakeUNet())
    probe = _probe(tmp_path)
    probe.install(pipe)

    _run(pipe, steps=2)
    probe.uninstall()

    stages = {entry["checkpoint"]: entry["stage"] for entry in probe.records if entry.get("tensor")}
    assert stages["ref01_unet_raw_output"] == "reference_unet"
    assert stages["step01_09_unet_raw_output"] == "unet_output"
    assert probe.reference_forwards == 1
    assert probe.current_step == 2


def test_a_non_finite_step_two_output_fails_only_the_step_two_check(tmp_path):
    outputs = [torch.randn((1, 4, 8, 8)),
               torch.randn(LATENT_SHAPE) * 0.01,
               torch.full(LATENT_SHAPE, float("nan"))]
    pipe = SimpleNamespace(scheduler=FakeScheduler(), unet=FakeUNet(outputs=outputs))
    probe = _probe(tmp_path)
    probe.install(pipe)

    _run(pipe, steps=2)
    probe.uninstall()
    gate = probe.finite_gate(expected_steps=2)

    assert gate["passed"] is False
    assert "step2_raw_unet_output_finite" in gate["failed_checks"]
    assert gate["checks"]["step1_raw_unet_output_finite"] is True
    assert gate["checks"]["reference_unet_output_finite"] is True


def test_a_non_finite_reference_output_is_classified_as_a_reference_failure(tmp_path):
    outputs = [torch.full((1, 4, 8, 8), float("nan")),
               torch.randn(LATENT_SHAPE) * 0.01,
               torch.randn(LATENT_SHAPE) * 0.01]
    pipe = SimpleNamespace(scheduler=FakeScheduler(), unet=FakeUNet(outputs=outputs))
    probe = _probe(tmp_path)
    probe.install(pipe)

    _run(pipe, steps=2)
    probe.uninstall()

    assert probe.classification() == "MVADAPTER_REFERENCE_UNET_OUTPUT_NONFINITE"
    assert probe.finite_gate(expected_steps=2)["checks"]["reference_unet_output_finite"] is False


def test_a_missing_step_is_reported_as_missing_not_as_passing(tmp_path):
    """A run that stopped after one step must not silently satisfy a two-step gate."""
    pipe = SimpleNamespace(scheduler=FakeScheduler(), unet=FakeUNet())
    probe = _probe(tmp_path)
    probe.install(pipe)

    _run(pipe, steps=1)
    probe.uninstall()
    gate = probe.finite_gate(expected_steps=2)

    assert gate["passed"] is False
    assert "step2_raw_unet_output_finite" in gate["missing_checks"]


def test_unchanged_latents_between_steps_fail_the_hash_check(tmp_path):
    """A scheduler that returns the sample untouched leaves both steps hashing the same."""
    scheduler = FakeScheduler()
    scheduler.step = lambda model_output, timestep, sample, **_k: SimpleNamespace(
        prev_sample=sample, pred_original_sample=sample)
    pipe = SimpleNamespace(scheduler=scheduler, unet=FakeUNet())
    probe = _probe(tmp_path)
    probe.install(pipe)

    _run(pipe, steps=2)
    probe.uninstall()
    gate = probe.finite_gate(expected_steps=2)

    assert gate["checks"]["latent_hashes_change_between_steps"] is False
    assert gate["passed"] is False


def test_six_identical_views_fail_the_variation_check(tmp_path):
    # A UNet that predicts the same noise for every view, fed identical latents: exactly the
    # "six identical PNGs" shape, one stage before the decode.
    uniform = torch.randn((1, 4, 8, 8)) * 0.01
    outputs = [torch.randn((1, 4, 8, 8))] + [uniform.repeat(VIEWS, 1, 1, 1) for _ in range(2)]
    pipe = SimpleNamespace(scheduler=FakeScheduler(), unet=FakeUNet(outputs=outputs))
    probe = _probe(tmp_path)
    probe.install(pipe)

    single = torch.randn((1, 4, 8, 8))
    _run(pipe, steps=2, latents=single.repeat(VIEWS, 1, 1, 1))
    probe.uninstall()
    gate = probe.finite_gate(expected_steps=2)

    assert gate["checks"]["views_not_all_identical"] is False
    assert gate["passed"] is False


def test_single_step_diagnostic_mode_keeps_its_original_labels(tmp_path):
    """The gate mode must not rename the checkpoints existing diagnostic receipts refer to."""
    pipe = SimpleNamespace(scheduler=FakeScheduler(), unet=FakeUNet())
    probe = StepSplitProbe(tmp_path, target_step=1, save_scheduler_inputs=False,
                           abort_on_first_failure=False)
    probe.install(pipe)

    _run(pipe, steps=2)
    probe.uninstall()

    labels = {entry["checkpoint"] for entry in probe.records}
    assert "09_unet_raw_output" in labels
    assert not any(label.startswith("step02_") for label in labels)


def test_gate_mode_does_not_abort_the_run(tmp_path):
    """The diagnostic stops at the first failure; the gate has to see every stage."""
    outputs = [torch.randn((1, 4, 8, 8)),
               torch.full(LATENT_SHAPE, float("nan")),
               torch.full(LATENT_SHAPE, float("nan"))]
    pipe = SimpleNamespace(scheduler=FakeScheduler(), unet=FakeUNet(outputs=outputs))
    probe = _probe(tmp_path)
    probe.install(pipe)

    _run(pipe, steps=2)
    probe.uninstall()

    assert probe.current_step == 2
    assert probe.finite_gate(expected_steps=2)["failed_checks"]
