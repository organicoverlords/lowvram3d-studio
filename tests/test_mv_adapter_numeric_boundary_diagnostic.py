from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("PIL")

from workers.diagnose_mv_adapter_numeric_boundary import (
    REQUIRED_BOUNDARIES,
    install_numeric_probes,
    missing_required_boundaries,
)
from workers.mv_adapter_numeric_probe import FirstNonfiniteTensor, NumericProbe


class FakeScheduler:
    def step(self, noise_pred, timestep, latents, **kwargs):
        return (latents + noise_pred,)


class FakeVae:
    def __init__(self, decoded):
        self._decoded = decoded
        self.seen_decode_inputs = []

    def decode(self, latents, *args, **kwargs):
        self.seen_decode_inputs.append(latents)
        return (self._decoded,)


class FakeOffloadedModule:
    """Module carrying an Accelerate-style hook and `_old_forward`."""

    def __init__(self, inner):
        self._hf_hook = object()
        self._old_forward = inner

    def __call__(self, *args, **kwargs):
        return self._old_forward(*args, **kwargs)


class FakePipe:
    def __init__(self, *, noise_preds, adapter_states, decoded):
        self._noise_preds = list(noise_preds)
        self.relay_calls = []
        self.scheduler = FakeScheduler()
        self.vae = FakeVae(decoded)

        def cond_forward(control_feature):
            return list(adapter_states)

        def unet_relay_forward(*args, **kwargs):
            # Stands in for the reference-cache relay: it must still run, and
            # still see the cross-attention kwargs, once the probe is layered on.
            cross = kwargs.get("cross_attention_kwargs") or {}
            self.relay_calls.append(dict(cross))
            if "cache_hidden_states" in cross:
                return (torch.zeros((1, 4, 8, 8)),)
            return (self._noise_preds.pop(0),)

        self.cond_encoder = FakeOffloadedModule(cond_forward)
        self.unet = FakeOffloadedModule(unet_relay_forward)

    def prepare_latents(self, latents):
        return latents

    def prepare_image_latents(self, latents, **kwargs):
        return latents

    def prepare_control_image(self, image, **kwargs):
        return image


def drive(pipe, *, steps, initial_noise, reference, control):
    """Replay the official pipeline's tensor boundary order."""

    latents = pipe.prepare_latents(initial_noise)
    pipe.prepare_image_latents(reference)
    pipe.unet(
        reference,
        cross_attention_kwargs={"cache_hidden_states": {}},
    )
    control_feature = pipe.prepare_control_image(control)
    pipe.cond_encoder(control_feature)
    for _ in range(steps):
        noise_pred = pipe.unet(
            latents,
            cross_attention_kwargs={"ref_hidden_states": {}},
        )[0]
        latents = pipe.scheduler.step(noise_pred, 0, latents)[0]
    return pipe.vae.decode(latents / 0.18215)


def build_pipe(*, steps=2, bad=None):
    noise_preds = []
    for index in range(steps):
        tensor = torch.full((2, 4, 8, 8), 0.1)
        if bad == f"unet_{index}":
            tensor[0, 0, 0, 0] = float("nan")
        noise_preds.append(tensor)

    adapter_states = [torch.full((2, 8, 4, 4), 0.2) for _ in range(3)]
    if bad == "adapter":
        adapter_states[1][0, 0, 0, 0] = float("inf")

    decoded = torch.full((2, 3, 16, 16), 0.5)
    if bad == "decode_output":
        decoded[0, 0, 0, 0] = float("nan")

    return FakePipe(
        noise_preds=noise_preds, adapter_states=adapter_states, decoded=decoded
    ), steps


def test_probes_cover_every_required_boundary_on_a_finite_run():
    pipe, steps = build_pipe(steps=2)
    probe = NumericProbe()
    install_numeric_probes(pipe, probe)

    drive(
        pipe,
        steps=steps,
        initial_noise=torch.zeros((2, 4, 8, 8)),
        reference=torch.ones((1, 4, 8, 8)),
        control=torch.full((2, 3, 16, 16), 0.3),
    )

    assert probe.first_nonfinite is None
    assert missing_required_boundaries(probe.summary()["probed_labels"]) == []
    assert probe.summary()["decision"] == (
        "PRESERVE_FINITE_DECODED_TENSOR_AND_PATCH_POSTPROCESS_PATH_ONLY"
    )


def test_probe_records_one_entry_per_denoising_step():
    pipe, steps = build_pipe(steps=4)
    probe = NumericProbe()
    install_numeric_probes(pipe, probe)
    drive(
        pipe,
        steps=steps,
        initial_noise=torch.zeros((2, 4, 8, 8)),
        reference=torch.ones((1, 4, 8, 8)),
        control=torch.full((2, 3, 16, 16), 0.3),
    )

    labels = probe.summary()["probed_labels"]
    assert [label for label in labels if label.startswith("unet_noise_pred")] == [
        f"unet_noise_pred_step_{index:02d}" for index in range(4)
    ]
    assert [label for label in labels if label.startswith("scheduler_latents")] == [
        f"scheduler_latents_step_{index:02d}" for index in range(4)
    ]
    assert [label for label in labels if label.startswith("adapter_state")] == [
        "adapter_state_00",
        "adapter_state_01",
        "adapter_state_02",
    ]


def test_probes_do_not_alter_any_tensor_value():
    pipe, steps = build_pipe(steps=2)
    probe = NumericProbe()
    install_numeric_probes(pipe, probe)

    decoded = drive(
        pipe,
        steps=steps,
        initial_noise=torch.zeros((2, 4, 8, 8)),
        reference=torch.ones((1, 4, 8, 8)),
        control=torch.full((2, 3, 16, 16), 0.3),
    )[0]

    # Two steps of +0.1 on zeros, decoded through the fake VAE.
    assert torch.allclose(decoded, torch.full((2, 3, 16, 16), 0.5))
    assert torch.allclose(
        pipe.vae.seen_decode_inputs[0], torch.full((2, 4, 8, 8), 0.2) / 0.18215
    )


def test_unet_probe_is_layered_outside_the_relay():
    pipe, steps = build_pipe(steps=1)
    probe = NumericProbe()
    placement = install_numeric_probes(pipe, probe)

    drive(
        pipe,
        steps=steps,
        initial_noise=torch.zeros((2, 4, 8, 8)),
        reference=torch.ones((1, 4, 8, 8)),
        control=torch.full((2, 3, 16, 16), 0.3),
    )

    assert placement["unet_probe_placement"] == "outside_relay_inside_accelerate_hook"
    assert placement["condition_encoder_probe_placement"] == "inside_accelerate_hook"
    # The relay still ran for the reference pass and for the denoising step.
    assert len(pipe.relay_calls) == 2
    assert "cache_hidden_states" in pipe.relay_calls[0]
    assert "ref_hidden_states" in pipe.relay_calls[1]


def test_fail_closed_at_the_first_nonfinite_unet_output():
    pipe, steps = build_pipe(steps=4, bad="unet_1")
    probe = NumericProbe()
    install_numeric_probes(pipe, probe)

    with pytest.raises(FirstNonfiniteTensor):
        drive(
            pipe,
            steps=steps,
            initial_noise=torch.zeros((2, 4, 8, 8)),
            reference=torch.ones((1, 4, 8, 8)),
            control=torch.full((2, 3, 16, 16), 0.3),
        )

    summary = probe.summary()
    assert summary["first_nonfinite_label"] == "unet_noise_pred_step_01"
    assert summary["boundary_category"] == "UNET_OUTPUT"
    assert summary["decision"] == (
        "STOP_PLAIN_I2MV_ROUTE_INSPECT_OFFICIAL_GEOMETRY_GUIDED_SD21_MV_ADAPTER"
    )
    # Later steps must not run once the gate has fired.
    assert "scheduler_latents_step_01" not in summary["probed_labels"]
    assert "vae_decode_input" not in summary["probed_labels"]


def test_fail_closed_at_a_nonfinite_adapter_state():
    pipe, steps = build_pipe(steps=2, bad="adapter")
    probe = NumericProbe()
    install_numeric_probes(pipe, probe)

    with pytest.raises(FirstNonfiniteTensor):
        drive(
            pipe,
            steps=steps,
            initial_noise=torch.zeros((2, 4, 8, 8)),
            reference=torch.ones((1, 4, 8, 8)),
            control=torch.full((2, 3, 16, 16), 0.3),
        )

    summary = probe.summary()
    assert summary["first_nonfinite_label"] == "adapter_state_01"
    assert summary["decision"] == (
        "INSPECT_CONTROL_IMAGE_NORMALISATION_AND_CONDITION_ENCODER_PRECISION"
    )


def test_fail_closed_at_a_nonfinite_vae_decode_output():
    pipe, steps = build_pipe(steps=2, bad="decode_output")
    probe = NumericProbe()
    install_numeric_probes(pipe, probe)

    with pytest.raises(FirstNonfiniteTensor):
        drive(
            pipe,
            steps=steps,
            initial_noise=torch.zeros((2, 4, 8, 8)),
            reference=torch.ones((1, 4, 8, 8)),
            control=torch.full((2, 3, 16, 16), 0.3),
        )

    summary = probe.summary()
    assert summary["first_nonfinite_label"] == "vae_decode_output"
    assert summary["decision"] == "REPAIR_VAE_SCALING_DTYPE_BOUNDARY_ONE_CORRECTED_RETRY"
    # The finite pre-decode boundaries were still captured as evidence.
    assert "final_pre_decode_latents" in summary["probed_labels"]
    assert "vae_decode_input" in summary["probed_labels"]


def test_missing_required_boundaries_reports_unreached_probes():
    missing = missing_required_boundaries(
        ["reference_latents", "initial_noise_latents", "unet_noise_pred_step_00"]
    )
    assert "vae_decode_input" in missing
    assert "reference_latents" not in missing
    assert set(missing).issubset(set(REQUIRED_BOUNDARIES))
