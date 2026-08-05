from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from workers.mv_adapter_condition_encoder_fp32 import (
    ConditionEncoderPrecisionError,
    install_fp32_condition_encoder_boundary,
    prepare_fp32_condition_encoder,
    validate_adapter_state_for_cast,
    verify_module_tensors_finite,
)


FP16_MAX = 65504.0


class FakeCondEncoder(torch.nn.Module):
    """Stand-in T2IAdapter returning a fixed list of residual states."""

    def __init__(self, states, *, weight_value=0.5):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.full((4,), weight_value))
        self.register_buffer("running_scale", torch.ones((2,)))
        self._states = states
        self.seen_input_dtypes: list[str] = []

    def forward(self, control):
        self.seen_input_dtypes.append(str(control.dtype))
        return [state.clone() for state in self._states]


class FakePipe:
    def __init__(self, cond_encoder):
        self.cond_encoder = cond_encoder


def offload(module):
    """Attach an Accelerate-style hook and `_old_forward`, as `cpu_offload` does."""

    module._hf_hook = object()
    module._old_forward = module.forward

    def hooked_forward(*args, **kwargs):
        return module._old_forward(*args, **kwargs)

    module.forward = hooked_forward
    return module


def build(states, *, weight_value=0.5, with_hook=True):
    encoder = FakeCondEncoder(states, weight_value=weight_value)
    pipe = FakePipe(encoder)
    prepare_fp32_condition_encoder(pipe, torch)
    if with_hook:
        offload(encoder)
    return pipe, encoder


# --- weight verification ------------------------------------------------


def test_finite_weights_pass_verification():
    encoder = FakeCondEncoder([torch.zeros((1, 4))])
    summary = verify_module_tensors_finite(encoder)

    assert summary["all_finite"] is True
    assert summary["parameter_count"] == 1
    assert summary["buffer_count"] == 1


def test_nonfinite_weights_fail_before_inference():
    encoder = FakeCondEncoder([torch.zeros((1, 4))])
    with torch.no_grad():
        encoder.weight[2] = float("nan")

    with pytest.raises(ConditionEncoderPrecisionError, match="nonfinite before inference"):
        verify_module_tensors_finite(encoder)


def test_nonfinite_buffers_also_fail():
    encoder = FakeCondEncoder([torch.zeros((1, 4))])
    encoder.running_scale[0] = float("inf")

    with pytest.raises(ConditionEncoderPrecisionError):
        verify_module_tensors_finite(encoder)


# --- FP32 preparation ---------------------------------------------------


def test_preparation_casts_parameters_and_buffers_to_fp32():
    encoder = FakeCondEncoder([torch.zeros((1, 4))]).to(dtype=torch.float16)
    pipe = FakePipe(encoder)

    summary = prepare_fp32_condition_encoder(pipe, torch)

    assert summary["dtypes_before"] == ["torch.float16"]
    assert summary["floating_dtypes_after"] == ["torch.float32"]
    assert encoder.weight.dtype == torch.float32
    assert encoder.running_scale.dtype == torch.float32
    assert summary["weights_finite_before_cast"] is True


def test_preparation_refuses_to_run_after_the_offload_hook_exists():
    encoder = offload(FakeCondEncoder([torch.zeros((1, 4))]))
    pipe = FakePipe(encoder)

    with pytest.raises(ConditionEncoderPrecisionError, match="already carries an Accelerate hook"):
        prepare_fp32_condition_encoder(pipe, torch)


def test_condition_encoder_parameters_remain_fp32_with_the_offload_hook_installed():
    states = [torch.full((1, 4), 0.25)]
    pipe, encoder = build(states)

    boundary = install_fp32_condition_encoder_boundary(
        pipe, torch, target_dtype=torch.float16
    )
    encoder.forward(torch.full((1, 4), 0.5, dtype=torch.float16))

    assert boundary["placement"] == "inside_accelerate_hook"
    assert getattr(encoder, "_hf_hook", None) is not None
    assert encoder.weight.dtype == torch.float32
    assert encoder.running_scale.dtype == torch.float32


# --- adapter state validation ------------------------------------------


def test_finite_fp32_state_is_recorded_and_castable():
    record = validate_adapter_state_for_cast(
        0, torch.tensor([[-2.0, 4.0]], dtype=torch.float32), torch.float16
    )
    statistics = record["statistics"]

    assert record["label"] == "adapter_state_00"
    assert statistics["shape"] == [1, 2]
    assert statistics["minimum"] == pytest.approx(-2.0)
    assert statistics["maximum"] == pytest.approx(4.0)
    assert statistics["mean"] == pytest.approx(1.0)
    assert statistics["standard_deviation"] == pytest.approx(3.0)
    assert statistics["absolute_maximum"] == pytest.approx(4.0)
    assert record["target_dtype_maximum"] == pytest.approx(FP16_MAX)


def test_nonfinite_state_fails_validation():
    with pytest.raises(ConditionEncoderPrecisionError, match="is nonfinite"):
        validate_adapter_state_for_cast(
            1, torch.tensor([0.0, float("nan")], dtype=torch.float32), torch.float16
        )


def test_state_outside_fp16_range_fails_before_casting():
    with pytest.raises(ConditionEncoderPrecisionError, match="exceeds .* range before cast"):
        validate_adapter_state_for_cast(
            2, torch.tensor([FP16_MAX * 2], dtype=torch.float32), torch.float16
        )


def test_state_at_the_fp16_limit_is_accepted():
    record = validate_adapter_state_for_cast(
        0, torch.tensor([FP16_MAX], dtype=torch.float32), torch.float16
    )
    assert record["statistics"]["absolute_maximum"] == pytest.approx(FP16_MAX)


# --- boundary behaviour -------------------------------------------------


def test_finite_fp32_outputs_become_finite_fp16_residuals():
    states = [
        torch.full((1, 4), 0.25, dtype=torch.float32),
        torch.full((1, 4), -3.5, dtype=torch.float32),
        torch.full((1, 4), 1000.0, dtype=torch.float32),
    ]
    pipe, encoder = build(states)
    boundary = install_fp32_condition_encoder_boundary(
        pipe, torch, target_dtype=torch.float16
    )

    residuals = encoder.forward(torch.full((1, 4), 0.5, dtype=torch.float16))

    assert isinstance(residuals, list)
    assert len(residuals) == 3
    assert all(item.dtype == torch.float16 for item in residuals)
    assert all(bool(torch.isfinite(item).all()) for item in residuals)
    assert torch.allclose(residuals[1], torch.full((1, 4), -3.5, dtype=torch.float16))
    assert boundary["converted_dtypes"] == ["torch.float16"] * 3
    assert [record["label"] for record in boundary["adapter_state_records"]] == [
        "adapter_state_00",
        "adapter_state_01",
        "adapter_state_02",
    ]


def test_control_input_enters_the_encoder_as_fp32():
    states = [torch.full((1, 4), 0.25, dtype=torch.float32)]
    pipe, encoder = build(states)
    boundary = install_fp32_condition_encoder_boundary(
        pipe, torch, target_dtype=torch.float16
    )

    encoder.forward(torch.full((1, 4), 0.5, dtype=torch.float16))

    assert encoder.seen_input_dtypes == ["torch.float32"]
    assert boundary["encoder_input_dtype"] == "torch.float16"
    assert boundary["encoder_compute_dtype"] == "torch.float32"
    assert boundary["input_statistics"]["dtype"] == "torch.float32"
    assert boundary["input_statistics"]["nonfinite_count"] == 0


def test_nonfinite_encoder_output_fails_before_the_unet():
    states = [
        torch.full((1, 4), 0.25, dtype=torch.float32),
        torch.full((1, 4), float("nan"), dtype=torch.float32),
    ]
    pipe, encoder = build(states)
    install_fp32_condition_encoder_boundary(pipe, torch, target_dtype=torch.float16)

    with pytest.raises(ConditionEncoderPrecisionError) as excinfo:
        encoder.forward(torch.full((1, 4), 0.5, dtype=torch.float16))

    assert "adapter_state_01 is nonfinite" in str(excinfo.value)


def test_out_of_fp16_range_output_fails_at_the_boundary():
    states = [torch.full((1, 4), FP16_MAX * 4, dtype=torch.float32)]
    pipe, encoder = build(states)
    install_fp32_condition_encoder_boundary(pipe, torch, target_dtype=torch.float16)

    with pytest.raises(ConditionEncoderPrecisionError, match="exceeds .* range before cast"):
        encoder.forward(torch.full((1, 4), 0.5, dtype=torch.float16))


def test_boundary_defaults_the_target_dtype_to_the_incoming_control_dtype():
    states = [torch.full((1, 4), 0.25, dtype=torch.float32)]
    pipe, encoder = build(states)
    boundary = install_fp32_condition_encoder_boundary(pipe, torch)

    residuals = encoder.forward(torch.full((1, 4), 0.5, dtype=torch.float16))

    assert boundary["resolved_target_dtype"] == "torch.float16"
    assert residuals[0].dtype == torch.float16


def test_boundary_refuses_to_install_without_the_offload_hook():
    states = [torch.full((1, 4), 0.25, dtype=torch.float32)]
    pipe, _encoder = build(states, with_hook=False)

    with pytest.raises(ConditionEncoderPrecisionError, match="CPU-offload hook is missing"):
        install_fp32_condition_encoder_boundary(pipe, torch, target_dtype=torch.float16)


def test_boundary_rejects_a_call_without_a_control_tensor():
    states = [torch.full((1, 4), 0.25, dtype=torch.float32)]
    pipe, encoder = build(states)
    install_fp32_condition_encoder_boundary(pipe, torch, target_dtype=torch.float16)

    with pytest.raises(ConditionEncoderPrecisionError, match="without a control tensor"):
        encoder.forward()


def test_residuals_stay_mutable_for_control_conditioning_scale():
    states = [torch.full((1, 4), 0.25, dtype=torch.float32)]
    pipe, encoder = build(states)
    install_fp32_condition_encoder_boundary(pipe, torch, target_dtype=torch.float16)

    residuals = encoder.forward(torch.full((1, 4), 0.5, dtype=torch.float16))
    # The official pipeline does exactly this after calling the encoder.
    for index, state in enumerate(residuals):
        residuals[index] = state * 1.0

    assert residuals[0].dtype == torch.float16
