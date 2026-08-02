"""Run the dynamic MV-Adapter condition encoder entirely in FP32.

The numeric-boundary diagnostic proved the first nonfinite tensor is
``adapter_state_00``: the ``T2IAdapter`` condition encoder receives a fully
finite FP16 control tensor in ``[0, 1]`` and returns residuals with zero finite
elements. Control normalisation matches the official ``do_normalize=False``
route and is deliberately left alone.

This module changes precision only, at one boundary:

* condition-encoder parameters and buffers are verified finite, then cast to
  FP32 *before* the Accelerate CPU-offload hook is installed, so the hook's
  weights map captures the FP32 tensors;
* the incoming control tensor is cast to FP32 at the encoder boundary;
* every returned adapter state is validated - finite, and within the target
  dtype's representable range - and only then cast back to the UNet latent
  dtype;
* FP32 temporaries are dropped as soon as each state is converted.

Nothing else moves: the UNet, the FP32 VAE boundaries, the reference-cache
relay, the scheduler, the weights, the seed, the camera order, the view count
and the prompts are untouched.
"""
from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mv_adapter_numeric_probe import tensor_statistics


class ConditionEncoderPrecisionError(RuntimeError):
    """Raised when the condition encoder cannot be trusted downstream.

    Always raised *before* the offending tensor can reach the UNet.
    """


def iter_named_tensors(module: Any):
    """Yield every parameter and buffer of a module as ``(kind, name, tensor)``."""

    for name, parameter in module.named_parameters():
        yield "parameter", name, parameter
    for name, buffer in module.named_buffers():
        yield "buffer", name, buffer


def verify_module_tensors_finite(module: Any) -> dict[str, Any]:
    """Fail closed if any condition-encoder parameter or buffer is nonfinite."""

    import torch

    parameter_count = 0
    buffer_count = 0
    element_count = 0
    nonfinite_tensors: list[dict[str, Any]] = []

    for kind, name, tensor in iter_named_tensors(module):
        if kind == "parameter":
            parameter_count += 1
        else:
            buffer_count += 1
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            continue
        element_count += int(tensor.numel())
        nonfinite = int((~torch.isfinite(tensor.detach())).sum().item())
        if nonfinite:
            nonfinite_tensors.append(
                {"kind": kind, "name": name, "nonfinite_count": nonfinite}
            )

    summary = {
        "parameter_count": parameter_count,
        "buffer_count": buffer_count,
        "element_count": element_count,
        "nonfinite_tensor_count": len(nonfinite_tensors),
        "nonfinite_tensors": nonfinite_tensors[:8],
        "all_finite": not nonfinite_tensors,
    }
    if nonfinite_tensors:
        first = nonfinite_tensors[0]
        raise ConditionEncoderPrecisionError(
            "condition-encoder weights are nonfinite before inference: "
            f"{first['kind']} {first['name']} "
            f"({first['nonfinite_count']} nonfinite elements)"
        )
    return summary


def prepare_fp32_condition_encoder(pipe: Any, torch_module: Any) -> dict[str, Any]:
    """Verify then cast the condition encoder to FP32.

    Must run *before* the Accelerate CPU-offload hook is installed. Casting a
    module after ``cpu_offload`` would leave the hook's weights map holding the
    original FP16 tensors, which are the ones actually restored per forward.
    """

    cond_encoder = pipe.cond_encoder
    if getattr(cond_encoder, "_hf_hook", None) is not None:
        raise ConditionEncoderPrecisionError(
            "condition encoder already carries an Accelerate hook; cast to FP32 "
            "before installing CPU offload or the hook's weights map keeps FP16"
        )

    dtypes_before = sorted(
        {str(tensor.dtype) for _kind, _name, tensor in iter_named_tensors(cond_encoder)}
    )
    finiteness = verify_module_tensors_finite(cond_encoder)

    cond_encoder.to(dtype=torch_module.float32)

    dtypes_after = sorted(
        {str(tensor.dtype) for _kind, _name, tensor in iter_named_tensors(cond_encoder)}
    )
    floating_after = sorted(
        {
            str(tensor.dtype)
            for _kind, _name, tensor in iter_named_tensors(cond_encoder)
            if tensor.is_floating_point()
        }
    )
    if floating_after not in ([], ["torch.float32"]):
        raise ConditionEncoderPrecisionError(
            f"condition encoder is not fully FP32 after cast: {floating_after}"
        )

    # Re-verify: a cast must not have introduced nonfinite values.
    verify_module_tensors_finite(cond_encoder)

    return {
        "dtypes_before": dtypes_before,
        "dtypes_after": dtypes_after,
        "floating_dtypes_after": floating_after,
        "weights_finite_before_cast": finiteness["all_finite"],
        "parameter_count": finiteness["parameter_count"],
        "buffer_count": finiteness["buffer_count"],
        "element_count": finiteness["element_count"],
    }


def validate_adapter_state_for_cast(
    index: int, state: Any, target_dtype: Any
) -> dict[str, Any]:
    """Fail closed unless a single adapter state is finite and castable.

    Returns the recorded statistics: shape, min, max, mean, std and absolute
    maximum, alongside the target dtype's representable maximum.
    """

    import torch

    if not isinstance(state, torch.Tensor):
        raise ConditionEncoderPrecisionError(
            f"adapter state {index} is not a tensor: {type(state)!r}"
        )

    statistics = tensor_statistics(state)
    target_maximum = float(torch.finfo(target_dtype).max)
    record = {
        "adapter_state_index": index,
        "label": f"adapter_state_{index:02d}",
        "target_dtype": str(target_dtype),
        "target_dtype_maximum": target_maximum,
        "statistics": statistics,
    }

    if statistics["nonfinite_count"] > 0:
        raise ConditionEncoderPrecisionError(
            f"FP32 condition-encoder output adapter_state_{index:02d} is nonfinite: "
            f"{statistics['nonfinite_count']}/"
            f"{statistics['nonfinite_count'] + statistics['finite_count']} elements"
        )

    absolute_maximum = statistics["absolute_maximum"]
    if absolute_maximum is not None and absolute_maximum > target_maximum:
        raise ConditionEncoderPrecisionError(
            f"adapter_state_{index:02d} exceeds {target_dtype} range before cast: "
            f"absolute_maximum={absolute_maximum} limit={target_maximum}"
        )
    return record


def install_fp32_condition_encoder_boundary(
    pipe: Any,
    torch_module: Any,
    *,
    target_dtype: Any = None,
    require_offload_hook: bool = True,
) -> dict[str, Any]:
    """Wrap the condition encoder so it runs in FP32 and returns validated states.

    Installed inside the Accelerate CPU-offload wrapper, exactly like the
    reference-cache relay, so the independent offload hook is preserved.
    """

    cond_encoder = pipe.cond_encoder
    hook = getattr(cond_encoder, "_hf_hook", None)
    old_forward = getattr(cond_encoder, "_old_forward", None)

    if hook is not None and callable(old_forward):
        inner_forward = old_forward
        placement = "inside_accelerate_hook"
    else:
        if require_offload_hook:
            raise ConditionEncoderPrecisionError(
                "condition-encoder CPU-offload hook is missing; refusing to "
                "install the FP32 boundary outside it"
            )
        inner_forward = cond_encoder.forward
        placement = "direct_forward"

    state: dict[str, Any] = {
        "installed": True,
        "placement": placement,
        "accelerate_hook": type(hook).__name__ if hook is not None else None,
        "call_count": 0,
        "encoder_input_dtype": None,
        "encoder_compute_dtype": str(torch_module.float32),
        "resolved_target_dtype": None if target_dtype is None else str(target_dtype),
        "input_statistics": None,
        "adapter_state_records": [],
        "converted_dtypes": [],
    }

    def fp32_condition_encoder_forward(*args, **kwargs):
        if not args or not isinstance(args[0], torch_module.Tensor):
            raise ConditionEncoderPrecisionError(
                "condition encoder was called without a control tensor"
            )

        control = args[0]
        resolved_target = target_dtype if target_dtype is not None else control.dtype
        state["call_count"] += 1
        state["encoder_input_dtype"] = str(control.dtype)
        state["resolved_target_dtype"] = str(resolved_target)

        control_fp32 = control.to(dtype=torch_module.float32)
        state["input_statistics"] = tensor_statistics(control_fp32)

        raw_states = inner_forward(control_fp32, *args[1:], **kwargs)
        del control_fp32

        sequence = (
            list(raw_states)
            if isinstance(raw_states, (list, tuple))
            else [raw_states]
        )
        del raw_states

        records: list[dict[str, Any]] = []
        converted: list[Any] = []
        for index in range(len(sequence)):
            adapter_state = sequence[index]
            # Drop the FP32 reference held by the sequence straight away so the
            # only live copy is the local one, freed at the end of this step.
            sequence[index] = None
            record = validate_adapter_state_for_cast(index, adapter_state, resolved_target)
            cast_state = adapter_state.to(dtype=resolved_target)
            del adapter_state

            cast_nonfinite = int(
                (~torch_module.isfinite(cast_state.detach())).sum().item()
            )
            if cast_nonfinite:
                raise ConditionEncoderPrecisionError(
                    f"adapter_state_{index:02d} became nonfinite when cast to "
                    f"{resolved_target}: {cast_nonfinite} elements"
                )
            record["cast_dtype"] = str(cast_state.dtype)
            record["cast_nonfinite_count"] = cast_nonfinite
            records.append(record)
            converted.append(cast_state)

        state["adapter_state_records"] = records
        state["converted_dtypes"] = [str(item.dtype) for item in converted]

        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()

        # The pipeline mutates this in place for control_conditioning_scale.
        return converted

    if placement == "inside_accelerate_hook":
        cond_encoder._old_forward = fp32_condition_encoder_forward
    else:
        cond_encoder.forward = fp32_condition_encoder_forward

    pipe._lowvram3d_condition_encoder_fp32_state = state
    return state
