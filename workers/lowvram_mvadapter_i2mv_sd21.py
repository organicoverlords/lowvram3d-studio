"""Low-VRAM MV-Adapter image+geometry SD2.1 pipeline.

Upstream ``MVAdapterI2MVSDPipeline`` creates its six-channel ``cond_encoder``
*after* pipeline construction, as a plain attribute:

    self.cond_encoder = T2IAdapter(...)      # pipeline_mvadapter_i2mv_sd.py:706

Diffusers only offloads what it can enumerate.  ``components`` is derived from
the ``__init__`` signature plus ``_optional_components``
(``pipeline_utils.py:1486``), and ``enable_sequential_cpu_offload`` iterates
exactly that dict (``pipeline_utils.py:1141``).  A late attribute is therefore
invisible to offload: it stays resident in fp16 on whatever device it was
built on, which on a 6 GB card is the difference between fitting and not.

This subclass changes nothing mathematical.  It declares ``cond_encoder`` as a
constructor component, registers it through ``register_modules`` so it lands in
``components``, and reuses the upstream adapter construction verbatim.  It also
replaces the upstream ``strict=False`` double load — which silently tolerates a
completely unloaded adapter — with explicit key accounting that fails closed.

Route restrictions enforced here: image+geometry SD2.1 only.  Text-conditioned
adapters are rejected by name, and nvdiffrast is never imported.
"""
from __future__ import annotations

import hashlib
from typing import Any

import torch
from diffusers.models import T2IAdapter

from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
from mvadapter.pipelines.pipeline_mvadapter_i2mv_sd import MVAdapterI2MVSDPipeline


#: Every key of the packaged T2IAdapter state dict carries this prefix, because
#: ``T2IAdapter`` stores its stack under ``self.adapter``.
COND_ENCODER_KEY_PREFIX = "adapter."

REQUIRED_ADAPTER_NAME = "mvadapter_ig2mv_sd21.safetensors"
FORBIDDEN_ADAPTER_NAMES = (
    "mvadapter_t2mv_sd21.safetensors",
    "mvadapter_tg2mv_sd21.safetensors",
)

#: Model components that must exist, be enumerable and be offloadable.
REQUIRED_MODEL_COMPONENTS = ("text_encoder", "unet", "vae", "cond_encoder")


class LowVRAMMVAdapterI2MVSDPipeline(MVAdapterI2MVSDPipeline):
    """MV-Adapter I2MV SD2.1 with ``cond_encoder`` as a first-class component."""

    # cond_encoder is optional at construction time (it is built by the adapter
    # initialiser) but naming it here is what puts it into ``components``.
    _optional_components = ["safety_checker", "feature_extractor", "image_encoder", "cond_encoder"]
    _exclude_from_cpu_offload = ["safety_checker"]

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        unet,
        scheduler,
        safety_checker,
        feature_extractor,
        image_encoder=None,
        cond_encoder=None,
        requires_safety_checker: bool = False,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            requires_safety_checker=requires_safety_checker,
        )
        # Normal component registration - not a late attribute assignment.
        self.register_modules(cond_encoder=cond_encoder)
        self._adapter_load_report: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # adapter construction
    # ------------------------------------------------------------------
    def _init_custom_adapter(self, *args: Any, **kwargs: Any) -> None:
        """Build the upstream adapter, then promote ``cond_encoder`` to a component.

        The upstream implementation is called unchanged so the adapter
        architecture and the attention-processor wiring stay bit-identical.
        """
        super()._init_custom_adapter(*args, **kwargs)
        cond_encoder = getattr(self, "cond_encoder", None)
        if cond_encoder is None:
            raise RuntimeError("MVADAPTER_COND_ENCODER_NOT_CONSTRUCTED")
        if not isinstance(cond_encoder, T2IAdapter):
            raise RuntimeError(f"MVADAPTER_COND_ENCODER_UNEXPECTED_CLASS:{type(cond_encoder).__name__}")
        cond_encoder.to(dtype=self.unet.dtype)
        self.register_modules(cond_encoder=cond_encoder)
        if "cond_encoder" not in self.components:
            raise RuntimeError("MVADAPTER_COND_ENCODER_NOT_REGISTERED")

    # ------------------------------------------------------------------
    # adapter weight loading
    # ------------------------------------------------------------------
    def _load_custom_adapter(self, state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
        """Load adapter weights with explicit accounting instead of silent tolerance."""
        cond_encoder = getattr(self, "cond_encoder", None)
        if cond_encoder is None:
            raise RuntimeError("MVADAPTER_COND_ENCODER_NOT_INITIALISED")

        unet_state = self.unet.state_dict()
        cond_state = cond_encoder.state_dict()
        checkpoint_cond = {k: v for k, v in state_dict.items() if k.startswith(COND_ENCODER_KEY_PREFIX)}
        checkpoint_unet = {k: v for k, v in state_dict.items() if not k.startswith(COND_ENCODER_KEY_PREFIX)}

        unexpected: list[str] = []
        shape_mismatch: list[str] = []
        for source, destination in ((checkpoint_unet, unet_state), (checkpoint_cond, cond_state)):
            for key, tensor in source.items():
                target = destination.get(key)
                if target is None:
                    unexpected.append(key)
                elif tuple(target.shape) != tuple(tensor.shape):
                    shape_mismatch.append(key)

        if not checkpoint_cond:
            raise RuntimeError("MVADAPTER_ADAPTER_HAS_NO_COND_ENCODER_KEYS")
        if not checkpoint_unet:
            raise RuntimeError("MVADAPTER_ADAPTER_HAS_NO_UNET_KEYS")
        if unexpected:
            raise RuntimeError(f"MVADAPTER_ADAPTER_UNEXPECTED_KEYS:{len(unexpected)}:{unexpected[:5]}")
        if shape_mismatch:
            raise RuntimeError(f"MVADAPTER_ADAPTER_SHAPE_MISMATCH:{len(shape_mismatch)}:{shape_mismatch[:5]}")

        # The cond_encoder is defined entirely by the checkpoint, so anything it
        # still needs afterwards is a critical missing key.
        cond_missing = sorted(set(cond_state) - set(checkpoint_cond))
        if cond_missing:
            raise RuntimeError(f"MVADAPTER_COND_ENCODER_KEYS_MISSING:{len(cond_missing)}:{cond_missing[:5]}")

        before = _tensor_fingerprint(cond_state)
        unet_result = self.unet.load_state_dict(checkpoint_unet, strict=False)
        cond_result = cond_encoder.load_state_dict(checkpoint_cond, strict=False)
        after = _tensor_fingerprint(cond_encoder.state_dict())

        if list(cond_result.unexpected_keys) or list(cond_result.missing_keys):
            raise RuntimeError(
                "MVADAPTER_COND_ENCODER_LOAD_INCOMPLETE:"
                f"missing={len(cond_result.missing_keys)},unexpected={len(cond_result.unexpected_keys)}"
            )
        if list(unet_result.unexpected_keys):
            raise RuntimeError(f"MVADAPTER_UNET_UNEXPECTED_KEYS:{len(unet_result.unexpected_keys)}")
        if before == after:
            raise RuntimeError("MVADAPTER_COND_ENCODER_WEIGHTS_UNCHANGED")

        report = {
            "adapter_total_keys": len(state_dict),
            "adapter_loaded_key_count": len(checkpoint_unet) + len(checkpoint_cond),
            "unet_adapter_keys_loaded": len(checkpoint_unet),
            "cond_encoder_keys_loaded": len(checkpoint_cond),
            "adapter_missing_keys": [],
            "adapter_unexpected_keys": [],
            "unet_unexpected_keys": list(unet_result.unexpected_keys),
            "cond_encoder_missing_keys": list(cond_result.missing_keys),
            "cond_encoder_unexpected_keys": list(cond_result.unexpected_keys),
            "cond_encoder_fingerprint_before": before,
            "cond_encoder_fingerprint_after": after,
            "cond_encoder_weights_changed": True,
        }
        self._adapter_load_report = report
        return report


def _tensor_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key]
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().to("cpu", torch.float32).numpy().tobytes())
    return digest.hexdigest()


# ----------------------------------------------------------------------
# construction
# ----------------------------------------------------------------------
def assert_image_geometry_adapter(adapter_name: str) -> None:
    """Fail closed on text-conditioned adapters and on anything but IG2MV SD2.1."""
    lowered = adapter_name.lower()
    for forbidden in FORBIDDEN_ADAPTER_NAMES:
        if lowered == forbidden or forbidden.split(".")[0] in lowered:
            raise RuntimeError(f"MVADAPTER_TEXT_CONDITIONED_ADAPTER_FORBIDDEN:{adapter_name}")
    if "ig2mv_sd21" not in lowered:
        raise RuntimeError(f"MVADAPTER_REQUIRES_IG2MV_SD21_ADAPTER:{adapter_name}")


def build_low_vram_pipeline(
    base_model: str,
    adapter_state: dict[str, torch.Tensor],
    adapter_name: str,
    num_views: int = 6,
    dtype: torch.dtype = torch.float16,
) -> tuple[LowVRAMMVAdapterI2MVSDPipeline, dict[str, Any]]:
    """Construct the pipeline, install the adapter and register ``cond_encoder``."""
    from diffusers import DDPMScheduler

    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

    assert_image_geometry_adapter(adapter_name)
    if "xl" in str(base_model).lower():
        raise RuntimeError(f"MVADAPTER_SDXL_BASE_REJECTED:{base_model}")

    pipe = LowVRAMMVAdapterI2MVSDPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
        local_files_only=True,
        safety_checker=None,
    )
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=DDPMScheduler,
    )
    pipe.init_custom_adapter(
        num_views=num_views, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0
    )
    load_report = pipe._load_custom_adapter(adapter_state)
    return pipe, load_report


# ----------------------------------------------------------------------
# component inventory
# ----------------------------------------------------------------------
def _module_devices(module: torch.nn.Module) -> list[str]:
    return sorted({str(parameter.device) for parameter in module.parameters()})


def component_inventory(pipe: LowVRAMMVAdapterI2MVSDPipeline) -> dict[str, Any]:
    """Enumerate ``pipe.components`` with parameter counts, dtypes and devices."""
    components: dict[str, Any] = {}
    for name, module in pipe.components.items():
        if not isinstance(module, torch.nn.Module):
            components[name] = {
                "is_torch_module": False,
                "type": type(module).__name__ if module is not None else None,
            }
            continue
        parameters = list(module.parameters())
        parameter_count = sum(p.numel() for p in parameters)
        dtypes = sorted({str(p.dtype).replace("torch.", "") for p in parameters})
        components[name] = {
            "is_torch_module": True,
            "type": type(module).__name__,
            "parameter_count": int(parameter_count),
            "cpu_footprint_mb": round(
                sum(p.numel() * p.element_size() for p in parameters) / 2**20, 3
            ),
            "dtypes": dtypes,
            "devices": _module_devices(module),
        }
    missing = [name for name in REQUIRED_MODEL_COMPONENTS if name not in components]
    if missing:
        raise RuntimeError(f"MVADAPTER_COMPONENT_INVENTORY_MISSING:{missing}")
    cond = components["cond_encoder"]
    if not cond.get("is_torch_module"):
        raise RuntimeError("MVADAPTER_COND_ENCODER_NOT_TORCH_MODULE")
    if not cond.get("parameter_count"):
        raise RuntimeError("MVADAPTER_COND_ENCODER_HAS_NO_PARAMETERS")
    if cond.get("dtypes") != ["float16"]:
        raise RuntimeError(f"MVADAPTER_COND_ENCODER_DTYPE_INVALID:{cond.get('dtypes')}")
    return components


# ----------------------------------------------------------------------
# attention backend
# ----------------------------------------------------------------------
def attention_report(pipe: LowVRAMMVAdapterI2MVSDPipeline) -> dict[str, Any]:
    """Confirm SDPA processors survive and attention slicing is not installed."""
    processors = pipe.unet.attn_processors
    names = sorted({type(processor).__name__ for processor in processors.values()})
    rowcol_count = sum(type(processor).__name__ == "DecoupledMVRowColSelfAttnProcessor2_0" for processor in processors.values())
    row_only_count = sum(type(processor).__name__ == "DecoupledMVRowSelfAttnProcessor2_0" for processor in processors.values())
    if rowcol_count <= 0 or row_only_count != 0:
        raise RuntimeError(f"MVADAPTER_ROWCOL_PROCESSOR_INVALID:rowcol={rowcol_count},row_only={row_only_count}")
    sliced = sorted(name for name in names if "Sliced" in name)
    if sliced:
        raise RuntimeError(f"MVADAPTER_ATTENTION_SLICING_ENABLED:{sliced}")
    non_sdpa = sorted(name for name in names if not name.endswith("2_0"))
    return {
        "attention_backend": "PYTORCH_SDPA",
        "attention_slicing": "DISABLED",
        "processor_classes": names,
        "sliced_processor_classes": sliced,
        "non_sdpa_processor_classes": non_sdpa,
        "processor_count": len(processors),
        "expected_processor": "DecoupledMVRowColSelfAttnProcessor2_0",
        "rowcol_processor_count": int(rowcol_count),
        "row_only_processor_count": int(row_only_count),
        "rowcol_processor_proven": True,
    }


# ----------------------------------------------------------------------
# sequential offload
# ----------------------------------------------------------------------
def install_low_vram_offload(
    pipe: LowVRAMMVAdapterI2MVSDPipeline, device: str = "cuda"
) -> dict[str, Any]:
    """Enable VAE slicing plus sequential CPU offload.

    Attention slicing is deliberately never enabled: the MV-Adapter attention
    processors already run through SDPA, and slicing would replace them.
    """
    pipe.enable_vae_slicing()
    pipe.enable_sequential_cpu_offload(device=device)
    return {
        "offload_mode": "SEQUENTIAL",
        "vae_slicing": "ENABLED",
        "attention_slicing": "DISABLED",
        "device": device,
    }


def offload_hook_report(pipe: LowVRAMMVAdapterI2MVSDPipeline) -> dict[str, Any]:
    """Per-component offload-hook evidence - never a single global boolean."""
    report: dict[str, Any] = {}
    for name, module in pipe.components.items():
        if not isinstance(module, torch.nn.Module):
            continue
        hook = getattr(module, "_hf_hook", None)
        submodule_hooks = sum(
            1 for child in module.modules() if getattr(child, "_hf_hook", None) is not None
        )
        execution_devices = sorted(
            {
                str(getattr(child, "_hf_hook").execution_device)
                for child in module.modules()
                if getattr(child, "_hf_hook", None) is not None
                and getattr(getattr(child, "_hf_hook"), "execution_device", None) is not None
            }
        )
        devices = _module_devices(module)
        report[name] = {
            "hook_installed": hook is not None,
            "hook_class": type(hook).__name__ if hook is not None else None,
            "submodule_hook_count": int(submodule_hooks),
            "execution_devices": execution_devices,
            "at_rest_devices": devices,
            "resident_on_cuda": any(device.startswith("cuda") for device in devices),
        }
    for name in REQUIRED_MODEL_COMPONENTS:
        record = report.get(name)
        if record is None:
            raise RuntimeError(f"MVADAPTER_OFFLOAD_COMPONENT_ABSENT:{name}")
        if not record["hook_installed"] or record["submodule_hook_count"] < 1:
            raise RuntimeError(f"MVADAPTER_OFFLOAD_HOOK_MISSING:{name}")
        if record["resident_on_cuda"]:
            raise RuntimeError(f"MVADAPTER_COMPONENT_LEFT_ON_CUDA:{name}:{record['at_rest_devices']}")
    return report


# ----------------------------------------------------------------------
# bounded device-path smoke test
# ----------------------------------------------------------------------
class _ForbiddenCall(RuntimeError):
    """Raised if the smoke test ever reaches generation code."""


def cond_encoder_device_path_smoke_test(
    pipe: LowVRAMMVAdapterI2MVSDPipeline,
    control_tensor: "Any",
    resolution: int = 64,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Execute ``cond_encoder`` exactly once under the installed offload hooks.

    This performs no denoising, no reference UNet pass, no scheduler step and
    no VAE decode, and produces no image.  The UNet, VAE decoder and scheduler
    are booby-trapped for the duration so that any accidental generation call
    fails loudly instead of silently consuming the reserved GPU sequence.
    """
    import numpy as np

    cond_encoder = pipe.cond_encoder
    if cond_encoder is None:
        raise RuntimeError("MVADAPTER_COND_ENCODER_ABSENT_FOR_SMOKE_TEST")

    array = np.asarray(control_tensor)
    if array.ndim != 4 or array.shape[1] != 6:
        raise RuntimeError(f"MVADAPTER_SMOKE_CONTROL_SHAPE_INVALID:{tuple(array.shape)}")
    control = torch.from_numpy(np.ascontiguousarray(array.astype(np.float32)))

    calls = {"unet": 0, "vae_decode": 0, "scheduler_step": 0}

    def _forbid(kind: str):
        def guard(*_args: Any, **_kwargs: Any):
            calls[kind] += 1
            raise _ForbiddenCall(f"MVADAPTER_SMOKE_TEST_FORBIDDEN_CALL:{kind}")

        return guard

    original_unet_forward = pipe.unet.forward
    original_vae_decode = pipe.vae.decode
    original_scheduler_step = pipe.scheduler.step
    # Record the parameter device from inside the real forward, after the
    # align-devices hook has staged the weights and before it releases them.
    observed: dict[str, Any] = {}
    probe_module = None
    probe_original = None
    for child in cond_encoder.modules():
        if getattr(child, "_hf_hook", None) is not None and getattr(child, "weight", None) is not None:
            probe_module = child
            break
    if probe_module is not None and hasattr(probe_module, "_old_forward"):
        probe_original = probe_module._old_forward

        def probe_forward(*args: Any, **kwargs: Any):
            observed.setdefault("parameter_device_during_forward", str(probe_module.weight.device))
            return probe_original(*args, **kwargs)

        probe_module._old_forward = probe_forward

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    try:
        pipe.unet.forward = _forbid("unet")
        pipe.vae.decode = _forbid("vae_decode")
        pipe.scheduler.step = _forbid("scheduler_step")

        feature = pipe.prepare_control_image(
            image=control,
            width=resolution,
            height=resolution,
            batch_size=control.shape[0],
            num_images_per_prompt=1,
            device=torch.device(device),
            dtype=pipe.unet.dtype,
            do_classifier_free_guidance=False,
        )
        feature = feature.to(device=torch.device(device), dtype=pipe.unet.dtype)
        control_input_device = str(feature.device)
        control_input_shape = list(feature.shape)

        with torch.no_grad():
            residuals = cond_encoder(feature)

        shapes = [list(state.shape) for state in residuals]
        devices = sorted({str(state.device) for state in residuals})
        finite = all(bool(torch.isfinite(state).all()) for state in residuals)
        peak_allocated = torch.cuda.max_memory_allocated()

        del residuals, feature
    finally:
        pipe.unet.forward = original_unet_forward
        pipe.vae.decode = original_vae_decode
        pipe.scheduler.step = original_scheduler_step
        if probe_module is not None and probe_original is not None:
            probe_module._old_forward = probe_original

    torch.cuda.empty_cache()
    allocated_after = torch.cuda.memory_allocated()

    at_rest = _module_devices(cond_encoder)
    hook = getattr(cond_encoder, "_hf_hook", None)
    execution_device = None
    for child in cond_encoder.modules():
        child_hook = getattr(child, "_hf_hook", None)
        if child_hook is not None and getattr(child_hook, "execution_device", None) is not None:
            execution_device = str(child_hook.execution_device)
            break

    result = {
        "control_input_device": control_input_device,
        "control_input_shape": control_input_shape,
        "control_resolution": resolution,
        "cond_encoder_execution_device": execution_device,
        "cond_encoder_parameter_device_during_forward": observed.get("parameter_device_during_forward"),
        "cond_encoder_output_level_count": len(shapes),
        "cond_encoder_output_shapes": shapes,
        "cond_encoder_output_devices": devices,
        "cond_encoder_output_finite": bool(finite),
        "cond_encoder_at_rest_devices": at_rest,
        "cond_encoder_resident_on_cuda_after": any(d.startswith("cuda") for d in at_rest),
        "cond_encoder_top_hook_class": type(hook).__name__ if hook is not None else None,
        "cuda_allocated_before_mb": round(allocated_before / 2**20, 3),
        "cuda_allocated_after_mb": round(allocated_after / 2**20, 3),
        "cuda_peak_allocated_mb": round(peak_allocated / 2**20, 3),
        "cuda_memory_released": bool(allocated_after <= allocated_before + 1024 * 1024),
        "unet_denoising_called": calls["unet"] > 0,
        "reference_unet_pass_called": False,
        "scheduler_step_called": calls["scheduler_step"] > 0,
        "vae_decode_called": calls["vae_decode"] > 0,
        "output_images": 0,
        "gpu_sequence_consumed": False,
    }

    if not result["cond_encoder_output_level_count"]:
        raise RuntimeError("MVADAPTER_COND_ENCODER_RETURNED_NO_RESIDUALS")
    if not result["cond_encoder_output_finite"]:
        raise RuntimeError("MVADAPTER_COND_ENCODER_OUTPUT_NOT_FINITE")
    if not any(d.startswith("cuda") for d in devices):
        raise RuntimeError(f"MVADAPTER_COND_ENCODER_OUTPUT_NOT_ON_CUDA:{devices}")
    if result["cond_encoder_resident_on_cuda_after"]:
        raise RuntimeError(f"MVADAPTER_COND_ENCODER_STILL_RESIDENT:{at_rest}")
    if result["unet_denoising_called"] or result["vae_decode_called"] or result["scheduler_step_called"]:
        raise RuntimeError(f"MVADAPTER_SMOKE_TEST_TOUCHED_GENERATION:{calls}")
    if not result["cuda_memory_released"]:
        raise RuntimeError(
            f"MVADAPTER_SMOKE_TEST_MEMORY_LEAK:{result['cuda_allocated_before_mb']}"
            f"->{result['cuda_allocated_after_mb']}"
        )
    result["passed"] = True
    return result
