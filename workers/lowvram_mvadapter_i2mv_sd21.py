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
import hashlib
from collections import Counter
from functools import wraps
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
ROWCOL_PROCESSOR_NAME = "DecoupledMVRowColSelfAttnProcessor2_0"


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

    def set_reference_latents_override(self, latents: torch.Tensor) -> None:
        """Use one explicitly owned, prevalidated reference-latent tensor."""
        if not isinstance(latents, torch.Tensor):
            raise TypeError("MVADAPTER_REFERENCE_LATENTS_OVERRIDE_NOT_TENSOR")
        self._lowvram_reference_latents_override = latents

    def clear_reference_latents_override(self) -> None:
        self._lowvram_reference_latents_override = None

    def set_reference_cache_override(self, cache: dict[str, torch.Tensor]) -> None:
        if not isinstance(cache, dict) or not cache:
            raise RuntimeError("MVADAPTER_REFERENCE_CACHE_OVERRIDE_EMPTY")
        self._lowvram_reference_cache_override = cache

    def clear_reference_cache_override(self) -> None:
        self._lowvram_reference_cache_override = None

    def prepare_image_latents(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        override = getattr(self, "_lowvram_reference_latents_override", None)
        if override is not None:
            dtype = kwargs.get("dtype")
            device = kwargs.get("device")
            if dtype is None and len(args) >= 5:
                dtype = args[4]
            if device is None and len(args) >= 6:
                device = args[5]
            requested_batch = kwargs.get("batch_size")
            if requested_batch is None and len(args) >= 3:
                requested_batch = args[2]
            images_per_prompt = kwargs.get("num_images_per_prompt", 1)
            if images_per_prompt is None and len(args) >= 4:
                images_per_prompt = args[3]
            requested_batch = int(requested_batch or 1) * int(images_per_prompt or 1)
            if requested_batch != int(override.shape[0]):
                raise RuntimeError(
                    f"MVADAPTER_REFERENCE_LATENTS_BATCH_MISMATCH:{requested_batch}:{override.shape[0]}"
                )
            return override.to(
                device=device if device is not None else override.device,
                dtype=dtype if dtype is not None else override.dtype,
            )
        return super().prepare_image_latents(*args, **kwargs)

    # ------------------------------------------------------------------
    # adapter construction
    # ------------------------------------------------------------------
    def _init_custom_adapter(self, *args: Any, **kwargs: Any) -> None:
        """Build the upstream adapter, then promote ``cond_encoder`` to a component.

        The upstream implementation is called unchanged so the adapter
        architecture and the attention-processor wiring stay bit-identical.
        """
        unet_dtype = self.unet.dtype
        super()._init_custom_adapter(*args, **kwargs)
        # The upstream processor factory creates its new MV/ref projections in
        # torch's default FP32.  The packaged SD2.1 route is FP16; leaving these
        # 160 parameters in FP32 produces a matmul dtype error on the first
        # denoising call.  Cast only the newly assembled UNet modules to the
        # already-proven UNet dtype; no model or generation setting changes.
        self.unet.to(dtype=unet_dtype)
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


def install_rowcol_reference_cache_compatibility() -> None:
    """Keep upstream reference-cache writes available across diffusers copies."""
    processor_class = DecoupledMVRowColSelfAttnProcessor2_0
    if getattr(processor_class, "_lowvram_cache_compatibility", False):
        return
    original_call = processor_class.__call__

    @wraps(original_call)
    def cache_safe_call(
        self: Any,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Any = None,
        attention_mask: Any = None,
        temb: Any = None,
        mv_scale: float = 1.0,
        ref_hidden_states: Any = None,
        ref_scale: float = 1.0,
        cache_hidden_states: Any = None,
        use_mv: bool = True,
        use_ref: bool = True,
        num_views: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        if cache_hidden_states is not None:
            self._lowvram_reference_hidden_state = hidden_states.detach().clone()
        elif use_ref:
            cached = getattr(self, "_lowvram_reference_hidden_state", None)
            if cached is not None and (ref_hidden_states is None or self.name not in ref_hidden_states):
                if cached.shape[0] == 1 and hidden_states.shape[0] > 1:
                    cached = cached.repeat_interleave(hidden_states.shape[0], dim=0)
                ref_hidden_states = dict(ref_hidden_states or {})
                ref_hidden_states[self.name] = cached
        return original_call(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            temb=temb,
            mv_scale=mv_scale,
            ref_hidden_states=ref_hidden_states,
            ref_scale=ref_scale,
            cache_hidden_states=cache_hidden_states,
            use_mv=use_mv,
            use_ref=use_ref,
            num_views=num_views,
            *args,
            **kwargs,
        )

    processor_class.__call__ = cache_safe_call
    processor_class._lowvram_cache_compatibility = True


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
        requires_safety_checker=False,
    )
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=DDPMScheduler,
    )
    install_rowcol_reference_cache_compatibility()
    pipe.init_custom_adapter(
        num_views=num_views, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0
    )
    load_report = pipe._load_custom_adapter(adapter_state)
    # Adapter construction creates new modules after from_pretrained().  Move
    # the complete assembled pipeline before offload hooks are installed, then
    # explicitly cover the two dynamically-created components.
    pipe.to(dtype=dtype)
    pipe.unet.to(dtype=dtype)
    pipe.cond_encoder.to(dtype=dtype)
    unet_dtypes = {parameter.dtype for parameter in pipe.unet.parameters()}
    if unet_dtypes != {dtype}:
        raise RuntimeError(f"MVADAPTER_UNET_DTYPE_INVALID:{sorted(str(value) for value in unet_dtypes)}")
    rowcol_dtype_inventory(pipe, required_dtype=dtype)
    return pipe, load_report


def _finite_numeric_record(tensor: torch.Tensor, name: str) -> dict[str, Any]:
    """Diagnostic statistics for the narrow FP32 reference-latent boundary."""
    record = tensor_dtype_record(name, tensor)
    finite_mask = torch.isfinite(tensor)
    finite_count = int(finite_mask.sum().item())
    record.update(
        {
            "finite": bool(finite_mask.all()),
            "finite_count": finite_count,
            "nan_count": int(torch.isnan(tensor).sum().item()),
            "positive_inf_count": int((torch.isinf(tensor) & (tensor > 0)).sum().item()),
            "negative_inf_count": int((torch.isinf(tensor) & (tensor < 0)).sum().item()),
        }
    )
    if finite_count:
        finite_values = tensor[finite_mask]
        record["minimum_finite_value"] = float(finite_values.min().item())
        record["maximum_finite_value"] = float(finite_values.max().item())
        record["absolute_maximum"] = float(finite_values.abs().max().item())
    else:
        record.update(
            {
                "minimum_finite_value": None,
                "maximum_finite_value": None,
                "absolute_maximum": None,
            }
        )
    return record


def assert_finite_reference_latents(tensor: torch.Tensor, name: str) -> dict[str, Any]:
    record = _finite_numeric_record(tensor, name)
    if not record["finite"]:
        raise RuntimeError(f"MVADAPTER_{name.upper()}_NONFINITE")
    return record


def prepare_reference_latents_fp32(
    pipe: LowVRAMMVAdapterI2MVSDPipeline,
    preprocessed_reference: torch.Tensor,
    generator: Any,
    device: str = "cuda:0",
    requested_dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Encode the reference image once in FP32, then return validated FP16 latents.

    This is deliberately performed before sequential-offload hooks are installed.
    The temporary FP32 VAE and working latents are released before production
    reference-cache or denoising execution begins.
    """
    if not isinstance(preprocessed_reference, torch.Tensor):
        raise TypeError("MVADAPTER_REFERENCE_IMAGE_NOT_TENSOR")
    target = torch.device(device)
    vae = pipe.vae
    prior_dtype = next(vae.parameters()).dtype
    report: dict[str, Any] = {
        "mode": "FP32_VAE_REFERENCE_ENCODE",
        "requested_latent_dtype": str(requested_dtype).replace("torch.", ""),
        "preprocessed_reference": assert_finite_reference_latents(
            preprocessed_reference, "preprocessed_reference_image"
        ),
    }
    latents_fp32 = None
    try:
        vae.to(device=target, dtype=torch.float32)
        with torch.no_grad():
            latents_fp32 = pipe.prepare_image_latents(
                preprocessed_reference.to(device=target, dtype=torch.float32),
                torch.zeros((1,), device=target, dtype=torch.long),
                batch_size=1,
                num_images_per_prompt=1,
                dtype=torch.float32,
                device=target,
                generator=generator,
                add_noise=False,
            )
        report["reference_latents_fp32"] = assert_finite_reference_latents(
            latents_fp32, "reference_latents_fp32"
        )
        latents_fp16 = latents_fp32.to(device=target, dtype=requested_dtype)
        report["reference_latents_fp16"] = assert_finite_reference_latents(
            latents_fp16, "reference_latents_fp16"
        )
    finally:
        del latents_fp32
        vae.to(device="cpu", dtype=prior_dtype)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pipe.set_reference_latents_override(latents_fp16)
    report["vae_restored_dtype"] = str(prior_dtype).replace("torch.", "")
    report["temporary_fp32_released"] = True
    return latents_fp16, report


def prepare_reference_cache_fp32(
    pipe: LowVRAMMVAdapterI2MVSDPipeline,
    reference_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    device: str = "cuda:0",
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Build the reference cache once in FP32 and return finite FP16 tensors."""
    target = torch.device(device)
    prior_dtype = next(pipe.unet.parameters()).dtype
    expected = sorted(
        name
        for name, processor in pipe.unet.attn_processors.items()
        if type(processor).__name__ == ROWCOL_PROCESSOR_NAME
    )
    cache_fp32: dict[str, torch.Tensor] = {}
    pipe.unet.to(device=target, dtype=torch.float32)
    try:
        with torch.no_grad():
            pipe.unet(
                reference_latents.to(device=target, dtype=torch.float32),
                torch.zeros((), device=target, dtype=torch.long),
                encoder_hidden_states=prompt_embeds.to(device=target, dtype=torch.float32),
                cross_attention_kwargs={
                    "cache_hidden_states": cache_fp32,
                    "use_mv": False,
                    "use_ref": False,
                    "num_views": 6,
                },
                return_dict=False,
            )
            torch.cuda.synchronize(target)
        actual = sorted(cache_fp32)
        if actual != expected:
            raise RuntimeError(f"MVADAPTER_REFERENCE_CACHE_KEYSET_MISMATCH:{actual}:{expected}")
        fp16_cache: dict[str, torch.Tensor] = {}
        entries: list[dict[str, Any]] = []
        for name in expected:
            value = cache_fp32[name]
            fp32_record = assert_finite_reference_latents(value, f"reference_cache_fp32_{name.replace('.', '_')}")
            cast = value.to(device=target, dtype=torch.float16)
            fp16_record = assert_finite_reference_latents(cast, f"reference_cache_fp16_{name.replace('.', '_')}")
            fp16_cache[name] = cast
            entries.append({"key": name, "fp32": fp32_record, "fp16": fp16_record})
    finally:
        del cache_fp32
        pipe.unet.to(device="cpu", dtype=prior_dtype)
        torch.cuda.empty_cache()
    pipe.set_reference_cache_override(fp16_cache)
    return fp16_cache, {
        "mode": "FP32_REFERENCE_UNET_CACHE_CAST_TO_FP16",
        "expected_key_count": len(expected),
        "actual_key_count": len(fp16_cache),
        "batch_before_expansion": 1,
        "dtype_before_cast": "float32",
        "dtype_after_cast": "float16",
        "entries": entries,
        "temporary_fp32_released": True,
    }


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


def _dtype_counts(parameters: list[torch.nn.Parameter]) -> dict[str, int]:
    return dict(sorted(Counter(str(parameter.dtype).replace("torch.", "") for parameter in parameters).items()))


def rowcol_dtype_inventory(
    pipe: LowVRAMMVAdapterI2MVSDPipeline,
    required_dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    """Inventory every parameter in every installed RowCol processor."""
    processors: list[dict[str, Any]] = []
    parameters: list[torch.nn.Parameter] = []
    mismatches: list[str] = []
    for name, processor in sorted(pipe.unet.attn_processors.items()):
        if type(processor).__name__ != ROWCOL_PROCESSOR_NAME:
            continue
        processor_parameters = list(processor.parameters()) if isinstance(processor, torch.nn.Module) else []
        parameters.extend(processor_parameters)
        processors.append(
            {
                "name": name,
                "class": type(processor).__name__,
                "parameter_count": int(sum(parameter.numel() for parameter in processor_parameters)),
                "dtypes": _dtype_counts(processor_parameters),
            }
        )
        for parameter_name, parameter in processor.named_parameters():
            if parameter.dtype != required_dtype:
                mismatches.append(f"{name}.{parameter_name}:{parameter.dtype}")
    if not processors:
        raise RuntimeError("MVADAPTER_ROWCOL_PROCESSOR_MISSING")
    if mismatches:
        raise RuntimeError(f"MVADAPTER_ROWCOL_DTYPE_MISMATCH:{mismatches[:8]}")
    return {
        "processor_class": ROWCOL_PROCESSOR_NAME,
        "processor_count": len(processors),
        "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
        "total_bytes": int(sum(parameter.numel() * parameter.element_size() for parameter in parameters)),
        "dtypes": _dtype_counts(parameters),
        "processors": processors,
        "required_dtype": str(required_dtype).replace("torch.", ""),
        "passed": True,
    }


def component_dtype_inventory(pipe: LowVRAMMVAdapterI2MVSDPipeline) -> dict[str, Any]:
    """Return the complete model-side dtype inventory before offload."""
    rowcol_ids = {
        id(parameter)
        for processor in pipe.unet.attn_processors.values()
        if type(processor).__name__ == ROWCOL_PROCESSOR_NAME
        for parameter in processor.parameters()
    }
    def record(module: torch.nn.Module, parameters: list[torch.nn.Parameter] | None = None) -> dict[str, Any]:
        values = list(module.parameters()) if parameters is None else parameters
        return {
            "parameter_count": int(sum(parameter.numel() for parameter in values)),
            "total_bytes": int(sum(parameter.numel() * parameter.element_size() for parameter in values)),
            "dtypes": _dtype_counts(values),
        }
    base_unet = [parameter for parameter in pipe.unet.parameters() if id(parameter) not in rowcol_ids]
    return {
        "text_encoder": record(pipe.text_encoder),
        "vae": record(pipe.vae),
        "unet_base_parameters": record(pipe.unet, base_unet),
        "rowcol_processor_parameters": rowcol_dtype_inventory(pipe),
        "cond_encoder": record(pipe.cond_encoder),
    }


def tensor_dtype_record(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    """Record a tensor boundary without synchronizing or scanning its values."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} is not a torch.Tensor")
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }


def tensor_group_record(name: str, tensors: list[torch.Tensor]) -> dict[str, Any]:
    """Summarize bytes, dtypes and devices for a tensor collection."""
    return {
        "name": name,
        "tensor_count": len(tensors),
        "total_bytes": int(sum(tensor.numel() * tensor.element_size() for tensor in tensors)),
        "dtypes": sorted({str(tensor.dtype).replace("torch.", "") for tensor in tensors}),
        "devices": sorted({str(tensor.device) for tensor in tensors}),
    }


def install_fp16_input_guards(pipe: LowVRAMMVAdapterI2MVSDPipeline) -> list[Any]:
    """Reject floating tensors entering FP16 UNet/adapter modules in flight."""
    handles: list[Any] = []

    def tensors(value: Any):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors(item)

    def guard(module_name: str, module: torch.nn.Module):
        expected = next((parameter.dtype for parameter in module.parameters() if parameter.is_floating_point()), None)
        if expected != torch.float16:
            return
        def _check(_module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
            for tensor in (*tensors(args), *tensors(kwargs)):
                if tensor.is_floating_point() and tensor.dtype != torch.float16:
                    raise RuntimeError(f"MVADAPTER_FP16_INPUT_DTYPE_MISMATCH:{module_name}:{tensor.dtype}")
        handles.append(module.register_forward_pre_hook(_check, with_kwargs=True))

    guard("unet", pipe.unet)
    guard("cond_encoder", pipe.cond_encoder)
    for name, processor in sorted(pipe.unet.attn_processors.items()):
        if type(processor).__name__ == ROWCOL_PROCESSOR_NAME and isinstance(processor, torch.nn.Module):
            guard(f"unet.attn_processors.{name}", processor)
    return handles


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
        residual_dtypes = sorted({str(state.dtype).replace("torch.", "") for state in residuals})
        finite = all(bool(torch.isfinite(state).all()) for state in residuals)
        peak_allocated = torch.cuda.max_memory_allocated()
        control_record = tensor_dtype_record("control_tensor_after_preprocessing", feature)
        residual_records = [tensor_dtype_record(f"cond_encoder_residual_{index}", state) for index, state in enumerate(residuals)]
        residual_summary = tensor_group_record("condition_residuals", list(residuals))

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
        "control_tensor_after_preprocessing": control_record,
        "control_resolution": resolution,
        "cond_encoder_execution_device": execution_device,
        "cond_encoder_parameter_device_during_forward": observed.get("parameter_device_during_forward"),
        "cond_encoder_output_level_count": len(shapes),
        "cond_encoder_output_shapes": shapes,
        "cond_encoder_output_devices": devices,
        "cond_encoder_output_dtypes": residual_dtypes,
        "cond_encoder_residual_outputs": residual_records,
        "condition_residual_summary": residual_summary,
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
    if result["cond_encoder_output_dtypes"] != ["float16"]:
        raise RuntimeError(f"MVADAPTER_COND_ENCODER_RESIDUAL_DTYPE_MISMATCH:{result['cond_encoder_output_dtypes']}")
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


def reference_unet_dtype_smoke_test(
    pipe: LowVRAMMVAdapterI2MVSDPipeline,
    device: str = "cuda:0",
    resolution: int = 64,
) -> dict[str, Any]:
    """Run exactly one tiny reference UNet forward with RowCol caching enabled."""
    if pipe.unet.dtype != torch.float16:
        raise RuntimeError(f"MVADAPTER_UNET_DTYPE_INVALID:{pipe.unet.dtype}")
    if resolution % 8 != 0 or resolution < 32:
        raise ValueError(f"MVADAPTER_REFERENCE_SMOKE_RESOLUTION_INVALID:{resolution}")
    target = torch.device(device)
    with torch.no_grad():
        prompt_embeds, _ = pipe.encode_prompt(
            "dtype smoke reference",
            target,
            1,
            False,
            None,
        )
        prompt_embeds = prompt_embeds.to(device=target, dtype=torch.float16)
        reference_image = torch.zeros((1, 3, resolution, resolution), device=target, dtype=torch.float16)
        timestep = torch.zeros((1,), device=target, dtype=torch.long)
        reference_latents = pipe.prepare_image_latents(
            reference_image,
            timestep,
            batch_size=1,
            num_images_per_prompt=1,
            dtype=torch.float16,
            device=target,
            add_noise=False,
        )
        cache_sink: dict[str, torch.Tensor] = {}

        class _ReferenceCache(dict[str, torch.Tensor]):
            # Diffusers copies cross-attention kwargs while descending through
            # transformer blocks.  Keep a shared sink so writes remain
            # observable even if a dependency materializes a plain copy.
            def __setitem__(self, key: str, value: torch.Tensor):
                cache_sink[key] = value
                super().__setitem__(key, value)

            def copy(self):
                return self

            def __deepcopy__(self, _memo: dict[int, Any]):
                return self

        cached_reference_hidden_states: dict[str, torch.Tensor] = _ReferenceCache()
        pipe.unet(
            reference_latents,
            torch.zeros((), device=target, dtype=torch.long),
            encoder_hidden_states=prompt_embeds[-1:],
            cross_attention_kwargs={
                "cache_hidden_states": cached_reference_hidden_states,
                "use_mv": False,
                "use_ref": False,
                "num_views": 1,
            },
            return_dict=False,
        )
        torch.cuda.synchronize(target)
    if not cache_sink:
        raise RuntimeError("MVADAPTER_REFERENCE_SMOKE_NO_CACHED_HIDDEN_STATES")
    hidden_records = [
        tensor_dtype_record(f"cached_reference_hidden_states.{name}", value)
        for name, value in sorted(cache_sink.items())
    ]
    all_records = [
        tensor_dtype_record("prompt_embeddings", prompt_embeds),
        tensor_dtype_record("reference_image_tensor", reference_image),
        tensor_dtype_record("reference_latents", reference_latents),
        *hidden_records,
    ]
    bad = [record for record in all_records if record["dtype"] != "float16"]
    if bad:
        raise RuntimeError(f"MVADAPTER_REFERENCE_SMOKE_DTYPE_MISMATCH:{bad}")
    return {
        "passed": True,
        "resolution": resolution,
        "rowcol_processor_active": True,
        "cache_hidden_states_enabled": True,
        "reference_unet_forward_count": 1,
        "denoising_steps": 0,
        "vae_decode": False,
        "output_images": 0,
        "gpu_sequence_consumed": False,
        "reference_cache_summary": tensor_group_record(
            "reference_cache", list(cache_sink.values())
        ),
        "tensor_inventory": all_records,
    }
