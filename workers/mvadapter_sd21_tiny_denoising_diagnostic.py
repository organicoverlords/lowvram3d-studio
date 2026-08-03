"""Fresh-process tiny real-denoising probe for the SD2.1 MV-Adapter route.

This worker is intentionally separate from the production runner.  It verifies
the reference-cache, condition-residual, RowCol and SDPA path at 64/96 pixels,
with zero scheduler steps and no VAE decode.  It never reads or writes the
production execution manifest and never produces quality-sequence images.
"""
from __future__ import annotations

import argparse
import contextvars
import gc
import hashlib
import json
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROWCOL_NAME = "DecoupledMVRowColSelfAttnProcessor2_0"
PROMPT = (
    "high quality clean albedo reference of the same tactical red panda character, "
    "consistent materials, consistent identity, flat neutral lighting"
)


class FirstReferenceNonFinite(RuntimeError):
    pass


class ReferenceOnlyComplete(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _memory(torch: Any) -> dict[str, float]:
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / 2**20, 3),
        "reserved_mb": round(torch.cuda.memory_reserved() / 2**20, 3),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 3),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 3),
    }


def _safe_memory(torch: Any) -> dict[str, Any]:
    try:
        return _memory(torch)
    except BaseException as exc:
        return {"memory_query_error": f"{type(exc).__name__}: {exc}"}


def _tensor_record(name: str, tensor: Any) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "contiguous": bool(tensor.is_contiguous()),
        "numel": int(tensor.numel()),
    }


def _finite_stats(tensor: Any) -> dict[str, Any]:
    """Diagnostic-only value scan for cache-contract failures."""
    finite = torch_isfinite = None
    try:
        import torch
        finite_mask = torch.isfinite(tensor)
        finite = bool(finite_mask.all())
        record: dict[str, Any] = {
            "finite": finite,
            "finite_count": int(finite_mask.sum().item()),
            "nan_count": int(torch.isnan(tensor).sum().item()),
            "inf_count": int(torch.isinf(tensor).sum().item()),
        }
        if tensor.numel():
            record["min"] = float(torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).min().item())
            record["max"] = float(torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0).max().item())
        return record
    except BaseException as exc:
        return {"finite": finite, "stats_error": f"{type(exc).__name__}: {exc}"}


def _first_tensor(torch: Any, value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _first_tensor(torch, item)
            if found is not None:
                return found
    if isinstance(value, dict):
        for item in value.values():
            found = _first_tensor(torch, item)
            if found is not None:
                return found
    return None


def _install_reference_numeric_hooks(torch: Any, pipe: Any, trace: IncrementalTrace) -> list[Any]:
    handles: list[Any] = []

    def hook(name: str):
        def record(_module: Any, inputs: Any, output: Any) -> None:
            value = _first_tensor(torch, output)
            if value is None:
                return
            output_stats = {**_tensor_record(f"{name}.output", value), **_finite_stats(value)}
            input_value = _first_tensor(torch, inputs)
            input_stats = (
                {**_tensor_record(f"{name}.input", input_value), **_finite_stats(input_value)}
                if input_value is not None else None
            )
            trace.write("reference_module_output", module=name, input=input_stats, output=output_stats)
            if output_stats.get("finite") is not True:
                raise FirstReferenceNonFinite(
                    f"FIRST_NONFINITE_MODULE={name};INPUT_ABS_MAX={input_stats.get('absolute_maximum') if input_stats else None};"
                    f"OUTPUT_ABS_MAX={output_stats.get('absolute_maximum')};INPUT_DTYPE={input_stats.get('dtype') if input_stats else None};"
                    f"OUTPUT_DTYPE={output_stats.get('dtype')}"
                )
        return record

    for name, module in pipe.unet.named_modules():
        if name:
            handles.append(module.register_forward_hook(hook(name)))
    return handles


class IncrementalTrace:
    """Append-and-flush evidence so a destroyed CUDA context leaves a trace."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("w", encoding="utf-8", buffering=1)

    def write(self, phase: str, **fields: Any) -> None:
        record = {"time": time.time(), "pid": os.getpid(), "phase": phase, **fields}
        try:
            self.handle.write(json.dumps(record, default=str) + "\n")
            self.handle.flush()
        except (OSError, UnicodeError, ValueError, BrokenPipeError):
            # The process may be in a destroyed CUDA context, but a failing
            # sink must not hide the primary exception from the caller.
            pass

    def close(self) -> None:
        try:
            self.handle.flush()
            self.handle.close()
        except (OSError, UnicodeError, ValueError, BrokenPipeError):
            pass


_sdpa_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "mvadapter_sdpa_context", default=None
)


def _install_sdpa_trace(torch: Any, trace: IncrementalTrace, device: Any) -> tuple[Any, Any]:
    import torch.nn.functional as functional

    original_sdpa = functional.scaled_dot_product_attention
    original_processor_call = None
    from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0

    def traced_sdpa(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        context = _sdpa_context.get()
        context = context or {"processor": "unknown", "branch": "base", "ordinal": 0}
        fields = {
            "processor": context.get("processor"),
            "branch": context.get("branch", "base"),
            "ordinal": int(context.get("ordinal", 0)),
            "q": _tensor_record("Q", query),
            "k": _tensor_record("K", key),
            "v": _tensor_record("V", value),
            "memory_before": _memory(torch),
        }
        trace.write("sdpa_started", **fields)
        try:
            output = original_sdpa(query, key, value, *args, **kwargs)
            torch.cuda.synchronize(device)
            trace.write(
                "sdpa_completed",
                processor=fields["processor"],
                branch=fields["branch"],
                ordinal=fields["ordinal"],
                output=_tensor_record("sdpa_output", output),
                memory_after=_memory(torch),
            )
            return output
        except BaseException as exc:
            trace.write(
                "sdpa_failed",
                processor=fields["processor"],
                branch=fields["branch"],
                ordinal=fields["ordinal"],
                error=f"{type(exc).__name__}: {exc}",
                memory_after=_memory(torch),
            )
            raise

    @__import__("functools").wraps(DecoupledMVRowColSelfAttnProcessor2_0.__call__)
    def traced_processor(self: Any, *args: Any, **kwargs: Any) -> Any:
        use_mv = bool(getattr(self, "use_mv", False) and kwargs.get("use_mv", True))
        use_ref = bool(getattr(self, "use_ref", False) and kwargs.get("use_ref", True))
        branches = ["base"]
        if use_mv:
            branches.extend(["mv_row", "mv_column"])
        if use_ref:
            branches.append("reference")
        state = {"processor": getattr(self, "name", "unknown"), "branches": branches, "ordinal": 0}
        cache_hidden_states = kwargs.get("cache_hidden_states")
        if cache_hidden_states is not None:
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[1] if len(args) > 1 else None
            if hidden_states is not None:
                cache_stats = {**_tensor_record("reference_cache_write", hidden_states), **_finite_stats(hidden_states)}
                trace.write("reference_cache_write", processor=state["processor"], stats=cache_stats)
                if cache_stats.get("finite") is not True:
                    raise FirstReferenceNonFinite(
                        f"FIRST_NONFINITE_CACHE_KEY={state['processor']};OUTPUT_ABS_MAX={cache_stats.get('absolute_maximum')}"
                    )
        token = _sdpa_context.set({"processor": state["processor"], "branches": branches, "branch": branches[0], "ordinal": 0})
        try:
            # The SDPA wrapper advances the ordinal after each call.  The
            # context is updated from this wrapper's call boundary below.
            return original_processor_call(self, *args, **kwargs)
        finally:
            _sdpa_context.reset(token)

    def branch_aware_sdpa(query: Any, key: Any, value: Any, *args: Any, **kwargs: Any) -> Any:
        context = _sdpa_context.get()
        if context is not None:
            # Each RowCol call has a deterministic SDPA order: base, row,
            # column, reference.  This avoids changing upstream attention code.
            ordinal = int(context.get("ordinal", 0))
            branches = context.get("branches", ["base"])
            context["branch"] = branches[min(ordinal, len(branches) - 1)]
            context["ordinal"] = ordinal + 1
        return traced_sdpa(query, key, value, *args, **kwargs)

    original_processor_call = DecoupledMVRowColSelfAttnProcessor2_0.__call__
    functional.scaled_dot_product_attention = branch_aware_sdpa
    DecoupledMVRowColSelfAttnProcessor2_0.__call__ = traced_processor
    return original_sdpa, original_processor_call


def _restore_sdpa_trace(torch: Any, originals: tuple[Any, Any]) -> None:
    import torch.nn.functional as functional
    from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0

    functional.scaled_dot_product_attention = originals[0]
    DecoupledMVRowColSelfAttnProcessor2_0.__call__ = originals[1]


def _validate_inputs(config_path: Path) -> dict[str, Any]:
    config = _json(config_path)
    selected = config["primary"]
    checks = (
        (Path(config["adapter"]), config["adapter_sha256"], "adapter"),
        (Path(config["mesh"]), config["immutable_mesh_sha256"], "mesh"),
        (Path(selected["conditioning_reference"]), selected["conditioning_reference_sha256"], "conditioning"),
        (Path(selected["control_tensor"]), selected["control_tensor_sha256"], "control"),
        (Path(selected["camera_contract"]), selected["camera_contract_sha256"], "camera_contract"),
    )
    hashes = {}
    for path, expected, label in checks:
        if not path.is_file():
            raise RuntimeError(f"DIAGNOSTIC_INPUT_MISSING:{label}:{path}")
        actual = _sha256(path)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"DIAGNOSTIC_INPUT_HASH_MISMATCH:{label}:{actual}:{expected}")
        hashes[label] = actual
    control = np.load(checks[3][0], allow_pickle=False)
    if tuple(control.shape) != (6, 6, int(selected["resolution"]), int(selected["resolution"])):
        raise RuntimeError(f"DIAGNOSTIC_CONTROL_SHAPE_INVALID:{tuple(control.shape)}")
    return {"config": config, "selected": selected, "hashes": hashes, "control_path": checks[3][0]}


def _cache_ownership(pipe: Any, cache: dict[str, Any], expected: list[str]) -> dict[str, Any]:
    entries = {}
    for name in expected:
        processor = pipe.unet.attn_processors[name]
        value = cache.get(name, getattr(processor, "_lowvram_reference_hidden_state", None))
        if value is not None:
            owner = "plain_module_attribute" if getattr(processor, "_lowvram_reference_hidden_state", None) is value else "explicit_pipeline_cache"
            entries[name] = {"ownership": owner, "tensor": value}
    missing = sorted(set(expected) - set(entries))
    unexpected = sorted(set(cache) - set(expected))
    return {"expected_keys": sorted(expected), "actual_keys": sorted(entries), "missing_keys": missing, "unexpected_keys": unexpected, "entries": entries}


def run_probe(
    config_path: Path,
    output_dir: Path,
    backend: str,
    offload: str,
    resolution: int,
    reference_unet_dtype: str = "fp16",
    reference_only: bool = False,
) -> dict[str, Any]:
    if resolution not in (64, 96) or resolution % 8:
        raise ValueError("DIAGNOSTIC_RESOLUTION_MUST_BE_64_OR_96")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"DIAGNOSTIC_OUTPUT_NOT_EMPTY:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = IncrementalTrace(output_dir / "sdpa_trace.jsonl")
    receipt: dict[str, Any] = {
        "schema": "lowvram3d_mvadapter_tiny_real_denoising_probe_v1",
        "config": str(config_path), "backend_requested": backend, "offload": offload,
        "resolution": resolution, "views": 6, "steps_requested": 0,
        "reference_unet_dtype": reference_unet_dtype,
        "reference_only": reference_only,
        "vae_decode": False, "output_images": 0, "production_manifest_updated": False,
        "gpu_sequence_consumed": False, "started": time.time(),
    }
    pipe = None
    originals = None
    guard_handles: list[Any] = []
    reference_numeric_handles: list[Any] = []
    pending_exc: BaseException | None = None
    try:
        inputs = _validate_inputs(config_path)
        receipt["input_hashes"] = inputs["hashes"]
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("DIAGNOSTIC_CUDA_UNAVAILABLE")
        target = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        trace.write("probe_started", backend=backend, offload=offload, resolution=resolution, memory=_memory(torch))
        upstream = Path(inputs["config"].get("mvadapter_source", r"C:\AI\mvadapter-upstream-inspection"))
        sys.path.append(str(upstream))
        import safetensors.torch
        from lowvram_mvadapter_i2mv_sd21 import (
            attention_report, build_low_vram_pipeline, component_dtype_inventory,
            component_inventory, install_fp16_input_guards, install_low_vram_offload,
            offload_hook_report, prepare_reference_cache_fp32, prepare_reference_latents_fp32,
            rowcol_dtype_inventory,
            tensor_dtype_record, tensor_group_record,
        )
        adapter_state = safetensors.torch.load_file(str(Path(inputs["config"]["adapter"])), device="cpu")
        pipe, adapter_report = build_low_vram_pipeline(
            inputs["config"]["base_model"], adapter_state, Path(inputs["config"]["adapter"]).name, num_views=6, dtype=torch.float16
        )
        receipt["adapter_report"] = adapter_report
        if reference_unet_dtype not in {"fp16", "fp32"}:
            raise ValueError(f"REFERENCE_UNET_DTYPE_INVALID:{reference_unet_dtype}")
        reference_dtype = torch.float32 if reference_unet_dtype == "fp32" else torch.float16
        receipt["component_inventory_before_offload"] = component_inventory(pipe)
        receipt["dtype_inventory_before_offload"] = component_dtype_inventory(pipe)
        receipt["rowcol_dtype_inventory"] = rowcol_dtype_inventory(pipe, torch.float16)
        receipt["attention"] = attention_report(pipe)
        if reference_dtype == torch.float32:
            pipe.unet.to(dtype=torch.float32)
            receipt["reference_unet_dtype_after_isolation_cast"] = "float32"
        receipt["sdpa_backend"] = {
            "requested": backend, "actual_api": "torch.nn.functional.scaled_dot_product_attention",
            "math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
            "mem_efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
        }
        selected = inputs["selected"]
        reference_image = Image.open(Path(selected["conditioning_reference"])).convert("RGB")
        reference_preprocessed = pipe.image_processor.preprocess(reference_image)
        reference_generator = torch.Generator(device=target).manual_seed(int(selected["seed"]))
        reference_latents, receipt["reference_vae_fp32"] = prepare_reference_latents_fp32(
            pipe,
            reference_preprocessed,
            generator=reference_generator,
            device="cuda:0",
            requested_dtype=torch.float16,
        )
        originals = _install_sdpa_trace(torch, trace, target)
        reference_prompt_embeds, _ = pipe.encode_prompt(PROMPT, torch.device("cpu"), 1, False, None)
        reference_cache, receipt["reference_cache_fp32"] = prepare_reference_cache_fp32(
            pipe,
            reference_latents,
            reference_prompt_embeds,
            device="cuda:0",
        )
        pipe.clear_reference_latents_override()
        trace.write(
            "reference_latents_fp32_validated",
            report=receipt["reference_vae_fp32"],
            memory=_memory(torch),
        )
        if offload == "sequential":
            receipt["offload_report"] = install_low_vram_offload(pipe, "cuda")
            receipt["offload_hooks"] = offload_hook_report(pipe)
        else:
            pipe.to(device=target, dtype=torch.float16)
            receipt["offload_report"] = {"offload_mode": "DISABLED", "device": str(target)}
            receipt["component_devices_resident"] = component_inventory(pipe)
        pipe.set_reference_cache_override(reference_cache)
        guard_handles = install_fp16_input_guards(pipe)
        with torch.no_grad():
            prompt_embeds, _ = pipe.encode_prompt(PROMPT, target, 1, False, None)
            prompt_embeds = prompt_embeds.to(target, dtype=torch.float16)
            trace.write("prompt_ready", tensor=_tensor_record("prompt_embeddings", prompt_embeds), memory=_memory(torch))
            trace.write("reference_inputs_ready", latents=_tensor_record("reference_latents", reference_latents), memory=_memory(torch))
            trace.write("reference_unet_completed", memory=_memory(torch), cache_entries=len(reference_cache), cache_owner="explicit_pipeline_cache")
            if reference_only:
                raise ReferenceOnlyComplete()
            expected = sorted(name for name, proc in pipe.unet.attn_processors.items() if type(proc).__name__ == ROWCOL_NAME and getattr(proc, "use_ref", False))
            receipt["reference_cache_ownership"] = {
                "owner": "explicit_pipeline_cache",
                "expected_keys": expected,
                "actual_keys": sorted(reference_cache),
                "missing_keys": sorted(set(expected) - set(reference_cache)),
                "unexpected_keys": sorted(set(reference_cache) - set(expected)),
            }
            cache_records = []
            for name in expected:
                if name not in reference_cache:
                    raise RuntimeError(f"REFERENCE_CACHE_KEY_MISSING:{name}")
                value = reference_cache[name]
                contract = {**_tensor_record(name, value), **_finite_stats(value)}
                receipt.setdefault("reference_cache_contract_checks", []).append(contract)
                if value.shape[0] != 1 or value.dtype != torch.float16 or value.device != target or not contract.get("finite", False):
                    raise RuntimeError(f"REFERENCE_CACHE_CONTRACT_INVALID:{name}:{contract}")
                cache_records.append(value)
            if sorted(reference_cache) != expected and not set(expected).issubset(reference_cache):
                raise RuntimeError(f"REFERENCE_CACHE_KEYSET_MISMATCH:{sorted(reference_cache)}:{expected}")
            receipt["reference_cache"] = {
                "batch_before_expansion": 1,
                "summary": tensor_group_record("reference_cache", cache_records),
                "entries": [_tensor_record(f"reference_cache.{name}", reference_cache[name]) for name in expected],
            }
            expanded_cache = {name: reference_cache[name].repeat_interleave(6, dim=0) for name in expected}
            if any(value.shape[0] != 6 for value in expanded_cache.values()):
                raise RuntimeError("REFERENCE_CACHE_EXPANSION_INVALID")
            control = torch.from_numpy(np.ascontiguousarray(np.load(inputs["control_path"], allow_pickle=False).astype(np.float32)))
            control_feature = pipe.prepare_control_image(control, resolution, resolution, 6, 1, target, torch.float16, False)
            control_feature = control_feature.to(target, dtype=torch.float16)
            trace.write("condition_encoder_started", tensor=_tensor_record("control_after_preprocessing", control_feature), memory=_memory(torch))
            residuals = pipe.cond_encoder(control_feature)
            torch.cuda.synchronize(target)
            if not residuals:
                raise RuntimeError("COND_RESIDUALS_EMPTY")
            if any(t.dtype != torch.float16 or t.device != target or not bool(torch.isfinite(t).all()) for t in residuals):
                raise RuntimeError("COND_RESIDUAL_CONTRACT_INVALID")
            receipt["condition_residuals"] = {"summary": tensor_group_record("condition_residuals", list(residuals)), "entries": [_tensor_record(f"condition_residual_{i}", t) for i, t in enumerate(residuals)]}
            trace.write("condition_encoder_completed", memory=_memory(torch), residuals=receipt["condition_residuals"])
            latent_size = resolution // 8
            latents = torch.randn((6, 4, latent_size, latent_size), device=target, dtype=torch.float16, generator=torch.Generator(device=target).manual_seed(int(selected["seed"])))
            denoise_prompt = prompt_embeds.repeat_interleave(6, dim=0)
            trace.write("denoising_unet_started", use_mv=True, use_ref=True, latent=_tensor_record("diffusion_latents", latents), prompt=_tensor_record("prompt_embeddings_expanded", denoise_prompt), memory=_memory(torch))
            pipe.unet(latents, torch.zeros((), device=target, dtype=torch.long), encoder_hidden_states=denoise_prompt, cross_attention_kwargs={"num_views": 6, "mv_scale": 1.0, "ref_hidden_states": expanded_cache, "ref_scale": 1.0, "use_mv": True, "use_ref": True}, down_intrablock_additional_residuals=list(residuals), return_dict=False)
            torch.cuda.synchronize(target)
            receipt["denoising_unet_forward_completed"] = True
            receipt["scheduler_steps_completed"] = 0
            trace.write("denoising_unet_completed", memory=_memory(torch))
        receipt["status"] = "PROVEN"
        receipt["classification"] = "TINY_REAL_DENOISING_FORWARD=PROVEN"
    except BaseException as exc:
        if isinstance(exc, ReferenceOnlyComplete):
            receipt["status"] = "PROVEN"
            receipt["classification"] = "REFERENCE_UNET_FORWARD_COMPLETED"
            trace.write("reference_only_completed", memory=_safe_memory(__import__("torch")))
        else:
            receipt["status"] = "REJECTED"
            receipt["classification"] = f"{type(exc).__name__}: {exc}"
            receipt["error"] = receipt["classification"]
            trace.write("probe_failed", error=receipt["classification"], memory=_safe_memory(__import__("torch")) if "torch" in sys.modules and getattr(__import__("torch"), "cuda", None) and __import__("torch").cuda.is_available() else {})
            pending_exc = exc
    finally:
        if originals is not None:
            try:
                import torch
                _restore_sdpa_trace(torch, originals)
            except Exception:
                pass
        for handle in guard_handles:
            try:
                handle.remove()
            except Exception:
                pass
        for handle in reference_numeric_handles:
            try:
                handle.remove()
            except Exception:
                pass
        if pipe is not None:
            try:
                pipe.clear_reference_cache_override()
            except Exception:
                pass
        trace.write("cleanup", memory=_safe_memory(__import__("torch")) if "torch" in sys.modules and getattr(__import__("torch"), "cuda", None) and __import__("torch").cuda.is_available() else {})
        trace.close()
        if pipe is not None:
            try:
                pipe.clear_reference_latents_override()
            except Exception:
                pass
            del pipe
        gc.collect()
        if "torch" in sys.modules and __import__("torch").cuda.is_available():
            __import__("torch").cuda.empty_cache()
    receipt["memory_final"] = _safe_memory(__import__("torch")) if "torch" in sys.modules and __import__("torch").cuda.is_available() else {}
    receipt["wall_seconds"] = round(time.time() - receipt["started"], 3)
    receipt.pop("started", None)
    (output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2, default=str) + "\n", encoding="utf-8")
    if pending_exc is not None:
        raise pending_exc
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("math", "auto"), required=True)
    parser.add_argument("--offload", choices=("sequential", "none"), default="sequential")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--reference-unet-dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    try:
        import torch
        context = nullcontext()
        if args.backend == "math":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            context = sdpa_kernel(SDPBackend.MATH)
        with context:
            result = run_probe(
                args.config,
                args.output_dir,
                args.backend,
                args.offload,
                args.resolution,
                args.reference_unet_dtype,
                args.reference_only,
            )
        print(json.dumps({"status": result["status"], "classification": result["classification"], "output_dir": str(args.output_dir)}))
        return 0
    except BaseException as exc:
        print(json.dumps({"status": "REJECTED", "error": f"{type(exc).__name__}: {exc}", "output_dir": str(args.output_dir)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
