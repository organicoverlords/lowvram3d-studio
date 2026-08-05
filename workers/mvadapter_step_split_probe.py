"""Split one denoising step into its individual operations and find where finite becomes NaN.

Per-step telemetry established that latents are already non-finite after step 1, which narrows the
question but does not answer it: a step is a scale_model_input, a UNet forward with several
conditioning tensors, and a scheduler update, and any of them could be the one that breaks.

This wraps the three call sites the pipeline actually uses -- ``scheduler.scale_model_input``,
``unet.forward`` and ``scheduler.step`` -- and records every tensor crossing them in chronological
order. The moment a checkpoint is non-finite while its inputs were finite, the offending operation
is named and the probe stops, so nothing downstream has to be guessed at.

The finite inputs to ``scheduler.step`` are written to NPZ so the scheduler update can be replayed
offline, with different dtypes, without paying for another UNet forward.

With ``target_step=None`` and ``abort_on_first_failure=False`` the same instrumentation becomes a
whole-run finite gate: every step is recorded under its own label, the pipeline is allowed to
finish, and ``finite_gate()`` reports named per-stage verdicts. That is the mode a repair proof
runs in -- the diagnostic mode answers "where does it break", the gate mode answers "is it still
broken anywhere".
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mvadapter_latent_telemetry import failures_for, per_view_stats, tensor_stats, views_identical


class StepSplitAbort(RuntimeError):
    """Raised to stop the pipeline as soon as the first non-finite checkpoint is captured."""


#: The failure labels that mean "the arithmetic broke". Only these abort a run.
#:
#: ``ALL_ZERO`` and ``NEAR_ZERO_STD`` describe a tensor whose values collapsed, which is a real
#: defect in a latent but a *documented expectation* elsewhere: this pipeline injects the reference
#: cache and returns ``torch.zeros_like(sample)`` from the in-pipeline reference forward instead of
#: running the UNet, so ``ref01_unet_raw_output`` is all-zero in every healthy run. Aborting on that
#: would kill a production run before step 1, and reporting it as non-finite -- which
#: ``classification()`` previously did -- names the wrong defect.
NONFINITE_FAILURES = ("NAN", "INF", "NON_FINITE_PRESENT")
FLAT_FAILURES = ("ALL_ZERO", "NEAR_ZERO_STD", "ALL_VIEWS_IDENTICAL")
REFERENCE_CACHE_CONTRACT_FAILURE = "REFERENCE_CACHE_CONTRACT_FAILED"


def _is_zero_or_flat(stats: dict[str, Any]) -> bool:
    return any(label in FLAT_FAILURES for label in failures_for(stats, expect_variation=True))


def _cache_contract(expected: dict[str, torch.Tensor] | None,
                    actual: Any) -> dict[str, Any]:
    """Validate the explicit reference-cache owner without inventing missing entries."""
    expected_keys = sorted(expected or {})
    actual_keys = sorted(actual) if isinstance(actual, dict) else []
    missing = sorted(set(expected_keys) - set(actual_keys))
    unexpected = sorted(set(actual_keys) - set(expected_keys))
    entries: list[dict[str, Any]] = []
    all_finite = True
    all_nonflat = True
    shape_dtype_device_ok = True
    nonfinite_present = False
    for key in expected_keys:
        value = actual.get(key) if isinstance(actual, dict) else None
        expected_value = expected[key]
        record: dict[str, Any] = {"key": key, "present": isinstance(value, torch.Tensor)}
        if isinstance(value, torch.Tensor):
            stats = tensor_stats(f"reference_cache.{key}", value)
            record["stats"] = stats
            finite = stats["finite_fraction"] == 1.0
            nonflat = not _is_zero_or_flat(stats)
            same_shape = list(value.shape) == list(expected_value.shape)
            same_dtype = value.dtype == expected_value.dtype
            same_device = str(value.device) == str(expected_value.device)
            record.update({"finite": finite, "nonflat": nonflat, "shape_match": same_shape,
                           "dtype_match": same_dtype, "device_match": same_device})
            all_finite &= finite
            all_nonflat &= nonflat
            shape_dtype_device_ok &= same_shape and same_dtype and same_device
            nonfinite_present |= not finite
        else:
            record.update({"finite": False, "nonflat": False, "shape_match": False,
                           "dtype_match": False, "device_match": False})
            all_finite = False
            all_nonflat = False
            shape_dtype_device_ok = False
        entries.append(record)
    valid = bool(
        expected is not None and not missing and not unexpected and entries and all_finite
        and all_nonflat and shape_dtype_device_ok
    )
    return {
        "contract": "EXPLICIT_REFERENCE_CACHE_FP16_BATCH1",
        "configured": expected is not None,
        "expected_key_count": len(expected_keys),
        "actual_key_count": len(actual_keys),
        "expected_keys": expected_keys,
        "actual_keys": actual_keys,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "all_finite": bool(all_finite),
        "nonfinite_present": bool(nonfinite_present),
        "all_nonflat": bool(all_nonflat),
        "shape_dtype_device_match": bool(shape_dtype_device_ok),
        "valid": valid,
        "entries": entries,
    }


class StepSplitProbe:
    """Instruments one denoising step and stops at the first finite-to-non-finite transition."""

    def __init__(self, root: Path, target_step: int | None = 1, save_scheduler_inputs: bool = True,
                 abort_on_first_failure: bool = True, expected_views: int = 6,
                 expected_reference_cache: dict[str, torch.Tensor] | None = None,
                 reference_output_contract: str = "CACHE_SIDE_EFFECT_WITH_ZERO_SENTINEL") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        #: ``None`` instruments every step instead of one, which is what the whole-run gate needs.
        self.target_step = None if target_step is None else int(target_step)
        # The reference image is pushed through the same UNet before denoising starts, with batch 1
        # and timestep 0. That forward is not part of step 1 and must not be mistaken for it, so it
        # is recorded under its own ``reference_unet`` stage with its own labels.
        self.expected_views = int(expected_views)
        self.reference_forwards = 0
        self.save_scheduler_inputs = save_scheduler_inputs
        self.abort_on_first_failure = abort_on_first_failure
        self.records: list[dict[str, Any]] = []
        self.current_step = 0
        self.first_failure: dict[str, Any] | None = None
        #: The earliest non-finite tensor specifically. Kept apart from ``first_failure`` because a
        #: collapsed-to-zero tensor and a NaN tensor are different defects with different fixes, and
        #: only one of them may stop a run.
        self.first_nonfinite: dict[str, Any] | None = None
        self.first_unexpected_flat: dict[str, Any] | None = None
        self.expected_reference_cache = expected_reference_cache
        self.reference_output_contract = reference_output_contract
        self.reference_cache_contract: dict[str, Any] | None = None
        self.reference_output_zero_sentinel: dict[str, Any] | None = None
        self.saved_npz: dict[str, str] = {}
        #: The two tensors either side of the current step's UNet forward, so a step-end callback can
        #: fold them into the chronological latent telemetry instead of a second file having to be
        #: read alongside it. Overwritten each step; never accumulated.
        self.latest_scaled_input: torch.Tensor | None = None
        self.latest_unet_output: torch.Tensor | None = None
        self.latest_scheduler_input_sample: torch.Tensor | None = None
        self.latest_processed_model_prediction: torch.Tensor | None = None
        self._installed: list[tuple[Any, str, Any]] = []
        self._sequence = 0

    def take_step_tensors(self) -> dict[str, torch.Tensor | None]:
        """Hand over the current step's UNet-boundary tensors and forget them."""
        tensors = {"scaled_input": self.latest_scaled_input, "noise_pred": self.latest_unet_output,
                   "scheduler_input_sample": self.latest_scheduler_input_sample,
                   "processed_model_prediction": self.latest_processed_model_prediction}
        self.latest_scaled_input = None
        self.latest_unet_output = None
        self.latest_scheduler_input_sample = None
        self.latest_processed_model_prediction = None
        return tensors

    # ------------------------------------------------------------------ recording
    def _failure_summary(self, label: str, stage: str, entry: dict[str, Any],
                         failures: list[str]) -> dict[str, Any]:
        previous = [item for item in self.records[:-1] if item.get("tensor")]
        last_clean = next((item["checkpoint"] for item in reversed(previous)
                           if not item["failures"]), None)
        summary = {
            "checkpoint": label,
            "stage": stage,
            "sequence": entry["sequence"],
            "failures": failures,
            "last_finite_checkpoint": last_clean,
            "nan_count": entry.get("stats", {}).get("nan_count"),
            "posinf_count": entry.get("stats", {}).get("posinf_count"),
            "neginf_count": entry.get("stats", {}).get("neginf_count"),
        }
        if self.first_failure is None:
            self.first_failure = summary
        if self.first_nonfinite is None and any(name in NONFINITE_FAILURES for name in failures):
            self.first_nonfinite = summary
        if self.first_unexpected_flat is None and any(name in FLAT_FAILURES for name in failures):
            self.first_unexpected_flat = summary
        return summary

    def record(self, label: str, value: Any, *, stage: str, expect_variation: bool = True,
               allow_flat: bool = False, extra_failures: list[str] | None = None) -> Any:
        if not isinstance(value, torch.Tensor):
            self._sequence += 1
            self.records.append({
                "sequence": self._sequence, "checkpoint": label, "stage": stage,
                "tensor": False, "repr": _describe_non_tensor(value), "failures": [],
            })
            return value
        stats = tensor_stats(label, value)
        views = per_view_stats(label, value) if value.dim() >= 1 and value.shape[0] > 1 else []
        failures = failures_for(stats, expect_variation=expect_variation, allow_flat=allow_flat)
        if extra_failures:
            failures.extend(item for item in extra_failures if item not in failures)
        self._sequence += 1
        entry = {
            "sequence": self._sequence,
            "timestamp": time.time(),
            "checkpoint": label,
            "stage": stage,
            "tensor": True,
            "stats": stats,
            "per_view": views,
            "all_views_identical": views_identical(views) if views else None,
            "failures": failures,
        }
        self.records.append(entry)
        if failures:
            self._failure_summary(label, stage, entry, failures)
        return value

    def record_event(self, label: str, stage: str, payload: dict[str, Any],
                     failures: list[str] | None = None) -> None:
        self._sequence += 1
        entry = {"sequence": self._sequence, "checkpoint": label, "stage": stage,
                 "timestamp": time.time(), "tensor": False, "payload": payload,
                 "failures": list(failures or [])}
        self.records.append(entry)
        if failures:
            self._failure_summary(label, stage, entry, list(failures))

    def _save(self, name: str, tensor: torch.Tensor) -> None:
        if not self.save_scheduler_inputs or not isinstance(tensor, torch.Tensor):
            return
        path = self.root / f"{name}.npz"
        np.savez_compressed(path, tensor=tensor.detach().float().cpu().numpy())
        self.saved_npz[name] = str(path)

    def _maybe_abort(self, reason: str) -> None:
        """Stop the run at the earliest non-finite tensor, and only for that.

        Aborting means no further UNet forwards, no scheduler steps and no decode -- which is the
        point: once a tensor is NaN everything downstream is black output that proves nothing and
        costs minutes of GPU time to produce.
        """
        if self.abort_on_first_failure and self.first_failure is not None:
            raise StepSplitAbort(
                f"STEP_SPLIT_FIRST_FAILURE:{reason}:{self.first_failure['checkpoint']}"
            )

    def _active(self) -> bool:
        return self.target_step is None or self.current_step + 1 == self.target_step

    def _prefix(self) -> str:
        """Label prefix for the step being recorded.

        In single-step mode the labels stay exactly as they were, so existing diagnostic receipts
        remain comparable. In whole-run mode every step needs its own namespace or step 2 would
        overwrite step 1 in any by-name lookup.
        """
        return "" if self.target_step is not None else f"step{self.current_step + 1:02d}_"

    # ------------------------------------------------------------------ installation
    def install(self, pipe: Any) -> None:
        scheduler = pipe.scheduler
        unet = pipe.unet

        original_scale = getattr(scheduler, "scale_model_input", None)
        original_step = scheduler.step
        original_forward = unet.forward

        def scale_model_input(sample: torch.Tensor, timestep: Any = None, *args: Any, **kwargs: Any):
            active = self._active()
            prefix = self._prefix()
            if active:
                self.record(f"{prefix}01_latents_before_scale_model_input", sample,
                            stage="scale_model_input")
            result = original_scale(sample, timestep, *args, **kwargs) if original_scale else sample
            if active:
                self.record(f"{prefix}02_scale_model_input_result", result, stage="scale_model_input")
                if isinstance(result, torch.Tensor):
                    self.latest_scaled_input = result.detach()
                self._maybe_abort("scale_model_input")
            return result

        def forward(sample: torch.Tensor, timestep: Any = None, encoder_hidden_states: Any = None,
                    *args: Any, **kwargs: Any):
            is_denoising = (isinstance(sample, torch.Tensor) and sample.dim() >= 1
                            and sample.shape[0] >= self.expected_views)
            active = self._active() and is_denoising
            module_probe = getattr(self, "module_probe", None)
            if module_probe is not None:
                module_probe.enabled = active
            if not is_denoising:
                # The reference pass gets its own stage and its own numbering. Folding it into the
                # step-1 checkpoints previously made a batch-1 timestep-0 forward look like the
                # first denoising forward.
                self.reference_forwards += 1
                reference_prefix = f"ref{self.reference_forwards:02d}_"
                self.record(f"{reference_prefix}unet_sample_input", sample, stage="reference_unet")
                result = original_forward(sample, timestep, encoder_hidden_states, *args, **kwargs)
                tensor = result[0] if isinstance(result, tuple) else getattr(result, "sample", result)
                cross = kwargs.get("cross_attention_kwargs") or {}
                cache_sink = cross.get("cache_hidden_states") if isinstance(cross, dict) else None
                self.reference_cache_contract = _cache_contract(self.expected_reference_cache, cache_sink)
                cache_configured = bool(self.reference_cache_contract.get("configured"))
                cache_failures = (
                    [] if not cache_configured or self.reference_cache_contract["valid"]
                    else [REFERENCE_CACHE_CONTRACT_FAILURE]
                )
                if cache_configured and self.reference_cache_contract["nonfinite_present"]:
                    cache_failures.extend(["NON_FINITE_PRESENT"])
                    for cache_entry in self.reference_cache_contract.get("entries", []):
                        stats = cache_entry.get("stats") or {}
                        if stats.get("has_nan"):
                            cache_failures.append("NAN")
                        if stats.get("has_inf"):
                            cache_failures.append("INF")
                output_stats = tensor_stats(f"{reference_prefix}unet_raw_output", tensor) if isinstance(tensor, torch.Tensor) else None
                output_is_zero = bool(output_stats is not None and output_stats["finite_fraction"] == 1.0 and _is_zero_or_flat(output_stats))
                sentinel_valid = bool(
                    output_is_zero
                    and self.reference_output_contract == "CACHE_SIDE_EFFECT_WITH_ZERO_SENTINEL"
                    and cache_configured and self.reference_cache_contract["valid"]
                )
                self.reference_output_zero_sentinel = {
                    "checkpoint": f"{reference_prefix}unet_raw_output",
                    "accepted": sentinel_valid,
                    "contract": self.reference_output_contract,
                    "cache_contract_valid": bool(self.reference_cache_contract["valid"]),
                    "finite": bool(output_stats and output_stats["finite_fraction"] == 1.0),
                    "all_zero_or_flat": output_is_zero,
                }
                output_failures = cache_failures if output_is_zero and not sentinel_valid else None
                self.record(
                    f"{reference_prefix}unet_raw_output", tensor, stage="reference_unet",
                    allow_flat=sentinel_valid, extra_failures=output_failures,
                )
                self.record_event(
                    f"{reference_prefix}reference_cache_contract", "reference_cache",
                    self.reference_cache_contract,
                    [] if not cache_configured or self.reference_cache_contract["valid"]
                    else [REFERENCE_CACHE_CONTRACT_FAILURE],
                )
                self._maybe_abort("reference_unet")
                return result
            prefix = self._prefix()
            if active:
                self.record(f"{prefix}03_unet_sample_input", sample, stage="unet_input")
                self.record(f"{prefix}04_unet_timestep", timestep, stage="unet_input")
                self.record(f"{prefix}05_unet_encoder_hidden_states", encoder_hidden_states,
                            stage="unet_input")
                for key, value in sorted(kwargs.items()):
                    if isinstance(value, torch.Tensor):
                        self.record(f"{prefix}06_unet_kwarg__{key}", value, stage="unet_input")
                    elif isinstance(value, (list, tuple)):
                        for index, item in enumerate(value):
                            if isinstance(item, torch.Tensor):
                                self.record(f"{prefix}07_unet_kwarg__{key}[{index}]", item,
                                            stage="unet_input")
                    elif isinstance(value, dict):
                        for sub_key, item in sorted(value.items()):
                            if isinstance(item, torch.Tensor):
                                self.record(f"{prefix}08_unet_kwarg__{key}__{sub_key}", item,
                                            stage="unet_input")
                self._maybe_abort("unet_input")
            result = original_forward(sample, timestep, encoder_hidden_states, *args, **kwargs)
            if active:
                tensor = result[0] if isinstance(result, tuple) else getattr(result, "sample", result)
                self.record(f"{prefix}09_unet_raw_output", tensor, stage="unet_output",
                            expect_variation=True)
                if isinstance(tensor, torch.Tensor):
                    self.latest_unet_output = tensor.detach()
                self._maybe_abort("unet_output")
            return result

        def step(model_output: torch.Tensor, timestep: Any, sample: torch.Tensor,
                 *args: Any, **kwargs: Any):
            active = self._active()
            prefix = self._prefix()
            if active:
                self.record(f"{prefix}10_model_output_into_scheduler", model_output,
                            stage="scheduler_input", expect_variation=True)
                self.record(f"{prefix}11_sample_into_scheduler", sample, stage="scheduler_input")
                self.record(f"{prefix}12_scheduler_timestep", timestep, stage="scheduler_input")
                self._save(f"{prefix}scheduler_input_model_output", model_output)
                self._save(f"{prefix}scheduler_input_sample", sample)
                self.latest_scheduler_input_sample = sample.detach() if isinstance(sample, torch.Tensor) else None
                self.latest_processed_model_prediction = model_output.detach() if isinstance(model_output, torch.Tensor) else None
                self.scheduler_timestep = _scalar(timestep)
                self._maybe_abort("scheduler_input")
            result = original_step(model_output, timestep, sample, *args, **kwargs)
            if active:
                predicted = getattr(result, "pred_original_sample", None)
                if isinstance(predicted, torch.Tensor):
                    self.record(f"{prefix}13_pred_original_sample", predicted, stage="scheduler_output")
                previous = result[0] if isinstance(result, tuple) else getattr(result, "prev_sample", None)
                if isinstance(previous, torch.Tensor):
                    self.record(f"{prefix}14_prev_sample", previous, stage="scheduler_output",
                                expect_variation=True)
                    self.record(f"{prefix}15_latent_for_next_iteration", previous,
                                stage="scheduler_output", expect_variation=True)
                self.current_step += 1
                self._maybe_abort("scheduler_output")
            else:
                self.current_step += 1
            return result

        if original_scale is not None:
            scheduler.scale_model_input = scale_model_input
            self._installed.append((scheduler, "scale_model_input", original_scale))
        scheduler.step = step
        self._installed.append((scheduler, "step", original_step))
        unet.forward = forward
        self._installed.append((unet, "forward", original_forward))

    def uninstall(self) -> None:
        for target, name, original in reversed(self._installed):
            setattr(target, name, original)
        self._installed.clear()

    # ------------------------------------------------------------------ output
    def classification(self) -> str:
        """What broke, named after the defect that actually occurred.

        Non-finiteness, unexpected flatness and cache-contract defects are separate outcomes. The
        valid reference zero sentinel produces none of them.
        """
        if self.first_failure is None:
            return ("MVADAPTER_ALL_RECORDED_CHECKPOINTS_FINITE" if self.target_step is None
                    else "MVADAPTER_STEP1_FINITE_FAILURE_MOVED_LATER")
        if self.first_nonfinite is not None:
            return "MVADAPTER_RUNTIME_REJECTED_NONFINITE"
        failures = set(self.first_failure.get("failures", []))
        if REFERENCE_CACHE_CONTRACT_FAILURE in failures:
            return "MVADAPTER_REFERENCE_CACHE_CONTRACT_FAILED"
        if self.first_unexpected_flat is not None:
            return "MVADAPTER_RUNTIME_REJECTED_FLAT_TENSOR"
        return "MVADAPTER_RUNTIME_REJECTED_NONFINITE"

    def finite_gate(self, expected_steps: int = 2) -> dict[str, Any]:
        """Named per-stage verdicts for a whole-run recording.

        The repair proof asks specific questions -- was the reference output finite, was the raw
        step-2 UNet output finite, did the latent hashes change between steps -- and answering them
        by eye from a few hundred checkpoints is how a partial repair gets called a full one.
        """
        by_label = {entry["checkpoint"]: entry for entry in self.records if entry.get("tensor")}

        def finite(label: str) -> bool | None:
            entry = by_label.get(label)
            if entry is None:
                return None
            return entry["stats"]["finite_fraction"] == 1.0

        checks: dict[str, bool | None] = {
            "reference_unet_output_finite": finite("ref01_unet_raw_output"),
        }
        if self.expected_reference_cache is not None:
            checks.update({
                "reference_zero_sentinel_accepted": (
                    None if self.reference_output_zero_sentinel is None
                    else bool(self.reference_output_zero_sentinel.get("accepted"))
                ),
                "reference_cache_contract_valid": (
                    None if self.reference_cache_contract is None
                    else bool(self.reference_cache_contract.get("valid"))
                ),
            })
        for step in range(1, int(expected_steps) + 1):
            prefix = f"step{step:02d}_"
            checks[f"step{step}_scaled_input_finite"] = finite(f"{prefix}02_scale_model_input_result")
            checks[f"step{step}_raw_unet_output_finite"] = finite(f"{prefix}09_unet_raw_output")
            checks[f"step{step}_scheduler_output_finite"] = finite(f"{prefix}14_prev_sample")
            checks[f"step{step}_latent_finite"] = finite(f"{prefix}15_latent_for_next_iteration")
        final_label = f"step{int(expected_steps):02d}_15_latent_for_next_iteration"
        checks["final_latent_finite"] = finite(final_label)

        latent_hashes = [
            (label, entry["stats"]["sha256"])
            for label, entry in by_label.items() if label.endswith("15_latent_for_next_iteration")
        ]
        checks["latent_hashes_change_between_steps"] = (
            len({digest for _, digest in latent_hashes}) == len(latent_hashes)
            if len(latent_hashes) > 1 else None
        )
        # Measured on the final latent, because that is the tensor the VAE actually decodes -- six
        # identical PNGs are six identical latents one stage earlier.
        final_latent = by_label.get(final_label)
        checks["views_not_all_identical"] = (
            (final_latent["all_views_identical"] is False)
            if final_latent is not None and final_latent.get("all_views_identical") is not None
            else None
        )
        return {
            "schema": "mvadapter_finite_gate_v1",
            "expected_steps": int(expected_steps),
            "reference_forwards": self.reference_forwards,
            "steps_recorded": self.current_step,
            "first_nonfinite": self.first_nonfinite,
            "checks": checks,
            "missing_checks": sorted(name for name, value in checks.items() if value is None),
            "failed_checks": sorted(name for name, value in checks.items() if value is False),
            "passed": bool(checks) and all(value is True for value in checks.values()),
            "latent_hashes": dict(latent_hashes),
            "reference_output_zero_sentinel": self.reference_output_zero_sentinel,
            "reference_cache_contract": self.reference_cache_contract,
        }

    def summary(self) -> dict[str, Any]:
        module_probe = getattr(self, "module_probe", None)
        return {
            "schema": "mvadapter_step_split_probe_v1",
            "unet_module_probe": module_probe.summary() if module_probe else None,
            "target_step": self.target_step,
            "classification": self.classification(),
            "first_nonfinite": self.first_nonfinite,
            "first_unexpected_flat": self.first_unexpected_flat,
            "first_failure": self.first_failure,
            "reference_output_zero_sentinel": self.reference_output_zero_sentinel,
            "reference_cache_contract": self.reference_cache_contract,
            "reference_cache_key_count_expected": (
                None if self.reference_cache_contract is None
                else self.reference_cache_contract.get("expected_key_count")
            ),
            "reference_cache_key_count_actual": (
                None if self.reference_cache_contract is None
                else self.reference_cache_contract.get("actual_key_count")
            ),
            "reference_cache_all_finite": (
                None if self.reference_cache_contract is None
                else self.reference_cache_contract.get("all_finite")
            ),
            "aborted": self.first_failure is not None and self.abort_on_first_failure,
            "checkpoints_recorded": len(self.records),
            "reference_forwards": self.reference_forwards,
            "saved_npz": self.saved_npz,
            "scheduler_timestep": getattr(self, "scheduler_timestep", None),
        }

    def write(self, path: Path | None = None) -> Path:
        destination = Path(path) if path else self.root / "step_split_probe.json"
        destination.write_text(
            json.dumps({**self.summary(), "records": self.records}, indent=2) + "\n",
            encoding="utf-8")
        return destination


class UNetModuleProbe:
    """Find the first UNet submodule whose inputs are finite and whose output is not.

    Forward hooks fire in execution order, so the first module that turns finite inputs into a
    non-finite output is the operation that actually breaks -- everything downstream of it is just
    propagating NaN and would otherwise look equally guilty.
    """

    def __init__(self, expected_views: int = 6, max_records: int = 400) -> None:
        self.expected_views = int(expected_views)
        self.max_records = int(max_records)
        self.enabled = False
        self.first_bad: dict[str, Any] | None = None
        self.trace: list[dict[str, Any]] = []
        self._handles: list[Any] = []
        self._order = 0

    def install(self, unet: Any) -> None:
        for name, module in unet.named_modules():
            if not name:
                continue
            self._handles.append(module.register_forward_hook(self._make_hook(name, module)))

    def _make_hook(self, name: str, module: Any):
        def hook(_module: Any, inputs: Any, output: Any) -> None:
            if not self.enabled:
                return
            in_tensors = [value for value in _flatten(inputs) if isinstance(value, torch.Tensor)]
            out_tensors = [value for value in _flatten(output) if isinstance(value, torch.Tensor)]
            if not out_tensors:
                return
            in_finite = all(bool(torch.isfinite(value).all()) for value in in_tensors
                            if value.is_floating_point())
            out_finite = all(bool(torch.isfinite(value).all()) for value in out_tensors
                             if value.is_floating_point())
            self._order += 1
            if self.first_bad is None and in_finite and not out_finite:
                self.first_bad = {
                    "module": name,
                    "module_class": type(module).__name__,
                    "execution_order": self._order,
                    "inputs": [_brief(value, f"in{index}") for index, value in enumerate(in_tensors)],
                    "outputs": [_brief(value, f"out{index}") for index, value in enumerate(out_tensors)],
                }
            if len(self.trace) < self.max_records:
                self.trace.append({
                    "order": self._order, "module": name, "module_class": type(module).__name__,
                    "input_finite": in_finite, "output_finite": out_finite,
                    "input_abs_max": _max_or_none(_abs_max(value) for value in in_tensors
                                                 if value.is_floating_point()),
                    "output_abs_max": _max_or_none(_abs_max(value) for value in out_tensors
                                                  if value.is_floating_point()),
                    "input_dtypes": sorted({str(v.dtype).replace("torch.", "") for v in in_tensors}),
                    "output_dtypes": sorted({str(v.dtype).replace("torch.", "") for v in out_tensors}),
                })
        return hook

    def uninstall(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def summary(self) -> dict[str, Any]:
        # The last module before the failure that still produced finite output, for context.
        preceding = [entry for entry in self.trace
                     if self.first_bad and entry["order"] < self.first_bad["execution_order"]
                     and entry["output_finite"]]
        return {
            "schema": "mvadapter_unet_module_probe_v1",
            "modules_traced": len(self.trace),
            "first_finite_to_nonfinite_module": self.first_bad,
            "last_finite_module_before_failure": preceding[-1] if preceding else None,
            "classification": ("MVADAPTER_UNET_FP16_OVERFLOW_LOCALIZED" if self.first_bad
                               else "MVADAPTER_UNET_FAILURE_NOT_LOCALIZED"),
        }


def _flatten(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten(item)
    elif hasattr(value, "sample") and isinstance(getattr(value, "sample"), torch.Tensor):
        yield value.sample


def _max_or_none(values) -> float | None:
    """An all-NaN tensor has no finite maximum, so None entries must not reach max()."""
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def _abs_max(tensor: torch.Tensor) -> float | None:
    finite = tensor[torch.isfinite(tensor)]
    return float(finite.abs().max().item()) if finite.numel() else None


def _brief(tensor: torch.Tensor, name: str) -> dict[str, Any]:
    finite = torch.isfinite(tensor)
    values = tensor[finite]
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "finite_fraction": round(float(finite.float().mean().item()), 8),
        "abs_max": float(values.abs().max().item()) if values.numel() else None,
        "std": float(values.std().item()) if values.numel() > 1 else None,
    }


def _describe_non_tensor(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, (int, float, bool, str)):
        return f"{type(value).__name__}={value}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{sorted(value)}]"
    return type(value).__name__


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().reshape(-1)[0].item()) if value.numel() else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
