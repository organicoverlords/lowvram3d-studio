"""Per-step latent telemetry for MV-Adapter runs.

The 384 attempt that produced six black images carried no per-step evidence, so there was no way to
tell from the receipt whether the latents were dead from initialisation, collapsed mid-denoise, or
were fine and destroyed at the decode. (They were fine: the VAE was returning NaN in float16. The
telemetry stays because the next failure will not be the same one.)

Records, at every requested checkpoint and for every view: shape, dtype, device, min, max, mean,
standard deviation, L1 and L2 norms, finite fraction, zero fraction, near-zero fraction and a
SHA-256 of the tensor. Selected latents are also written as compressed NPZ.

Records numerical non-finiteness separately from finite flatness. The reference UNet in this route
uses a documented zero return sentinel while writing the useful reference cache through a side
effect, so the reference sentinel is validated by the step probe rather than mistaken for a
non-finite failure.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

NEAR_ZERO = 1e-6
FLAT_STD = 1e-6


def tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    as_float = detached.float()
    finite = torch.isfinite(as_float)
    values = as_float[finite]
    payload: dict[str, Any] = {
        "name": name,
        "shape": list(detached.shape),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "device": str(detached.device),
        "numel": int(detached.numel()),
        "finite_fraction": round(float(finite.float().mean().item()), 8),
        "has_nan": bool(torch.isnan(as_float).any().item()),
        "has_inf": bool(torch.isinf(as_float).any().item()),
    }
    if values.numel():
        payload.update({
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
            "std": float(values.std().item()) if values.numel() > 1 else 0.0,
            "l1_norm": float(values.abs().sum().item()),
            "l2_norm": float(values.pow(2).sum().sqrt().item()),
            "zero_fraction": round(float((values == 0).float().mean().item()), 8),
            "near_zero_fraction": round(float((values.abs() < NEAR_ZERO).float().mean().item()), 8),
        })
    else:
        # No finite values at all. Reporting zero_fraction 1.0 here previously made an all-NaN
        # tensor also read as ALL_ZERO, which is a different defect with a different fix.
        payload.update({"min": None, "max": None, "mean": None, "std": None,
                        "l1_norm": None, "l2_norm": None,
                        "zero_fraction": None, "near_zero_fraction": None})
    payload.update({
        "nan_count": int(torch.isnan(as_float).sum().item()),
        "posinf_count": int(torch.isposinf(as_float).sum().item()),
        "neginf_count": int(torch.isneginf(as_float).sum().item()),
        "abs_max": float(values.abs().max().item()) if values.numel() else None,
    })
    payload["sha256"] = hashlib.sha256(
        np.ascontiguousarray(as_float.cpu().numpy()).tobytes()).hexdigest()
    return payload


def per_view_stats(name: str, tensor: torch.Tensor) -> list[dict[str, Any]]:
    if tensor.dim() == 0:
        return []
    return [tensor_stats(f"{name}[view{index}]", tensor[index]) for index in range(tensor.shape[0])]


def failures_for(stats: dict[str, Any], expect_variation: bool = True,
                 allow_flat: bool = False) -> list[str]:
    """Failure labels for one tensor.

    A non-finite tensor is reported only as non-finite. It is deliberately not also labelled
    ALL_ZERO or NEAR_ZERO_STD: those describe a tensor whose values collapsed, which is a different
    defect with a different fix, and conflating them sent the first diagnosis at the wrong stage.
    """
    failures: list[str] = []
    if stats["has_nan"]:
        failures.append("NAN")
    if stats["has_inf"]:
        failures.append("INF")
    if stats["finite_fraction"] < 1.0:
        failures.append("NON_FINITE_PRESENT")
        return failures
    # Index-like tensors -- timesteps, step counters -- are integer and often scalar, and zero is a
    # legitimate value for them. Only floating-point tensors with real spatial extent can be
    # "collapsed to zero" in the sense this check is looking for.
    index_like = stats["dtype"].startswith(("int", "uint", "bool")) or not stats["shape"]
    if not index_like and not allow_flat:
        if stats.get("zero_fraction") is not None and stats["zero_fraction"] >= 0.999:
            failures.append("ALL_ZERO")
        if expect_variation and stats.get("std") is not None and stats["std"] < FLAT_STD:
            failures.append("NEAR_ZERO_STD")
    return failures


def views_identical(view_records: Iterable[dict[str, Any]]) -> bool | None:
    """Whether every view hashes the same, or None when the comparison is not meaningful.

    Two all-NaN views hash identically because their bytes are identical, which says nothing about
    whether the model produced the same content. Equality is only reported when every view is
    finite.
    """
    records = list(view_records)
    if not records:
        return None
    if any(record["finite_fraction"] < 1.0 for record in records):
        return None
    return len({record["sha256"] for record in records}) == 1


class LatentTelemetry:
    """Collects checkpoints during a run and writes them out as JSON plus NPZ."""

    def __init__(self, root: Path, snapshot_steps: tuple[int, ...] = (1, 5, 10, 15, 18, 19, 20),
                 save_tensors: bool = True) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.snapshot_steps = set(snapshot_steps)
        self.save_tensors = save_tensors
        self.checkpoints: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.violations: list[dict[str, Any]] = []
        self.first_nonfinite_record: dict[str, Any] | None = None
        self.first_unexpected_flat_record: dict[str, Any] | None = None
        self.probe_summary: dict[str, Any] | None = None
        self.probe_records: list[dict[str, Any]] = []
        # Checkpoints and steps interleave in real time, so ordering has to come from a single
        # counter. Scanning checkpoints before steps previously reported the final blank image as
        # the first failure when the latents had already died at step 1.
        self._sequence = 0

    def _next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def _track_failures(self, entry: dict[str, Any], kind: str, at: Any) -> None:
        failures = list(entry.get("failures") or [])
        if not failures:
            return
        record = {"kind": kind, "at": at, "sequence": entry["sequence"],
                  "failures": failures}
        self.violations.append(record)
        if self.first_nonfinite_record is None and any(
            label in {"NAN", "INF", "NON_FINITE_PRESENT"} for label in failures
        ):
            self.first_nonfinite_record = record
        if self.first_unexpected_flat_record is None and any(
            label in {"ALL_ZERO", "NEAR_ZERO_STD", "ALL_VIEWS_IDENTICAL"}
            for label in failures
        ):
            self.first_unexpected_flat_record = record

    def attach_probe(self, summary: dict[str, Any] | None,
                     records: list[dict[str, Any]] | None = None) -> None:
        """Embed the probe's chronological boundary record in this single telemetry artifact."""
        self.probe_summary = summary
        self.probe_records = list(records or [])

    # ---------------------------------------------------------------- recording
    def record(self, label: str, tensor: torch.Tensor, *, expect_variation: bool = True,
               allow_flat: bool = False, save: bool = False,
               extra: dict[str, Any] | None = None) -> dict[str, Any]:
        stats = tensor_stats(label, tensor)
        views = per_view_stats(label, tensor)
        entry: dict[str, Any] = {
            "sequence": self._next_sequence(),
            "timestamp": time.time(),
            "checkpoint": label,
            "aggregate": stats,
            "per_view": views,
            "all_views_identical": views_identical(views) if views else None,
            "failures": failures_for(stats, expect_variation, allow_flat),
        }
        if entry["all_views_identical"] is True and expect_variation and len(views) > 1:
            entry["failures"].append("ALL_VIEWS_IDENTICAL")
        if extra:
            entry.update(extra)
        if save and self.save_tensors:
            path = self.root / f"{label}.npz"
            np.savez_compressed(path, tensor=tensor.detach().float().cpu().numpy())
            entry["npz"] = str(path)
        self.checkpoints.append(entry)
        self._track_failures(entry, "checkpoint", label)
        return entry

    def record_step(self, step: int, timestep: Any, latents: torch.Tensor,
                    scheduler: Any = None, noise_pred: torch.Tensor | None = None,
                    condition_residuals: Any = None,
                    scaled_input: torch.Tensor | None = None,
                    scheduler_input_sample: torch.Tensor | None = None,
                    processed_model_prediction: torch.Tensor | None = None) -> dict[str, Any]:
        """One chronological record per denoising step.

        ``scaled_input`` and ``noise_pred`` are the tensors on either side of the UNet forward. They
        are captured by the step probe and handed in here rather than recomputed, so a single
        artifact carries the whole step -- timestep, noise level, what went into the UNet, what came
        out, and what the scheduler made of it -- instead of forcing two files to be read together.
        """
        entry: dict[str, Any] = {
            "sequence": self._next_sequence(),
            "timestamp": time.time(),
            "step": int(step),
            "timestep": _scalar(timestep),
            "scheduler_class": type(scheduler).__name__ if scheduler is not None else None,
            "sigma": _sigma_for(scheduler, timestep),
            "scheduler_output_latents": tensor_stats(f"step{step:02d}_latents", latents),
        }
        if scaled_input is not None:
            entry["unet_scaled_input"] = tensor_stats(f"step{step:02d}_scaled_input", scaled_input)
        if scheduler_input_sample is not None:
            entry["scheduler_input_sample"] = tensor_stats(
                f"step{step:02d}_scheduler_input_sample", scheduler_input_sample
            )
        if processed_model_prediction is not None:
            entry["processed_model_prediction"] = tensor_stats(
                f"step{step:02d}_processed_model_prediction", processed_model_prediction
            )
        entry["per_view"] = per_view_stats(f"step{step:02d}_latents", latents)
        entry["all_views_identical"] = views_identical(entry["per_view"]) if entry["per_view"] else None
        entry["failures"] = failures_for(entry["scheduler_output_latents"])
        if entry["all_views_identical"] is True:
            entry["failures"].append("ALL_VIEWS_IDENTICAL")
        if noise_pred is not None:
            entry["unet_noise_prediction"] = tensor_stats(f"step{step:02d}_noise_pred", noise_pred)
            entry["failures"].extend(
                failure for failure in failures_for(entry["unet_noise_prediction"])
                if failure not in entry["failures"]
            )
        for key in ("unet_scaled_input", "scheduler_input_sample", "processed_model_prediction"):
            stats = entry.get(key)
            if stats is not None:
                entry["failures"].extend(
                    failure for failure in failures_for(stats)
                    if failure not in entry["failures"]
                )
        if condition_residuals is not None:
            entry["condition_residuals"] = _residual_stats(condition_residuals)
        if step in self.snapshot_steps and self.save_tensors:
            path = self.root / f"step{step:02d}_latents.npz"
            np.savez_compressed(path, latents=latents.detach().float().cpu().numpy())
            entry["npz"] = str(path)
        self.steps.append(entry)
        self._track_failures(entry, "step", step)
        return entry

    # ---------------------------------------------------------------- output
    def first_failure(self) -> dict[str, Any] | None:
        """Earliest failing record in real time, across both checkpoints and steps."""
        candidates = [
            {"kind": "checkpoint", "at": entry["checkpoint"],
             "sequence": entry["sequence"], "timestamp": entry.get("timestamp"),
             "failures": entry["failures"]}
            for entry in self.checkpoints if entry["failures"]
        ] + [
            {"kind": "step", "at": entry["step"],
             "sequence": entry["sequence"], "timestamp": entry.get("timestamp"),
             "failures": entry["failures"]}
            for entry in self.steps if entry["failures"]
        ]
        if self.probe_summary and self.probe_summary.get("first_failure"):
            probe_failure = dict(self.probe_summary["first_failure"])
            probe_record = next(
                (item for item in self.probe_records
                 if item.get("checkpoint") == probe_failure.get("checkpoint")),
                None,
            )
            candidates.append({
                "kind": "probe",
                "at": probe_failure.get("checkpoint"),
                "sequence": probe_failure.get("sequence"),
                "timestamp": (probe_record or {}).get("timestamp") or float("inf"),
                "failures": probe_failure.get("failures", []),
            })
        if not candidates:
            return None
        return min(candidates, key=lambda entry: (entry.get("timestamp", float("inf")),
                                                  entry.get("sequence", 0)))

    def chronological_records(self) -> list[dict[str, Any]]:
        """Return a compact merged index for telemetry and probe records."""
        records: list[dict[str, Any]] = []
        for entry in self.checkpoints:
            records.append({"source": "telemetry", "kind": "checkpoint",
                            "checkpoint": entry["checkpoint"], "stage": entry.get("stage"),
                            "sequence": entry["sequence"], "timestamp": entry.get("timestamp")})
        for entry in self.steps:
            records.append({"source": "telemetry", "kind": "step",
                            "checkpoint": f"step{entry['step']:02d}", "stage": "denoising_step",
                            "sequence": entry["sequence"], "timestamp": entry.get("timestamp")})
        for entry in self.probe_records:
            records.append({"source": "probe", "kind": "checkpoint",
                            "checkpoint": entry.get("checkpoint"), "stage": entry.get("stage"),
                            "sequence": entry.get("sequence"), "timestamp": entry.get("timestamp")})
        records.sort(key=lambda entry: (entry.get("timestamp") or float("inf"),
                                        entry.get("sequence", 0)))
        for index, entry in enumerate(records, 1):
            entry["chronological_sequence"] = index
        return records

    def summary(self) -> dict[str, Any]:
        hashes = [entry["scheduler_output_latents"]["sha256"] for entry in self.steps]
        return {
            "schema": "mvadapter_latent_telemetry_v1",
            "checkpoint_count": len(self.checkpoints),
            "step_count": len(self.steps),
            "step_hashes_all_distinct": len(set(hashes)) == len(hashes) if hashes else None,
            "violations": self.violations,
            "first_failure": self.first_failure(),
            "first_nonfinite": self.first_nonfinite_record,
            "first_unexpected_flat": self.first_unexpected_flat_record,
            "probe_summary": self.probe_summary,
            "chronological_record_count": len(self.chronological_records()),
            "chronological_records": self.chronological_records(),
            "clean": not self.violations,
        }

    def write(self, path: Path | None = None) -> Path:
        destination = Path(path) if path else self.root / "latent_telemetry.json"
        payload = {**self.summary(), "checkpoints": self.checkpoints, "steps": self.steps,
                   "probe_records": self.probe_records}
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return destination


def _scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().reshape(-1)[0].item()) if value.numel() else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sigma_for(scheduler: Any, timestep: Any) -> float | None:
    """Noise level for the step, however this scheduler happens to express it."""
    if scheduler is None:
        return None
    sigmas = getattr(scheduler, "sigmas", None)
    step_index = getattr(scheduler, "step_index", None)
    if sigmas is not None and step_index is not None:
        try:
            return float(sigmas[step_index])
        except (IndexError, TypeError, ValueError):
            return None
    alphas = getattr(scheduler, "alphas_cumprod", None)
    index = _scalar(timestep)
    if alphas is not None and index is not None:
        try:
            alpha = float(alphas[int(index)])
            return float(((1.0 - alpha) / max(alpha, 1e-12)) ** 0.5)
        except (IndexError, TypeError, ValueError):
            return None
    return None


def _residual_stats(residuals: Any) -> Any:
    if isinstance(residuals, torch.Tensor):
        return tensor_stats("condition_residual", residuals)
    if isinstance(residuals, (list, tuple)):
        return [tensor_stats(f"condition_residual[{index}]", value)
                for index, value in enumerate(residuals) if isinstance(value, torch.Tensor)]
    return None
