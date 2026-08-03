"""Per-step latent telemetry for MV-Adapter runs.

The 384 attempt that produced six black images carried no per-step evidence, so there was no way to
tell from the receipt whether the latents were dead from initialisation, collapsed mid-denoise, or
were fine and destroyed at the decode. (They were fine: the VAE was returning NaN in float16. The
telemetry stays because the next failure will not be the same one.)

Records, at every requested checkpoint and for every view: shape, dtype, device, min, max, mean,
standard deviation, L1 and L2 norms, finite fraction, zero fraction, near-zero fraction and a
SHA-256 of the tensor. Selected latents are also written as compressed NPZ.

Trips immediately on NaN, Inf, an all-zero tensor, a near-zero standard deviation, or identical
latent hashes across all views where variation is expected -- the last of which is what "six
identical black PNGs" looks like one stage earlier.
"""
from __future__ import annotations

import hashlib
import json
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
        payload.update({"min": None, "max": None, "mean": None, "std": None,
                        "l1_norm": None, "l2_norm": None,
                        "zero_fraction": 1.0, "near_zero_fraction": 1.0})
    payload["sha256"] = hashlib.sha256(
        np.ascontiguousarray(as_float.cpu().numpy()).tobytes()).hexdigest()
    return payload


def per_view_stats(name: str, tensor: torch.Tensor) -> list[dict[str, Any]]:
    if tensor.dim() == 0:
        return []
    return [tensor_stats(f"{name}[view{index}]", tensor[index]) for index in range(tensor.shape[0])]


def failures_for(stats: dict[str, Any], expect_variation: bool = True) -> list[str]:
    failures: list[str] = []
    if stats["has_nan"]:
        failures.append("NAN")
    if stats["has_inf"]:
        failures.append("INF")
    if stats["finite_fraction"] < 1.0:
        failures.append("NON_FINITE_PRESENT")
    if stats.get("zero_fraction") is not None and stats["zero_fraction"] >= 0.999:
        failures.append("ALL_ZERO")
    if expect_variation and stats.get("std") is not None and stats["std"] < FLAT_STD:
        failures.append("NEAR_ZERO_STD")
    return failures


def views_identical(view_records: Iterable[dict[str, Any]]) -> bool:
    hashes = {record["sha256"] for record in view_records}
    return len(hashes) == 1


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

    # ---------------------------------------------------------------- recording
    def record(self, label: str, tensor: torch.Tensor, *, expect_variation: bool = True,
               save: bool = False, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        stats = tensor_stats(label, tensor)
        views = per_view_stats(label, tensor)
        entry: dict[str, Any] = {
            "checkpoint": label,
            "aggregate": stats,
            "per_view": views,
            "all_views_identical": views_identical(views) if views else None,
            "failures": failures_for(stats, expect_variation),
        }
        if entry["all_views_identical"] and expect_variation and len(views) > 1:
            entry["failures"].append("ALL_VIEWS_IDENTICAL")
        if extra:
            entry.update(extra)
        if save and self.save_tensors:
            path = self.root / f"{label}.npz"
            np.savez_compressed(path, tensor=tensor.detach().float().cpu().numpy())
            entry["npz"] = str(path)
        self.checkpoints.append(entry)
        if entry["failures"]:
            self.violations.append({"checkpoint": label, "failures": entry["failures"]})
        return entry

    def record_step(self, step: int, timestep: Any, latents: torch.Tensor,
                    scheduler: Any = None, noise_pred: torch.Tensor | None = None,
                    condition_residuals: Any = None) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "step": int(step),
            "timestep": _scalar(timestep),
            "scheduler_class": type(scheduler).__name__ if scheduler is not None else None,
            "sigma": _sigma_for(scheduler, timestep),
            "scheduler_output_latents": tensor_stats(f"step{step:02d}_latents", latents),
        }
        entry["per_view"] = per_view_stats(f"step{step:02d}_latents", latents)
        entry["all_views_identical"] = views_identical(entry["per_view"]) if entry["per_view"] else None
        entry["failures"] = failures_for(entry["scheduler_output_latents"])
        if entry["all_views_identical"]:
            entry["failures"].append("ALL_VIEWS_IDENTICAL")
        if noise_pred is not None:
            entry["unet_noise_prediction"] = tensor_stats(f"step{step:02d}_noise_pred", noise_pred)
        if condition_residuals is not None:
            entry["condition_residuals"] = _residual_stats(condition_residuals)
        if step in self.snapshot_steps and self.save_tensors:
            path = self.root / f"step{step:02d}_latents.npz"
            np.savez_compressed(path, latents=latents.detach().float().cpu().numpy())
            entry["npz"] = str(path)
        self.steps.append(entry)
        if entry["failures"]:
            self.violations.append({"step": step, "failures": entry["failures"]})
        return entry

    # ---------------------------------------------------------------- output
    def first_failure(self) -> dict[str, Any] | None:
        for entry in self.checkpoints:
            if entry["failures"]:
                return {"kind": "checkpoint", "at": entry["checkpoint"], "failures": entry["failures"]}
        for entry in self.steps:
            if entry["failures"]:
                return {"kind": "step", "at": entry["step"], "failures": entry["failures"]}
        return None

    def summary(self) -> dict[str, Any]:
        hashes = [entry["scheduler_output_latents"]["sha256"] for entry in self.steps]
        return {
            "schema": "mvadapter_latent_telemetry_v1",
            "checkpoint_count": len(self.checkpoints),
            "step_count": len(self.steps),
            "step_hashes_all_distinct": len(set(hashes)) == len(hashes) if hashes else None,
            "violations": self.violations,
            "first_failure": self.first_failure(),
            "clean": not self.violations,
        }

    def write(self, path: Path | None = None) -> Path:
        destination = Path(path) if path else self.root / "latent_telemetry.json"
        payload = {**self.summary(), "checkpoints": self.checkpoints, "steps": self.steps}
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
