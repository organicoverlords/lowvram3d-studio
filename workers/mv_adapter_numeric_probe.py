"""Bounded finite-statistics collector and first-nonfinite fail-closed gate.

This module is deliberately free of pipeline knowledge so it can be unit tested
without CUDA, MV-Adapter or Diffusers. The driver
``diagnose_mv_adapter_numeric_boundary`` wires it into the proven failing
SD2.1 image-to-multiview run.

Statistics are computed over the *finite* elements only. A tensor containing a
single NaN would otherwise poison ``min``/``max``/``mean``/``std`` and hide the
magnitude information that identifies an overflow boundary.
"""
from __future__ import annotations

from typing import Any


STATISTIC_FIELDS = (
    "shape",
    "dtype",
    "device",
    "finite_count",
    "nonfinite_count",
    "finite_ratio",
    "minimum",
    "maximum",
    "mean",
    "standard_deviation",
    "absolute_maximum",
)

# Ordered longest-prefix-first so ``final_pre_decode_latents`` is not shadowed.
BOUNDARY_CATEGORIES = (
    ("reference_latents", "REFERENCE_LATENTS"),
    ("reference_unet_output", "UNET_OUTPUT"),
    ("control_image_prepared", "CONTROL_ENCODER_INPUT"),
    ("condition_encoder_input", "CONTROL_ENCODER_INPUT"),
    ("adapter_state", "CONTROL_ENCODER_OUTPUT"),
    ("initial_noise_latents", "INITIAL_NOISE"),
    ("unet_noise_pred", "UNET_OUTPUT"),
    ("final_pre_decode_latents", "SCHEDULER_LATENTS"),
    ("scheduler_latents", "SCHEDULER_LATENTS"),
    ("vae_decode_input", "VAE_DECODE_INPUT"),
    ("vae_decode_output", "VAE_DECODE_OUTPUT"),
)

# Decision rules fixed before the run so the outcome cannot be rationalised
# after seeing the numbers. ``task_rule`` records whether the decision is
# verbatim from the task brief or an explicit extension of it.
BOUNDARY_DECISIONS = {
    "REFERENCE_LATENTS": (
        "INSPECT_VAE_ENCODE_REFERENCE_LATENT_BOUNDARY",
        False,
    ),
    "CONTROL_ENCODER_INPUT": (
        "INSPECT_CONTROL_IMAGE_NORMALISATION_AND_CONDITION_ENCODER_PRECISION",
        True,
    ),
    "CONTROL_ENCODER_OUTPUT": (
        "INSPECT_CONTROL_IMAGE_NORMALISATION_AND_CONDITION_ENCODER_PRECISION",
        True,
    ),
    "INITIAL_NOISE": (
        "STOP_PLAIN_I2MV_ROUTE_INSPECT_OFFICIAL_GEOMETRY_GUIDED_SD21_MV_ADAPTER",
        False,
    ),
    "UNET_OUTPUT": (
        "STOP_PLAIN_I2MV_ROUTE_INSPECT_OFFICIAL_GEOMETRY_GUIDED_SD21_MV_ADAPTER",
        True,
    ),
    "SCHEDULER_LATENTS": (
        "STOP_PLAIN_I2MV_ROUTE_INSPECT_OFFICIAL_GEOMETRY_GUIDED_SD21_MV_ADAPTER",
        True,
    ),
    "VAE_DECODE_INPUT": (
        "REPAIR_VAE_SCALING_DTYPE_BOUNDARY_ONE_CORRECTED_RETRY",
        True,
    ),
    "VAE_DECODE_OUTPUT": (
        "REPAIR_VAE_SCALING_DTYPE_BOUNDARY_ONE_CORRECTED_RETRY",
        True,
    ),
}

NO_NONFINITE_DECISION = (
    "PRESERVE_FINITE_DECODED_TENSOR_AND_PATCH_POSTPROCESS_PATH_ONLY",
    True,
)


class FirstNonfiniteTensor(RuntimeError):
    """Raised the moment a probed tensor contains NaN or Inf."""

    def __init__(self, label: str, entry: dict[str, Any]) -> None:
        statistics = entry["statistics"]
        super().__init__(
            f"first nonfinite tensor at boundary {label!r}: "
            f"nonfinite={statistics['nonfinite_count']} "
            f"finite_ratio={statistics['finite_ratio']}"
        )
        self.label = label
        self.entry = entry


def classify_boundary(label: str) -> str:
    """Map a probe label to the coarse boundary category used for decisions."""

    for prefix, category in BOUNDARY_CATEGORIES:
        if label.startswith(prefix):
            return category
    return "UNCLASSIFIED"


def decide_next_action(label: str | None) -> dict[str, Any]:
    """Return the pre-registered follow-up decision for a boundary label."""

    if label is None:
        decision, from_task_rules = NO_NONFINITE_DECISION
        return {
            "first_nonfinite_label": None,
            "boundary_category": "NONE_ALL_PROBED_TENSORS_FINITE",
            "decision": decision,
            "decision_from_task_rules": from_task_rules,
        }

    category = classify_boundary(label)
    decision, from_task_rules = BOUNDARY_DECISIONS.get(
        category, ("STOP_AND_REPORT_UNCLASSIFIED_BOUNDARY", False)
    )
    return {
        "first_nonfinite_label": label,
        "boundary_category": category,
        "decision": decision,
        "decision_from_task_rules": from_task_rules,
    }


def tensor_statistics(tensor: Any) -> dict[str, Any]:
    """Collect the fixed statistic set for one tensor.

    ``minimum``/``maximum``/``mean``/``standard_deviation``/``absolute_maximum``
    are computed in float32 over finite elements only, and are ``None`` when the
    tensor has no finite elements at all.
    """

    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"probe expected a tensor, received {type(tensor)!r}")

    values = tensor.detach()
    total = int(values.numel())
    finite_mask = torch.isfinite(values)
    finite_count = int(finite_mask.sum().item())
    nonfinite_count = total - finite_count

    statistics: dict[str, Any] = {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "device": str(values.device),
        "finite_count": finite_count,
        "nonfinite_count": nonfinite_count,
        "finite_ratio": (finite_count / total) if total else None,
        "minimum": None,
        "maximum": None,
        "mean": None,
        "standard_deviation": None,
        "absolute_maximum": None,
    }

    if finite_count == 0:
        return statistics

    if nonfinite_count == 0:
        finite_values = values.reshape(-1).to(torch.float32)
    else:
        finite_values = values[finite_mask].to(torch.float32)

    statistics["minimum"] = float(finite_values.min().item())
    statistics["maximum"] = float(finite_values.max().item())
    statistics["mean"] = float(finite_values.mean().item())
    # Population standard deviation: the unbiased estimator is NaN for a single
    # finite element, which would be indistinguishable from a real nonfinite.
    statistics["standard_deviation"] = float(finite_values.std(unbiased=False).item())
    statistics["absolute_maximum"] = float(finite_values.abs().max().item())
    return statistics


class NumericProbe:
    """Ordered tensor-statistics recorder with a first-nonfinite fail-closed gate."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.first_nonfinite: dict[str, Any] | None = None

    @property
    def first_nonfinite_label(self) -> str | None:
        return None if self.first_nonfinite is None else self.first_nonfinite["label"]

    def record(self, label: str, tensor: Any, **context: Any) -> dict[str, Any]:
        """Record one tensor boundary, raising immediately if it is nonfinite."""

        statistics = tensor_statistics(tensor)
        entry: dict[str, Any] = {
            "order": len(self.records) + 1,
            "label": label,
            "boundary_category": classify_boundary(label),
            "statistics": statistics,
        }
        entry.update(context)
        self.records.append(entry)

        if statistics["nonfinite_count"] > 0:
            if self.first_nonfinite is None:
                self.first_nonfinite = entry
            raise FirstNonfiniteTensor(label, entry)
        return entry

    def summary(self) -> dict[str, Any]:
        return {
            "probe_record_count": len(self.records),
            "nonfinite_boundary_found": self.first_nonfinite is not None,
            "probed_labels": [entry["label"] for entry in self.records],
            "records": self.records,
            **decide_next_action(self.first_nonfinite_label),
        }
