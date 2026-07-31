"""Content validation for generated multi-view image sets.

Exists because a diffusion worker can fail in a way that no exit code reveals. On job 26a37e41 the
SD2.1 UNet produced non-finite latents on a GTX 1660 SUPER (Turing fp16), diffusers cast them with
``(images * 255).round().astype("uint8")`` -- NaN casts to 0 -- and the worker wrote six
byte-identical pure-black PNGs alongside ``"success": true``. Every downstream stage then behaved
correctly on garbage input, and the failure only became visible as an all-black textured model.

A "did the process exit 0" check cannot catch that. These predicates inspect pixels instead, and
are kept free of torch/diffusers imports so they can be unit-tested directly.
"""
from __future__ import annotations

import numpy as np

# Fractions of full scale, applied to images normalised to [0, 1].
BLACK_PIXEL_LEVEL = 2.0 / 255.0
MIN_NONBLACK_FRACTION = 0.005
MIN_UNIQUE_COLOURS_8BIT = 64
MIN_STD = 1.0 / 255.0


def as_unit_float(image: np.ndarray) -> np.ndarray:
    """Normalise uint8 or float image data to float64 in [0, 1] without hiding non-finite values."""
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return array.astype(np.float64) / 255.0
    return array.astype(np.float64)


def view_statistics(image: np.ndarray) -> dict:
    array = as_unit_float(image)
    rgb = array[..., :3] if array.ndim == 3 and array.shape[-1] >= 3 else array
    finite = np.isfinite(rgb)
    safe = np.where(finite, rgb, 0.0)
    flat = safe.reshape(-1, safe.shape[-1]) if safe.ndim == 3 else safe.reshape(-1, 1)
    return {
        "all_finite": bool(finite.all()),
        "nonfinite_count": int((~finite).sum()),
        "min": float(safe.min()),
        "max": float(safe.max()),
        "mean": float(safe.mean()),
        "std": float(safe.std()),
        "unique_colours_8bit": int(len(np.unique(np.round(flat * 255).astype(np.int16), axis=0))),
        "nonblack_fraction": float((safe.max(axis=-1) > BLACK_PIXEL_LEVEL).mean()),
    }


def is_effectively_blank(statistics: dict) -> bool:
    """Black, near-black, or flat enough to carry no usable surface detail."""
    return (
        statistics["nonblack_fraction"] < MIN_NONBLACK_FRACTION
        or statistics["unique_colours_8bit"] < MIN_UNIQUE_COLOURS_8BIT
        or statistics["std"] < MIN_STD
    )


def validate_generated_views(views: dict[str, np.ndarray], required: tuple[str, ...]) -> list[str]:
    """Return the reasons this view set must not be recorded as a success; empty means it may be.

    Ordering matters for diagnosis: non-finite data is reported before blankness, because NaN is
    the cause and the black pixels are only its visible symptom after the uint8 cast.
    """
    problems: list[str] = []

    missing = [name for name in required if name not in views]
    if missing:
        problems.append(f"missing required views: {sorted(missing)}")

    statistics = {name: view_statistics(image) for name, image in views.items()}

    nonfinite = sorted(name for name, s in statistics.items() if not s["all_finite"])
    if nonfinite:
        counts = {name: statistics[name]["nonfinite_count"] for name in nonfinite}
        problems.append(f"non-finite pixels in generated views: {counts}")

    present_required = [name for name in required if name in views]
    if present_required:
        blank = [name for name in present_required if is_effectively_blank(statistics[name])]
        if len(blank) == len(present_required):
            problems.append(
                f"all required views are effectively black or constant: {sorted(blank)}"
            )
        elif blank:
            problems.append(f"required views are effectively black or constant: {sorted(blank)}")

    digests: dict[bytes, list[str]] = {}
    for name in present_required:
        key = np.ascontiguousarray(as_unit_float(views[name])).tobytes()
        digests.setdefault(key, []).append(name)
    duplicates = sorted(sorted(names) for names in digests.values() if len(names) > 1)
    if duplicates:
        problems.append(f"required views are byte-identical: {duplicates}")

    return problems
