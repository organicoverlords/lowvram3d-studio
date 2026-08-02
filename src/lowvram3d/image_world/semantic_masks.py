"""Semantic separation contracts for image-to-world reconstruction.

Terrain reconstruction must never consume a generic foreground mask.  This
module validates mutually constrained class probabilities and derives a
conservative terrain candidate mask with an explicit unresolved region.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .contracts import ContractError

SEMANTIC_CLASSES = (
    "terrain",
    "water",
    "sky",
    "vegetation",
    "structure",
    "residual",
)


@dataclass(frozen=True)
class SemanticMaskSet:
    probabilities: Mapping[str, np.ndarray]
    terrain_candidate: np.ndarray
    unresolved: np.ndarray
    confidence: np.ndarray
    class_index: np.ndarray
    terrain_threshold: float
    exclusion_threshold: float

    def validate(self) -> None:
        missing = [name for name in SEMANTIC_CLASSES if name not in self.probabilities]
        if missing:
            raise ContractError(f"missing semantic classes: {missing}")
        shapes = {np.asarray(value).shape for value in self.probabilities.values()}
        if len(shapes) != 1:
            raise ContractError("all semantic probability maps must share one shape")
        shape = next(iter(shapes))
        if len(shape) != 2:
            raise ContractError("semantic maps must be 2D")
        for name, value in self.probabilities.items():
            array = np.asarray(value)
            if not np.isfinite(array).all():
                raise ContractError(f"{name} contains non-finite values")
            if array.min(initial=0.0) < 0.0 or array.max(initial=0.0) > 1.0:
                raise ContractError(f"{name} probabilities must be in [0, 1]")
        for name in ("terrain_candidate", "unresolved", "confidence", "class_index"):
            if np.asarray(getattr(self, name)).shape != shape:
                raise ContractError(f"{name} shape must match semantic maps")
        if np.any(self.terrain_candidate & self.unresolved):
            raise ContractError("terrain candidates cannot also be unresolved")


def build_semantic_mask_set(
    probabilities: Mapping[str, np.ndarray],
    *,
    valid_mask: np.ndarray | None = None,
    terrain_threshold: float = 0.60,
    exclusion_threshold: float = 0.35,
    minimum_margin: float = 0.15,
) -> SemanticMaskSet:
    """Build a conservative terrain mask from per-class probabilities.

    A pixel is accepted only when terrain exceeds ``terrain_threshold``, every
    exclusion class stays below ``exclusion_threshold``, and terrain wins by at
    least ``minimum_margin``.  Everything else remains unresolved rather than
    being silently assigned to terrain.
    """

    if not 0.0 <= terrain_threshold <= 1.0:
        raise ContractError("terrain_threshold must be in [0, 1]")
    if not 0.0 <= exclusion_threshold <= 1.0:
        raise ContractError("exclusion_threshold must be in [0, 1]")
    if not 0.0 <= minimum_margin <= 1.0:
        raise ContractError("minimum_margin must be in [0, 1]")

    arrays: dict[str, np.ndarray] = {}
    for name in SEMANTIC_CLASSES:
        if name not in probabilities:
            raise ContractError(f"missing semantic class: {name}")
        array = np.asarray(probabilities[name], dtype=np.float32)
        if array.ndim != 2:
            raise ContractError(f"{name} probability map must be 2D")
        if not np.isfinite(array).all():
            raise ContractError(f"{name} contains non-finite values")
        if np.any((array < 0.0) | (array > 1.0)):
            raise ContractError(f"{name} probabilities must be in [0, 1]")
        arrays[name] = array

    shape = arrays[SEMANTIC_CLASSES[0]].shape
    if any(array.shape != shape for array in arrays.values()):
        raise ContractError("semantic probability maps have mismatched shapes")

    valid = np.ones(shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if valid.shape != shape:
        raise ContractError("valid_mask shape must match semantic maps")

    stack = np.stack([arrays[name] for name in SEMANTIC_CLASSES], axis=0)
    class_index = np.argmax(stack, axis=0).astype(np.uint8)
    sorted_scores = np.sort(stack, axis=0)
    confidence = sorted_scores[-1].astype(np.float32)
    margin = (sorted_scores[-1] - sorted_scores[-2]).astype(np.float32)

    exclusion_names = ("water", "sky", "vegetation", "structure", "residual")
    exclusion_max = np.maximum.reduce([arrays[name] for name in exclusion_names])
    terrain_candidate = (
        valid
        & (arrays["terrain"] >= terrain_threshold)
        & (exclusion_max <= exclusion_threshold)
        & (margin >= minimum_margin)
        & (class_index == 0)
    )
    unresolved = valid & ~terrain_candidate

    result = SemanticMaskSet(
        probabilities=arrays,
        terrain_candidate=terrain_candidate,
        unresolved=unresolved,
        confidence=confidence,
        class_index=class_index,
        terrain_threshold=terrain_threshold,
        exclusion_threshold=exclusion_threshold,
    )
    result.validate()
    return result


def mask_report(mask_set: SemanticMaskSet) -> dict[str, object]:
    mask_set.validate()
    valid_count = int(mask_set.terrain_candidate.size)
    class_fractions = {
        name: float((mask_set.class_index == index).sum() / valid_count)
        for index, name in enumerate(SEMANTIC_CLASSES)
    }
    return {
        "classification": "SEMANTIC_SEPARATION_NOT_MODEL_QUALITY_PROOF",
        "terrain_candidate_fraction": float(mask_set.terrain_candidate.mean()),
        "unresolved_fraction": float(mask_set.unresolved.mean()),
        "mean_confidence": float(mask_set.confidence.mean()),
        "class_fractions": class_fractions,
        "terrain_threshold": mask_set.terrain_threshold,
        "exclusion_threshold": mask_set.exclusion_threshold,
        "promotion_allowed": False,
    }
