"""Validation contracts for specialist vision-model outputs.

These validators deliberately accept plain JSON-compatible dictionaries so Florence,
EdgeTAM, DA3, pose estimators, and future implementations can run in isolated environments.
They validate shape, range, provenance, and finite metrics; they do not claim semantic quality.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from .contracts import ContractError


SPECIALIST_TASKS = frozenset({"labels", "masks", "depth", "pose", "embedding"})


def validate_specialist_output(data: Mapping[str, Any]) -> dict[str, Any]:
    if data.get("schema") != "vision_specialist_output_v1":
        raise ContractError("unsupported specialist output schema")
    task = str(data.get("task", ""))
    if task not in SPECIALIST_TASKS:
        raise ContractError(f"unsupported specialist task: {task!r}")
    for key in ("model_id", "source_artifact_id"):
        if not str(data.get(key, "")).strip():
            raise ContractError(f"{key} is required")
    observations = data.get("observations")
    if not isinstance(observations, list):
        raise ContractError("observations must be a list")
    validators = {
        "labels": _validate_labels,
        "masks": _validate_masks,
        "depth": _validate_depth,
        "pose": _validate_pose,
        "embedding": _validate_embedding,
    }
    validators[task](observations, data.get("metrics", {}))
    return dict(data)


def _validate_labels(observations: list[Any], metrics: Any) -> None:
    for index, item in enumerate(observations):
        _require_mapping(item, index)
        if not str(item.get("label", "")).strip():
            raise ContractError(f"label observation {index} has no label")
        _confidence(item, index)
        if "bbox_norm_xyxy" in item:
            _bbox(item["bbox_norm_xyxy"], index)


def _validate_masks(observations: list[Any], metrics: Any) -> None:
    for index, item in enumerate(observations):
        _require_mapping(item, index)
        _confidence(item, index)
        area = _finite_number(item.get("area_fraction"), f"mask observation {index} area_fraction")
        if not 0.0 <= area <= 1.0:
            raise ContractError(f"mask observation {index} area_fraction is outside [0, 1]")
        _bbox(item.get("bbox_norm_xyxy"), index)
        if not str(item.get("mask_artifact_id", "")).strip():
            raise ContractError(f"mask observation {index} has no mask_artifact_id")


def _validate_depth(observations: list[Any], metrics: Any) -> None:
    if observations:
        raise ContractError("depth output stores aggregate values in metrics, not observations")
    if not isinstance(metrics, Mapping):
        raise ContractError("depth metrics must be an object")
    valid = _finite_number(metrics.get("valid_fraction"), "depth valid_fraction")
    if not 0.0 <= valid <= 1.0:
        raise ContractError("depth valid_fraction is outside [0, 1]")
    for optional in ("aligned_l1", "normal_angular_error_deg", "silhouette_iou"):
        if optional in metrics:
            value = _finite_number(metrics[optional], f"depth {optional}")
            if value < 0:
                raise ContractError(f"depth {optional} must be non-negative")


def _validate_pose(observations: list[Any], metrics: Any) -> None:
    for index, item in enumerate(observations):
        _require_mapping(item, index)
        if not str(item.get("joint", "")).strip():
            raise ContractError(f"pose observation {index} has no joint")
        _confidence(item, index)
        point = item.get("point_norm_xy")
        if not isinstance(point, list) or len(point) != 2:
            raise ContractError(f"pose observation {index} point_norm_xy must contain two values")
        for value in point:
            number = _finite_number(value, f"pose observation {index} point")
            if not 0.0 <= number <= 1.0:
                raise ContractError(f"pose observation {index} point is outside [0, 1]")


def _validate_embedding(observations: list[Any], metrics: Any) -> None:
    if len(observations) != 1 or not isinstance(observations[0], Mapping):
        raise ContractError("embedding output requires exactly one observation")
    dimension = int(observations[0].get("dimension", 0))
    if dimension <= 0:
        raise ContractError("embedding dimension must be positive")
    if not str(observations[0].get("embedding_artifact_id", "")).strip():
        raise ContractError("embedding_artifact_id is required")


def _require_mapping(item: Any, index: int) -> None:
    if not isinstance(item, Mapping):
        raise ContractError(f"observation {index} must be an object")


def _confidence(item: Mapping[str, Any], index: int) -> float:
    value = _finite_number(item.get("confidence"), f"observation {index} confidence")
    if not 0.0 <= value <= 1.0:
        raise ContractError(f"observation {index} confidence is outside [0, 1]")
    return value


def _bbox(value: Any, index: int) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ContractError(f"observation {index} bbox_norm_xyxy must contain four values")
    x0, y0, x1, y1 = (_finite_number(item, f"observation {index} bbox") for item in value)
    if not all(0.0 <= item <= 1.0 for item in (x0, y0, x1, y1)):
        raise ContractError(f"observation {index} bbox is outside [0, 1]")
    if x1 <= x0 or y1 <= y0:
        raise ContractError(f"observation {index} bbox has non-positive area")


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} must be numeric") from exc
    if not math.isfinite(number):
        raise ContractError(f"{name} must be finite")
    return number
