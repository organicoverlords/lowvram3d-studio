"""Pure-Python validation helpers for the image-to-scene smoke lane."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_input_receipt(path: Path, *, dimensions: tuple[int, int], mode: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    width, height = dimensions
    if width <= 0 or height <= 0:
        raise ValueError("IMAGE_DIMENSIONS_INVALID")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "dimensions": [int(width), int(height)],
        "mode": mode,
        "finite_dimensions": True,
    }


def require_dedicated_content_root(root: str, *, forbidden: Iterable[str] = ()) -> str:
    normalized = root.replace("\\", "/").rstrip("/")
    if not normalized.startswith("/Game/"):
        raise ValueError("UNREAL_CONTENT_ROOT_MUST_BE_GAME_PATH")
    if any(normalized == item.rstrip("/") or normalized.startswith(item.rstrip("/") + "/") for item in forbidden):
        raise ValueError("UNREAL_EXISTING_CONTENT_ROOT_FORBIDDEN")
    if normalized.count("/") < 3:
        raise ValueError("UNREAL_CONTENT_ROOT_NOT_DEDICATED")
    return normalized + "/"


def uniform_scale(source_dimensions: Iterable[float], target_max_dimension: float) -> float:
    dims = [float(value) for value in source_dimensions]
    if len(dims) != 3 or not all(math.isfinite(value) and value > 0 for value in dims):
        raise ValueError("SOURCE_DIMENSIONS_INVALID")
    target = float(target_max_dimension)
    if not math.isfinite(target) or target <= 0:
        raise ValueError("TARGET_DIMENSION_INVALID")
    return target / max(dims)


def upright_axis_conversion(source_up: str, target_up: str = "Z") -> dict[str, Any]:
    source = str(source_up).upper()
    target = str(target_up).upper()
    if source not in {"X", "Y", "Z"} or target not in {"X", "Y", "Z"}:
        raise ValueError("UP_AXIS_INVALID")
    return {
        "source_up_axis": source,
        "target_up_axis": target,
        "rotation_required": source != target,
        "handedness_preserved": True,
    }


def ground_placement(actor_min_z: float, ground_z: float = 0.0) -> float:
    value = float(actor_min_z)
    ground = float(ground_z)
    if not math.isfinite(value) or not math.isfinite(ground):
        raise ValueError("GROUND_PLACEMENT_NONFINITE")
    return ground - value


def validate_material_texture_binding(binding: dict[str, Any]) -> dict[str, Any]:
    required = ("material", "texture", "bound_to_base_color")
    if any(key not in binding for key in required):
        raise ValueError("MATERIAL_BINDING_RECEIPT_INCOMPLETE")
    if not binding["material"] or not binding["texture"] or binding["bound_to_base_color"] is not True:
        raise ValueError("MATERIAL_TEXTURE_BINDING_INVALID")
    return {"passed": True, **binding}


def validate_collision(collision: dict[str, Any]) -> dict[str, Any]:
    if collision.get("applicability") == "NOT_APPLICABLE_IMAGE_TO_SCENE":
        return {"passed": True, **collision}
    if int(collision.get("primitive_count", 0)) <= 0:
        raise ValueError("COLLISION_MISSING")
    if collision.get("method") == "per_poly" and collision.get("dense_asset"):
        raise ValueError("DENSE_ASSET_PER_POLY_COLLISION_FORBIDDEN")
    return {"passed": True, **collision}


def compare_reload_receipt(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    keys = ("map", "image_texture", "image_surface", "camera_primary", "camera_secondary")
    mismatches = [key for key in keys if before.get(key) != after.get(key)]
    return {"passed": not mismatches, "mismatches": mismatches}


def render_manifest(output_dir: Path, views: list[dict[str, Any]], *, resolution: tuple[int, int]) -> dict[str, Any]:
    if len(views) < 2:
        raise ValueError("RENDER_VIEW_COUNT_TOO_SMALL")
    width, height = resolution
    if width < 320 or height < 180:
        raise ValueError("RENDER_RESOLUTION_TOO_SMALL")
    manifest = {
        "schema": "scene_pipeline_render_manifest_v1",
        "resolution": [width, height],
        "views": views,
        "output_dir": str(output_dir),
        "all_named": all(bool(view.get("name")) and bool(view.get("path")) for view in views),
    }
    manifest["passed"] = bool(manifest["all_named"])
    return manifest


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
