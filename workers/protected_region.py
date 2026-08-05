"""Protected-region masks in source-view image space, loaded from asset configuration.

The panda's face is the one part of this asset where a grazing side observation must never
outvote the frontal one. That is an asset fact, not a property of the fusion algorithm, so
the geometry of the region lives in a config file and this module only knows how to turn
whatever it finds there into a mask. No asset coordinate appears in fusion code.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def load(config_path: Path) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema") != "lowvram3d_protected_region_v1":
        raise RuntimeError("PROTECTED_REGION_SCHEMA_INVALID")
    if not config.get("regions"):
        raise RuntimeError("PROTECTED_REGION_EMPTY")
    return config


def _ellipse_polygon(centre, radii, steps: int = 96) -> list[tuple[float, float]]:
    angles = np.linspace(0.0, 2.0 * np.pi, steps, endpoint=False)
    return [(float(centre[0] + radii[0] * np.cos(a)),
             float(centre[1] + radii[1] * np.sin(a))) for a in angles]


def region_polygon(region: dict) -> list[tuple[float, float]]:
    shape = str(region.get("shape", "polygon"))
    if shape == "ellipse":
        return _ellipse_polygon(region["centre_normalised"], region["radii_normalised"])
    if shape == "polygon":
        return [(float(x), float(y)) for x, y in region["points_normalised"]]
    raise RuntimeError(f"PROTECTED_REGION_SHAPE_UNSUPPORTED:{shape}")


def _feather(mask: np.ndarray, radius: int) -> np.ndarray:
    """Box-blur the hard mask a few times so the region's edge is a ramp, not a step."""
    if radius <= 0:
        return mask.astype(np.float32)
    blurred = mask.astype(np.float32)
    for _ in range(3):
        padded = np.pad(blurred, radius, mode="edge")
        cumulative = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        cumulative = np.pad(cumulative, ((1, 0), (1, 0)), mode="constant")
        size = 2 * radius + 1
        window = (cumulative[size:, size:] - cumulative[:-size, size:]
                  - cumulative[size:, :-size] + cumulative[:-size, :-size])
        blurred = window / float(size * size)
    return np.clip(blurred, 0.0, 1.0)


def build_masks(config: dict, size: int) -> dict[str, dict]:
    """One float mask per region, in the source view's own pixel space."""
    masks = {}
    for region in config["regions"]:
        image = Image.new("L", (size, size), 0)
        polygon = [(x * (size - 1), y * (size - 1)) for x, y in region_polygon(region)]
        ImageDraw.Draw(image).polygon(polygon, fill=255)
        hard = np.asarray(image) > 127
        radius = int(round(float(region.get("feather_fraction", 0.0)) * size))
        masks[str(region["name"])] = {
            "weight": _feather(hard, radius),
            "hard": hard,
            "owner_semantic": str(region.get("owner_semantic", "")),
            "forbidden_owner_semantics": [str(s) for s in
                                          region.get("forbidden_owner_semantics", [])],
            "priority": float(region.get("priority", 1.0)),
        }
    return masks


def overlay(image: np.ndarray, masks: dict[str, dict]) -> Image.Image:
    pixels = np.asarray(image).astype(np.float32).copy()
    for record in masks.values():
        weight = record["weight"][..., None]
        pixels = pixels * (1.0 - 0.45 * weight) + np.array([255.0, 60.0, 60.0]) * 0.45 * weight
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))
