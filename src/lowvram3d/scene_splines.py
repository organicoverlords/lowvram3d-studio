"""Deterministic, unpromoted scene spline specifications.

The spline layer is semantic authoring data for later river exclusion and
bridge-axis work. It does not generate water, bridges, PCG, or collision.
Coordinates are authored in metres and converted by the Unreal adapter.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "scene_spline_spec_v1"
DEFAULT_OUTPUT = Path("evidence/latest-scene-splines/scene_spline_spec.json")
REQUIRED_TAGS = {"scene_spec_generated", "unpromoted"}


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _point(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return [_finite(item, f"{name}[{index}]") for index, item in enumerate(value)]


def build_scene_spline_spec() -> dict[str, Any]:
    """Return the fixed bounded Castlegrounds spline specification."""

    return {
        "schema_version": SCHEMA_VERSION,
        "scene_id": "castlegrounds_hybrid_authoritative_v1",
        "coordinate_system": {"units": "meters", "up_axis": "Z", "unreal_scale_cm_per_unit": 100.0},
        "classification": "PROVEN",
        "promotion": False,
        "splines": [
            {
                "id": "river_main",
                "kind": "river_centerline",
                "points_m": [[-30.0, 45.0, 0.0], [0.0, 50.0, 0.0], [30.0, 58.0, 0.0]],
                "width_m": 12.0,
                "exclusion_radius_m": 3.0,
                "tags": ["water", "no_build", "scene_spec_generated", "unpromoted"],
            },
            {
                "id": "bridge_axis_main",
                "kind": "bridge_axis",
                "points_m": [[-10.0, 31.0, 4.0], [10.0, 31.0, 4.0]],
                "width_m": 3.0,
                "tags": ["crossing", "scene_spec_generated", "unpromoted"],
            },
        ],
    }


def validate_scene_spline_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if spec.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if spec.get("classification") != "PROVEN":
        errors.append("spline spec must be PROVEN")
    if spec.get("promotion") is not False:
        errors.append("spline spec must remain unpromoted")
    coordinates = spec.get("coordinate_system")
    if not isinstance(coordinates, Mapping) or coordinates.get("units") != "meters" or coordinates.get("up_axis") != "Z":
        errors.append("coordinate system must be metres with Z up")
    if not isinstance(spec.get("splines"), list):
        errors.append("splines must be a list")
        return {"scene_spline_spec_valid": False, "errors": errors}
    ids: set[str] = set()
    for index, spline in enumerate(spec["splines"]):
        prefix = f"splines[{index}]"
        if not isinstance(spline, Mapping):
            errors.append(f"{prefix} must be an object")
            continue
        spline_id = spline.get("id")
        if not isinstance(spline_id, str) or not spline_id:
            errors.append(f"{prefix}.id is required")
        elif spline_id in ids:
            errors.append(f"duplicate spline id: {spline_id}")
        else:
            ids.add(spline_id)
        points = spline.get("points_m")
        if not isinstance(points, list) or len(points) < 2:
            errors.append(f"{prefix}.points_m requires at least two points")
        else:
            for point_index, point in enumerate(points):
                try:
                    _point(point, f"{prefix}.points_m[{point_index}]")
                except ValueError as exc:
                    errors.append(str(exc))
        try:
            width = _finite(spline.get("width_m"), f"{prefix}.width_m")
            if width <= 0:
                errors.append(f"{prefix}.width_m must be positive")
        except (TypeError, ValueError) as exc:
            errors.append(str(exc))
        if "exclusion_radius_m" in spline:
            try:
                radius = _finite(spline.get("exclusion_radius_m"), f"{prefix}.exclusion_radius_m")
                if radius < 0:
                    errors.append(f"{prefix}.exclusion_radius_m cannot be negative")
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        tags = set(spline.get("tags", [])) if isinstance(spline.get("tags"), list) else set()
        if not REQUIRED_TAGS.issubset(tags):
            errors.append(f"{prefix}.tags must include {sorted(REQUIRED_TAGS)}")
        if spline_id == "river_main" and "water" not in tags:
            errors.append("river_main must be tagged water")
        if spline_id == "bridge_axis_main" and "crossing" not in tags:
            errors.append("bridge_axis_main must be tagged crossing")
    required = {"river_main", "bridge_axis_main"}
    if ids != required:
        errors.append(f"required spline ids are {sorted(required)}")
    return {"scene_spline_spec_valid": not errors, "errors": errors}


def write_scene_spline_spec(output: str | Path = DEFAULT_OUTPUT) -> Path:
    spec = build_scene_spline_spec()
    validation = validate_scene_spline_spec(spec)
    if not validation["scene_spline_spec_valid"]:
        raise ValueError(validation["errors"])
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the bounded Castlegrounds scene spline spec")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    path = write_scene_spline_spec(args.output)
    print("SCENE_SPLINE_SPEC=PROVEN")
    print(f"SPEC_PATH={path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
