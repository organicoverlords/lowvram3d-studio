"""Plan one deterministic Castlegrounds gameplay smoke proxy.

This module is deliberately a planner only.  It does not load Blender, Unreal,
MoGe, or a source mesh.  The authoritative Hybrid SceneSpec remains the only
source of placement, camera, units, and semantic IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .scene_spec import validate_scene_spec


DEFAULT_SPEC = Path("evidence/latest-scene-hybrid/authoritative_hybrid_scene_spec.json")
DEFAULT_OUTPUT_DIR = Path("evidence/latest-scene-gameplay-proxy")
ASSET_ID = "castle_proxy"
REGION_ID = "castle_core"
LANDMARK_ID = "castle_base"
DEPTH_BAND_ID = "castle"
SEED = 20260803
MAX_DIMENSION_M = 50.0
MIN_DIMENSION_M = 0.5


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: Any, name: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be a 3-vector")
    return [_finite(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read authoritative SceneSpec: {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("authoritative SceneSpec must be an object")
    return value


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _find_by_id(items: Any, item_id: str, collection: str) -> dict[str, Any]:
    if not isinstance(items, list):
        raise ValueError(f"authoritative SceneSpec has no valid {collection} list")
    matches = [item for item in items if isinstance(item, dict) and item.get("id") == item_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {collection} with id {item_id!r}, found {len(matches)}")
    return matches[0]


def _dimensions_m(region: Mapping[str, Any], depth_band: Mapping[str, Any], camera: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, float]:
    """Derive a conservative box from the normalized image region.

    Formula (all values are metres):

    ``depth_mid = (band.near + band.far) / 2``
    ``frame_width = 2 * depth_mid * tan(horizontal_fov / 2)``
    ``frame_height = frame_width / source_aspect``
    ``width = bbox_width * frame_width``
    ``height = bbox_height * frame_height``
    ``depth = clamp(depth_mid * 0.25, 2, 12)``

    This uses only the authoritative region, castle depth band, camera FOV, and
    source aspect ratio.  The depth fraction is intentionally conservative for
    a smoke collision proxy, not a reconstruction claim.
    """

    bbox = region.get("bbox_norm_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("castle_core requires bbox_norm_xyxy")
    x0, y0, x1, y1 = [_finite(item, "castle_core.bbox_norm_xyxy") for item in bbox]
    bbox_width = x1 - x0
    bbox_height = y1 - y0
    if not 0 < bbox_width <= 1 or not 0 < bbox_height <= 1 or not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
        raise ValueError("castle_core bbox must be finite, ordered, and normalized")

    near_m = _finite(depth_band.get("near_m"), "castle depth near_m")
    far_m = _finite(depth_band.get("far_m"), "castle depth far_m")
    fov_deg = _finite(camera.get("field_of_view_deg"), "authoritative horizontal FOV")
    if not 0 < near_m < far_m or not 0 < fov_deg < 180:
        raise ValueError("castle depth band or horizontal FOV is invalid")
    dimensions_px = source.get("dimensions_px")
    if not isinstance(dimensions_px, list) or len(dimensions_px) != 2:
        raise ValueError("source dimensions_px must be [width, height]")
    source_width = _finite(dimensions_px[0], "source width")
    source_height = _finite(dimensions_px[1], "source height")
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")

    depth_mid = (near_m + far_m) / 2.0
    source_aspect = source_width / source_height
    frame_width = 2.0 * depth_mid * math.tan(math.radians(fov_deg) / 2.0)
    frame_height = frame_width / source_aspect
    dimensions = {
        "width_m": bbox_width * frame_width,
        "height_m": bbox_height * frame_height,
        "depth_m": min(12.0, max(2.0, depth_mid * 0.25)),
    }
    for name, value in dimensions.items():
        if not MIN_DIMENSION_M <= value <= MAX_DIMENSION_M or not math.isfinite(value):
            raise ValueError(f"planned {name} is outside conservative bounds: {value}")
    return dimensions


def build_gameplay_proxy_plan(spec: Mapping[str, Any], *, spec_path: str | Path = DEFAULT_SPEC) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(plan, receipt)`` or fail closed with ``ValueError``."""

    validation = validate_scene_spec(spec)
    if not validation["scene_spec_valid"]:
        raise ValueError(f"authoritative SceneSpec is invalid: {validation['errors']}")

    asset = _find_by_id(spec.get("assets"), ASSET_ID, "asset")
    region = _find_by_id(spec.get("regions"), REGION_ID, "region")
    landmark = _find_by_id(spec.get("landmarks"), LANDMARK_ID, "landmark")
    depth_band = _find_by_id(spec.get("depth_bands"), DEPTH_BAND_ID, "depth band")
    camera_root = spec.get("camera")
    if not isinstance(camera_root, dict):
        raise ValueError("authoritative SceneSpec has no camera object")
    camera = camera_root.get("source_camera")
    if not isinstance(camera, dict):
        raise ValueError("authoritative SceneSpec has no source camera")
    coordinate_system = spec.get("coordinate_system")
    if not isinstance(coordinate_system, dict) or coordinate_system.get("units") != "meters" or coordinate_system.get("up_axis") != "Z":
        raise ValueError("unsupported SceneSpec coordinate system")
    cm_per_unit = _finite(coordinate_system.get("unreal_scale_cm_per_unit"), "Unreal scale")
    if cm_per_unit != 100.0:
        raise ValueError("expected Unreal scale of exactly 100 cm per metre")

    if asset.get("representation") != "gameplay_proxy" or asset.get("collision") == "none":
        raise ValueError("castle_proxy must be a collidable gameplay proxy")
    if asset.get("source_uri") or asset.get("id") == "source_mesh_v2":
        raise ValueError("source mesh cannot be requested as gameplay collision")
    if asset.get("promotion") is True or "promoted" in asset.get("tags", []):
        raise ValueError("gameplay proxy must not be promoted")
    if region.get("depth_band") != DEPTH_BAND_ID or not region.get("walkable"):
        raise ValueError("castle_core is not a walkable castle region")
    base_anchor_m = _vector(landmark.get("world_m"), "castle_base.world_m")
    dimensions_m = _dimensions_m(region, depth_band, camera, spec.get("source", {}))
    actor_center_m = [base_anchor_m[0], base_anchor_m[1], base_anchor_m[2] + dimensions_m["height_m"] / 2.0]
    dimensions_cm = {
        key.replace("_m", "_cm"): value * cm_per_unit
        for key, value in dimensions_m.items()
    }
    actor_center_cm = [value * cm_per_unit for value in actor_center_m]
    source_aspect = _finite(spec["source"]["dimensions_px"][0], "source width") / _finite(spec["source"]["dimensions_px"][1], "source height")
    source_hash = _canonical_hash(spec)
    tags = ["gameplay_proxy", "replaceable", "unpromoted", "scene_spec_generated", "castle_proxy"]
    plan = {
        "schema_version": "gameplay_proxy_plan_v1",
        "classification": "PROVEN",
        "asset_id": ASSET_ID,
        "region_id": REGION_ID,
        "landmark_id": LANDMARK_ID,
        "depth_band_id": DEPTH_BAND_ID,
        "primitive_type": "engine_cube",
        "base_anchor_m": base_anchor_m,
        "actor_center_m": actor_center_m,
        "actor_center_cm": actor_center_cm,
        "dimensions_m": dimensions_m,
        "dimensions_cm": dimensions_cm,
        "rotation_deg_xyz": [0.0, 0.0, 0.0],
        "collision": "blocking/simple",
        "navigation": "walkable intent",
        "promotion": False,
        "tags": tags,
        "authoritative_camera": {
            "id": camera.get("id"),
            "horizontal_fov_deg": _finite(camera.get("field_of_view_deg"), "authoritative horizontal FOV"),
            "position_m": _vector(camera.get("position_m"), "camera position"),
            "look_at_m": _vector(camera.get("look_at_m"), "camera look_at"),
            "source_aspect": source_aspect,
        },
        "unreal_unit_conversion": {"source_units": "meters", "unreal_units": "centimetres", "cm_per_m": cm_per_unit},
        "source_scene_spec_path": str(Path(spec_path).resolve()),
        "source_scene_spec_sha256": source_hash,
        "deterministic_seed": SEED,
        "source_mesh_collision": False,
        "formula": "depth_mid=(near_m+far_m)/2; frame_width=2*depth_mid*tan(horizontal_fov/2); frame_height=frame_width/source_aspect; width=bbox_width*frame_width; height=bbox_height*frame_height; depth=clamp(depth_mid*0.25,2,12)",
    }
    receipt = {
        "schema_version": "gameplay_proxy_plan_receipt_v1",
        "classification": "PROVEN",
        "plan_path": "evidence/latest-scene-gameplay-proxy/gameplay_proxy_plan.json",
        "scene_spec_validation": validation,
        "derived_from_scene_spec": True,
        "selected_asset": ASSET_ID,
        "selected_region": REGION_ID,
        "selected_landmark": LANDMARK_ID,
        "selected_depth_band": DEPTH_BAND_ID,
        "authoritative_camera_fov_deg": plan["authoritative_camera"]["horizontal_fov_deg"],
        "source_aspect": source_aspect,
        "depth_mid_m": (float(depth_band["near_m"]) + float(depth_band["far_m"])) / 2.0,
        "formula": plan["formula"],
        "source_scene_spec_sha256": source_hash,
        "source_mesh_requested_as_collision": False,
        "promotion": False,
        "navigation_proof": "intent_only_not_full_navigation_proof",
    }
    return plan, receipt


def write_gameplay_proxy_plan(spec_path: str | Path = DEFAULT_SPEC, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> tuple[Path, Path]:
    plan, receipt = build_gameplay_proxy_plan(_read_json(spec_path), spec_path=spec_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "gameplay_proxy_plan.json"
    receipt_path = output / "gameplay_proxy_plan_receipt.json"
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan_path, receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan the bounded Castlegrounds gameplay proxy")
    parser.add_argument("--scene-spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    plan_path, receipt_path = write_gameplay_proxy_plan(args.scene_spec, args.output_dir)
    print(f"GAMEPLAY_PROXY_PLAN=PROVEN")
    print(f"PLAN_PATH={plan_path.resolve()}")
    print(f"RECEIPT_PATH={receipt_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
