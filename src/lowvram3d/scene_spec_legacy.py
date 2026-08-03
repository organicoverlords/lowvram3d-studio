"""Build SceneSpec v1 from preserved image-to-scene evidence.

This adapter is intentionally deterministic and CPU-only. It does not rerun
MoGe, Blender, Unreal, or any neural model. It translates already-recorded
camera, source, landmark, and selected-mesh contracts into the reusable
SceneSpec format and emits preservation hashes for the migrated contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .scene_spec import validate_scene_spec


def _read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return value


def _contract_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _depth_bands(depth_ranges: dict[str, Any], far_m: float) -> list[dict[str, Any]]:
    values: list[tuple[str, float]] = []
    for name, depth in depth_ranges.items():
        _require(isinstance(name, str) and name, "depth range names must be non-empty strings")
        _require(isinstance(depth, (int, float)) and depth > 0, f"invalid depth value for {name}")
        values.append((name, float(depth)))
    values.sort(key=lambda item: item[1])
    _require(values, "at least one depth range is required")

    bands: list[dict[str, Any]] = []
    near = 0.1
    for index, (name, centre) in enumerate(values):
        if index + 1 < len(values):
            boundary = (centre + values[index + 1][1]) / 2.0
        else:
            boundary = max(centre + max(5.0, centre * 0.5), far_m)
        boundary = max(boundary, near + 0.1)
        bands.append({"id": name, "near_m": near, "far_m": boundary})
        near = boundary
    return bands


def _region_specs(
    interpretation_regions: dict[str, Any],
    depth_band_ids: set[str],
) -> list[dict[str, Any]]:
    representation = {
        "foreground": "gameplay_proxy",
        "midground": "editable_mesh",
        "background": "visual_shell",
        "sky": "sky",
        "walkable_surface": "gameplay_proxy",
    }
    preferred_band = {
        "foreground": "foreground",
        "midground": "castle",
        "background": "background",
        "walkable_surface": "foreground",
    }
    regions: list[dict[str, Any]] = []
    for name, description in interpretation_regions.items():
        if not isinstance(name, str) or not isinstance(description, str):
            continue
        item: dict[str, Any] = {
            "id": name,
            "label": description,
            "representation": representation.get(name, "visual_shell"),
            "confidence": 0.5,
            "tags": ["legacy_evidence"],
            "walkable": name == "walkable_surface",
            "interactive": name in {"midground", "walkable_surface"},
        }
        band = preferred_band.get(name)
        if band in depth_band_ids and item["representation"] != "sky":
            item["depth_band"] = band
        regions.append(item)
    _require(regions, "legacy interpretation contains no usable regions")
    return regions


def build_scene_spec_from_legacy(
    interpretation: dict[str, Any],
    geometry: dict[str, Any],
    selected_asset: dict[str, Any],
    source_mesh: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_a = interpretation.get("source")
    source_b = selected_asset.get("source")
    camera = interpretation.get("camera_estimate")
    regions = interpretation.get("regions")
    depth_ranges = geometry.get("depth_ranges")
    landmarks = geometry.get("landmarks")

    _require(isinstance(source_a, dict), "scene interpretation source is missing")
    _require(isinstance(source_b, dict), "selected asset source is missing")
    _require(isinstance(camera, dict), "camera estimate is missing")
    _require(isinstance(regions, dict), "region interpretation is missing")
    _require(isinstance(depth_ranges, dict), "geometry depth ranges are missing")
    _require(isinstance(landmarks, list), "geometry landmarks are missing")

    _require(source_a.get("sha256") == source_b.get("sha256"), "source SHA-256 mismatch")
    _require(source_a.get("dimensions") == source_b.get("dimensions"), "source dimensions mismatch")
    _require(source_mesh.get("schema_version") == "scene_source_mesh_contract_v1", "mesh contract schema mismatch")

    projection = camera.get("projection")
    _require(projection in {"perspective", "orthographic"}, "unsupported camera projection")
    source_camera: dict[str, Any] = {
        "id": "source",
        "projection": projection,
        "position_m": camera.get("camera_position"),
        "look_at_m": camera.get("look_at"),
        "principal_point_norm": camera.get("principal_point"),
        "near_m": camera.get("near_plane"),
        "far_m": camera.get("far_plane"),
        "purpose": "source_match",
    }
    if projection == "perspective":
        source_camera["field_of_view_deg"] = camera.get("field_of_view_deg")
    else:
        source_camera["ortho_scale_m"] = camera.get("ortho_scale_m")

    far_m = float(camera.get("far_plane", 500.0))
    depth_bands = _depth_bands(depth_ranges, far_m)
    depth_band_ids = {band["id"] for band in depth_bands}

    migrated_landmarks: list[dict[str, Any]] = []
    for item in landmarks:
        _require(isinstance(item, dict), "landmark must be an object")
        name = item.get("name")
        world = item.get("world")
        _require(isinstance(name, str) and name, "landmark name is missing")
        _require(isinstance(world, list) and len(world) == 3, f"invalid landmark world for {name}")
        migrated_landmarks.append({"id": name, "role": "layout_anchor", "world_m": world})

    transform = source_mesh.get("transform")
    _require(isinstance(transform, dict), "mesh transform is missing")
    mesh_path = source_mesh.get("path")
    mesh_sha = source_mesh.get("sha256")
    triangles = source_mesh.get("triangles")
    _require(isinstance(mesh_path, str) and mesh_path, "mesh path is missing")
    _require(isinstance(mesh_sha, str) and len(mesh_sha) == 64, "mesh SHA-256 is invalid")
    _require(isinstance(triangles, int) and triangles > 0, "mesh triangle count is invalid")

    scene_id = Path(str(source_a.get("path", "scene"))).stem.lower().replace(" ", "_")
    scene_id = "".join(char for char in scene_id if char.isalnum() or char in "_-")
    if len(scene_id) < 3:
        scene_id = "legacy_scene"
    scene_id = f"{scene_id}_source_locked_v1"[:64]

    spec: dict[str, Any] = {
        "schema_version": "scene_spec_v1",
        "scene_id": scene_id,
        "intent": {
            "mode": "source_locked_2p5d",
            "target_engine": "unreal_5_8",
            "deterministic_seed": 20260803,
            "quality_tier": "smoke",
            "description": "Deterministic migration of preserved image-to-scene evidence.",
        },
        "source": {
            "kind": "image",
            "path": source_a.get("path"),
            "sha256": source_a.get("sha256"),
            "dimensions_px": source_a.get("dimensions"),
            "notes": "Migrated without rerunning geometry or depth inference.",
        },
        "coordinate_system": {
            "units": "meters",
            "up_axis": "Z",
            "handedness": "right",
            "unreal_scale_cm_per_unit": 100,
        },
        "camera": {
            "source_camera": source_camera,
            "required_views": [dict(source_camera)],
            "envelope": {"max_translation_m": 0.0, "max_yaw_deg": 0.0, "max_pitch_deg": 0.0},
        },
        "depth_bands": depth_bands,
        "regions": _region_specs(regions, depth_band_ids),
        "landmarks": migrated_landmarks,
        "splines": [],
        "assets": [
            {
                "id": source_mesh.get("asset_id", "source_mesh_v2"),
                "semantic_class": source_mesh.get("semantic_class", "source_visible_shell"),
                "representation": "visual_shell",
                "source_uri": mesh_path,
                "transform": transform,
                "collision": source_mesh.get("collision", "none"),
                "navigation": source_mesh.get("navigation", "ignored"),
                "lod_policy": source_mesh.get("lod_policy", "nanite"),
                "material_policy": source_mesh.get("material_policy", "unlit_projection"),
                "triangle_budget": triangles,
                "region_id": "background" if "background" in {r["id"] for r in _region_specs(regions, depth_band_ids)} else None,
                "tags": ["legacy_evidence", "source_camera_locked", f"sha256:{mesh_sha}"],
            }
        ],
        "populations": [],
        "pcg": {
            "policy": "reference_driven_incremental",
            "reference_graphs": ["Project/PCG/Reference/LegacySourceShell"],
            "layers": [],
        },
        "budgets": {
            "gpu_vram_mb": 6144,
            "max_total_triangles": max(triangles, 1),
            "max_texture_memory_mb": 1536,
            "max_gpu_population_instances": 0,
            "max_draw_calls_target": 500,
            "allow_concurrent_neural_and_unreal_gpu": False,
        },
        "uncertainties": [
            {
                "id": "unseen_surfaces",
                "description": "Unseen geometry is not reconstructed by the source-locked migration.",
                "severity": "high",
                "fallback": "Use independent gameplay proxies outside the source camera envelope.",
            }
        ],
        "proof": {
            "required_gates": [
                "SCENE_SPEC_VALID",
                "SOURCE_CAMERA_COVERAGE_PROVEN",
                "UNREAL_INTERCHANGE_IMPORT_PROVEN",
                "UNREAL_SAVE_RELOAD_PROVEN",
                "UNREAL_SOURCE_RENDER_PROVEN",
                "UNREAL_PARALLAX_PROVEN",
            ],
            "current_classification": "PARTIAL",
            "evidence_root": "proof/scene/20260803-image-to-scene-smoke/",
        },
        "outputs": {"source_shell_glb": mesh_path},
    }
    if spec["assets"][0]["region_id"] is None:
        spec["assets"][0].pop("region_id")

    camera_before = {
        "projection": camera.get("projection"),
        "field_of_view_deg": camera.get("field_of_view_deg"),
        "principal_point": camera.get("principal_point"),
        "camera_position": camera.get("camera_position"),
        "look_at": camera.get("look_at"),
        "near_plane": camera.get("near_plane"),
        "far_plane": camera.get("far_plane"),
    }
    camera_after = {
        "projection": source_camera.get("projection"),
        "field_of_view_deg": source_camera.get("field_of_view_deg"),
        "principal_point": source_camera.get("principal_point_norm"),
        "camera_position": source_camera.get("position_m"),
        "look_at": source_camera.get("look_at_m"),
        "near_plane": source_camera.get("near_m"),
        "far_plane": source_camera.get("far_m"),
    }
    landmark_before = [{"name": item["name"], "world": item["world"]} for item in landmarks]
    landmark_after = [{"name": item["id"], "world": item["world_m"]} for item in migrated_landmarks]

    validation = validate_scene_spec(spec)
    receipt = {
        "schema_version": "scene_spec_legacy_migration_receipt_v1",
        "classification": "PROVEN" if validation["scene_spec_valid"] else "REJECTED",
        "scene_spec_valid": validation["scene_spec_valid"],
        "validation_errors": validation["errors"],
        "source_sha256_preserved": spec["source"]["sha256"] == source_a.get("sha256"),
        "source_dimensions_preserved": spec["source"]["dimensions_px"] == source_a.get("dimensions"),
        "camera_contract_sha256_before": _contract_hash(camera_before),
        "camera_contract_sha256_after": _contract_hash(camera_after),
        "camera_contract_preserved": _contract_hash(camera_before) == _contract_hash(camera_after),
        "landmark_contract_sha256_before": _contract_hash(landmark_before),
        "landmark_contract_sha256_after": _contract_hash(landmark_after),
        "landmark_contract_preserved": _contract_hash(landmark_before) == _contract_hash(landmark_after),
        "mesh_identity_preserved": spec["assets"][0]["source_uri"] == mesh_path
        and f"sha256:{mesh_sha}" in spec["assets"][0]["tags"],
        "transform_contract_sha256_before": _contract_hash(transform),
        "transform_contract_sha256_after": _contract_hash(spec["assets"][0]["transform"]),
        "transform_contract_preserved": _contract_hash(transform)
        == _contract_hash(spec["assets"][0]["transform"]),
        "neural_work_started": False,
        "blender_work_started": False,
        "unreal_work_started": False,
    }
    if not all(
        [
            receipt["scene_spec_valid"],
            receipt["source_sha256_preserved"],
            receipt["source_dimensions_preserved"],
            receipt["camera_contract_preserved"],
            receipt["landmark_contract_preserved"],
            receipt["mesh_identity_preserved"],
            receipt["transform_contract_preserved"],
        ]
    ):
        receipt["classification"] = "REJECTED"
    return spec, receipt


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate preserved scene evidence into SceneSpec v1")
    parser.add_argument("--interpretation", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--selected-asset", required=True)
    parser.add_argument("--source-mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()

    try:
        spec, receipt = build_scene_spec_from_legacy(
            _read_json(args.interpretation),
            _read_json(args.geometry),
            _read_json(args.selected_asset),
            _read_json(args.source_mesh),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        receipt = {
            "schema_version": "scene_spec_legacy_migration_receipt_v1",
            "classification": "REJECTED",
            "scene_spec_valid": False,
            "error": str(exc),
            "neural_work_started": False,
            "blender_work_started": False,
            "unreal_work_started": False,
        }
        Path(args.receipt).parent.mkdir(parents=True, exist_ok=True)
        Path(args.receipt).write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
        print("LEGACY_SCENE_SPEC_MIGRATION=REJECTED")
        return 2

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2, sort_keys=True), encoding="utf-8")
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(f"LEGACY_SCENE_SPEC_MIGRATION={receipt['classification']}")
    return 0 if receipt["classification"] == "PROVEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
