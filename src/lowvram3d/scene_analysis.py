"""CPU-safe image-derived SceneSpec bootstrap for arbitrary single images."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quality_metrics(path: Path, dimensions: tuple[int, int]) -> dict[str, Any]:
    try:
        from PIL import Image, ImageStat
        image = Image.open(path).convert("RGB")
        stat = ImageStat.Stat(image)
        return {
            "decode": "PROVEN",
            "dimensions_px": [int(image.width), int(image.height)],
            "mean_rgb": [round(float(value), 4) for value in stat.mean],
            "stddev_rgb": [round(float(value), 4) for value in stat.stddev],
            "alpha_present": "A" in Image.open(path).getbands(),
            "exposure_uncertainty": "NOT_ESTIMATED",
        }
    except Exception as exc:
        return {"decode": "NOT_PROVEN", "dimensions_px": [int(dimensions[0]), int(dimensions[1])], "error": type(exc).__name__}


def _apply_depth(bundle: dict[str, Any], depth: dict[str, Any]) -> dict[str, Any]:
    """Replace the bootstrap's placeholder depth and camera with measured values.

    Without this the bundle keeps `confidence: 0.0` and an unresolved camera,
    and downstream asset strategy selects `source_projection` -- the source
    image on a shell, which matches the source view by construction and carries
    no geometry.
    """
    near, far = depth.get("depth_range", [0.1, 100.0])
    camera = depth.get("camera", {})

    bundle["analysis_mode"] = "moge_depth"
    bundle["depth"] = {
        "representation": "metric_point_map",
        "confidence": 0.9,
        "estimator": depth.get("model"),
        "mesh_glb": depth.get("output_glb"),
        "mesh_sha256": depth.get("output_sha256"),
        "triangles": depth.get("triangles"),
        "vertices": depth.get("vertices"),
        "quads_dropped_at_depth_edges": depth.get("quads_dropped_at_depth_edges"),
        "masked_pixel_fraction": depth.get("masked_pixel_fraction"),
    }
    bundle["camera"] = {
        "contract_status": "RESOLVED",
        "projection": camera.get("projection", "perspective"),
        "source_view_id": "source",
        "horizontal_fov_deg": camera.get("fov_x_deg"),
        "vertical_fov_deg": camera.get("fov_y_deg"),
        "aspect_ratio": camera.get("aspect_ratio"),
        "convention": camera.get("convention"),
        "uncertainty": 0.1,
    }
    bundle["depth_bands"] = [{"id": "moge_measured", "near_m": float(near),
                              "far_m": float(far), "confidence": 0.9}]
    bundle["world_scale"] = {"status": "RESOLVED_UP_TO_SCALE", "units": "meters",
                             "confidence": 0.6}
    bundle["regions"] = [{
        "id": "reconstructed_geometry_001",
        "layer_type": "reconstructed_geometry",
        "representation": "depth_mesh",
        "confidence": 0.9,
        "bbox_norm_xyxy": [0.0, 0.0, 1.0, 1.0],
        "uncertainty": "single_view_reconstruction_is_unobserved_behind_surfaces",
    }]
    bundle["resource_budgets"]["gpu_work_requested"] = True
    bundle["uncertainties"] = [{
        "id": "single_view_occlusion",
        "description": "A single view cannot observe surfaces behind what it sees; "
                       "the mesh is open where depth edges were culled",
        "fallback": "leave_holes_rather_than_weld_across_depth_edges",
        "severity": "medium",
    }]
    return bundle


def analyze_image(path: str | Path, scene_id: str, dimensions: tuple[int, int],
                  mode: str = "cpu_bootstrap",
                  depth_receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    width, height = (int(dimensions[0]), int(dimensions[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    quality = _quality_metrics(source, (width, height))
    bundle = {
        "schema_version": "analysis_bundle_v1",
        "classification": "PROVEN",
        "scene_id": str(scene_id),
        "analysis_mode": mode,
        "source": {"path": str(source.resolve()), "sha256": _sha256(source), "dimensions_px": [width, height], "kind": "single_image"},
        "image_quality": quality,
        "camera": {"contract_status": "REQUIRES_ANALYSIS", "projection": "perspective", "source_view_id": "source", "hypothesis": "unresolved_single_view_camera", "uncertainty": 1.0},
        "depth": {"representation": "unknown_depth_bands", "confidence": 0.0, "uncertainty": "requires_depth_estimator"},
        "surface_orientation": {"representation": "unknown_normals_or_slopes", "confidence": 0.0, "uncertainty": "requires_surface_estimator"},
        "regions": [{"id": "visual_shell_001", "layer_type": "visual_shell", "representation": "visual_shell", "confidence": 0.5, "bbox_norm_xyxy": [0.0, 0.0, 1.0, 1.0], "uncertainty": "semantic_analysis_not_instantiated"}],
        "object_instances": [],
        "structural_features": [],
        "material_regions": [],
        "visibility": [{"region_id": "visual_shell_001", "classification": "fully_visible", "completion_policy": "visual_shell_only"}],
        "support_relationships": [],
        "world_scale": {"status": "UNRESOLVED", "units": "meters", "confidence": 0.0},
        "depth_bands": [{"id": "unknown_001", "near_m": 0.1, "far_m": 100.0, "confidence": 0.2}],
        "coordinate_system": {"units": "meters", "up_axis": "Z", "handedness": "right"},
        "uncertainties": [{"id": "semantic_bootstrap", "description": "CPU bootstrap has not inferred reliable semantic regions", "fallback": "visual_shell_and_unresolved", "severity": "high"}],
        "resource_budgets": {"max_vram_mb": 6144, "max_triangles": 1500000, "gpu_work_requested": False},
    }
    if depth_receipt and depth_receipt.get("available"):
        return _apply_depth(bundle, depth_receipt)
    if depth_receipt is not None:
        bundle["uncertainties"].append({
            "id": "depth_estimator_unavailable",
            "description": str(depth_receipt.get("reason", "depth stage did not run")),
            "fallback": "cpu_bootstrap_visual_shell",
            "severity": "high",
        })
    return bundle
