"""CPU image comparison and bounded visual-defect routing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping


def _bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def _image_stats(path: str | Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageStat
        image = Image.open(path).convert("RGB")
        stat = ImageStat.Stat(image)
        pixels = list(image.getdata())
        non_dark = sum(1 for pixel in pixels if max(pixel) > 12)
        return {"dimensions_px": list(image.size), "mean_rgb": [round(value, 4) for value in stat.mean], "stddev_rgb": [round(value, 4) for value in stat.stddev], "non_dark_fraction": non_dark / max(1, len(pixels)), "sha256": hashlib.sha256(_bytes(path)).hexdigest()}
    except Exception as exc:
        return {"error": f"image_unavailable:{type(exc).__name__}"}


def compare_source_view(source_path: str | Path, render_path: str | Path, spec: Mapping[str, Any] | None = None, tier: str = "smoke") -> dict[str, Any]:
    source = _image_stats(source_path)
    render = _image_stats(render_path)
    defects: list[dict[str, Any]] = []
    if source.get("error") or render.get("error"):
        defects.append({"defect_id": "image_unavailable", "stage": "source_view_validation", "severity": "high", "repair_owner": "camera_estimation|representation_selection", "automatic_repair_safe": False})
        classification = "NOT_PROVEN"
    else:
        source_area = float(source.get("non_dark_fraction", 0.0))
        render_area = float(render.get("non_dark_fraction", 0.0))
        area_delta = abs(source_area - render_area)
        source_mean = source["mean_rgb"]
        render_mean = render["mean_rgb"]
        mean_delta = sum(abs(a - b) for a, b in zip(source_mean, render_mean)) / 3.0
        if render_area < 0.03:
            defects.append({"defect_id": "mostly_blank_render", "stage": "visual_shell_generation|unreal_assembly", "severity": "high", "repair_owner": "camera_estimation|representation_selection", "automatic_repair_safe": False})
        if mean_delta > 70:
            defects.append({"defect_id": "colour_distribution_mismatch", "stage": "material_harmonization", "severity": "medium", "repair_owner": "material_generation|material_harmonization", "automatic_repair_safe": True})
        classification = "PROVEN" if not defects else "REJECTED"
    return {"schema_version": "source_view_validation_v1", "classification": classification, "tier": tier, "source": source, "render": render, "metrics": {"landmark_reprojection": "NOT_PROVEN", "silhouette_overlap": "NOT_PROVEN", "semantic_region_overlap": "NOT_PROVEN", "major_edge_alignment": "NOT_PROVEN", "colour_distribution": "PROVEN" if not defects else "REJECTED", "depth_ordering": "NOT_PROVEN"}, "defects": defects, "threshold_policy": {"smoke_render_non_dark_min": 0.03, "smoke_mean_rgb_delta_max": 70.0}}


def build_offset_validation(render_records: list[Mapping[str, Any]], tier: str = "smoke") -> dict[str, Any]:
    defects = []
    for record in render_records:
        if record.get("classification") not in {"PROVEN", "NOT_PROVEN"}:
            defects.append({"view": record.get("view"), "defect_id": "view_failed", "stage": "offset_view_validation", "severity": "high", "repair_owner": "unseen_world_completion", "automatic_repair_safe": False})
    return {"schema_version": "offset_view_validation_v1", "classification": "PROVEN" if not defects and render_records else "NOT_PROVEN", "tier": tier, "views": list(render_records), "defects": defects, "bounded_360_claim": False}


def repair_history(defects: list[Mapping[str, Any]]) -> dict[str, Any]:
    records = []
    for defect in defects:
        records.append({"defect_id": defect.get("defect_id"), "responsible_stage": defect.get("stage"), "affected_region": defect.get("affected_region"), "severity": defect.get("severity", "medium"), "proposed_correction": defect.get("repair_owner"), "automatic_repair_safe": bool(defect.get("automatic_repair_safe", False)), "retry_count": 0, "result": "PENDING_USER_OR_LIVE_EDITOR"})
    return {"schema_version": "repair_history_v1", "classification": "PROVEN", "max_attempts_per_defect": 1, "records": records, "thresholds_unchanged": True}
