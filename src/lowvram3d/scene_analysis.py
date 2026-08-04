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


def analyze_image(path: str | Path, scene_id: str, dimensions: tuple[int, int], mode: str = "cpu_bootstrap") -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    width, height = (int(dimensions[0]), int(dimensions[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return {
        "schema_version": "scene_analysis_v1",
        "classification": "PROVEN",
        "scene_id": str(scene_id),
        "analysis_mode": mode,
        "source": {"path": str(source.resolve()), "sha256": _sha256(source), "dimensions_px": [width, height]},
        "camera": {"contract_status": "REQUIRES_ANALYSIS", "projection": "perspective", "source_view_id": "source"},
        "regions": [{"id": "visual_shell_001", "layer_type": "visual_shell", "representation": "visual_shell", "confidence": 0.5, "bbox_norm_xyxy": [0.0, 0.0, 1.0, 1.0], "uncertainty": "semantic_analysis_not_instantiated"}],
        "depth_bands": [{"id": "unknown_001", "near_m": 0.1, "far_m": 100.0, "confidence": 0.2}],
        "coordinate_system": {"units": "meters", "up_axis": "Z", "handedness": "right"},
        "uncertainties": [{"id": "semantic_bootstrap", "description": "CPU bootstrap has not inferred reliable semantic regions", "fallback": "visual_shell_and_unresolved", "severity": "high"}],
        "resource_budgets": {"max_vram_mb": 6144, "max_triangles": 1500000, "gpu_work_requested": False},
    }
