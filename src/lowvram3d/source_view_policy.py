"""Fail-closed policy for deterministic Unreal source-view captures.

This module is deliberately Unreal-independent.  The editor-side capture
adapter records its facts here, while CPU tests can verify the contract
without launching Unreal or touching a map.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

EXPECTED_CAMERA_LABEL = "Castlegrounds_Camera_Source"
EXPECTED_FOV_DEG = 66.50838470458984
EXPECTED_ASPECT = 4.0 / 3.0


@dataclass(frozen=True)
class CaptureContract:
    camera_label: str
    width: int
    height: int
    fov_deg: float
    projection: str = "perspective"

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


def validate_capture_contract(record: Mapping[str, Any], contract: CaptureContract) -> dict[str, Any]:
    """Validate named-camera identity and the exact 4:3 capture contract."""

    defects: list[dict[str, Any]] = []
    if record.get("camera_label") != contract.camera_label:
        defects.append({"code": "NAMED_CAMERA_NOT_USED", "actual": record.get("camera_label"), "expected": contract.camera_label})
    if bool(record.get("player_camera_used", False)):
        defects.append({"code": "PLAYER_CAMERA_DEPENDENCY", "actual": True})
    if bool(record.get("slate_capture", False)) or bool(record.get("editor_ui_visible", False)):
        defects.append({"code": "EDITOR_UI_OR_SLATE_CAPTURE", "actual": True})
    width = int(record.get("width", 0) or 0)
    height = int(record.get("height", 0) or 0)
    if (width, height) != (contract.width, contract.height):
        defects.append({"code": "DIMENSIONS_MISMATCH", "actual": [width, height], "expected": [contract.width, contract.height]})
    if height <= 0 or abs((width / height) - contract.aspect_ratio) > 1e-6:
        defects.append({"code": "ASPECT_RATIO_MISMATCH", "actual": (width / height) if height else None, "expected": contract.aspect_ratio})
    if abs(float(record.get("fov_deg", 0.0)) - contract.fov_deg) > 1e-4:
        defects.append({"code": "FOV_MISMATCH", "actual": record.get("fov_deg"), "expected": contract.fov_deg})
    if str(record.get("projection", "")).lower() != contract.projection:
        defects.append({"code": "PROJECTION_MISMATCH", "actual": record.get("projection"), "expected": contract.projection})
    if not record.get("source_shell_visible", False):
        defects.append({"code": "SOURCE_SHELL_HIDDEN"})
    if record.get("preview_material_visible", False):
        defects.append({"code": "PREVIEW_MATERIAL_VISIBLE"})
    if record.get("proxy_occluding_shell", False):
        defects.append({"code": "PROXY_OCCLUDES_SOURCE_SHELL"})
    return {
        "classification": "PROVEN" if not defects else "REJECTED",
        "contract": {
            "camera_label": contract.camera_label,
            "dimensions_px": [contract.width, contract.height],
            "aspect_ratio": contract.aspect_ratio,
            "fov_deg": contract.fov_deg,
            "projection": contract.projection,
        },
        "defects": defects,
    }


def validate_image_evidence(stats: Mapping[str, Any], *, min_non_dark: float = 0.03, min_stddev: float = 2.0, min_mean_luminance: float = 16.0) -> dict[str, Any]:
    """Reject missing, stale, black, or near-constant capture evidence."""

    defects: list[dict[str, Any]] = []
    dimensions = stats.get("dimensions_px") or []
    if len(dimensions) != 2 or int(dimensions[0]) <= 0 or int(dimensions[1]) <= 0:
        defects.append({"code": "INVALID_IMAGE_DIMENSIONS"})
    if float(stats.get("non_dark_fraction", 0.0)) < min_non_dark:
        defects.append({"code": "NEARLY_BLACK_CAPTURE", "non_dark_fraction": stats.get("non_dark_fraction")})
    stddev = stats.get("stddev_rgb") or []
    if len(stddev) != 3 or max(float(value) for value in stddev) < min_stddev:
        defects.append({"code": "LOW_IMAGE_VARIANCE", "stddev_rgb": stddev})
    if "mean_luminance" in stats and float(stats.get("mean_luminance") or 0.0) < min_mean_luminance:
        defects.append({"code": "LOW_MEAN_LUMINANCE", "mean_luminance": stats.get("mean_luminance"), "minimum": min_mean_luminance})
    if not stats.get("sha256"):
        defects.append({"code": "MISSING_IMAGE_HASH"})
    return {"classification": "PROVEN" if not defects else "REJECTED", "defects": defects}


def measure_image_file(path: str | Path) -> dict[str, Any]:
    """Measure a rendered image on CPU before it can enter visual QA.

    This deliberately records luminance occupancy and channel variation rather
    than treating a non-empty PNG as proof.  It keeps black, stale, and nearly
    constant Unreal captures fail-closed without requiring Unreal or a GPU.
    """

    image_path = Path(path)
    result: dict[str, Any] = {
        "path": str(image_path.resolve()),
        "exists": image_path.is_file(),
        "bytes": image_path.stat().st_size if image_path.is_file() else 0,
        "dimensions_px": None,
        "mode": None,
        "non_dark_fraction": 0.0,
        "mean_luminance": 0.0,
        "stddev_rgb": [],
        "sha256": None,
    }
    if not image_path.is_file():
        result["classification"] = "REJECTED"
        result["defects"] = [{"code": "MISSING_IMAGE_FILE"}]
        return result
    import hashlib

    digest = hashlib.sha256()
    with image_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    result["sha256"] = digest.hexdigest()
    try:
        from PIL import Image
        import numpy as np

        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            result["dimensions_px"] = [int(image.width), int(image.height)]
            result["mode"] = image.mode
        luminance = rgb.mean(axis=2)
        result["non_dark_fraction"] = float((luminance > 8.0).mean())
        result["mean_luminance"] = float(luminance.mean())
        result["stddev_rgb"] = [float(value) for value in rgb.reshape(-1, 3).std(axis=0)]
    except Exception as exc:
        result["defects"] = [{"code": "IMAGE_DECODE_FAILED", "error": f"{type(exc).__name__}: {exc}"}]
        result["classification"] = "REJECTED"
        return result
    verdict = validate_image_evidence(result)
    result["classification"] = verdict["classification"]
    result["defects"] = verdict["defects"]
    return result


def validate_capture_evidence(record: Mapping[str, Any], image_path: str | Path, contract: CaptureContract, material_audit: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Combine camera, pixel, and material gates into one fail-closed receipt."""

    contract_result = validate_capture_contract(record, contract)
    image_result = measure_image_file(image_path)
    material_result = validate_material_audit(material_audit or {})
    defects = [
        *contract_result["defects"],
        *image_result.get("defects", []),
        *material_result["defects"],
    ]
    return {
        "classification": "PROVEN" if not defects else "REJECTED",
        "contract": contract_result,
        "image": image_result,
        "materials": material_result,
        "defects": defects,
    }


def validate_material_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Reject engine/default/preview material slots in a source capture."""

    defects = []
    for slot in audit.get("slots", []):
        path = str(slot.get("path") or "")
        if not path:
            defects.append({"code": "MISSING_MATERIAL", "slot": slot.get("slot")})
        if bool(slot.get("engine_placeholder", False)) or bool(slot.get("preview_material", False)):
            defects.append({"code": "PLACEHOLDER_MATERIAL", "slot": slot.get("slot"), "path": path})
        if not bool(slot.get("unlit", False)):
            defects.append({"code": "SOURCE_MATERIAL_NOT_UNLIT", "slot": slot.get("slot"), "path": path})
    if not audit.get("slots"):
        defects.append({"code": "NO_MATERIAL_SLOTS"})
    return {"classification": "PROVEN" if not defects else "REJECTED", "defects": defects}


def capture_contract_for(width: int = 1448, height: int = 1086) -> CaptureContract:
    """Build the authoritative source contract, allowing only exact 4:3 sizes."""

    if width <= 0 or height <= 0 or abs((width / height) - EXPECTED_ASPECT) > 1e-6:
        raise ValueError("source capture resolution must be exact 4:3")
    return CaptureContract(EXPECTED_CAMERA_LABEL, width, height, EXPECTED_FOV_DEG)
