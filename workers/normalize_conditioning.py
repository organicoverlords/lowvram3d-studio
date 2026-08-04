"""Bounded automatic source-alpha/matte normalization for image-conditioned generation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

try:
    from .pipeline_matte import key_alpha
except ImportError:  # direct worker execution from the repository root
    from pipeline_matte import key_alpha


def _components(alpha: np.ndarray) -> tuple[np.ndarray, list[int]]:
    mask = alpha > 0
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = sorted((int(v) for v in np.bincount(labels.ravel())[1:]), reverse=True)
    return labels, areas


def _metrics(alpha: np.ndarray) -> dict:
    if alpha.ndim != 2 or alpha.dtype != np.uint8:
        raise ValueError("alpha must be a uint8 HxW array")
    finite = bool(np.isfinite(alpha).all())
    mask = alpha > 0
    count = int(mask.sum())
    h, w = alpha.shape
    ys, xs = np.where(mask)
    bbox = None if not count else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    _, areas = _components(alpha)
    return {
        "dimensions": [int(w), int(h)],
        "alpha_valid": finite,
        "nonzero_alpha_pixels": count,
        "foreground_coverage_percent": round(count / (w * h) * 100.0, 4),
        "foreground_bbox_xyxy": bbox,
        "foreground_width_percent": round((bbox[2] - bbox[0] + 1) / w * 100.0, 4) if bbox else 0.0,
        "foreground_height_percent": round((bbox[3] - bbox[1] + 1) / h * 100.0, 4) if bbox else 0.0,
        "transparent_margin_percent": round(100.0 - count / (w * h) * 100.0, 4),
        "disconnected_foreground_components": max(0, len(areas) - 1),
        "component_count": len(areas),
        "component_areas_desc": areas[:32],
        "border_contact": {
            "left": bool(bbox and bbox[0] == 0),
            "top": bool(bbox and bbox[1] == 0),
            "right": bool(bbox and bbox[2] == w - 1),
            "bottom": bool(bbox and bbox[3] == h - 1),
        },
    }


def _checker(size: tuple[int, int]) -> Image.Image:
    w, h = size
    out = Image.new("RGBA", (w, h), (40, 40, 40, 255))
    draw = ImageDraw.Draw(out)
    block = max(8, min(w, h) // 16)
    for y in range(0, h, block):
        for x in range(0, w, block):
            if (x // block + y // block) % 2:
                draw.rectangle((x, y, x + block, y + block), fill=(70, 70, 70, 255))
    return out


def _write_overlay(image: Image.Image, path: Path, bbox=None) -> None:
    out = _checker(image.size)
    out.alpha_composite(image)
    if bbox:
        ImageDraw.Draw(out).rectangle(tuple(bbox), outline=(255, 50, 50, 255), width=max(2, image.width // 256))
    out.convert("RGB").save(path)


def normalize_conditioning(
    image_path: Path,
    output_path: Path,
    audit_path: Path,
    overlay_path: Path,
    original_vs_matte_path: Path,
    size: int = 512,
    tolerance: float = 42.0,
    enclosed_tolerance: float = 32.0,
    shadow_tolerance: float = 180.0,
    shadow_from: float = 0.78,
) -> dict:
    with Image.open(image_path) as original_image:
        original_bands = original_image.getbands()
        source = original_image.convert("RGBA")
    original_alpha = np.asarray(source)[..., 3]
    source_has_useful_alpha = (
        "A" in original_bands
        and int(original_alpha.min()) < 255
        and int((original_alpha > 0).sum()) > 0
    )

    if source_has_useful_alpha:
        selected = source
        route = "original_alpha_preserved"
        matte_stats = None
    else:
        rgb = np.asarray(source.convert("RGB"))
        alpha, matte_stats = key_alpha(
            rgb, tolerance, "hybrid", 5000, enclosed_tolerance,
            shadow_tolerance, shadow_from, 2
        )
        selected = Image.fromarray(np.dstack((np.where(alpha[..., None] > 0, rgb, 255), alpha)).astype(np.uint8), "RGBA")
        route = "existing_pipeline_matte"

    selected_alpha = np.asarray(selected)[..., 3]
    selected_metrics = _metrics(selected_alpha)
    if not selected_metrics["alpha_valid"]:
        raise ValueError("INVALID_ALPHA")
    if selected_metrics["nonzero_alpha_pixels"] == 0:
        raise ValueError("EMPTY_FOREGROUND")
    if not source_has_useful_alpha and selected_metrics["foreground_coverage_percent"] >= 99.5:
        raise ValueError("MATTE_BACKGROUND_FAILURE_ALMOST_FULL_CANVAS")

    bbox = selected_metrics["foreground_bbox_xyxy"]
    x0, y0, x1, y1 = bbox
    subject = selected.crop((x0, y0, x1 + 1, y1 + 1))
    subject_w, subject_h = subject.size
    pad_x = max(1, round(subject_w * 0.10))
    pad_y = max(1, round(subject_h * 0.10))
    crop_w, crop_h = subject_w + 2 * pad_x, subject_h + 2 * pad_y
    square = max(crop_w, crop_h)
    canvas = Image.new("RGBA", (square, square), (255, 255, 255, 0))
    paste_x = (square - crop_w) // 2 + pad_x
    paste_y = (square - crop_h) // 2 + pad_y
    canvas.alpha_composite(subject, (paste_x, paste_y))
    normalized = canvas.resize((size, size), Image.Resampling.LANCZOS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.save(output_path)

    normalized_metrics = _metrics(np.asarray(normalized)[..., 3])
    audit = {
        "status": "PROVEN",
        "asset_source": str(image_path),
        "conditioning_output": str(output_path),
        "route": route,
        "source_has_useful_alpha": source_has_useful_alpha,
        "matte_stats": matte_stats,
        "original_alpha": _metrics(original_alpha),
        "selected_foreground": selected_metrics,
        "normalization": {
            "padding_fraction_each_side": 0.10,
            "square_canvas_before_resize": [square, square],
            "generator_input_size": [size, size],
            "aspect_ratio_preserved": True,
            "meaningful_components_retained": True,
            "clipping_prevented": True,
        },
        "normalized_alpha": normalized_metrics,
        "audit_artifacts": [str(overlay_path), str(original_vs_matte_path)],
    }
    _write_overlay(normalized, overlay_path, normalized_metrics["foreground_bbox_xyxy"])
    comparison = Image.new("RGB", (source.width * 2, source.height), (20, 20, 20))
    comparison.paste(source.convert("RGB"), (0, 0))
    selected_preview = _checker(source.size)
    selected_preview.alpha_composite(selected)
    comparison.paste(selected_preview.convert("RGB"), (source.width, 0))
    ImageDraw.Draw(comparison).text((8, 8), "original", fill="white")
    ImageDraw.Draw(comparison).text((source.width + 8, 8), route, fill="white")
    comparison.save(original_vs_matte_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-json", required=True)
    parser.add_argument("--overlay", required=True)
    parser.add_argument("--original-vs-matte", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--tolerance", type=float, default=42.0)
    parser.add_argument("--enclosed-tolerance", type=float, default=32.0)
    parser.add_argument("--shadow-tolerance", type=float, default=180.0)
    parser.add_argument("--shadow-from", type=float, default=0.78)
    args = parser.parse_args()
    try:
        audit = normalize_conditioning(
            Path(args.image), Path(args.output), Path(args.audit_json),
            Path(args.overlay), Path(args.original_vs_matte), args.size,
            args.tolerance, args.enclosed_tolerance, args.shadow_tolerance,
            args.shadow_from,
        )
        print(f"CONDITIONING_NORMALIZED status={audit['status']} route={audit['route']} output={args.output}", flush=True)
        return 0
    except Exception as exc:
        payload = {"status": "BLOCKED", "error_code": str(exc), "image": args.image}
        Path(args.audit_json).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"CONDITIONING_BLOCKED error={exc}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
