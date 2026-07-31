from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

VIEW_ORDER = ("front", "right", "back", "left", "top", "bottom")
COLORS = {
    "red": (150, 55, 45), "blue": (55, 85, 145), "green": (55, 125, 70),
    "yellow": (180, 150, 45), "black": (35, 35, 35), "white": (205, 205, 200),
    "brown": (110, 75, 45), "gray": (110, 115, 120), "grey": (110, 115, 120),
    "orange": (175, 95, 35), "purple": (105, 65, 135), "gold": (155, 120, 45),
}


def palette_color(prompt: str) -> tuple[int, int, int]:
    lowered = prompt.lower()
    for name, color in COLORS.items():
        if re.search(rf"\b{name}\b", lowered):
            return color
    return (105, 110, 110)


def estimate_background(rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    return np.median(border.astype(np.float32), axis=0)


def foreground_rgba(path: Path, size: int) -> tuple[Image.Image, dict[str, object]]:
    opened = ImageOps.exif_transpose(Image.open(path))
    existing_alpha = "A" in opened.getbands()
    rgba = opened.convert("RGBA")
    rgba.thumbnail((size * 2, size * 2), Image.Resampling.LANCZOS)
    alpha_array = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    meaningful_alpha = existing_alpha and bool(np.any(alpha_array < 250)) and bool(np.any(alpha_array > 8))
    if meaningful_alpha:
        alpha_image = rgba.getchannel("A")
        bbox = alpha_image.point(lambda value: 255 if value > 8 else 0).getbbox()
        report = {
            "backend": "preserved_source_alpha",
            "foreground_bbox_found": bool(bbox),
            "transparent_fraction": round(float(np.mean(alpha_array < 8)), 6),
        }
    else:
        source = rgba.convert("RGB")
        rgb = np.asarray(source).astype(np.float32)
        background = estimate_background(rgb)
        distance = np.linalg.norm(rgb - background[None, None, :], axis=2)
        border_distances = np.concatenate((distance[0], distance[-1], distance[:, 0], distance[:, -1]))
        noise = float(np.percentile(border_distances, 90))
        low = max(8.0, noise * 1.4)
        high = max(low + 18.0, noise * 3.5 + 20.0)
        alpha = np.clip((distance - low) / max(high - low, 1e-6), 0.0, 1.0)
        alpha_image = Image.fromarray(np.uint8(alpha * 255), mode="L").filter(ImageFilter.GaussianBlur(1.0))
        rgba = source.convert("RGBA")
        rgba.putalpha(alpha_image)
        bbox = alpha_image.point(lambda value: 255 if value > 20 else 0).getbbox()
        report = {
            "backend": "border_colour_estimation",
            "background_rgb": [round(float(value), 2) for value in background],
            "threshold_low": round(low, 2),
            "threshold_high": round(high, 2),
            "foreground_bbox_found": bool(bbox),
        }
    if bbox:
        rgba = rgba.crop(bbox)
    rgba.thumbnail((int(size * 0.92), int(size * 0.92)), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(rgba, ((size - rgba.width) // 2, (size - rgba.height) // 2))
    return canvas, report


def flat_fallback(size: int, prompt: str) -> Image.Image:
    color = palette_color(prompt)
    return Image.new("RGBA", (size, size), (*color, 255))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    source_path = Path(args.source_image) if args.source_image else None
    if source_path and source_path.is_file():
        source, background_report = foreground_rgba(source_path, args.size)
    else:
        source = flat_fallback(args.size, args.prompt)
        background_report = {"foreground_bbox_found": False, "fallback": "prompt_palette"}

    variants = {
        "front": source,
        "right": ImageEnhance.Color(ImageOps.mirror(source)).enhance(0.86),
        "back": ImageEnhance.Brightness(ImageEnhance.Color(ImageOps.mirror(source)).enhance(0.72)).enhance(0.78),
        "left": ImageEnhance.Color(source).enhance(0.86),
        "top": ImageEnhance.Brightness(source.filter(ImageFilter.GaussianBlur(1.5))).enhance(1.02),
        "bottom": ImageEnhance.Brightness(source.filter(ImageFilter.GaussianBlur(2.5))).enhance(0.62),
    }
    for name, image in variants.items():
        image.save(output / f"{name}.png")
    contact = Image.new("RGBA", (args.size * 6, args.size), (0, 0, 0, 0))
    for index, name in enumerate(VIEW_ORDER):
        contact.alpha_composite(variants[name], (index * args.size, 0))
    contact.save(output / "contact_sheet.png")
    (output / "worker_receipt.json").write_text(json.dumps({
        "success": True,
        "backend": "deterministic_source_projection_views",
        "source_image": str(source_path) if source_path else None,
        "view_order": VIEW_ORDER,
        "background_removal": background_report,
        "limitations": [
            "Unseen views are mirrored and palette-preserving approximations, not semantic reconstructions.",
            "MV-Adapter is preferred when it passes the target-machine VRAM gate.",
        ],
    }, indent=2), encoding="utf-8")

    # Only "front" is a real observation of the subject; every other view is a mirrored/blurred
    # approximation with no semantic content. The raster texture route reads this file to decide
    # which views may project real pixels vs which may only contribute low-frequency fill colour —
    # projecting a mirrored view as if it were real puts the front face onto the back of the mesh.
    real_view = "front" if (source_path and source_path.is_file()) else None
    (output / "view_metadata.json").write_text(json.dumps({
        "job_type": "single_image" if real_view else "prompt_only",
        "views": [
            {
                "view": name,
                "source_type": "real" if name == real_view else "mirrored",
                "confidence": 1.0 if name == real_view else 0.0,
            }
            for name in VIEW_ORDER
        ],
        "policy": {
            "semantic_projection": ["real", "generated"],
            "low_frequency_fill_only": ["mirrored", "synthetic"],
            "rule": "A front face must never be projected onto rear-facing polygons.",
        },
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
