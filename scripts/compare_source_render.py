"""Compare an Unreal render against the source image it was reconstructed from.

Also answers the orientation question directly: the render is scored against the
source under all four axis-flip candidates, so a mirrored or 180-rotated import
is diagnosed by measurement rather than by eye.

Both images are normalised before comparison. The Unreal render is far darker
than the source, and raw pixel difference would be dominated by that global
exposure gap rather than by geometry.

    py -3.12 scripts/compare_source_render.py --render <png> --source <png> --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

TRANSFORMS = {
    "identity": lambda im: im,
    "vflip": lambda im: im.transpose(Image.FLIP_TOP_BOTTOM),
    "hflip": lambda im: im.transpose(Image.FLIP_LEFT_RIGHT),
    "rot180": lambda im: im.transpose(Image.ROTATE_180),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise(gray: np.ndarray) -> np.ndarray:
    """Percentile-stretch to remove the global exposure gap."""
    low, high = np.percentile(gray, 1.0), np.percentile(gray, 99.0)
    if high - low < 1e-6:
        return np.zeros_like(gray)
    return np.clip((gray - low) / (high - low), 0.0, 1.0)


def edges(image: Image.Image) -> np.ndarray:
    filtered = image.convert("L").filter(ImageFilter.FIND_EDGES)
    array = np.asarray(filtered, dtype=np.float64) / 255.0
    return (array > np.percentile(array, 90.0)).astype(np.float64)


def score(render: Image.Image, source: Image.Image) -> dict[str, float]:
    r = normalise(np.asarray(render.convert("L"), dtype=np.float64) / 255.0)
    s = normalise(np.asarray(source.convert("L"), dtype=np.float64) / 255.0)

    correlation = float(np.corrcoef(r.ravel(), s.ravel())[0, 1])

    re, se = edges(render), edges(source)
    union = float(np.logical_or(re, se).sum())
    edge_iou = float(np.logical_and(re, se).sum() / union) if union else 0.0

    # Foreground silhouette: what the reconstruction placed vs what the source
    # shows, both taken as "brighter than the frame median".
    rf = r > np.median(r)
    sf = s > np.median(s)
    sil_union = float(np.logical_or(rf, sf).sum())
    silhouette_iou = float(np.logical_and(rf, sf).sum() / sil_union) if sil_union else 0.0

    return {
        "correlation": correlation,
        "edge_iou": edge_iou,
        "silhouette_iou": silhouette_iou,
        "combined": correlation + edge_iou + silhouette_iou,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    render_path, source_path = Path(args.render), Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    render = Image.open(render_path).convert("RGB")
    source = Image.open(source_path).convert("RGB")
    if source.size != render.size:
        source = source.resize(render.size, Image.LANCZOS)

    scores = {name: score(fn(render), source) for name, fn in TRANSFORMS.items()}
    best = max(scores, key=lambda k: scores[k]["combined"])
    oriented = TRANSFORMS[best](render)

    r = np.asarray(oriented, dtype=np.float64)
    s = np.asarray(source, dtype=np.float64)
    delta = np.abs(r - s)

    gray = np.asarray(oriented.convert("L"), dtype=np.float64)
    report = {
        "render": str(render_path),
        "render_sha256": sha256(render_path),
        "source": str(source_path),
        "dimensions": list(render.size),
        "aspect_ratio": render.size[0] / render.size[1],
        "aspect_is_4_3": abs(render.size[0] / render.size[1] - 4 / 3) < 1e-3,
        "best_orientation": best,
        "orientation_scores": scores,
        "mean_abs_difference": float(delta.mean()),
        "p50_abs_difference": float(np.percentile(delta, 50)),
        "p95_abs_difference": float(np.percentile(delta, 95)),
        "render_mean_luma": float(gray.mean()),
        "render_luma_std": float(gray.std()),
        "non_dark_fraction": float((gray > 16).mean()),
        "rgb_range": [int(np.asarray(oriented).min()), int(np.asarray(oriented).max())],
    }

    # Side-by-side sheet: source | render | difference.
    width, height = render.size
    sheet = Image.new("RGB", (width * 3, height + 28), (16, 16, 18))
    sheet.paste(source, (0, 28))
    sheet.paste(oriented, (width, 28))
    sheet.paste(Image.fromarray(delta.astype(np.uint8)), (width * 2, 28))
    draw = ImageDraw.Draw(sheet)
    for index, label in enumerate(
            ["source image", f"unreal render ({best})", "absolute difference"]):
        draw.text((index * width + 10, 8), label, fill=(235, 235, 235))

    comparison = out_dir / "source_camera_comparison.png"
    difference = out_dir / "source_camera_difference.png"
    sheet.save(comparison)
    ImageChops.difference(oriented, source).save(difference)

    report["comparison_png"] = str(comparison)
    report["difference_png"] = str(difference)
    report["comparison_sha256"] = sha256(comparison)

    (out_dir / "source_camera_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
