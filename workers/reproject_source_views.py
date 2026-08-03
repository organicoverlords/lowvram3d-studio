"""Unlit reprojection of a textured GLB through the exact source cameras, with metrics.

Blender renders answer "does it look right under lights"; they cannot answer "did the atlas
keep what the source view contained", because shading, colour management and a different
camera all move the pixels. This re-renders the textured mesh through the bundle's own
camera matrices with the base colour sampled directly and no lighting at all, so every
difference from the MV-Adapter source image is attributable to fusion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_mvadapter_cpu_controls import PROJECTION_SPAN
from mesh_io import read_glb
from multiview_texture_projection import semantic_of
from render_control_bundle_texture import base_colour_image, file_prefix


def reproject(bundle: Path, view: dict, uv: np.ndarray, tris: np.ndarray,
              texture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample the atlas through the view's own triangle-ID and barycentric buffers."""
    prefix = file_prefix(view)
    ids = np.load(bundle / f"{prefix}_triangle_ids.npy")
    bary = np.load(bundle / f"{prefix}_barycentric.npy")
    visible = ids >= 0
    canvas = np.zeros(ids.shape + (3,), np.float32)
    if visible.any():
        pixel_uv = np.einsum("nc,ncd->nd", bary[visible], uv[tris[ids[visible]]])
        height, width = texture.shape[:2]
        xs = np.clip((pixel_uv[:, 0] % 1.0) * (width - 1), 0, width - 1)
        ys = np.clip((pixel_uv[:, 1] % 1.0) * (height - 1), 0, height - 1)
        x0 = np.floor(xs).astype(np.int64)
        y0 = np.floor(ys).astype(np.int64)
        fx = (xs - x0)[:, None]
        fy = (ys - y0)[:, None]
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        top = texture[y0, x0] * (1 - fx) + texture[y0, x1] * fx
        bottom = texture[y1, x0] * (1 - fx) + texture[y1, x1] * fx
        canvas[visible] = top * (1 - fy) + bottom * fy
    return canvas, visible


def _grey(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float64) @ np.array([0.2126, 0.7152, 0.0722])


def _box(image: np.ndarray, radius: int) -> np.ndarray:
    padded = np.pad(image, radius, mode="edge")
    cumulative = np.cumsum(np.cumsum(padded, axis=0), axis=1)
    cumulative = np.pad(cumulative, ((1, 0), (1, 0)), mode="constant")
    size = 2 * radius + 1
    window = (cumulative[size:, size:] - cumulative[:-size, size:]
              - cumulative[size:, :-size] + cumulative[:-size, :-size])
    return window / float(size * size)


def _gradient(image: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(image)
    return np.hypot(gx, gy)


def metrics(candidate: np.ndarray, reference: np.ndarray, mask: np.ndarray,
            weight: np.ndarray | None = None) -> dict:
    """SSIM and detail-retention statistics restricted to ``mask``."""
    selection = mask if weight is None else (mask & (weight > 0.5))
    if selection.sum() < 32:
        return {"pixels": int(selection.sum()), "insufficient_support": True}

    a, b = _grey(candidate), _grey(reference)
    mu_a, mu_b = _box(a, 4), _box(b, 4)
    va = np.maximum(_box(a * a, 4) - mu_a ** 2, 0.0)
    vb = np.maximum(_box(b * b, 4) - mu_b ** 2, 0.0)
    cov = _box(a * b, 4) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ssim_map = (((2 * mu_a * mu_b + c1) * (2 * cov + c2))
                / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))

    ga, gb = _gradient(a), _gradient(b)
    error = np.abs(candidate.astype(np.float64) - reference.astype(np.float64)).mean(axis=2)
    mse = float(((a[selection] - b[selection]) ** 2).mean())
    dark_reference = selection & (b < np.percentile(b[selection], 20))
    edge_reference = selection & (gb > np.percentile(gb[selection], 80))

    def correlation(x, y):
        x = x[selection] - x[selection].mean()
        y = y[selection] - y[selection].mean()
        denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
        return float(x @ y / denominator) if denominator > 0 else 0.0

    return {
        "pixels": int(selection.sum()),
        "ssim": float(ssim_map[selection].mean()),
        "psnr": float(10.0 * np.log10((255.0 ** 2) / mse)) if mse > 0 else float("inf"),
        "mean_absolute_colour_error": float(error[selection].mean()),
        "gradient_magnitude_retention": float(ga[selection].mean()
                                              / max(gb[selection].mean(), 1e-9)),
        "high_frequency_energy_retention": float(
            np.std(a[selection] - _box(a, 4)[selection])
            / max(np.std(b[selection] - _box(b, 4)[selection]), 1e-9)),
        "edge_correlation": correlation(ga, gb),
        "local_contrast": float(np.std(a[selection])),
        "local_contrast_reference": float(np.std(b[selection])),
        "dark_feature_retention": (
            float(a[dark_reference].mean() / max(b[dark_reference].mean(), 1e-9))
            if dark_reference.any() else None),
        "edge_density": float((ga[selection] > np.percentile(gb[selection], 80)).mean()),
        "edge_density_reference": float(
            (gb[selection] > np.percentile(gb[selection], 80)).mean()),
        "coverage": float(mask.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="textured GLB to reproject")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--views-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--region-config", default=None)
    parser.add_argument("--label", default="candidate")
    parser.add_argument("--write-images", action="store_true")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = json.loads((bundle / "camera_contract.json").read_text(encoding="utf-8"))
    receipt = json.loads(Path(args.views_receipt).read_text(encoding="utf-8"))
    sources = {str(item["name"]): Path(item["path"]) for item in receipt["output_images"]}

    _positions, _normals, uv, tris = read_glb(Path(args.mesh))
    texture = np.asarray(base_colour_image(Path(args.mesh))).astype(np.float32)

    region_masks = {}
    if args.region_config:
        import protected_region
        config = protected_region.load(Path(args.region_config))
        region_masks = protected_region.build_masks(config, int(config["source_image_size"]))

    per_view = {}
    for view in sorted(contract["views"], key=lambda item: int(item["index"])):
        index = int(view["index"])
        semantic = semantic_of(view)
        source_path = sources.get(f"view_{index}_{semantic}.png")
        if source_path is None:
            raise RuntimeError(f"REPROJECT_SOURCE_MISSING:{index}:{semantic}")
        reference = np.asarray(Image.open(source_path).convert("RGB")).astype(np.float32)
        candidate, visible = reproject(bundle, view, uv, tris, texture)
        record = {"semantic": semantic, "raw_index": index,
                  "source_image": str(source_path),
                  "whole_foreground": metrics(candidate, reference, visible)}
        for name, mask in region_masks.items():
            if mask["owner_semantic"] and mask["owner_semantic"] != semantic:
                continue
            record[f"region_{name}"] = metrics(candidate, reference, visible, mask["weight"])
        per_view[semantic] = record

        if args.write_images:
            Image.fromarray(np.clip(candidate, 0, 255).astype(np.uint8)).save(
                output_dir / f"{args.label}_{semantic}_reprojection.png")
            difference = np.abs(candidate - reference).mean(axis=2)
            difference[~visible] = 0
            Image.fromarray(
                np.clip(difference * 3.0, 0, 255).astype(np.uint8)).save(
                output_dir / f"{args.label}_{semantic}_reprojection_difference.png")

    report = {
        "schema": "source_view_reprojection_qa_v1",
        "label": args.label,
        "mesh": str(args.mesh),
        "bundle": str(bundle),
        "unlit": True,
        "lighting": "NONE",
        "colour_management": "NONE_RAW_BASE_COLOUR",
        "camera": "EXACT_SOURCE_CONTROL_CAMERAS",
        "region_config": args.region_config,
        "views": per_view,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    face = per_view.get("front", {}).get("region_face")
    print(f"REPROJECTION_QA {args.label} front_ssim="
          f"{per_view.get('front', {}).get('whole_foreground', {}).get('ssim', 0):.4f}"
          + (f" face_ssim={face['ssim']:.4f}" if face and "ssim" in face else ""), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
