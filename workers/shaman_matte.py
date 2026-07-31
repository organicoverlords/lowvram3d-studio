"""Background key for the shaman anchor that preserves detached subject parts.

The rembg/u2net matte used by the default Mini Turbo path destroys this particular source: it
drops every ornament hanging on a cord from the antler pole, erases the cords themselves, and
collapses the staff's ring head into an opaque black blob. Those failures then show up in the
generated geometry as missing silhouette features plus floating debris.

The source is a studio plate with a smooth, near-uniform background, so a far more faithful matte
is a border flood-fill key: classify pixels close to the background colour, keep only the
background-coloured regions that are *connected to the image border*, and treat everything else as
subject. Interior detail and fully detached ornaments both survive, because neither touches the
border.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def border_background_colour(rgb: np.ndarray, band: int = 8) -> np.ndarray:
    edges = np.concatenate(
        [
            rgb[:band].reshape(-1, 3),
            rgb[-band:].reshape(-1, 3),
            rgb[:, :band].reshape(-1, 3),
            rgb[:, -band:].reshape(-1, 3),
        ]
    )
    return np.median(edges, axis=0)


def key_alpha(
    rgb: np.ndarray,
    tolerance: float,
    mode: str = "hybrid",
    enclosed_min_area: int = 1500,
    enclosed_tolerance: float | None = None,
    shadow_tolerance: float | None = None,
    shadow_from: float = 0.82,
) -> tuple[np.ndarray, dict]:
    background = border_background_colour(rgb)
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=-1)

    # The cast shadow on the floor is background that is too dark for the plate tolerance. It only
    # ever occupies the bottom band, where the sole subject content is the dark bird feet, so a
    # looser threshold applied to those rows removes the shadow without touching the character.
    threshold = np.full(distance.shape, float(tolerance), dtype=np.float32)
    if shadow_tolerance is not None and shadow_tolerance > tolerance:
        first_row = int(distance.shape[0] * shadow_from)
        threshold[first_row:] = float(shadow_tolerance)
    candidate = distance < threshold

    if mode == "hybrid":
        # Border-connected background at a generous tolerance removes the plate and the soft floor
        # shadow without risk to the character, because interior pixels are only reachable through
        # a background-coloured path. Raising tolerance here cannot perforate the body.
        labels, count = ndimage.label(candidate)
        border_labels = set(labels[0].tolist()) | set(labels[-1].tolist())
        border_labels |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
        border_labels.discard(0)
        background_mask = np.isin(labels, list(border_labels)) if border_labels else np.zeros_like(candidate)

        # The pockets enclosed by the antler pole, the hanging cords and the staff ring are real
        # background but never touch the border. Remove them by area so that genuine background
        # panels key out while the small background-coloured speckles inside the robes - which are
        # what perforates the body in a pure colour key - stay part of the subject.
        # Enclosed background needs a tight threshold across the torso, where a loose one punches
        # holes through pale robe fabric, but a loose one in the floor band, where the shadow
        # pocket trapped between the legs is bright and large. Reuse the same row-dependent
        # profile as the plate key.
        enclosed_threshold = np.full(
            distance.shape,
            float(enclosed_tolerance if enclosed_tolerance is not None else tolerance),
            dtype=np.float32,
        )
        if shadow_tolerance is not None:
            enclosed_threshold[int(distance.shape[0] * shadow_from):] = float(shadow_tolerance)
        enclosed = ndimage.label((distance < enclosed_threshold) & ~background_mask)
        enclosed_labels, enclosed_count = enclosed
        if enclosed_count:
            areas = ndimage.sum(
                np.ones_like(enclosed_labels, dtype=bool),
                enclosed_labels,
                range(1, enclosed_count + 1),
            )
            large = [i + 1 for i, area in enumerate(areas) if area >= enclosed_min_area]
            if large:
                background_mask |= np.isin(enclosed_labels, large)
    elif mode == "flood":
        # Border-connected background only. Preserves interior detail but also traps genuine
        # background in the pockets enclosed by the antler pole, the hanging cords and the staff,
        # which then survives as grey slab geometry.
        labels, count = ndimage.label(candidate)
        border_labels = set(labels[0].tolist()) | set(labels[-1].tolist())
        border_labels |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
        border_labels.discard(0)
        background_mask = np.isin(labels, list(border_labels)) if border_labels else np.zeros_like(candidate)
    else:
        # Pure colour key. Every background-coloured pixel is background wherever it sits, so the
        # enclosed pockets and the ring hole in the staff head all key out correctly. The subject
        # is textured and saturated enough that nothing on the character falls inside tolerance.
        count = 0
        background_mask = candidate

    alpha = np.where(background_mask, 0, 255).astype(np.uint8)
    subject = alpha > 0
    kept_labels, kept_count = ndimage.label(subject)
    sizes = ndimage.sum(subject, kept_labels, range(1, kept_count + 1))
    stats = {
        "tolerance": tolerance,
        "background_rgb": [float(v) for v in background],
        "candidate_regions": int(count),
        "subject_pixel_fraction": float(subject.mean()),
        "subject_components": int(kept_count),
        "largest_component_fraction": float(sizes.max() / subject.sum()) if kept_count else 0.0,
        "detached_components": int((sizes >= 40).sum() - 1) if kept_count else 0,
    }
    return alpha, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=40.0)
    parser.add_argument("--preview", default="")
    parser.add_argument("--stats-json", default="")
    parser.add_argument("--sweep", default="", help="comma separated tolerances to preview instead of writing output")
    parser.add_argument("--enclosed-min-area", type=int, default=1500)
    parser.add_argument("--enclosed-tolerance", type=float, default=None)
    parser.add_argument("--shadow-tolerance", type=float, default=None)
    parser.add_argument("--shadow-from", type=float, default=0.82)
    parser.add_argument("--mode", choices=("hybrid", "colour", "flood"), default="hybrid")
    args = parser.parse_args()

    source = Image.open(args.image).convert("RGB")
    rgb = np.asarray(source)

    if args.sweep:
        report = []
        for raw in args.sweep.split(","):
            tolerance = float(raw)
            alpha, stats = key_alpha(rgb, tolerance, args.mode, args.enclosed_min_area, args.enclosed_tolerance, args.shadow_tolerance, args.shadow_from)
            preview = Image.new("RGB", source.size, (255, 0, 255))
            preview.paste(source, mask=Image.fromarray(alpha, "L"))
            path = Path(args.output).with_name(f"matte_{args.mode}_t{int(tolerance)}.png")
            preview.save(path)
            stats["preview"] = str(path)
            report.append(stats)
            print(
                f"tolerance={tolerance:5.1f} subject={stats['subject_pixel_fraction']*100:5.2f}% "
                f"components={stats['subject_components']:5d} detached>=40px={stats['detached_components']:4d} "
                f"largest={stats['largest_component_fraction']*100:5.2f}% -> {path.name}",
                flush=True,
            )
        if args.stats_json:
            Path(args.stats_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    alpha, stats = key_alpha(rgb, args.tolerance, args.mode, args.enclosed_min_area, args.enclosed_tolerance, args.shadow_tolerance, args.shadow_from)
    rgba = np.dstack([rgb, alpha])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(output)
    if args.preview:
        preview = Image.new("RGB", source.size, (255, 0, 255))
        preview.paste(source, mask=Image.fromarray(alpha, "L"))
        Image.fromarray(np.asarray(preview)).save(args.preview)
    if args.stats_json:
        Path(args.stats_json).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"MATTE_WRITTEN {output} subject={stats['subject_pixel_fraction']*100:.2f}% "
        f"components={stats['subject_components']} detached={stats['detached_components']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



