"""Decide which generated view carries the photograph, and by how much.

The composite bake needs one number it cannot guess: which of the six generated
views shows the same face of the subject the photograph shows. Get it wrong and
the photograph is registered onto the wrong side, which is exactly the failure
that produced the first barn composite.

An earlier attempt at this compared the photograph's matte against the *control*
masks -- the silhouettes the renderer produces from the mesh. On the boat that
picked index 1 by a margin of 0.013, which is not a decision, it is a coin toss
with extra steps. The controls are also the weakest available evidence: every
view's silhouette comes from the same mesh, so they differ only in outline.

This compares against the *generated images* instead, and on two axes rather
than one:

  silhouette  -- IoU of the mattes after fitting bounding boxes, as before
  appearance  -- correlation of luminance inside the shared coverage

Appearance is the axis that actually separates them, because a wrong-side view
can share an outline but cannot share where the dark and light regions fall. A
riverboat's paddlewheel is a black disc at one end; the mirrored view puts it at
the other, and no amount of outline agreement hides that.

Both are reported for every view. If the two axes disagree the answer is not
trustworthy and this says so rather than picking one.

    py -3.12 workers/identify_photographed_view.py \\
        --photograph crop512.png --matte matte.png --views DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Below this the winning view is not meaningfully ahead of the runner-up and
#: the caller should not act on it. 0.013 -- the boat's old control-mask margin
#: -- must land on the wrong side of this.
MIN_MARGIN = 0.05


def _load_matte(path, size):
    import numpy as np
    from PIL import Image

    image = Image.open(path)
    if image.mode == "RGBA":
        alpha = np.asarray(image.split()[-1], dtype=np.float32) / 255.0
    else:
        alpha = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    mask = alpha > 0.5
    return mask


def _fit_to(mask, target_shape):
    """Scale a mask so its bounding box fills the target's bounding box.

    Registration by bounding box, not by centroid: the photograph is a crop of
    unknown padding and the generated views are rendered to a fixed frame, so
    the only thing the two share is the extent of the subject.
    """
    import numpy as np
    from PIL import Image

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return np.zeros(target_shape, dtype=bool), None
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    crop = mask[y0:y1 + 1, x0:x1 + 1]
    resized = np.asarray(
        Image.fromarray(crop.astype(np.uint8) * 255).resize(
            (target_shape[1], target_shape[0]), Image.BILINEAR)) > 127
    return resized, [int(x0), int(y0), int(x1), int(y1)]


def _fit_image_to(rgb, mask, target_shape):
    import numpy as np
    from PIL import Image

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    crop = rgb[y0:y1 + 1, x0:x1 + 1]
    return np.asarray(Image.fromarray(crop).resize(
        (target_shape[1], target_shape[0]), Image.BILINEAR), dtype=np.float32)


def _luminance(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photograph", required=True)
    parser.add_argument("--matte", required=True)
    parser.add_argument("--views", required=True,
                        help="Directory of view_N_<name>.png from six-view inference.")
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image

    views_dir = Path(args.views).resolve()
    view_paths = sorted(views_dir.glob("view_*_*.png"))
    if not view_paths:
        raise SystemExit(f"NO_VIEWS_FOUND:{views_dir}")

    photo_rgb = np.asarray(Image.open(args.photograph).convert("RGB"))
    photo_mask = _load_matte(args.matte, None)
    if photo_mask.shape != photo_rgb.shape[:2]:
        photo_mask = np.asarray(Image.fromarray(
            photo_mask.astype(np.uint8) * 255).resize(
            (photo_rgb.shape[1], photo_rgb.shape[0]), Image.NEAREST)) > 127

    results = []
    for path in view_paths:
        index = int(path.stem.split("_")[1])
        name = path.stem.split("_", 2)[2]
        view = Image.open(path).convert("RGB")
        view_rgb = np.asarray(view, dtype=np.uint8)
        shape = view_rgb.shape[:2]

        # The generated views sit on a flat grey plate. Foreground is anything
        # that is not that plate, measured against the corner pixel rather than
        # a hardcoded value, so a change of plate colour cannot silently empty
        # the mask.
        plate = view_rgb[2, 2].astype(np.float32)
        distance = np.linalg.norm(view_rgb.astype(np.float32) - plate, axis=2)
        view_mask = distance > 18.0

        fitted_photo_mask, _ = _fit_to(photo_mask, shape)
        union = np.logical_or(fitted_photo_mask, view_mask).sum()
        intersection = np.logical_and(fitted_photo_mask, view_mask).sum()
        iou = float(intersection) / float(max(union, 1))

        shared = np.logical_and(fitted_photo_mask, view_mask)
        if shared.sum() < 64:
            correlation = 0.0
        else:
            fitted_photo_rgb = _fit_image_to(photo_rgb, photo_mask, shape)
            a = _luminance(fitted_photo_rgb)[shared]
            b = _luminance(view_rgb.astype(np.float32))[shared]
            a = a - a.mean()
            b = b - b.mean()
            denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
            correlation = float((a * b).sum() / denominator) if denominator > 0 else 0.0

        results.append({
            "index": index,
            "name": name,
            "silhouette_iou": round(iou, 6),
            "appearance_correlation": round(correlation, 6),
            "view_coverage": round(float(view_mask.mean()), 6),
        })

    by_iou = sorted(results, key=lambda r: -r["silhouette_iou"])
    by_corr = sorted(results, key=lambda r: -r["appearance_correlation"])
    iou_margin = by_iou[0]["silhouette_iou"] - by_iou[1]["silhouette_iou"]
    corr_margin = by_corr[0]["appearance_correlation"] - by_corr[1]["appearance_correlation"]
    agree = by_iou[0]["index"] == by_corr[0]["index"]

    report = {
        "schema_version": "photographed_view_identification_v1",
        "photograph": str(Path(args.photograph).resolve()),
        "views": str(views_dir),
        "per_view": sorted(results, key=lambda r: r["index"]),
        "silhouette_winner": by_iou[0]["index"],
        "silhouette_margin": round(iou_margin, 6),
        "appearance_winner": by_corr[0]["index"],
        "appearance_margin": round(corr_margin, 6),
        "axes_agree": bool(agree),
        "decisive": bool(agree and corr_margin >= MIN_MARGIN),
        "photographed_view": by_corr[0]["index"] if agree else None,
        "classification": ("PROVEN" if (agree and corr_margin >= MIN_MARGIN)
                           else "INCONCLUSIVE"),
    }
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
