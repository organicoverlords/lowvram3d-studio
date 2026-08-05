"""Correct the global exposure and white balance of generated views.

The six views MV-Adapter produced for the riverboat came back neutral grey --
view 0 is [124.5, 123.7, 121.4], the three channels within three levels of each
other -- while the photograph that conditioned them is warm brown at
[47.5, 44.1, 38.1]. That is not a subtle grade. It is the generator ignoring the
reference's colour and falling back on its own prior.

The fix here is deliberately the smallest one that can be right: a single
per-channel gain in linear light, fitted on the one view whose pose matches the
photograph, and applied identically to all six.

Why linear, and why it matters
------------------------------
Measured in sRGB the gap looks like 2.8x. In linear light it is about 7.8x --
roughly three stops. Fitting the gain on sRGB values would therefore
under-correct the shadows and over-correct the highlights, because sRGB is a
transfer curve and a gain applied through it is not a gain at all. Light
multiplies linearly; the encoding does not.

Why one global gain rather than per-view
----------------------------------------
Each view supplies a different patch of the same UV atlas. Normalising each view
to its own statistics pushes the same surface albedo through a different function
depending on which camera happened to see it, which writes a step into the
texture at every view boundary -- a seam manufactured by the correction itself.
A single gain commutes with all of that: every inter-view and inter-texel ratio
survives it untouched.

That property is also what protects genuine variation. The top view is brighter
because it faces the light and the bottom is darker because it does not; a global
gain cannot flatten that difference, and the receipt checks it did not, by
comparing the top/bottom ratio before and after. If that ratio moves, the
correction was not global and the run should be rejected.

What this cannot do
-------------------
Highlights that were already clipped in the generated views are gone. Scaling
moves the clipped plateau down but recovers no detail inside it, and the fit
itself is biased by those pixels, so they are excluded from the statistics rather
than allowed to poison the gain. Clipping is reported, not silently corrected.

    py -3.12 workers/match_view_colour.py --views DIR --photograph P.png \\
        --matte M.png --reference-view 0 --output DIR_matched
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Pixels at or above this (0-1, sRGB) are treated as clipped and excluded from
#: the fit. 250/255 rather than 255 because the blown regions in these views
#: shoulder off just below the maximum.
CLIP_CEILING = 250.0 / 255.0

#: And at the bottom, where quantisation dominates the linear values.
CLIP_FLOOR = 2.0 / 255.0

#: The top/bottom brightness ratio must survive the correction, because a single
#: global gain cannot change a ratio between two views.
#:
#: The threshold is loose on purpose, and the gap it sits in is enormous. The
#: failure being guarded against is per-view normalisation, which drives every
#: view toward the same mean and so drives this ratio toward 1.0 -- at an
#: observed ratio of 6.6 that is a drift near 0.85. What lands below the
#: threshold is 8-bit quantisation: the correction darkens by 5-8x, and the
#: darkest view's values are then rounded on a coarse grid, which perturbs its
#: mean slightly more than the bright view's. Measured drift is 0.0005 on the
#: real views and 0.02 on a synthetic worst case with a 7.8x gain. Anything in
#: between is quantisation; nothing in between is a per-view transform.
MAX_RATIO_DRIFT = 0.10

#: Below this linear mean, the darkest view has been crushed into so few 8-bit
#: levels that its shadow detail is largely gone. Not a failure -- the gain is
#: still correct -- but it must be visible in the receipt rather than discovered
#: later in a texture.
SHADOW_CRUSH_LINEAR = 0.004


def _srgb_to_linear(x):
    import numpy as np

    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    import numpy as np

    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * (x ** (1 / 2.4)) - 0.055)


def _foreground(view_rgb):
    """Subject mask for a generated view, keyed off its own plate corner."""
    import numpy as np

    plate = view_rgb[2, 2].astype(np.float32)
    return np.linalg.norm(view_rgb.astype(np.float32) - plate, axis=2) > 18.0


def _mean_linear(rgb_uint8, mask):
    """Per-channel linear mean over unclipped foreground pixels."""
    import numpy as np

    linear = _srgb_to_linear(rgb_uint8.astype(np.float64) / 255.0)
    srgb = rgb_uint8.astype(np.float64) / 255.0
    # Excluded per-channel: a pixel blown in red still carries usable green.
    usable = mask[..., None] & (srgb < CLIP_CEILING) & (srgb > CLIP_FLOOR)
    means = []
    for channel in range(3):
        values = linear[..., channel][usable[..., channel]]
        means.append(float(values.mean()) if values.size else float("nan"))
    clipped = float((mask[..., None] & (srgb >= CLIP_CEILING)).sum()
                    / max(mask.sum() * 3, 1))
    return means, clipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", required=True)
    parser.add_argument("--photograph", required=True)
    parser.add_argument("--matte", required=True)
    parser.add_argument("--reference-view", type=int, required=True,
                        help="Index of the view whose pose matches the photograph.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image

    views_dir = Path(args.views).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    view_paths = sorted(views_dir.glob("view_*_*.png"),
                        key=lambda p: int(p.stem.split("_")[1]))
    if not view_paths:
        raise SystemExit(f"NO_VIEWS_FOUND:{views_dir}")

    # The photograph's matte is RGBA and its alpha is the subject. Reading it
    # through convert("L") silently selects the background instead, because the
    # colour channels render as the subject on white -- a mistake that is
    # invisible by eye and produced a 4x wrong luminance earlier in this project.
    matte = Image.open(args.matte)
    if matte.mode != "RGBA":
        raise SystemExit(f"MATTE_NOT_RGBA:{matte.mode}")
    photo_mask = np.asarray(matte)[..., 3] > 127
    photo_rgb = np.asarray(matte.convert("RGB"))

    photo_mean, photo_clipped = _mean_linear(photo_rgb, photo_mask)

    reference_path = next(p for p in view_paths
                          if int(p.stem.split("_")[1]) == args.reference_view)
    reference_rgb = np.asarray(Image.open(reference_path).convert("RGB"))
    reference_mean, reference_clipped = _mean_linear(
        reference_rgb, _foreground(reference_rgb))

    gain = [photo_mean[c] / reference_mean[c] for c in range(3)]

    def _index(path):
        return int(path.stem.split("_")[1])

    # Key each view's subject once, on the original, and reuse that mask for the
    # corrected image. Re-keying after correction would be a measurement bug
    # rather than a measurement: `_foreground` thresholds an absolute colour
    # distance from the plate, and a gain of ~0.2 shrinks every such distance by
    # 5x, so the threshold would then reject most of the subject and the "after"
    # mean would be taken over a biased subset. That alone reported a 23% drift
    # in the top/bottom ratio -- for a transform that, being a single global
    # gain, cannot change a ratio at all.
    masks = {}
    for path in view_paths:
        masks[_index(path)] = _foreground(
            np.asarray(Image.open(path).convert("RGB")))

    def _brightness(path):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        mask = masks[_index(path)]
        linear = _srgb_to_linear(rgb.astype(np.float64) / 255.0)
        return float(linear[mask].mean()) if mask.any() else float("nan")

    before = {_index(p): _brightness(p) for p in view_paths}

    written = []
    for path in view_paths:
        rgb = np.asarray(Image.open(path).convert("RGB"))
        linear = _srgb_to_linear(rgb.astype(np.float64) / 255.0)
        corrected = _linear_to_srgb(linear * np.asarray(gain))
        out_path = output_dir / path.name
        Image.fromarray(np.round(corrected * 255.0).astype(np.uint8)).save(out_path)
        written.append(out_path)

    after = {_index(p): _brightness(p) for p in written}

    # The guard: a global gain cannot change the ratio between two views. If it
    # did, the correction was per-view and genuine lighting variation has been
    # flattened.
    top, bottom = 4, 5
    ratio_before = before.get(top, float("nan")) / max(before.get(bottom, 1e-9), 1e-9)
    ratio_after = after.get(top, float("nan")) / max(after.get(bottom, 1e-9), 1e-9)
    drift = abs(ratio_after - ratio_before) / max(abs(ratio_before), 1e-9)

    result_rgb = np.asarray(Image.open(output_dir / reference_path.name).convert("RGB"))
    result_mean, _ = _mean_linear(result_rgb, masks[args.reference_view])
    achieved_srgb = [round(float(v), 1) for v in
                     (_linear_to_srgb(np.asarray(result_mean)) * 255.0)]
    target_srgb = [round(float(v), 1) for v in
                   (_linear_to_srgb(np.asarray(photo_mean)) * 255.0)]

    darkest = min(after.values()) if after else float("nan")
    crushed = bool(darkest < SHADOW_CRUSH_LINEAR)

    report = {
        "schema_version": "view_colour_match_v1",
        "classification": "PROVEN" if drift <= MAX_RATIO_DRIFT else "REJECTED_NOT_GLOBAL",
        "darkest_view_mean_linear_after": round(float(darkest), 6),
        "shadow_crush_warning": crushed,
        "shadow_crush_note": (
            "the darkest view now occupies very few 8-bit levels; the gain is "
            "correct but its shadow detail is largely quantised away"
            if crushed else "no"),
        "views": str(views_dir),
        "output": str(output_dir),
        "reference_view": args.reference_view,
        "space": "linear light, sRGB EOTF applied and inverted",
        "photograph_mean_linear": [round(v, 6) for v in photo_mean],
        "reference_view_mean_linear": [round(v, 6) for v in reference_mean],
        "gain_per_channel": [round(v, 6) for v in gain],
        "gain_sRGB_equivalent_ratio": round(
            float(np.mean(reference_mean) / max(np.mean(photo_mean), 1e-9)), 3),
        "achieved_reference_mean_srgb": achieved_srgb,
        "target_photograph_mean_srgb": target_srgb,
        "photograph_clipped_fraction": round(photo_clipped, 6),
        "reference_view_clipped_fraction": round(reference_clipped, 6),
        "top_bottom_ratio_before": round(ratio_before, 6),
        "top_bottom_ratio_after": round(ratio_after, 6),
        "top_bottom_ratio_drift": round(drift, 6),
        "genuine_variation_preserved": bool(drift <= MAX_RATIO_DRIFT),
        "clipped_highlights_are_unrecoverable": True,
    }
    report_path = Path(args.report) if args.report else output_dir / "colour_match.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if drift <= MAX_RATIO_DRIFT else 2


if __name__ == "__main__":
    raise SystemExit(main())
