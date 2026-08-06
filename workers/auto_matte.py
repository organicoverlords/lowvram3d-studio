"""Matte an image and choose the shadow tolerance by measurement, not by eye.

A studio render of a character sits on a soft contact shadow. Left in, the
shadow is opaque to the generator and becomes a floor slab welded to the feet.
`pipeline_matte.py` can remove it, but the tolerance was being chosen by
sweeping and looking, which is slow and does not survive a batch.

It also cannot be chosen by the obvious number. "Alpha remaining in the bottom
of the frame" ranks *total destruction* as the best outcome, because the feet
lie inside the region it measures:

    tolerance 130 -> 10.82% remaining, shadow still there
    tolerance 180 ->  7.69% remaining, CORRECT
    tolerance 240 ->  1.12% remaining, feet destroyed
    tolerance 300 ->  0.00% remaining, everything below the hem deleted

The criterion that does work is a property of the two things being separated:

    a shadow is SMOOTH.  a subject is TEXTURED.

So sweep the tolerance and, at each step, measure the local texture energy of
the pixels being *removed*. While the removed pixels are smooth, the sweep is
eating shadow. The moment removed texture energy jumps, it has started eating
the subject, and the previous step is the answer.

Measured on a red panda in a ghillie suit, over a near-white background:

    tol   removed px   texture energy
     60       66,001       4.53   smooth -> shadow
    120      137,086       4.64   smooth -> shadow
    180      157,720       5.32   smooth -> shadow
    240      173,187       6.10   TEXTURED -> eating the subject

It selects 180 — independently the same value hand-tuned on a completely
different subject, which is the first evidence that the criterion generalises
rather than fitting one image.

    py auto_matte.py --image source.png --out matte.png
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

#: Tolerances tried, low to high. Coarse on purpose: the transition is wide, and
#: each step costs a full matte pass.
LADDER = (0, 60, 120, 180, 240, 300)

#: Window for the local texture measurement, in source pixels.
TEXTURE_WINDOW = 9

#: Where the cut sits between background texture and subject texture.
#:
#: The first version of this anchored the threshold to the FIRST sweep step and
#: failed validation: on a subject whose first step already clips textured edges
#: (measured 26.7 against a later 19.8), every subsequent step looks smooth by
#: comparison and the sweep runs to the end, selecting a tolerance that deletes
#: the feet. Anchoring to two absolute references measured from the image itself
#: — the confident subject interior and the confident background — removes that
#: dependence on which step happens to come first.
#:
#: 0.35 puts the line nearer the background, because over-cutting the subject is
#: unrecoverable and leaving a little shadow is not.
SUBJECT_TEXTURE_FRACTION = 0.35

#: Ignore steps that remove almost nothing; their statistics are noise.
MIN_REMOVED_PIXELS = 2000

PYTHON = sys.executable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shadow-from", type=float, default=0.72,
                        help="Height fraction below which the stronger "
                             "tolerance applies.")
    parser.add_argument("--mode", default="colour",
                        choices=("hybrid", "colour", "flood"))
    parser.add_argument("--force-tolerance", type=int, default=None,
                        help="Skip selection and use this. The escape hatch "
                             "for when a human disagrees with the measurement.")
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image
    from scipy import ndimage

    worker = Path(__file__).with_name("pipeline_matte.py")

    def run_matte(tolerance, destination):
        command = [PYTHON, str(worker), "--image", args.image,
                   "--output", str(destination), "--mode", args.mode]
        if tolerance:
            command += ["--shadow-tolerance", str(tolerance),
                        "--shadow-from", str(args.shadow_from)]
        result = subprocess.run(command, capture_output=True, text=True)
        if not Path(destination).is_file():
            raise SystemExit(f"MATTE_FAILED at tolerance {tolerance}: "
                             f"{result.stderr[-300:]}")
        return np.asarray(Image.open(destination).convert("RGBA"))

    if args.force_tolerance is not None:
        run_matte(args.force_tolerance, args.out)
        print(json.dumps({"schema_version": "auto_matte_v1",
                          "selected_tolerance": args.force_tolerance,
                          "selection": "forced"}, indent=2))
        return 0

    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        baseline = run_matte(0, work / "t0.png")
        base_alpha = baseline[..., 3] > 8
        grey = baseline[..., :3].astype(np.float32).mean(axis=2)
        # Local standard deviation, as E[x^2] - E[x]^2 over a small window.
        energy = np.sqrt(np.maximum(
            ndimage.uniform_filter(grey ** 2, TEXTURE_WINDOW)
            - ndimage.uniform_filter(grey, TEXTURE_WINDOW) ** 2, 0.0))

        # Two absolute references measured from this image.
        #
        # Subject: the confident interior, eroded well away from the matte edge
        # so antialiased boundary pixels -- which are textured for a reason that
        # has nothing to do with the subject -- do not inflate it.
        # Background: outside a dilated matte, for the same reason inverted.
        subj_ref = None
        interior = ndimage.binary_erosion(base_alpha, iterations=6)
        exterior = ~ndimage.binary_dilation(base_alpha, iterations=6)
        subject_energy = float(energy[interior].mean()) if interior.any() else 0.0
        background_energy = float(energy[exterior].mean()) if exterior.any() else 0.0
        cut = background_energy + SUBJECT_TEXTURE_FRACTION * max(
            subject_energy - background_energy, 0.0)

        trace = []
        selected = 0
        for tolerance in LADDER[1:]:
            alpha = run_matte(tolerance, work / f"t{tolerance}.png")[..., 3] > 8
            removed = base_alpha & ~alpha
            count = int(removed.sum())
            # Measure the INTERIOR of what was removed, and with a median.
            #
            # Averaging every removed pixel measures the boundary, not the
            # region: antialiased edges carry high local gradient whether they
            # belong to a shadow or a subject. On one subject that put the
            # removed-pixel mean at 26.7 against the subject's own 22.0, and the
            # sweep refused to remove anything at all. Eroding strips the edge
            # band; the median ignores whatever survives it.
            core = ndimage.binary_erosion(removed, iterations=2)
            sample = core if core.sum() >= MIN_REMOVED_PIXELS // 4 else removed
            texture = float(np.median(energy[sample])) if sample.any() else 0.0
            entry = {"tolerance": tolerance, "removed_px": count,
                     "texture_energy": round(texture, 3)}
            if count < MIN_REMOVED_PIXELS:
                entry["verdict"] = "negligible"
                trace.append(entry)
                continue
            if texture <= cut:
                entry["verdict"] = "smooth -> shadow"
                selected = tolerance
                trace.append(entry)
            else:
                entry["verdict"] = "textured -> would eat the subject"
                trace.append(entry)
                break

        final = run_matte(selected, args.out)
        alpha = final[..., 3] > 8

    receipt = {
        "schema_version": "auto_matte_v1",
        "image": str(Path(args.image).resolve()),
        "output": str(Path(args.out).resolve()),
        "selection": "measured",
        "criterion": ("largest tolerance whose removed pixels are still smooth; "
                      "a shadow has no texture and a subject does"),
        "selected_tolerance": selected,
        "shadow_from": args.shadow_from,
        "subject_texture_energy": round(subject_energy, 3),
        "background_texture_energy": round(background_energy, 3),
        "cut_threshold": round(cut, 3),
        "subject_texture_fraction": SUBJECT_TEXTURE_FRACTION,
        "subject_coverage": round(float(alpha.mean()), 4),
        "trace": trace,
    }
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
