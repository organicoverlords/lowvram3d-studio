"""Build the tiny visual-QA regression fixtures and their manifests.

Crops are cut once and stored verbatim so a regression is reproducible without re-rendering.
Nothing here writes to a canonical input; it only reads them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SOURCE_ART = Path(r"C:\Users\Lauri\Downloads\ChatGPT Image 29.7.2026 klo 20.00.45.png")
REJECTED = Path(r"C:\AI\LowVRAM3D-benchmarks\shaman-staff-hole-repair\rejected-20260802-generic-donut\evidence")
OUT = Path(__file__).resolve().parent
CROPS = OUT / "crops"

# Staff head in the concept art: the bone torus at upper left, which already shows a real hole.
SOURCE_BOX = (45, 70, 255, 280)
# The close-up renders are centred on the ring by the repair camera.
FRONT_BOX = (300, 300, 740, 740)
OBLIQUE_BOX = (470, 210, 950, 690)


def crop(source: Path, box, target: Path, size: int = 512) -> Path:
    image = Image.open(source).convert("RGB").crop(box)
    if max(image.size) > size:
        scale = size / float(max(image.size))
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    return target


def synthetic_good_repair(render: Path, box, target: Path, size: int = 512) -> Path:
    """Punch an opening sized to the source proportion into the unchanged baseline crop.

    This is a 2D stand-in for a correctly scaled repair, used only to prove the gate accepts a
    faithful candidate. It is never fed back into the 3D pipeline.
    """
    from PIL import ImageDraw

    image = Image.open(render).convert("RGB").crop(box)
    if max(image.size) > size:
        scale = size / float(max(image.size))
        image = image.resize((round(image.width * scale), round(image.height * scale)),
                             Image.LANCZOS)
    width, height = image.size
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    border = np.concatenate([pixels[0, :, :], pixels[-1, :, :],
                             pixels[:, 0, :], pixels[:, -1, :]])
    background = tuple(int(round(v * 255)) for v in np.median(border, axis=0))

    mask = np.linalg.norm(pixels - np.median(border, axis=0), axis=2) > 0.12
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0

    # Calibrate against the SAME measurement the gate uses. Drawing a nominally 33% ellipse is not
    # enough: it merges with the existing dark recess, so the measured opening comes out larger.
    # Search for the radius whose measured opening fraction matches the source's.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "workers"))
    from deterministic_visual_gate import (  # noqa: E402
        enclosed_opening_diameter, foreground_mask, load_image, subject_diameter,
    )

    source_img = load_image(str(CROPS / "source_staff_head.png"))
    source_mask = foreground_mask(source_img)
    target_fraction = enclosed_opening_diameter(source_mask) / subject_diameter(source_mask)

    best, best_error = None, None
    for radius in np.linspace(4.0, min(width, height) * 0.34, 22):
        trial = image.copy()
        ImageDraw.Draw(trial).ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius], fill=background
        )
        probe = np.asarray(trial, np.float32) / 255.0
        probe_mask = foreground_mask(probe)
        outer = subject_diameter(probe_mask)
        fraction = enclosed_opening_diameter(probe_mask) / outer if outer > 0 else 0.0
        error = abs(fraction - target_fraction)
        if best_error is None or error < best_error:
            best, best_error = trial, error
    print(f"  synthetic good repair: target_fraction={target_fraction:.4f} error={best_error:.4f}")

    target.parent.mkdir(parents=True, exist_ok=True)
    best.save(target)
    return target


def synthetic_collateral(good: Path, target: Path) -> Path:
    """The good repair, plus an obvious change far outside the repair ROI."""
    from PIL import ImageDraw

    image = Image.open(good).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle([int(width * 0.02), int(height * 0.70),
                    int(width * 0.32), int(height * 0.97)], fill=(210, 210, 210))
    image.save(target)
    return target


def main() -> int:
    for path in (SOURCE_ART, REJECTED):
        if not path.exists():
            print(f"MISSING INPUT: {path}", file=sys.stderr)
            return 2
    CROPS.mkdir(parents=True, exist_ok=True)

    source = crop(SOURCE_ART, SOURCE_BOX, CROPS / "source_staff_head.png")
    before_front = crop(REJECTED / "before_staff_front.png", FRONT_BOX, CROPS / "before_front.png")
    after_front = crop(REJECTED / "after_staff_front.png", FRONT_BOX, CROPS / "candidate_front_oversized.png")
    before_obl = crop(REJECTED / "before_staff_oblique.png", OBLIQUE_BOX, CROPS / "before_oblique.png")
    after_obl = crop(REJECTED / "after_staff_oblique.png", OBLIQUE_BOX, CROPS / "candidate_oblique_oversized.png")
    # An unrelated region of the character, for the mismatch control.
    unrelated = crop(REJECTED / "before_full_character.png", (430, 620, 780, 970), CROPS / "unrelated_robe.png")

    # A synthetic *good* repair: the baseline with an opening matching the source proportion.
    # Without a positive fixture the gate could pass its negatives by rejecting everything.
    good = synthetic_good_repair(REJECTED / "before_staff_front.png", FRONT_BOX,
                                 CROPS / "candidate_front_good.png")
    # A collateral-change fixture: the good repair plus damage well outside the repair ROI.
    collateral = synthetic_collateral(CROPS / "candidate_front_good.png",
                                      CROPS / "candidate_front_collateral.png")

    constraints = [
        "the opening must stay inside the original recessed centre",
        "preserve the raised inner lip and layered carving",
        "opening should be roughly 30-40 percent of the outer disc diameter",
        "must not look like a generic machine-cut donut",
    ]

    fixtures = {
        # 1. the real negative: oversized generic hole
        # Front view: an opening is only measurable head-on. The oblique view shows the bore wall,
        # so the hole is not background there and no enclosed region can be found.
        "staff_hole_rejected": {
            "source_crop": str(source),
            "before_crop": str(before_front),
            "candidate_crop": str(after_front),
            "feature_name": "staff ring through-hole",
            "expected_description": (
                "a small organic opening inside the original recess, keeping the raised inner lip"
            ),
            "constraints": constraints,
        },
        # 2. control: identical before/candidate
        "control_no_change": {
            "source_crop": str(source),
            "before_crop": str(before_front),
            "candidate_crop": str(before_front),
            "feature_name": "staff ring through-hole",
            "expected_description": "unchanged baseline; no repair was applied",
            "constraints": constraints,
            # An unchanged candidate is the correct outcome here, not a failed repair, and it
            # must not be reported as collateral damage.
            "require_change": False,
        },
        # 3. control: front view of the same oversized repair (second angle on the negative)
        "control_front_view": {
            "source_crop": str(source),
            "before_crop": str(before_front),
            "candidate_crop": str(after_front),
            "feature_name": "staff ring through-hole",
            "expected_description": (
                "a small organic opening inside the original recess, keeping the raised inner lip"
            ),
            "constraints": constraints,
        },
        # 4. control: unrelated crop mismatch
        "control_unrelated_crop": {
            "source_crop": str(source),
            "before_crop": str(before_front),
            "candidate_crop": str(unrelated),
            "feature_name": "staff ring through-hole",
            "expected_description": "the staff ring, not a piece of robe",
            "constraints": constraints,
        },
        # 5. positive control: a correctly scaled repair must PASS
        "control_good_repair": {
            "source_crop": str(source),
            "before_crop": str(before_front),
            "candidate_crop": str(good),
            "feature_name": "staff ring through-hole",
            "expected_description": "opening matching the source proportion",
            "constraints": constraints,
            "repair_roi": [0.28, 0.28, 0.72, 0.72],
        },
        # 6. collateral-change fixture: good repair, damage outside the ROI
        "control_collateral_change": {
            "source_crop": str(source),
            "before_crop": str(before_front),
            "candidate_crop": str(collateral),
            "feature_name": "staff ring through-hole",
            "expected_description": "opening matching the source, but with damage outside the ROI",
            "constraints": constraints,
            "repair_roi": [0.28, 0.28, 0.72, 0.72],
        },
        # 7. control: missing image on disk
        "control_missing_image": {
            "source_crop": str(source),
            "before_crop": str(before_obl),
            "candidate_crop": str(CROPS / "does_not_exist.png"),
            "feature_name": "staff ring through-hole",
            "expected_description": "candidate image is absent; the judge must not invent a verdict",
            "constraints": constraints,
        },
    }

    for name, manifest in fixtures.items():
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"FIXTURE {name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
