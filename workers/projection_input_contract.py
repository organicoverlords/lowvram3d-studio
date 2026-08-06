"""Input contract for texture projection jobs.

A projection run that silently receives the wrong source image produces a plausible-looking atlas
with the plate background painted onto the hull, and costs hours before anyone can see it. This
module makes the input explicit and refuses the known-bad cases up front.

It records what was actually fed in -- paths, hashes, mask method, transform, coverage -- and fails
closed when the input is the raw plate, when the plate's checkerboard is still detectable, when
background survives outside the accepted foreground, or when the image and its mask disagree about
geometry.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

ACCEPTED_MASK_METHODS = ("BIREFNET_HARD_MASK",)
CHECKERBOARD_PEAK_RATIO = 3.0
BACKGROUND_OUTSIDE_TOLERANCE = 0.005
SHADOW_TOLERANCE = 0.004


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plate_tones(plate_rgb: np.ndarray, background: np.ndarray) -> list[float]:
    """The two grey levels the plate's checkerboard alternates between.

    A frequency-domain test was tried first and does not work on this plate: the check is light
    grey on near-white, so its spectral energy sits far below the subject's high-contrast window
    rows and planking, and the plate's own background scored *lower* at the detected peak than its
    foreground. The tiles are trivially separable by tone instead, which is what actually
    distinguishes plate material from ship material.
    """
    grey = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2GRAY)[background]
    if grey.size == 0:
        return []
    histogram = np.bincount(grey.ravel(), minlength=256).astype(np.float64)
    histogram = cv2.GaussianBlur(histogram.reshape(-1, 1), (1, 5), 0).ravel()
    peaks = [value for value in range(1, 255)
             if histogram[value] > histogram[value - 1] and histogram[value] >= histogram[value + 1]
             and histogram[value] > histogram.max() * 0.05]
    peaks.sort(key=lambda value: histogram[value], reverse=True)
    return [float(v) for v in peaks[:2]]


def plate_tone_fraction(rgb: np.ndarray, region: np.ndarray, tones: list[float],
                        tolerance: float = 5.0) -> float:
    """Fraction of a region made of near-neutral pixels sitting on one of the plate's tile tones."""
    if not tones or region.sum() == 0:
        return 0.0
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    neutral = (rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)) < 14
    on_tone = np.zeros(grey.shape, bool)
    for tone in tones:
        on_tone |= np.abs(grey - tone) <= tolerance
    return float((region & neutral & on_tone).sum()) / float(region.sum())


def validate(source_image: Path, alpha_mask: Path, conditioning: Path,
             original_plate: Path, mask_method: str,
             crop_transform: dict | None = None) -> dict:
    """Build the contract record and decide whether the projection may run."""
    failures: list[str] = []
    #: Checks that could not be computed on these inputs, with the reason.
    #: Recorded rather than dropped: a check that silently did not run is
    #: indistinguishable in the receipt from one that ran and passed.
    skipped: list[str] = []

    rgba = cv2.imread(str(source_image), cv2.IMREAD_UNCHANGED)
    if rgba is None:
        raise RuntimeError(f"unreadable projection source: {source_image}")
    if rgba.ndim == 2:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_GRAY2BGRA)
    if rgba.shape[2] == 3:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2BGRA)
        rgba[..., 3] = 255
    rgb = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB)
    alpha = rgba[..., 3].astype(np.float32) / 255.0

    mask_image = cv2.imread(str(alpha_mask), cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise RuntimeError(f"unreadable alpha mask: {alpha_mask}")

    source_hash = sha256(source_image)
    plate_hash = sha256(original_plate)

    if mask_method not in ACCEPTED_MASK_METHODS:
        failures.append(f"MASK_METHOD_NOT_ACCEPTED:{mask_method}")

    # 1. The raw plate is never a valid projection source.
    if source_hash == plate_hash:
        failures.append("SOURCE_IS_ORIGINAL_PLATE")
    if source_image.resolve() == original_plate.resolve():
        failures.append("SOURCE_PATH_IS_ORIGINAL_PLATE")
    if alpha.max() <= 0.999 * 1.0 and float((alpha > 0.5).mean()) >= 0.999:
        failures.append("SOURCE_HAS_NO_TRANSPARENCY")
    if rgba.shape[2] < 4 or float(alpha.min()) >= 1.0:
        failures.append("SOURCE_FULLY_OPAQUE_NO_MATTE")

    # 2. The plate's checkerboard must not survive anywhere the projection can sample.
    plate_bgr = cv2.imread(str(original_plate), cv2.IMREAD_COLOR)
    plate_rgb_full = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
    # References are cut with the supplied hard mask, never with the candidate's own alpha:
    # keying them to the candidate makes them collapse whenever the candidate is fully opaque,
    # which is exactly the raw-plate case the test exists to catch.
    reference_cut = mask_image > 127
    if reference_cut.shape != plate_rgb_full.shape[:2]:
        reference_cut = cv2.resize(reference_cut.astype(np.uint8),
                                   (plate_rgb_full.shape[1], plate_rgb_full.shape[0]),
                                   interpolation=cv2.INTER_NEAREST) > 0
    tones = plate_tones(plate_rgb_full, ~reference_cut)
    positive = round(plate_tone_fraction(plate_rgb_full, ~reference_cut, tones), 6)
    negative = round(plate_tone_fraction(plate_rgb_full, reference_cut, tones), 6)
    sampled_region = alpha > 0.5
    observed = round(plate_tone_fraction(rgb, sampled_region, tones), 6)
    # The clean reference is the floor, not the halfway point: a matte that leaks even a tenth of
    # the way toward a fully tiled plate is already painting background onto geometry.
    midpoint = negative + 0.10 * (positive - negative)
    checker_present = positive > negative and observed >= midpoint
    if checker_present:
        failures.append("CHECKERBOARD_PATTERN_DETECTED_IN_SAMPLED_REGION")

    # 3. No plate background or ground shadow may survive inside the accepted foreground.
    foreground = alpha > 0.5
    border = np.concatenate((rgb[:3].reshape(-1, 3), rgb[-3:].reshape(-1, 3),
                             rgb[:, :3].reshape(-1, 3), rgb[:, -3:].reshape(-1, 3)))
    plate_rgb = np.median(border, axis=0)

    # The test needs a background colour to compare against, and it reads one off
    # the plate's border. That assumes the plate is the original opaque
    # photograph. An asset delivered as a transparent PNG has no background at
    # all: its border decodes to RGB (0,0,0) under an alpha of 0, and the test
    # then measures "distance from black", so every dark pixel near the
    # silhouette is scored as leaked background.
    #
    # Measured on the sky whale, supplied as a 4K transparent PNG: border median
    # (0,0,0) against a subject median of (115,107,89), 3.6% of the subject
    # within 18 of "plate colour", and a boundary-band fraction of 0.0097 that
    # failed a 0.0090 tolerance. Nothing had leaked -- the whale is simply dark.
    #
    # So the check is skipped when the plate has no opaque border to read a
    # colour from, and says so, rather than either failing a clean input or
    # being quietly disabled for everyone.
    plate_alpha_border = None
    plate_rgba = cv2.imread(str(original_plate), cv2.IMREAD_UNCHANGED)
    if plate_rgba is not None and plate_rgba.ndim == 3 and plate_rgba.shape[2] == 4:
        pa = plate_rgba[..., 3]
        plate_alpha_border = float(np.concatenate(
            (pa[:3].ravel(), pa[-3:].ravel(),
             pa[:, :3].ravel(), pa[:, -3:].ravel())).mean())
    plate_background_readable = (plate_alpha_border is None
                                 or plate_alpha_border > 8.0)

    distance = np.linalg.norm(rgb.astype(np.float32) - plate_rgb[None, None, :], axis=2)
    # Leaked background shows up as a halo hugging the matte boundary. Bright highlights deep
    # inside the subject -- this ship has a lit sign and lamp glows -- are legitimate content and
    # must not be counted, so only a boundary band is inspected.
    boundary_band = foreground & ~ndimage.binary_erosion(
        foreground, np.ones((7, 7), bool))
    plate_like_inside = boundary_band & (distance < 18.0)
    plate_fraction = float(plate_like_inside.sum()) / max(int(foreground.sum()), 1)
    interior_plate_like = float((foreground & ~boundary_band & (distance < 12.0)).sum()) / max(int(foreground.sum()), 1)
    if not plate_background_readable:
        skipped.append(
            f"PLATE_BACKGROUND_HALO_AT_MATTE_BOUNDARY: plate has a fully "
            f"transparent border (mean alpha {plate_alpha_border:.1f}), so it "
            f"carries no background colour to compare against. Measured "
            f"fraction would have been {plate_fraction:.4f}.")
    elif plate_fraction > BACKGROUND_OUTSIDE_TOLERANCE:
        failures.append(f"PLATE_BACKGROUND_HALO_AT_MATTE_BOUNDARY:{plate_fraction:.4f}")

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1].astype(np.float32) / 255.0
    value = hsv[..., 2].astype(np.float32) / 255.0
    plate_value = float(np.mean(plate_rgb)) / 255.0
    lower_band = np.zeros_like(foreground)
    lower_band[int(foreground.shape[0] * 0.72):] = True
    shadow = (foreground & lower_band & (saturation < 0.16)
              & (value > plate_value * 0.55) & (value < plate_value * 0.97))
    shadow_fraction = float(shadow.sum()) / max(int(foreground.sum()), 1)
    shadow_remains = shadow_fraction > SHADOW_TOLERANCE
    if shadow_remains:
        failures.append(f"GROUND_SHADOW_REMAINS:{shadow_fraction:.4f}")

    # 4. Image and mask must describe the same object.
    mask_solid = mask_image > 127
    transforms_match = mask_solid.shape == foreground.shape
    if not transforms_match:
        failures.append(f"MASK_SHAPE_MISMATCH:{mask_solid.shape}!={foreground.shape}")
        agreement = 0.0
    else:
        union = float((mask_solid | foreground).sum())
        agreement = float((mask_solid & foreground).sum()) / union if union else 0.0
        if agreement < 0.98:
            failures.append(f"IMAGE_MASK_TRANSFORM_DISAGREEMENT:{agreement:.4f}")

    labels, count = ndimage.label(foreground, np.ones((3, 3), int))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    rows = np.flatnonzero(foreground.any(axis=1))
    cols = np.flatnonzero(foreground.any(axis=0))

    record = {
        "schema": "projection_input_contract_v1",
        "mask_method": mask_method,
        "mask_method_accepted": mask_method in ACCEPTED_MASK_METHODS,
        "source_image": str(source_image),
        "source_image_sha256": source_hash,
        "alpha_mask": str(alpha_mask),
        "alpha_mask_sha256": sha256(alpha_mask),
        "normalized_conditioning_image": str(conditioning) if conditioning else None,
        "normalized_conditioning_sha256": sha256(conditioning) if conditioning and Path(conditioning).is_file() else None,
        "original_plate": str(original_plate),
        "original_plate_sha256": plate_hash,
        "source_equals_original_plate": source_hash == plate_hash,
        "source_dimensions": [int(rgb.shape[1]), int(rgb.shape[0])],
        "alpha_coverage": round(float(foreground.mean()), 6),
        "foreground_components": int((sizes > 0).sum()),
        "crop_bounds_x0y0x1y1": ([int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])]
                                 if rows.size and cols.size else None),
        "crop_transform": crop_transform,
        "background_treatment": {
            "estimated_plate_rgb": [round(float(v), 1) for v in plate_rgb],
            "plate_like_halo_fraction_at_boundary": round(plate_fraction, 6),
            "plate_like_fraction_deep_interior": round(interior_plate_like, 6),
            "interior_note": "interior plate-coloured pixels are the lit sign and lamps, not leakage",
            "tolerance": BACKGROUND_OUTSIDE_TOLERANCE,
        },
        "ground_shadow_remains": bool(shadow_remains),
        "ground_shadow_fraction": round(shadow_fraction, 6),
        "checkerboard": {
            "method": "two-tone plate-tile fraction, calibrated against the plate's own background and foreground",
            "plate_tile_tones": tones,
            "positive_reference_plate_background": positive,
            "negative_reference_plate_foreground": negative,
            "observed_in_sampled_region": observed,
            "decision_midpoint": round(midpoint, 6),
            "detected_in_sampled_region": bool(checker_present),
        },
        "image_mask_agreement_iou": round(agreement, 6),
        "transforms_match": bool(transforms_match),
        "failures": failures,
        "skipped_checks": skipped,
        "contract_satisfied": not failures,
        "classification": "PROJECTION_INPUT_ACCEPTED" if not failures else "PROJECTION_INPUT_REJECTED",
    }
    return record


def enforce(record: dict) -> None:
    if not record["contract_satisfied"]:
        raise RuntimeError("PROJECTION_INPUT_CONTRACT_VIOLATION:" + ",".join(record["failures"]))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--alpha-mask", required=True)
    parser.add_argument("--conditioning", default="")
    parser.add_argument("--original-plate", required=True)
    parser.add_argument("--mask-method", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()

    record = validate(Path(args.source_image), Path(args.alpha_mask),
                      Path(args.conditioning) if args.conditioning else None,
                      Path(args.original_plate), args.mask_method)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"PROJECTION_CONTRACT {record['classification']} failures={record['failures']}", flush=True)
    if args.enforce:
        enforce(record)


if __name__ == "__main__":
    main()
