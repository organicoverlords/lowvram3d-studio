"""Gate a reconstruction against the source illustration's shape.

This is the check whose absence let a wrong head through every stage. GEOMETRY_QA measured axis
ratio, detached shards and thin-feature survival - all properties of the mesh on its own. None of
them asks the only question that matters after generation: does this actually look like the thing in
the picture? A smooth rounded lump where the source has a narrow bird skull satisfies every
self-referential check perfectly, and the error was not caught until a human looked at a texture
render eight stages later and said the face was the wrong shape.

Measures silhouette agreement from the resolved front direction, overall and in horizontal bands, so
a body that fits while the head does not is reported separately rather than averaged away.

DIAGNOSTIC ONLY - NOT WIRED AS A GATE. The default thresholds below are placeholders, not calibrated
values, and must not be treated as validated. Calibrating them needs source/model pairs that are
independent of the asset being judged; on the machine this was written for, 29 of the 30 recoverable
pairs were the very model under suspicion, and a threshold fitted to that corpus would simply
reproduce the verdict it was supposed to test. Until independent pairs exist this reports numbers
and nothing blocks on them.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from mesh_io import read_glb
from shaman_texture_views import mask_iou, project, rasterise, subject_bbox, warp_to_frame

SOURCE_SILHOUETTE_MISMATCH = "SOURCE_SILHOUETTE_MISMATCH"
# Deliberately "upper" rather than "face". The top band of this subject is mostly antler bar and
# hanging cords, where a pixel of misregistration wrecks IoU, so this number is evidence about the
# upper silhouette as a whole and must not be quoted as a measurement of facial shape.
UPPER_SHAPE_MISMATCH = "UPPER_SHAPE_MISMATCH"


def band_iou(mesh_mask: np.ndarray, source_mask: np.ndarray, lo: float, hi: float) -> float:
    """Silhouette agreement restricted to a horizontal band of the subject, top-relative."""
    rows = np.nonzero(mesh_mask.any(axis=1) | source_mask.any(axis=1))[0]
    if rows.size == 0:
        return 0.0
    top, bottom = int(rows.min()), int(rows.max())
    height = max(bottom - top + 1, 1)
    y0 = top + int(round(lo * height))
    y1 = top + int(round(hi * height))
    if y1 <= y0:
        return 0.0
    return mask_iou(mesh_mask[y0:y1], source_mask[y0:y1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--debug-image", default="")
    parser.add_argument("--raster-size", type=int, default=1024)
    parser.add_argument("--front-direction",
                        choices=("+z", "-z"),
                        default=os.environ.get("LOWVRAM3D_FRONT_DIRECTION", "+z").lower())
    parser.add_argument("--min-overall-iou", type=float, default=0.72)
    parser.add_argument("--min-head-iou", type=float, default=0.60)
    parser.add_argument("--head-band", type=float, default=0.26,
                        help="fraction of subject height, measured from the top, treated as head")
    args = parser.parse_args()

    positions, _, _, tris = read_glb(Path(args.mesh))
    positions = positions.astype(np.float64)
    centre = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
    verts = positions - centre
    ortho = float((verts.max(axis=0) - verts.min(axis=0)).max())

    source = cv2.imread(args.source, cv2.IMREAD_UNCHANGED)
    if source is None:
        raise RuntimeError(f"could not read {args.source}")
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGRA)
    if source.ndim == 3 and source.shape[2] == 4:
        source_mask = source[:, :, 3] > 127
    else:
        # Key the background from the border, not from a brightness threshold. The illustration sits
        # on a mid-olive backdrop, so `min(rgb) < 245` selects the entire frame - which silently
        # turns every silhouette comparison into a comparison against a filled rectangle and reports
        # a confident, meaningless number.
        from lowvram3d.asset_profiles import foreground_mask
        source_mask = foreground_mask(source)
    if source_mask.mean() > 0.92:
        raise RuntimeError("source mask covers almost the whole frame; background keying failed")

    direction = np.array([0.0, 0.0, 1.0]) if args.front_direction == "+z" else np.array([0.0, 0.0, -1.0])
    screen, depth = project(verts, direction, ortho)
    _, silhouette = rasterise(screen, depth, tris, args.raster_size)

    warped = warp_to_frame(
        source_mask.astype(np.uint8) * 255,
        subject_bbox(source_mask),
        subject_bbox(silhouette),
        args.raster_size,
        interpolation=cv2.INTER_NEAREST,
    ) > 127

    # Refine the fit before measuring. Corner-to-corner bounding-box alignment is set by whatever
    # reaches furthest - here the staff and the outermost hanging charms - so a mesh whose props sit
    # slightly wider is scaled down bodily, and every silhouette then disagrees by a constant margin
    # that has nothing to do with shape. Measured on this asset the unrefined fit scored 0.53 for
    # every candidate and every seed, a spread of 0.003, which is a registration floor being
    # reported as if it were reconstruction quality.
    best = (mask_iou(warped, silhouette), 1.0, 0, 0)
    source_uint = warped.astype(np.uint8)
    for scale_step in np.arange(0.86, 1.15, 0.02):
        matrix = cv2.getRotationMatrix2D(
            (args.raster_size / 2.0, args.raster_size / 2.0), 0.0, float(scale_step))
        scaled = cv2.warpAffine(source_uint, matrix, (args.raster_size, args.raster_size),
                                flags=cv2.INTER_NEAREST)
        for dy in range(-40, 41, 8):
            for dx in range(-40, 41, 8):
                shifted = np.roll(np.roll(scaled, dy, axis=0), dx, axis=1)
                score = mask_iou(shifted > 0, silhouette)
                if score > best[0]:
                    best = (score, float(scale_step), dy, dx)

    _, best_scale, best_dy, best_dx = best
    matrix = cv2.getRotationMatrix2D(
        (args.raster_size / 2.0, args.raster_size / 2.0), 0.0, best_scale)
    aligned = cv2.warpAffine(source_uint, matrix, (args.raster_size, args.raster_size),
                             flags=cv2.INTER_NEAREST)
    warped = np.roll(np.roll(aligned, best_dy, axis=0), best_dx, axis=1) > 0

    overall = mask_iou(warped, silhouette)
    upper = band_iou(silhouette, warped, 0.0, args.head_band)
    torso = band_iou(silhouette, warped, args.head_band, 0.65)
    lower = band_iou(silhouette, warped, 0.65, 1.0)

    if args.debug_image:
        overlay = np.zeros((args.raster_size, args.raster_size, 3), np.uint8)
        overlay[..., 1] = silhouette.astype(np.uint8) * 255      # mesh in green
        overlay[..., 2] = warped.astype(np.uint8) * 255          # source in red
        Path(args.debug_image).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.debug_image, overlay)

    failure_codes = []
    if overall < args.min_overall_iou:
        failure_codes.append(SOURCE_SILHOUETTE_MISMATCH)
    if upper < args.min_head_iou:
        failure_codes.append(UPPER_SHAPE_MISMATCH)

    report = {
        "mesh": args.mesh,
        "source": args.source,
        "front_direction": args.front_direction,
        "silhouette_iou": {
            "overall": round(float(overall), 5),
            "upper": round(float(upper), 5),
            "torso": round(float(torso), 5),
            "lower": round(float(lower), 5),
        },
        "score": round(float(0.6 * overall + 0.4 * upper), 5),
        "registration": {"scale": round(best_scale, 4), "shift_y": best_dy, "shift_x": best_dx,
                         "refined": True},
        "thresholds": {"overall": args.min_overall_iou, "head": args.min_head_iou,
                       "head_band_fraction": args.head_band},
        "debug_image": args.debug_image or None,
        "failure_codes": failure_codes,
        "passed": not failure_codes,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SOURCE_FIDELITY passed={report['passed']} overall={overall:.4f} upper={upper:.4f} "
          f"torso={torso:.4f} lower={lower:.4f} score={report['score']:.4f} "
          f"codes={failure_codes}", flush=True)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
