"""Landmark-independent test of which camera actually faces the character.

Naming a camera "front" does not make it one.  A bilaterally symmetric character has one
sagittal plane, so only the two views looking along its facing axis produce a silhouette
that mirrors about the vertical image axis; every side view does not.  That test needs no
texture, no labels and no hand-picked landmarks, so it can contradict a declared contract.

It resolves the facing *axis*, not its sign.  Distinguishing muzzle from tail needs a
landmark, so this worker reports the axis and refuses to invent the rest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def mirror_scores(mask: np.ndarray) -> dict:
    """Intersection-over-union of the mask with its flip about each image axis."""
    def iou(a: np.ndarray, b: np.ndarray) -> float:
        union = np.count_nonzero(a | b)
        return float(np.count_nonzero(a & b) / union) if union else 0.0

    # Centre the mask first: an off-centre subject scores badly for reasons that have
    # nothing to do with symmetry.
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return {"vertical_axis_mirror_iou": 0.0, "horizontal_axis_mirror_iou": 0.0}
    centred = np.roll(mask, (int(mask.shape[0] // 2 - ys.mean()),
                             int(mask.shape[1] // 2 - xs.mean())), axis=(0, 1))
    return {
        "vertical_axis_mirror_iou": iou(centred, centred[:, ::-1]),
        "horizontal_axis_mirror_iou": iou(centred, centred[::-1, :]),
    }


def file_prefix(view: dict) -> str:
    """On-disk array prefix, which stops being the semantic label once a bundle is relabelled."""
    return str(view.get("control_file_prefix") or view["semantic_name"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sagittal-margin", type=float, default=0.04,
                        help="How far the facing pair must lead the rest to count as proven.")
    args = parser.parse_args()

    bundle = Path(args.bundle)
    contract = json.loads((bundle / "camera_contract.json").read_text(encoding="utf-8"))

    views = []
    for view in sorted(contract["views"], key=lambda item: int(item["index"])):
        prefix = file_prefix(view)
        mask = np.asarray(Image.open(bundle / f"{prefix}_mask.png").convert("L")) > 127
        depth = np.load(bundle / f"{prefix}_depth.npy")
        finite = mask & np.isfinite(depth)
        scores = mirror_scores(mask)
        views.append({
            "raw_index": int(view["index"]),
            "bundle_prefix": prefix,
            "declared_semantic": view.get("proven_semantic"),
            "azimuth_deg": view["azimuth_deg"],
            "elevation_deg": view["elevation_deg"],
            "camera_direction_control_space": view["camera_direction"],
            "camera_up_control_space": view["camera_up"],
            "foreground_pixels": int(mask.sum()),
            "nearest_depth": float(depth[finite].min()) if finite.any() else None,
            **scores,
        })

    ranked = sorted(views, key=lambda item: item["vertical_axis_mirror_iou"], reverse=True)
    facing_pair = ranked[:2]
    margin = (facing_pair[1]["vertical_axis_mirror_iou"]
              - ranked[2]["vertical_axis_mirror_iou"]) if len(ranked) > 2 else 0.0
    opposed = np.allclose(
        np.asarray(facing_pair[0]["camera_direction_control_space"], dtype=float),
        -np.asarray(facing_pair[1]["camera_direction_control_space"], dtype=float), atol=1e-3)
    axis_proven = bool(margin >= args.sagittal_margin and opposed)

    declared_facing = {item["raw_index"] for item in views
                       if str(item["declared_semantic"]) in ("front", "rear")}
    measured_facing = {item["raw_index"] for item in facing_pair}
    contract_agrees = declared_facing == measured_facing

    report = {
        "schema": "camera_semantics_symmetry_v1",
        "bundle": str(bundle),
        "method": "bilateral silhouette symmetry about the vertical image axis",
        "views": views,
        "measured_facing_axis_raw_indices": sorted(measured_facing),
        "measured_facing_axis_direction": facing_pair[0]["camera_direction_control_space"],
        "facing_pair_are_opposed": bool(opposed),
        "symmetry_margin_over_next_view": float(margin),
        "facing_axis_proven": axis_proven,
        "declared_front_rear_raw_indices": sorted(declared_facing),
        "declared_contract_agrees_with_measurement": bool(contract_agrees),
        "facing_sign_resolved": False,
        "facing_sign_note": (
            "Symmetry fixes the axis, not which end is the muzzle. Front vs rear needs a "
            "landmark call, which this worker deliberately does not guess."),
        "classification": (
            "PANDA_CAMERA_SEMANTICS_PROVEN" if axis_proven and contract_agrees
            else "PANDA_CAMERA_SEMANTICS_USER_REVIEW_REQUIRED"),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CAMERA_SEMANTICS {report['classification']} "
          f"facing_axis_raw={sorted(measured_facing)} margin={margin:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
