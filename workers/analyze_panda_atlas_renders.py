"""Analyze six transparent PNG renders and build one labeled contact sheet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


SEMANTICS = ("front", "left", "right", "rear", "top", "bottom")


def classify_pixels(image: np.ndarray) -> dict:
    if image is None or image.ndim != 3 or image.shape[2] < 4:
        raise ValueError("render must be an RGBA image")
    bgra = image
    bgr = bgra[..., :3]
    alpha = bgra[..., 3]
    opaque = alpha > 16
    b, g, r = [bgr[..., index] for index in range(3)]
    magenta = opaque & (r >= 230) & (g <= 35) & (b >= 230)
    black = opaque & (r <= 20) & (g <= 20) & (b <= 20)
    white = opaque & (r >= 235) & (g >= 235) & (b >= 235)
    return {
        "opaque_pixels": int(opaque.sum()),
        "debug_magenta_pixels": int(magenta.sum()),
        "unexpected_black_pixels": int(black.sum()),
        "near_white_pixels": int(white.sum()),
        "debug_magenta_fraction": float(magenta.sum() / max(int(opaque.sum()), 1)),
        "unexpected_black_fraction": float(black.sum() / max(int(opaque.sum()), 1)),
        "magenta_mask": magenta,
        "black_mask": black,
    }


def composite_on_dark(image: np.ndarray) -> np.ndarray:
    bgr = image[..., :3].astype(np.float32)
    alpha = image[..., 3:4].astype(np.float32) / 255.0
    background = np.full_like(bgr, 20.0)
    return np.clip(bgr * alpha + background * (1.0 - alpha), 0, 255).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-contact-sheet", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--kind", choices=("synthetic", "orientation", "real"), default="synthetic")
    args = parser.parse_args()

    render_dir = Path(args.render_dir)
    report_path = Path(args.output_report)
    contact_path = Path(args.output_contact_sheet)
    records = []
    panels = []
    total_magenta = 0
    total_black = 0
    total_opaque = 0

    for semantic in SEMANTICS:
        path = render_dir / f"{args.prefix}_{semantic}.png"
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise SystemExit(f"RENDER_MISSING:{path}")
        if image.shape[2] == 3:
            alpha = np.full(image.shape[:2] + (1,), 255, np.uint8)
            image = np.concatenate([image, alpha], axis=2)
        stats = classify_pixels(image)
        magenta = stats.pop("magenta_mask")
        black = stats.pop("black_mask")
        total_magenta += stats["debug_magenta_pixels"]
        total_black += stats["unexpected_black_pixels"]
        total_opaque += stats["opaque_pixels"]

        panel = composite_on_dark(image)
        overlay = panel.copy()
        overlay[magenta] = (255, 255, 255)
        overlay[black] = (0, 0, 255)
        cv2.putText(
            overlay,
            f"{semantic} M={stats['debug_magenta_pixels']} B={stats['unexpected_black_pixels']}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
        panels.append(overlay)
        records.append({"semantic": semantic, "path": str(path), **stats})

    height = max(panel.shape[0] for panel in panels)
    width = max(panel.shape[1] for panel in panels)
    sheet = np.full((height * 2, width * 3, 3), 18, np.uint8)
    for index, panel in enumerate(panels):
        row, col = divmod(index, 3)
        sheet[row * height: row * height + panel.shape[0],
              col * width: col * width + panel.shape[1]] = panel
    contact_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(contact_path), sheet):
        raise RuntimeError(f"CONTACT_SHEET_WRITE_FAILED:{contact_path}")

    if args.kind == "synthetic":
        classification = (
            "PROVEN"
            if total_magenta == 0 and total_black == 0 and total_opaque > 0
            else "REJECTED"
        )
    else:
        classification = "MEASURED"

    report = {
        "schema": "panda_atlas_contract_render_analysis_v1",
        "kind": args.kind,
        "render_dir": str(render_dir),
        "prefix": args.prefix,
        "views": records,
        "totals": {
            "opaque_pixels": total_opaque,
            "debug_magenta_pixels": total_magenta,
            "unexpected_black_pixels": total_black,
            "debug_magenta_fraction": float(total_magenta / max(total_opaque, 1)),
            "unexpected_black_fraction": float(total_black / max(total_opaque, 1)),
        },
        "classification": {
            "SYNTHETIC_UNIQUE_TRIANGLE_RENDER": classification if args.kind == "synthetic" else "NOT_APPLICABLE",
            "VISIBLE_UNOWNED_ATLAS_SAMPLING_ZERO": (
                "PROVEN" if args.kind == "synthetic" and total_magenta == 0
                else "REJECTED" if args.kind == "synthetic"
                else "NOT_APPLICABLE"
            ),
            "VISIBLE_UNEXPECTED_BLACK_ZERO": (
                "PROVEN" if args.kind == "synthetic" and total_black == 0
                else "REJECTED" if args.kind == "synthetic"
                else "MEASURED"
            ),
        },
        "contact_sheet": str(contact_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"ATLAS_RENDER_ANALYSIS kind={args.kind} magenta={total_magenta} "
        f"black={total_black} classification={classification}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
