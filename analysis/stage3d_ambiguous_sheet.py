"""STAGE 3D: ONE numbered contact sheet covering every ambiguous component."""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

CACHE = sys.argv[1]
DECISIONS = sys.argv[2]
IDDIR = sys.argv[3]
OUT = sys.argv[4]

decisions = json.load(open(DECISIONS, encoding="utf-8"))
ambiguous = sorted(
    [d for d in decisions["components"] if d["action"] == "AMBIGUOUS"],
    key=lambda d: -d["faces"],
)
ambiguous_ids = [d["component_id"] for d in ambiguous]
main_id = decisions["main_component_id"]
views = ["front", "right", "back", "left", "top", "underside"]

panels = []
for view in views:
    idbuffer = np.load(os.path.join(IDDIR, f"idbuffer_{view}.npy")) if os.path.exists(
        os.path.join(IDDIR, f"idbuffer_{view}.npy")
    ) else None
    colour_path = os.path.join(IDDIR, f"componentid_{view}.png")
    main_path = os.path.join(IDDIR, f"maskmain_{view}.png")
    main_mask = cv2.imread(main_path, cv2.IMREAD_GRAYSCALE) > 127
    canvas = np.zeros((*main_mask.shape, 3), np.uint8)
    canvas[main_mask] = (70, 70, 70)

    if idbuffer is not None:
        for order, component in enumerate(ambiguous_ids):
            pixels = idbuffer == component
            if not pixels.any():
                continue
            hue = int(179 * order / max(len(ambiguous_ids), 1))
            colour = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
            canvas[pixels] = colour
            ys, xs = np.nonzero(pixels)
            cv2.putText(canvas, str(component), (int(xs.mean()) + 6, int(ys.mean())),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    canvas = cv2.copyMakeBorder(canvas, 0, 26, 0, 0, cv2.BORDER_CONSTANT, value=(15, 15, 15))
    cv2.putText(canvas, view, (5, canvas.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    panels.append(canvas)

sheet = np.concatenate(
    [np.concatenate(panels[:3], axis=1), np.concatenate(panels[3:], axis=1)], axis=0
)
cv2.imwrite(OUT, sheet)
print("AMBIGUOUS_IDS " + json.dumps(ambiguous_ids))
for d in ambiguous:
    print(f"  id={d['component_id']:3d} faces={d['faces']:6d} elong={d['elongation']:5.2f} "
          f"islands={d['island_views']} outside={d['aggregate_outside_dilated_percent']:5.1f}% "
          f"visible_px={d['total_visible_pixels']:6d} dist={d['distance_as_model_ratio']:.4f}")
