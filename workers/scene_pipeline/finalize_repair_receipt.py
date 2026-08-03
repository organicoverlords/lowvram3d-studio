"""Add exact render-pixel cause counts to an existing repair receipt."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
PROOF = Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"


def main() -> None:
    receipt = json.loads((ROOT / "rejection_reason_counts.json").read_text(encoding="utf-8"))
    valid = np.load(ROOT / "mask.npy").astype(bool)
    old = cv2.imread(str(ROOT / "blender_exact_source_cull_off.png"), cv2.IMREAD_COLOR)
    old_on = cv2.imread(str(ROOT / "blender_exact_source_cull_on.png"), cv2.IMREAD_COLOR)
    old_visible = old.mean(axis=2) > 3.0
    on_visible = old_on.mean(axis=2) > 3.0
    coverage = cv2.imread(str(ROOT / "source_face_coverage_mask.png"), cv2.IMREAD_GRAYSCALE) > 0
    cells = receipt["baseline_balanced"]["cell_stats"]
    receipt["missing_cause_distribution"] = {
        "valid_vertex_pixels": int(valid.sum()),
        "valid_pixels_without_rasterized_balanced_face": int(np.count_nonzero(valid & ~coverage)),
        "blender_cull_off_missing_pixels": int(np.count_nonzero(valid & ~old_visible)),
        "blender_cull_on_missing_pixels": int(np.count_nonzero(valid & ~on_visible)),
        "invalid_mask_boundary_cells": cells["invalid_mask_boundary"],
        "both_triangles_rejected_cells": cells["both_triangles_rejected"],
        "one_triangle_rejected_cells": cells["one_triangle_rejected"],
        "winding_culling": "ALL_SOURCE_FACES_REJECTED_WITH_CULL_ON" if np.count_nonzero(valid & ~on_visible) == int(valid.sum()) else "NOT_TOTAL",
        "glb_rasterization_precision": "NOT_OBSERVED_AS_PRIMARY_CAUSE",
    }
    payload = json.dumps(receipt, indent=2) + "\n"
    (ROOT / "rejection_reason_counts.json").write_text(payload, encoding="utf-8")
    (PROOF / "rejection_reason_counts.json").write_text(payload, encoding="utf-8")
    print(json.dumps(receipt["missing_cause_distribution"], indent=2))


if __name__ == "__main__":
    main()
