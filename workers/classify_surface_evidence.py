"""Classify triangle/view visibility from exact raster evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def classify(manifest: dict, *, triangle_count: int, facing_threshold: float = 0.15,
             min_projected_pixels: int = 1) -> tuple[dict[str, np.ndarray], dict]:
    views = list(manifest.get("views", []))
    states = np.full((len(views), triangle_count), 4, np.uint8)  # UNOBSERVED
    confidence = np.zeros((len(views), triangle_count), np.float32)
    pixels = np.zeros((len(views), triangle_count), np.int32)
    classes = []
    for vi, item in enumerate(views):
        classes.append(str(item.get("source_class", item.get("semantic_source_class", "UNKNOWN"))))
        with np.load(item["path"], allow_pickle=False) as data:
            ids = np.asarray(data["triangle_id"], np.int64)
            visible = np.asarray(data.get("visible", ids >= 0), bool)
            alpha = np.asarray(data.get("source_alpha", np.ones_like(ids, np.float32)), np.float32)
            facing = np.asarray(data.get("normal_facing", np.ones_like(ids, np.float32)), np.float32)
            exact = np.asarray(data.get("exact_triangle_id_match", ids >= 0), bool)
            valid = (ids >= 0) & (ids < triangle_count) & visible & exact \
                & (alpha > 0.0) & np.isfinite(facing) & (facing > facing_threshold)
            ids_valid = ids[valid]
            counts = np.bincount(ids_valid, minlength=triangle_count)
            pixels[vi] = counts
            score = np.zeros(triangle_count, np.float32)
            np.add.at(score, ids_valid, np.clip(facing[valid], 0, 1) * alpha[valid])
            confidence[vi] = score
            direct = counts >= int(min_projected_pixels)
            state_value = 3 if str(classes[-1]).startswith("GENERATED") else 1
            states[vi, direct] = np.uint8(state_value)
    report = {
        "schema": "surface_evidence_v1",
        "triangle_count": int(triangle_count),
        "view_count": len(views),
        "direct_observed_pairs": int(np.count_nonzero(states == 1)),
        "generated_observed_pairs": int(np.count_nonzero(states == 3)),
        "unobserved_pairs": int(np.count_nonzero(states == 4)),
        "exact_id_required": True,
        "deterministic": True,
    }
    return {"evidence_state": states, "confidence": confidence, "projected_pixels": pixels}, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-manifest", required=True)
    parser.add_argument("--triangle-count", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--facing-threshold", type=float, default=0.15)
    args = parser.parse_args()
    manifest = json.loads(Path(args.evidence_manifest).read_text(encoding="utf-8"))
    arrays, report = classify(manifest, triangle_count=args.triangle_count,
                              facing_threshold=args.facing_threshold)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
