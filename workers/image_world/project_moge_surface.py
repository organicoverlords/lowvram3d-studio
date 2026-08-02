"""Convert saved MoGe arrays into an unclassified top-down surface baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from lowvram3d.image_world.surface_projection import project_moge_surface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid-size", type=int, default=513)
    parser.add_argument("--minimum-surface-alignment", type=float, default=0.15)
    parser.add_argument("--smoothing-iterations", type=int, default=32)
    parser.add_argument("--stream-minimum-cells", type=float)
    parser.add_argument("--allow-up-fallback", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    geometry = Path(args.geometry).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "surface-projection-report.json"
    started = time.perf_counter()

    try:
        points = np.load(geometry / "points.npy", allow_pickle=False)
        normals = np.load(geometry / "normal.npy", allow_pickle=False)
        mask = np.load(geometry / "mask.npy", allow_pickle=False).astype(bool)
        result = project_moge_surface(
            points,
            normals,
            mask,
            grid_size=args.grid_size,
            minimum_surface_alignment=args.minimum_surface_alignment,
            smoothing_iterations=args.smoothing_iterations,
            stream_minimum_cells=args.stream_minimum_cells,
            allow_up_fallback=args.allow_up_fallback,
        )

        arrays = output / "arrays"
        arrays.mkdir(parents=True, exist_ok=True)
        _save(arrays / "observed-height.npy", result.observation.height, np.float32)
        _save(arrays / "observed-mask.npy", result.observation.observed_mask, np.uint8)
        _save(arrays / "sample-count.npy", result.observation.sample_count, np.int32)
        _save(arrays / "variance.npy", result.observation.variance, np.float32)
        _save(arrays / "confidence.npy", result.observation.confidence, np.float32)
        _save(arrays / "completed-height.npy", result.completed.height, np.float32)
        _save(arrays / "generated-mask.npy", result.completed.generated_mask, np.uint8)
        _save(arrays / "hydrology-height.npy", result.hydrology_height, np.float32)
        _save(arrays / "slope-degrees.npy", result.slope, np.float32)
        _save(arrays / "flow-direction.npy", result.flow_direction, np.int8)
        _save(arrays / "flow-accumulation.npy", result.flow_accumulation, np.float32)
        _save(arrays / "stream-mask.npy", result.stream_mask, np.uint8)
        _save(arrays / "source-candidate-mask.npy", result.candidate_mask, np.uint8)
        _save(arrays / "source-up-alignment.npy", result.alignment, np.float32)

        previews = output / "previews"
        previews.mkdir(parents=True, exist_ok=True)
        import cv2

        _write_gray(cv2, previews / "height.png", _scalar_preview(result.completed.height))
        _write_gray(cv2, previews / "observed.png", result.observation.observed_mask.astype(np.uint8) * 255)
        _write_gray(cv2, previews / "generated.png", result.completed.generated_mask.astype(np.uint8) * 255)
        _write_gray(cv2, previews / "slope.png", _scalar_preview(result.slope, low=0.0, high=75.0))
        log_flow = np.log1p(result.flow_accumulation)
        _write_gray(cv2, previews / "flow-accumulation.png", _scalar_preview(log_flow))
        _write_gray(cv2, previews / "streams.png", result.stream_mask.astype(np.uint8) * 255)

        report = {
            "status": "PASS_BASELINE",
            "classification": result.classification,
            "terrain_semantics_proven": False,
            "promotion_allowed": False,
            "frame": asdict(result.frame),
            "grid_size": args.grid_size,
            "xy_bounds": list(result.observation.xy_bounds),
            "cell_size": result.cell_size,
            "candidate_pixel_fraction": float(result.candidate_mask.mean()),
            "observed_cell_fraction": float(result.observation.observed_mask.mean()),
            "generated_cell_fraction": float(result.completed.generated_mask.mean()),
            "stream_cell_fraction": float(result.stream_mask.mean()),
            "stream_minimum_cells": result.stream_minimum_cells,
            "minimum_surface_alignment": args.minimum_surface_alignment,
            "smoothing_iterations": args.smoothing_iterations,
            "wall_time_seconds": time.perf_counter() - started,
            "warnings": [
                "MoGe validity plus normal alignment is not a semantic terrain mask.",
                "Roofs, water and other horizontal non-terrain surfaces may remain.",
                "This artifact is a geometry diagnostic and cannot be promoted to Unreal Landscape.",
            ],
            "errors": [],
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("IMAGE_WORLD_SURFACE_PROJECTION_BASELINE_PASS")
        return 0
    except Exception as exc:
        report = {
            "status": "FAILED",
            "classification": "SURFACE_PROJECTION_FAILED",
            "terrain_semantics_proven": False,
            "promotion_allowed": False,
            "wall_time_seconds": time.perf_counter() - started,
            "errors": [f"{type(exc).__name__}: {exc}", traceback.format_exc()],
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(report["errors"][0], file=sys.stderr)
        return 2


def _save(path: Path, value: np.ndarray, dtype) -> None:
    np.save(path, np.asarray(value, dtype=dtype), allow_pickle=False)


def _scalar_preview(
    values: np.ndarray,
    *,
    low: float | None = None,
    high: float | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("preview array has no finite values")
    samples = array[finite]
    if low is None:
        low = float(np.percentile(samples, 2.0))
    if high is None:
        high = float(np.percentile(samples, 98.0))
    if not high > low:
        high = low + 1.0
    normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    image = np.zeros(array.shape, dtype=np.uint8)
    image[finite] = np.rint(normalized[finite] * 255.0).astype(np.uint8)
    return image


def _write_gray(cv2_module, path: Path, image: np.ndarray) -> None:
    if not cv2_module.imwrite(str(path), np.asarray(image, dtype=np.uint8)):
        raise RuntimeError(f"OpenCV failed to write {path}")


if __name__ == "__main__":
    sys.exit(main())
