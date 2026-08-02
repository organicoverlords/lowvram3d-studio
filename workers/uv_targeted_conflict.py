"""Bounded exact UV ownership diagnostic for a selected pair of triangle classes.

The repository-wide overlap gate can fail closed when the complete atlas has too many
candidate pairs.  This diagnostic asks the narrower question needed by projection repair:
do front-observed triangles and rear-dominant triangles share positive-area UV ownership?
It uses UV bounding-box bins only to reduce candidates, then uses the same exact convex
clipping rule as :mod:`lowvram3d.uv_overlap`.  A shared edge is therefore not a conflict.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "workers"))

from lowvram3d.uv_overlap import (  # noqa: E402
    AREA_EPSILON_UV,
    _clip_convex,
    _polygon_area,
)
from uv_exact_validate import load_uv_mesh  # noqa: E402


def _signed_area(triangles: np.ndarray) -> np.ndarray:
    return 0.5 * (
        (triangles[:, 1, 0] - triangles[:, 0, 0])
        * (triangles[:, 2, 1] - triangles[:, 0, 1])
        - (triangles[:, 2, 0] - triangles[:, 0, 0])
        * (triangles[:, 1, 1] - triangles[:, 0, 1])
    )


def _candidate_pairs(
    triangles: np.ndarray,
    front_ids: np.ndarray,
    rear_ids: np.ndarray,
    grid_size: int,
    max_candidates: int,
) -> tuple[list[tuple[int, int]], int, bool]:
    """Return bbox candidates, with a hard fail-closed cap."""
    low = np.clip(np.floor(np.nanmin(triangles, axis=1) * grid_size).astype(np.int64), 0, grid_size - 1)
    high = np.clip(np.floor(np.nanmax(triangles, axis=1) * grid_size).astype(np.int64), 0, grid_size - 1)
    buckets: dict[int, list[int]] = {}
    for triangle_id in front_ids.tolist():
        for cell_x in range(int(low[triangle_id, 0]), int(high[triangle_id, 0]) + 1):
            for cell_y in range(int(low[triangle_id, 1]), int(high[triangle_id, 1]) + 1):
                buckets.setdefault(cell_x * grid_size + cell_y, []).append(int(triangle_id))

    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for triangle_id in rear_ids.tolist():
        candidates: set[int] = set()
        for cell_x in range(int(low[triangle_id, 0]), int(high[triangle_id, 0]) + 1):
            for cell_y in range(int(low[triangle_id, 1]), int(high[triangle_id, 1]) + 1):
                candidates.update(buckets.get(cell_x * grid_size + cell_y, ()))
        for front_id in candidates:
            pair = (front_id, int(triangle_id))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                if len(pairs) > max_candidates:
                    return pairs, len(pairs), True
    return pairs, len(pairs), False


def diagnose(
    uv_triangles: np.ndarray,
    front_observed: np.ndarray,
    rear_dominant: np.ndarray,
    *,
    atlas_resolution: int = 1024,
    grid_size: int = 1024,
    max_candidates: int = 2_000_000,
    max_reported_pairs: int = 20_000,
    timeout_seconds: float = 180.0,
) -> dict:
    started = time.monotonic()
    triangles = np.asarray(uv_triangles, dtype=np.float64)
    front = np.asarray(front_observed, dtype=bool)
    rear = np.asarray(rear_dominant, dtype=bool)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 2):
        raise ValueError("uv_triangles must have shape (F, 3, 2)")
    if len(front) != len(triangles) or len(rear) != len(triangles):
        raise ValueError("classification masks must match triangle count")
    finite = np.isfinite(triangles).all(axis=(1, 2))
    valid = finite & (np.abs(_signed_area(triangles)) > AREA_EPSILON_UV)
    front_ids = np.flatnonzero(front & valid)
    rear_ids = np.flatnonzero(rear & ~front & valid)

    pairs, candidate_count, capped = _candidate_pairs(
        triangles, front_ids, rear_ids, grid_size, max_candidates
    )
    report = {
        "schema": "targeted_uv_conflict_report_v1",
        "atlas_resolution": int(atlas_resolution),
        "grid_size": int(grid_size),
        "triangle_count": int(len(triangles)),
        "front_observed_triangle_count": int(len(front_ids)),
        "rear_dominant_triangle_count": int(len(rear_ids)),
        "invalid_uv_triangle_count": int((~valid).sum()),
        "candidate_pair_count": int(candidate_count),
        "candidate_cap": int(max_candidates),
        "candidate_cap_exceeded": bool(capped),
        "tested_pair_count": 0,
        "positive_overlap_pair_count": 0,
        "positive_overlap_total_area_uv": 0.0,
        "positive_overlap_total_texels_equivalent": 0.0,
        "reported_pairs": [],
        "timed_out": False,
        "success": False,
        "classification": "NOT_PROVEN",
    }
    if capped:
        report["error"] = "candidate cap exceeded; exact conflict result is not proven"
        report["elapsed_seconds"] = time.monotonic() - started
        return report

    total_area = 0.0
    tested = 0
    positive = 0
    for front_id, rear_id in pairs:
        if time.monotonic() - started > timeout_seconds:
            report["timed_out"] = True
            report["error"] = "timeout during exact conflict testing; result is not proven"
            break
        a = triangles[front_id]
        b = triangles[rear_id]
        if (
            a[:, 0].max() < b[:, 0].min()
            or b[:, 0].max() < a[:, 0].min()
            or a[:, 1].max() < b[:, 1].min()
            or b[:, 1].max() < a[:, 1].min()
        ):
            continue
        tested += 1
        area = _polygon_area(_clip_convex(a.copy(), b.copy()))
        if area > AREA_EPSILON_UV:
            positive += 1
            total_area += area
            if len(report["reported_pairs"]) < max_reported_pairs:
                report["reported_pairs"].append(
                    {
                        "front_triangle": int(front_id),
                        "rear_triangle": int(rear_id),
                        "overlap_area_uv": float(area),
                        "overlap_texels_equivalent": float(area * atlas_resolution * atlas_resolution),
                    }
                )

    report["tested_pair_count"] = int(tested)
    report["positive_overlap_pair_count"] = int(positive)
    report["positive_overlap_total_area_uv"] = float(total_area)
    report["positive_overlap_total_texels_equivalent"] = float(
        total_area * atlas_resolution * atlas_resolution
    )
    report["reported_pair_cap"] = int(max_reported_pairs)
    report["reported_pairs_truncated"] = positive > len(report["reported_pairs"])
    report["success"] = not report["timed_out"]
    report["classification"] = "PROVEN" if report["success"] else "NOT_PROVEN"
    report["elapsed_seconds"] = time.monotonic() - started
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--observed", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--atlas-resolution", type=int, default=1024)
    parser.add_argument("--grid-size", type=int, default=1024)
    parser.add_argument("--max-candidates", type=int, default=2_000_000)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    args = parser.parse_args()

    _, uv = load_uv_mesh(Path(args.input))
    provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    observed = np.load(args.observed)
    rear = np.asarray(provenance["rear_dominant"], dtype=bool)
    report = diagnose(
        uv,
        observed,
        rear,
        atlas_resolution=args.atlas_resolution,
        grid_size=args.grid_size,
        max_candidates=args.max_candidates,
        timeout_seconds=args.timeout_seconds,
    )
    report["input"] = str(Path(args.input))
    report["provenance"] = str(Path(args.provenance))
    report["observed"] = str(Path(args.observed))
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "TARGETED_UV_CONFLICT "
        f"candidates={report['candidate_pair_count']} tested={report['tested_pair_count']} "
        f"positive_pairs={report['positive_overlap_pair_count']} "
        f"texels={report['positive_overlap_total_texels_equivalent']:.3f} "
        f"success={report['success']} classification={report['classification']}",
        flush=True,
    )
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
