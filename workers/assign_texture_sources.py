"""Assign authoritative source views only to directly observed triangles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def mesh_adjacency(triangles: np.ndarray) -> list[list[int]]:
    edges: dict[tuple[int, int], list[int]] = {}
    for tid, tri in enumerate(np.asarray(triangles, np.int64)):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edges.setdefault(tuple(sorted((int(a), int(b)))), []).append(tid)
    result = [[] for _ in range(len(triangles))]
    for values in edges.values():
        if len(values) == 2:
            a, b = values; result[a].append(b); result[b].append(a)
    return result


def assign(evidence: dict[str, np.ndarray], triangles: np.ndarray, *, passes: int = 5,
           regions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    states = np.asarray(evidence["evidence_state"], np.uint8)
    confidence = np.asarray(evidence["confidence"], np.float32)
    view_count, triangle_count = states.shape
    direct = np.isin(states, (1, 3))
    unary = np.where(direct, -confidence, np.inf).astype(np.float64)
    labels = np.full(triangle_count, -1, np.int32)
    for tid in range(triangle_count):
        candidates = np.flatnonzero(direct[:, tid])
        if candidates.size:
            labels[tid] = int(candidates[np.lexsort((candidates, -confidence[candidates, tid]))][0])
    adjacency = mesh_adjacency(triangles)
    changes = []
    for _ in range(max(1, min(int(passes), 5))):
        changed = 0
        for tid in range(triangle_count):
            candidates = np.flatnonzero(direct[:, tid])
            if not candidates.size:
                continue
            best = labels[tid]; best_cost = np.inf
            for view in candidates.tolist():
                cost = float(unary[view, tid])
                for other in adjacency[tid]:
                    if labels[other] < 0 or labels[other] == view:
                        continue
                    if regions is not None and regions[other] != regions[tid]:
                        continue
                    cost += 0.15
                if cost < best_cost or (cost == best_cost and (best < 0 or view < best)):
                    best, best_cost = view, cost
            if best != labels[tid]:
                labels[tid] = best; changed += 1
        changes.append(changed)
        if changed == 0:
            break
    selected_conf = np.zeros(triangle_count, np.float32)
    observed = labels >= 0
    selected_conf[observed] = confidence[labels[observed], np.flatnonzero(observed)]
    report = {
        "schema": "texture_source_assignment_v1",
        "triangle_count": int(triangle_count), "view_count": int(view_count),
        "assigned_triangles": int(observed.sum()),
        "unobserved_triangles": int((~observed).sum()),
        "icm_pass_changes": changes, "max_passes": 5,
        "unobserved_receive_no_view": True, "deterministic": True,
    }
    return labels, selected_conf, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--regions", default="")
    parser.add_argument("--passes", type=int, default=5)
    args = parser.parse_args()
    from mesh_io import read_glb
    _v, _n, _u, triangles = read_glb(Path(args.mesh))
    with np.load(args.evidence, allow_pickle=False) as data:
        evidence = {key: data[key] for key in data.files}
    regions = np.load(args.regions) if args.regions else None
    labels, confidence, report = assign(evidence, triangles, passes=args.passes, regions=regions)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "primary_view_per_triangle.npy", labels)
    np.save(out / "primary_view_confidence.npy", confidence)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
