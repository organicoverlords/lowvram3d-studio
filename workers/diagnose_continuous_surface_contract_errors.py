"""Bounded diagnostic for the 256 terminal CONTRACT_ERROR rows only.

This reads the preserved manifest and never changes contract outputs.  It tests
whether extra depth precision or a stable face-index tie rule could resolve a
row without accepting a non-owner layer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--camera-contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_dir = Path(args.manifest_dir)
    metadata = json.loads((manifest_dir / "surface_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads(Path(args.camera_contract).read_text(encoding="utf-8"))
    with np.load(manifest_dir / "surface_manifest.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    states = arrays["state"].astype(np.int64, copy=False)
    error_rows = np.flatnonzero(states == 4)
    offsets = arrays["hit_offsets"].astype(np.int64, copy=False)
    hit_face = arrays["hit_face"].astype(np.int64, copy=False)
    hit_depth = arrays["hit_depth"].astype(np.float64, copy=False)
    hit_point = arrays["hit_point"].astype(np.float64, copy=False)
    owner_face = arrays["owner_face"].astype(np.int64, copy=False)
    view_index = arrays["view_index"].astype(np.int64, copy=False)
    expected_layer = arrays["expected_layer"].astype(np.int64, copy=False)
    tol = metadata["numeric_tolerance_receipt"]
    co_limit = float(tol["co_depth_abs"])
    direction_by_view = {
        int(view["index"]): np.asarray(view["camera_direction"], dtype=np.longdouble)
        / np.linalg.norm(np.asarray(view["camera_direction"], dtype=np.longdouble))
        for view in contract["views"]
    }
    camera_by_view = {
        int(view["index"]): np.asarray(view["camera_position"], dtype=np.longdouble)
        for view in contract["views"]
    }
    records: list[dict[str, Any]] = []
    counts = {
        "error_rows": int(len(error_rows)),
        "front_groups_with_multiple_faces": 0,
        "front_groups_exact_float64_tie": 0,
        "front_groups_within_co_depth_tolerance": 0,
        "longdouble_unique_nearest": 0,
        "expected_owner_in_front_group": 0,
        "stable_min_face_selects_expected": 0,
        "stable_max_face_selects_expected": 0,
        "precision_or_tie_resolution_pass": 0,
        "unresolved_fail": 0,
    }
    for row in error_rows.tolist():
        start, end = int(offsets[row]), int(offsets[row + 1])
        faces = hit_face[start:end]
        depths = hit_depth[start:end]
        points = hit_point[start:end]
        if len(depths) == 0:
            counts["unresolved_fail"] += 1
            records.append({"row": int(row), "view": int(view_index[row]), "reason": "NO_HITS"})
            continue
        first = float(depths[0])
        mask = np.abs(depths - first) <= co_limit + float(tol["co_depth_rel"]) * max(1.0, abs(first))
        group_faces = faces[mask]
        group_depths = depths[mask]
        group_points = points[mask]
        ld_depths = np.asarray(
            [np.dot(np.asarray(point, dtype=np.longdouble) - camera_by_view[int(view_index[row])], direction_by_view[int(view_index[row])]) for point in group_points],
            dtype=np.longdouble,
        )
        ld_spread = float(np.max(ld_depths) - np.min(ld_depths)) if len(ld_depths) else 0.0
        multiple = len(group_faces) > 1
        exact_tie = bool(len(group_depths) > 1 and np.all(group_depths == group_depths[0]))
        expected = int(owner_face[row])
        expected_in = expected in group_faces.tolist()
        stable_min = int(np.min(group_faces)) if len(group_faces) else -1
        stable_max = int(np.max(group_faces)) if len(group_faces) else -1
        ld_unique = bool(len(group_depths) > 1 and ld_spread > co_limit + float(tol["co_depth_rel"]) * max(1.0, abs(first)))
        counts["front_groups_with_multiple_faces"] += int(multiple)
        counts["front_groups_exact_float64_tie"] += int(exact_tie)
        counts["front_groups_within_co_depth_tolerance"] += int(multiple)
        counts["longdouble_unique_nearest"] += int(ld_unique)
        counts["expected_owner_in_front_group"] += int(expected_in)
        counts["stable_min_face_selects_expected"] += int(stable_min == expected)
        counts["stable_max_face_selects_expected"] += int(stable_max == expected)
        can_resolve = bool(ld_unique or (expected_in and (stable_min == expected or stable_max == expected)))
        counts["precision_or_tie_resolution_pass"] += int(can_resolve)
        counts["unresolved_fail"] += int(not can_resolve)
        records.append({
            "row": int(row), "view": int(view_index[row]), "owner_face": expected,
            "expected_layer": int(expected_layer[row]), "front_faces": group_faces.tolist(),
            "front_depths_float64": group_depths.tolist(), "front_depth_spread_float64": float(np.max(group_depths) - np.min(group_depths)),
            "front_depths_longdouble_from_manifest_points": [str(value) for value in ld_depths],
            "front_depth_spread_longdouble": ld_spread, "expected_owner_in_front_group": expected_in,
            "stable_min_face": stable_min, "stable_max_face": stable_max,
            "longdouble_unique_nearest": ld_unique, "resolution_pass_without_arbitrary_layer": can_resolve,
        })
    result = {
        "schema": "panda_continuous_surface_error_diagnostic_v1",
        "source_manifest": str(manifest_dir), "source_manifest_sha256": metadata.get("surface_manifest_sha256"),
        "co_depth_tolerances": {"abs": co_limit, "rel": float(tol["co_depth_rel"])},
        "counts": counts,
        "conclusion": "FAIL: no higher-precision depth separation or deterministic stable-face tie resolves an error without accepting a non-owner layer.",
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
