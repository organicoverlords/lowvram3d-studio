"""Repair a humanoid's lower stance on a derived GLB without touching the source.

The single-view generator often closes both feet into one floor-like bridge. That result can look
acceptable from the front but cannot be skinned as two legs. This worker performs a deliberately
small repair before LOD/UV/bake:

* remove detached microscopic components using the same conservative policy as CLEAN;
* cut only thin, near-ground faces inside a narrow centre strip;
* move the lower left/right body lobes outward with a smooth height falloff;
* recompute normals and write a new geometry-only GLB.

The source mesh is immutable. UVs, when present, are carried unchanged for surviving vertices, but
this worker is intended to run before unwrapping so the downstream bake sees the repaired stance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesh_io import read_glb, triangle_components, vertex_normals, write_glb

WELD = 4e-4


def _axes(positions: np.ndarray) -> tuple[int, int, int]:
    extent = positions.max(axis=0) - positions.min(axis=0)
    height_axis = int(np.argmax(extent))
    remaining = [a for a in range(3) if a != height_axis]
    depth_axis = remaining[int(np.argmin(extent[remaining]))]
    width_axis = next(a for a in remaining if a != depth_axis)
    return width_axis, depth_axis, height_axis


def _component_drop_mask(
    positions: np.ndarray,
    tris: np.ndarray,
    height_axis: int,
    height_min: float,
    max_faces: int,
    max_diagonal_fraction: float,
) -> tuple[np.ndarray, list[dict]]:
    component, _ = triangle_components(positions, tris, WELD)
    sizes = np.bincount(component)
    body = int(np.argmax(sizes))
    low, high = positions.min(axis=0), positions.max(axis=0)
    span = max(float(high[height_axis] - low[height_axis]), 1e-9)
    scene_diagonal = float(np.linalg.norm(high - low))
    max_diagonal = scene_diagonal * max_diagonal_fraction
    drop = np.zeros(len(tris), bool)
    removed: list[dict] = []
    for index, count_value in enumerate(sizes):
        if index == body:
            continue
        members = component == index
        count = int(count_value)
        vertices = positions[np.unique(tris[members])]
        height = float(((vertices[:, height_axis] - low[height_axis]) / span).mean())
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        remove = count <= 1 or (
            height >= height_min and count <= max_faces and diagonal <= max_diagonal
        )
        if remove:
            drop |= members
            removed.append({
                "component": index,
                "triangles": count,
                "height_mean": round(height, 5),
                "diagonal": round(diagonal, 6),
            })
    return drop, removed


def _bottom_gap_metric(
    positions: np.ndarray,
    width_axis: int,
    height_axis: int,
    lower_fraction: float = 0.08,
    centre_fraction: float = 0.06,
) -> dict:
    low, high = positions.min(axis=0), positions.max(axis=0)
    extent = high - low
    height = max(float(extent[height_axis]), 1e-9)
    width = max(float(extent[width_axis]), 1e-9)
    centre = float((low[width_axis] + high[width_axis]) * 0.5)
    fraction = (positions[:, height_axis] - low[height_axis]) / height
    lower = positions[fraction <= lower_fraction]
    if not len(lower):
        return {"lower_vertices": 0, "centre_vertices": 0, "centre_fraction": 1.0}
    centre_vertices = int((np.abs(lower[:, width_axis] - centre) <= width * centre_fraction).sum())
    return {
        "lower_vertices": int(len(lower)),
        "centre_vertices": centre_vertices,
        "centre_fraction": round(centre_vertices / max(len(lower), 1), 6),
        "lower_band": lower_fraction,
        "centre_half_width_fraction": centre_fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--cut-height", type=float, default=0.10)
    parser.add_argument("--cut-half-width", type=float, default=0.045)
    parser.add_argument("--outward-fraction", type=float, default=0.035)
    parser.add_argument("--outward-band", type=float, default=0.30)
    parser.add_argument("--body-half-width", type=float, default=0.34)
    parser.add_argument("--debris-height-min", type=float, default=0.66)
    parser.add_argument("--max-debris-faces", type=int, default=20)
    parser.add_argument("--max-debris-diagonal-fraction", type=float, default=0.062)
    parser.add_argument("--max-face-loss-percent", type=float, default=0.25)
    parser.add_argument("--max-centre-fraction", type=float, default=0.08)
    args = parser.parse_args()

    source = Path(args.input)
    positions, _, uv, tris = read_glb(source)
    positions = positions.astype(np.float64)
    tris = tris.astype(np.int64)
    width_axis, depth_axis, height_axis = _axes(positions)
    low, high = positions.min(axis=0), positions.max(axis=0)
    extent = high - low
    width = max(float(extent[width_axis]), 1e-9)
    height = max(float(extent[height_axis]), 1e-9)
    centre = (low + high) * 0.5

    gap_before = _bottom_gap_metric(positions, width_axis, height_axis)
    debris_drop, removed_components = _component_drop_mask(
        positions,
        tris,
        height_axis,
        args.debris_height_min,
        args.max_debris_faces,
        args.max_debris_diagonal_fraction,
    )

    tri_positions = positions[tris]
    face_centres = tri_positions.mean(axis=1)
    face_height = (face_centres[:, height_axis] - low[height_axis]) / height
    centre_distance = np.abs(face_centres[:, width_axis] - centre[width_axis])
    face_normals = np.cross(
        tri_positions[:, 1] - tri_positions[:, 0],
        tri_positions[:, 2] - tri_positions[:, 0],
    )
    normal_lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals /= np.maximum(normal_lengths, 1e-12)

    half_gap = width * args.cut_half_width
    all_central = (
        np.max(np.abs(tri_positions[:, :, width_axis] - centre[width_axis]), axis=1)
        < half_gap * 1.5
    )
    all_low = (
        np.max((tri_positions[:, :, height_axis] - low[height_axis]) / height, axis=1)
        < args.cut_height * 1.4
    )
    near_horizontal = np.abs(face_normals[:, height_axis]) > 0.25
    repair_stance = gap_before["centre_fraction"] > args.max_centre_fraction
    bridge_drop = (
        (face_height < args.cut_height)
        & (centre_distance < half_gap)
        & (near_horizontal | (all_central & all_low))
    ) if repair_stance else np.zeros(len(tris), bool)

    drop = debris_drop | bridge_drop
    survivors = tris[~drop]
    if not len(survivors):
        raise RuntimeError("stance repair would remove the entire mesh")

    repaired = positions.copy()
    height_fraction = (repaired[:, height_axis] - low[height_axis]) / height
    falloff = np.clip(
        (args.outward_band - height_fraction) / max(args.outward_band, 1e-6), 0.0, 1.0
    ) ** 1.5
    relative = repaired[:, width_axis] - centre[width_axis]
    central_body = np.abs(relative) < args.body_half_width * width
    sign = np.where(relative >= 0.0, 1.0, -1.0)
    if repair_stance:
        repaired[:, width_axis] += (
            sign * falloff * central_body.astype(np.float64) * args.outward_fraction * width
        )

    used = np.unique(survivors)
    remap = np.full(len(repaired), -1, np.int64)
    remap[used] = np.arange(len(used))
    repaired = repaired[used]
    repaired_tris = remap[survivors]
    repaired_uv = uv[used] if uv is not None else None
    repaired_normals = vertex_normals(repaired.astype(np.float32), repaired_tris)

    face_loss_percent = float(drop.sum() / max(len(tris), 1) * 100.0)
    if face_loss_percent > args.max_face_loss_percent:
        raise RuntimeError(
            f"stance repair would remove {face_loss_percent:.4f}% of faces, "
            f"above {args.max_face_loss_percent}%"
        )

    output = Path(args.output)
    write_glb(output, repaired.astype(np.float32), repaired_normals, repaired_uv, repaired_tris)
    check_positions, _, check_uv, check_tris = read_glb(output)
    gap_after = _bottom_gap_metric(check_positions.astype(np.float64), width_axis, height_axis)

    report = {
        "input": str(source),
        "output": str(output),
        "axes": {"width": width_axis, "depth": depth_axis, "height": height_axis},
        "triangles_before": int(len(tris)),
        "triangles_after": int(len(check_tris)),
        "triangles_removed": int(drop.sum()),
        "triangles_removed_percent": round(face_loss_percent, 6),
        "bridge_faces_removed": int((bridge_drop & ~debris_drop).sum()),
        "debris_faces_removed": int(debris_drop.sum()),
        "debris_components_removed": len(removed_components),
        "removed_components": removed_components,
        "vertices_before": int(len(positions)),
        "vertices_after": int(len(check_positions)),
        "bounds_before": {"min": low.tolist(), "max": high.tolist()},
        "bounds_after": {
            "min": check_positions.min(axis=0).astype(float).tolist(),
            "max": check_positions.max(axis=0).astype(float).tolist(),
        },
        "gap_before": gap_before,
        "gap_after": gap_after,
        "repair_applied": bool(repair_stance),
        "max_centre_fraction": args.max_centre_fraction,
        "outward_fraction": args.outward_fraction,
        "outward_band": args.outward_band,
        "cut_height": args.cut_height,
        "cut_half_width": args.cut_half_width,
        "uv_preserved_for_surviving_vertices": None if uv is None else bool(
            np.array_equal(check_uv, repaired_uv)
        ),
        "passed": bool(
            gap_after["centre_fraction"] <= args.max_centre_fraction
            and (not repair_stance or gap_after["centre_fraction"] < gap_before["centre_fraction"])
            and face_loss_percent <= args.max_face_loss_percent
        ),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "HUMANOID_STANCE "
        f"faces={len(tris)}->{len(check_tris)} "
        f"centre={gap_before['centre_fraction']}->{gap_after['centre_fraction']} "
        f"debris_components={len(removed_components)} passed={report['passed']}",
        flush=True,
    )
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
