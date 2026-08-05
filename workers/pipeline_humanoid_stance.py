"""Conservatively separate fused humanoid feet and remove microscopic detached shards.

The source is immutable. A derived GLB is written only when face loss stays below the configured
cap and the lower-centre occupancy improves to the riggable threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesh_io import read_glb, triangle_components, vertex_normals, write_glb

WELD = 4e-4


def axes(points: np.ndarray) -> tuple[int, int, int]:
    extent = points.max(0) - points.min(0)
    height = int(np.argmax(extent))
    rest = [axis for axis in range(3) if axis != height]
    depth = rest[int(np.argmin(extent[rest]))]
    width = next(axis for axis in rest if axis != depth)
    return width, depth, height


def gap_metric(points: np.ndarray, width_axis: int, height_axis: int) -> dict:
    low, high = points.min(0), points.max(0)
    extent = high - low
    height = max(float(extent[height_axis]), 1e-9)
    width = max(float(extent[width_axis]), 1e-9)
    centre = float((low[width_axis] + high[width_axis]) * 0.5)
    lower = points[(points[:, height_axis] - low[height_axis]) / height <= 0.08]
    central = int((np.abs(lower[:, width_axis] - centre) <= width * 0.06).sum()) if len(lower) else 0
    return {
        "lower_vertices": int(len(lower)),
        "centre_vertices": central,
        "centre_fraction": round(central / max(len(lower), 1), 6),
        "lower_band": 0.08,
        "centre_half_width_fraction": 0.06,
    }


def debris_mask(points, faces, height_axis, height_min, max_faces, diagonal_fraction):
    labels, _ = triangle_components(points, faces, WELD)
    sizes = np.bincount(labels)
    body = int(np.argmax(sizes))
    low, high = points.min(0), points.max(0)
    height = max(float(high[height_axis] - low[height_axis]), 1e-9)
    max_diagonal = float(np.linalg.norm(high - low)) * diagonal_fraction
    drop = np.zeros(len(faces), bool)
    removed = []
    for component, size in enumerate(sizes):
        if component == body:
            continue
        members = labels == component
        vertices = points[np.unique(faces[members])]
        relative_height = float(((vertices[:, height_axis] - low[height_axis]) / height).mean())
        diagonal = float(np.linalg.norm(vertices.max(0) - vertices.min(0)))
        if int(size) <= 1 or (
            relative_height >= height_min and int(size) <= max_faces and diagonal <= max_diagonal
        ):
            drop |= members
            removed.append({
                "component": component,
                "triangles": int(size),
                "height_mean": round(relative_height, 5),
                "diagonal": round(diagonal, 6),
            })
    return drop, removed


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
    parser.add_argument("--max-face-loss-percent", type=float, default=0.75)
    parser.add_argument("--max-centre-fraction", type=float, default=0.08)
    args = parser.parse_args()

    source = Path(args.input)
    points, _, uv, faces = read_glb(source)
    points = points.astype(np.float64)
    faces = faces.astype(np.int64)
    width_axis, depth_axis, height_axis = axes(points)
    low, high = points.min(0), points.max(0)
    extent = high - low
    width = max(float(extent[width_axis]), 1e-9)
    height = max(float(extent[height_axis]), 1e-9)
    centre = (low + high) * 0.5
    before = gap_metric(points, width_axis, height_axis)

    debris, removed = debris_mask(
        points, faces, height_axis, args.debris_height_min,
        args.max_debris_faces, args.max_debris_diagonal_fraction,
    )
    triangles = points[faces]
    centres = triangles.mean(1)
    face_height = (centres[:, height_axis] - low[height_axis]) / height
    centre_distance = np.abs(centres[:, width_axis] - centre[width_axis])
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    half_gap = width * args.cut_half_width
    all_central = (
        np.max(np.abs(triangles[:, :, width_axis] - centre[width_axis]), 1) < half_gap * 1.5
    )
    all_low = (
        np.max((triangles[:, :, height_axis] - low[height_axis]) / height, 1)
        < args.cut_height * 1.4
    )
    needs_stance = before["centre_fraction"] > args.max_centre_fraction
    bridge = (
        (face_height < args.cut_height)
        & (centre_distance < half_gap)
        & ((np.abs(normals[:, height_axis]) > 0.25) | (all_central & all_low))
    ) if needs_stance else np.zeros(len(faces), bool)
    drop = debris | bridge
    survivors = faces[~drop]

    repaired = points.copy()
    if needs_stance:
        fraction = (repaired[:, height_axis] - low[height_axis]) / height
        falloff = np.clip(
            (args.outward_band - fraction) / max(args.outward_band, 1e-6), 0, 1
        ) ** 1.5
        relative = repaired[:, width_axis] - centre[width_axis]
        central_body = np.abs(relative) < args.body_half_width * width
        repaired[:, width_axis] += (
            np.where(relative >= 0, 1.0, -1.0)
            * falloff * central_body * args.outward_fraction * width
        )

    used = np.unique(survivors)
    remap = np.full(len(repaired), -1, np.int64)
    remap[used] = np.arange(len(used))
    repaired = repaired[used]
    repaired_faces = remap[survivors]
    repaired_uv = uv[used] if uv is not None else None
    after = gap_metric(repaired, width_axis, height_axis)
    face_loss = float(drop.sum() / max(len(faces), 1) * 100)
    passed = bool(
        len(repaired_faces)
        and face_loss <= args.max_face_loss_percent
        and after["centre_fraction"] <= args.max_centre_fraction
        and (not needs_stance or after["centre_fraction"] < before["centre_fraction"])
    )
    report = {
        "input": str(source),
        "output": str(args.output),
        "axes": {"width": width_axis, "depth": depth_axis, "height": height_axis},
        "triangles_before": int(len(faces)),
        "triangles_after": int(len(repaired_faces)),
        "triangles_removed": int(drop.sum()),
        "triangles_removed_percent": round(face_loss, 6),
        "bridge_faces_removed": int((bridge & ~debris).sum()),
        "debris_faces_removed": int(debris.sum()),
        "debris_components_removed": len(removed),
        "removed_components": removed,
        "vertices_before": int(len(points)),
        "vertices_after": int(len(repaired)),
        "gap_before": before,
        "gap_after": after,
        "repair_applied": bool(needs_stance),
        "max_centre_fraction": args.max_centre_fraction,
        "max_face_loss_percent": args.max_face_loss_percent,
        "passed": passed,
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if passed:
        write_glb(
            Path(args.output),
            repaired.astype(np.float32),
            vertex_normals(repaired.astype(np.float32), repaired_faces),
            repaired_uv,
            repaired_faces,
        )
    print(
        f"HUMANOID_STANCE faces={len(faces)}->{len(repaired_faces)} "
        f"centre={before['centre_fraction']}->{after['centre_fraction']} "
        f"loss={face_loss:.4f}% debris_components={len(removed)} passed={passed}",
        flush=True,
    )
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
