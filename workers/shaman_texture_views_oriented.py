"""Build the projection view with an explicit, auditable front direction.

The original single-image route selected +Z versus -Z solely from silhouette IoU. That is unsafe for
shallow near-symmetric reconstructions: the back can win by a few percent while still being the wrong
semantic side. This worker reuses the proven raster/registration implementation but adds:

* an explicit ``--front-direction`` override (also accepted through
  ``LOWVRAM3D_FRONT_DIRECTION``);
* fail-closed automatic selection when the two silhouette scores are too close;
* a report recording the automatic winner, override, margin, and final projection hemisphere.

No UVs, vertex positions, normals, or bakes are changed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from mesh_io import read_glb
from shaman_texture_views import (
    dense_mask_refine,
    mask_iou,
    project,
    rasterise,
    refine_box,
    subject_bbox,
    warp_to_frame,
)

AUTO_AMBIGUITY_MARGIN = 0.08


def rasterise_face_ids(
    screen: np.ndarray, depth: np.ndarray, tris: np.ndarray, size: int
) -> np.ndarray:
    """Return the exact frontmost triangle ID at every projected pixel.

    The projection uses the same texel-centre convention and depth sign as ``rasterise``.
    Background is -1.  Ties are resolved deterministically in favour of the lowest triangle
    index so repeated runs cannot change provenance at shared silhouette pixels.
    """
    px = np.empty_like(screen)
    px[:, 0] = screen[:, 0] * (size - 1)
    px[:, 1] = screen[:, 1] * (size - 1)
    zbuffer = np.full((size, size), np.inf, np.float64)
    entries: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []
    for tri in tris:
        a = px[tri]
        x_lo, y_lo = np.maximum(np.floor(a.min(0)).astype(int), 0)
        x_hi, y_hi = np.minimum(np.ceil(a.max(0)).astype(int), size - 1)
        if x_hi < x_lo or y_hi < y_lo:
            entries.append(None)
            continue
        xs, ys = np.meshgrid(np.arange(x_lo, x_hi + 1), np.arange(y_lo, y_hi + 1))
        xs, ys = xs.ravel(), ys.ravel()
        (x0, y0), (x1, y1), (x2, y2) = a
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            entries.append(None)
            continue
        fx, fy = xs + 0.5, ys + 0.5
        w0 = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) / den
        w1 = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            entries.append(None)
            continue
        xs, ys = xs[inside], ys[inside]
        d = w0[inside] * depth[tri[0]] + w1[inside] * depth[tri[1]] + w2[inside] * depth[tri[2]]
        np.minimum.at(zbuffer, (ys, xs), d)
        entries.append((xs, ys, d))

    face_ids = np.full((size, size), -1, np.int32)
    tolerance = max(float(np.ptp(depth)) * 1e-7, 1e-8)
    for triangle_id, entry in enumerate(entries):
        if entry is None:
            continue
        xs, ys, d = entry
        visible = d <= zbuffer[ys, xs] + tolerance
        if not visible.any():
            continue
        current = face_ids[ys[visible], xs[visible]]
        replace = (current < 0) | (triangle_id < current)
        if replace.any():
            yi, xi = ys[visible][replace], xs[visible][replace]
            face_ids[yi, xi] = triangle_id
    return face_ids


def encode_face_id_png(face_ids: np.ndarray) -> np.ndarray:
    """Encode signed triangle IDs as a lossless 24-bit BGR diagnostic image."""
    encoded = np.maximum(face_ids.astype(np.int64) + 1, 0)
    return np.dstack([
        (encoded & 255).astype(np.uint8),
        ((encoded >> 8) & 255).astype(np.uint8),
        ((encoded >> 16) & 255).astype(np.uint8),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--views-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--raster-size", type=int, default=1536)
    parser.add_argument("--view-size", type=int, default=2048)
    parser.add_argument(
        "--front-direction",
        choices=("auto", "+z", "-z"),
        default=os.environ.get("LOWVRAM3D_FRONT_DIRECTION", "auto").lower(),
    )
    parser.add_argument("--auto-ambiguity-margin", type=float, default=AUTO_AMBIGUITY_MARGIN)
    args = parser.parse_args()

    positions, _, uv, tris = read_glb(Path(args.mesh))
    if uv is None:
        raise RuntimeError("projection mesh has no UVs")
    positions = positions.astype(np.float64)
    centre = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
    verts = positions - centre
    ortho = float((verts.max(axis=0) - verts.min(axis=0)).max())

    edge1 = verts[tris[:, 1]] - verts[tris[:, 0]]
    edge2 = verts[tris[:, 2]] - verts[tris[:, 0]]
    face_normals = np.cross(edge1, edge2)
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)

    source = cv2.imread(args.source, cv2.IMREAD_UNCHANGED)
    if source is None:
        raise RuntimeError(f"could not read {args.source}")
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGRA)
    if source.shape[2] == 4:
        source_bgr = source[:, :, :3]
        source_mask = source[:, :, 3] > 127
    else:
        source_bgr = source[:, :, :3]
        source_mask = source_bgr.min(axis=2) < 245

    candidates = {
        "+z": np.array([0.0, 0.0, 1.0]),
        "-z": np.array([0.0, 0.0, -1.0]),
    }
    scored: dict[str, dict] = {}
    source_box = subject_bbox(source_mask)
    for name, direction in candidates.items():
        screen, depth = project(verts, direction, ortho)
        _, silhouette = rasterise(screen, depth, tris, args.raster_size)
        mesh_box = subject_bbox(silhouette)
        warped = warp_to_frame(
            source_mask.astype(np.uint8) * 255,
            source_box,
            mesh_box,
            args.raster_size,
            interpolation=cv2.INTER_NEAREST,
        )
        scored[name] = {
            "iou": mask_iou(warped > 127, silhouette),
            "silhouette": silhouette,
            "mesh_box": mesh_box,
        }

    auto_front = max(scored, key=lambda key: scored[key]["iou"])
    score_margin = abs(float(scored["+z"]["iou"]) - float(scored["-z"]["iou"]))
    ambiguous = score_margin < float(args.auto_ambiguity_margin)
    if args.front_direction == "auto":
        front = auto_front
        selection_policy = "silhouette_iou_auto"
    else:
        front = args.front_direction
        selection_policy = "explicit_semantic_override"

    direction = candidates[front]
    screen, depth = project(verts, direction, ortho)
    visible, silhouette = rasterise(screen, depth, tris, args.raster_size)
    face_ids = rasterise_face_ids(screen, depth, tris, args.view_size)
    mesh_box, refined_affine_iou = refine_box(
        source_mask,
        silhouette,
        source_box,
        scored[front]["mesh_box"],
        args.raster_size,
    )
    view, view_alpha, dense_report = dense_mask_refine(
        source_bgr,
        source_mask,
        source_box,
        mesh_box,
        silhouette,
        args.view_size,
        args.raster_size,
    )

    views_dir = Path(args.views_dir)
    views_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(views_dir / "front.png"), np.dstack([view, view_alpha]))
    np.save(views_dir / "face_id_front.npy", face_ids)
    cv2.imwrite(str(views_dir / "face_id_front.png"), encode_face_id_png(face_ids))
    (views_dir / "view_metadata.json").write_text(
        json.dumps(
            {
                "policy": {"semantic_projection": ["real", "generated"]},
                "front_direction": front,
                "selection_policy": selection_policy,
                "views": [{"view": "front", "source_type": "real", "confidence": 1.0}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cam = direction * (ortho * 3.0)
    np.savez_compressed(
        args.output_npz,
        verts=verts.astype(np.float32),
        tris=tris.astype(np.int32),
        uvs=uv[tris].astype(np.float32),
        normals=face_normals.astype(np.float32),
        view_names=np.array(["front"]),
        view_locs=np.array([cam], np.float32),
        ortho_scale=np.float32(ortho),
        vis_front=visible,
        face_id_front=face_ids,
    )

    facing = face_normals @ direction
    final_registration_iou = (
        dense_report["dense_iou"] if dense_report.get("accepted") else dense_report["affine_iou"]
    )
    failure_codes: list[str] = []
    if final_registration_iou < 0.58:
        failure_codes.append("TEXTURE_MISREGISTRATION")
    if args.front_direction == "auto" and ambiguous:
        failure_codes.append("FRONT_DIRECTION_AMBIGUOUS")

    report = {
        "mesh": args.mesh,
        "source": args.source,
        "front_direction": front,
        "front_direction_request": args.front_direction,
        "front_direction_auto_winner": auto_front,
        "front_direction_selection_policy": selection_policy,
        "front_direction_score_margin": round(score_margin, 6),
        "front_direction_ambiguous": ambiguous,
        "auto_ambiguity_margin": float(args.auto_ambiguity_margin),
        "silhouette_iou_bbox_fit": {
            key: round(float(value["iou"]), 6) for key, value in scored.items()
        },
        "silhouette_iou_refined_affine": round(refined_affine_iou, 6),
        "dense_registration": dense_report,
        "final_registration_iou": round(float(final_registration_iou), 6),
        "registration_gate_passed": not failure_codes,
        "failure_codes": failure_codes,
        "ortho_scale": ortho,
        "centre_offset": [float(v) for v in centre],
        "triangles": int(len(tris)),
        "visible_triangles": int(visible.sum()),
        "visible_percent": round(float(visible.mean() * 100), 3),
        "front_facing_triangles": int((facing > 0.15).sum()),
        "occluded_but_front_facing": int(((facing > 0.15) & ~visible).sum()),
        "mesh_bbox_pixels": list(mesh_box),
        "source_bbox_pixels": list(source_box),
        "raster_size": args.raster_size,
        "view_size": args.view_size,
        "face_id_buffer": {
            "path_npy": str(views_dir / "face_id_front.npy"),
            "path_png": str(views_dir / "face_id_front.png"),
            "background_id": -1,
            "frontmost_pixels": int(np.count_nonzero(face_ids >= 0)),
            "triangles_represented": int(np.unique(face_ids[face_ids >= 0]).size),
        },
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"VIEWS_BUILT front={front} request={args.front_direction} auto={auto_front} "
        f"margin={score_margin:.4f} visible={visible.sum()}/{len(tris)} "
        f"registration_iou={final_registration_iou:.4f} codes={failure_codes}",
        flush=True,
    )
    raise SystemExit(0 if report["registration_gate_passed"] else 2)


if __name__ == "__main__":
    main()
