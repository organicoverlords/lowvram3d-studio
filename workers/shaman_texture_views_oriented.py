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
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lowvram3d.anchor_provenance import GEOMETRY_HASH_FRAME, geometry_sha256  # noqa: E402

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
    )
    geometry_hash = geometry_sha256(verts, tris)

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
        "geometry_hash_frame": GEOMETRY_HASH_FRAME,
        "geometry_sha256": geometry_hash,
        "triangles": int(len(tris)),
        "visible_triangles": int(visible.sum()),
        "visible_percent": round(float(visible.mean() * 100), 3),
        "front_facing_triangles": int((facing > 0.15).sum()),
        "occluded_but_front_facing": int(((facing > 0.15) & ~visible).sum()),
        "mesh_bbox_pixels": list(mesh_box),
        "source_bbox_pixels": list(source_box),
        "raster_size": args.raster_size,
        "view_size": args.view_size,
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
