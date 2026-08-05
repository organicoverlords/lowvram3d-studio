"""Build a fail-closed four-view projection bundle for raster_project.py.

The previous shaman texture route projected one front image and synthesized the
remaining 84% of the surface.  This worker accepts explicit front/right/back/left
images, registers each against the corresponding mesh silhouette, determines the
correct +/-Z front and +/-X side assignments as paired decisions, and writes the
same NPZ/view_metadata contract consumed by ``raster_project.py``.

No image is mirrored and no missing view is fabricated.  Every supplied view must
pass registration before the bundle is emitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lowvram3d.asset_profiles import foreground_mask  # noqa: E402
from mesh_io import read_glb  # noqa: E402
from shaman_texture_views import (  # noqa: E402
    dense_mask_refine,
    mask_iou,
    project,
    rasterise,
    refine_box,
    subject_bbox,
    warp_to_frame,
)

VIEW_NAMES = ("front", "right", "back", "left")
DEFAULT_MIN_REGISTRATION_IOU = 0.50
DEFAULT_MIN_VISIBLE_UNION_PERCENT = 45.0


def _load_image(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read view image {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 4:
        bgr = image[:, :, :3]
    else:
        bgr = image[:, :, :3]
    mask = foreground_mask(image)
    if not mask.any():
        raise RuntimeError(f"view image has an empty foreground mask: {path}")
    return bgr, mask.astype(bool)


def _score_direction(
    mask: np.ndarray,
    direction: np.ndarray,
    verts: np.ndarray,
    tris: np.ndarray,
    ortho: float,
    raster_size: int,
) -> dict[str, Any]:
    screen, depth = project(verts, direction, ortho)
    visible, silhouette = rasterise(screen, depth, tris, raster_size)
    source_box = subject_bbox(mask)
    mesh_box = subject_bbox(silhouette)
    warped = warp_to_frame(
        mask.astype(np.uint8) * 255,
        source_box,
        mesh_box,
        raster_size,
        interpolation=cv2.INTER_NEAREST,
    )
    return {
        "iou": mask_iou(warped > 127, silhouette),
        "visible": visible,
        "silhouette": silhouette,
        "source_box": source_box,
        "mesh_box": mesh_box,
    }


def _choose_pair(
    first_name: str,
    second_name: str,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    verts: np.ndarray,
    tris: np.ndarray,
    ortho: float,
    raster_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    first_positive = _score_direction(
        first_mask, positive, verts, tris, ortho, raster_size
    )
    second_negative = _score_direction(
        second_mask, negative, verts, tris, ortho, raster_size
    )
    first_negative = _score_direction(
        first_mask, negative, verts, tris, ortho, raster_size
    )
    second_positive = _score_direction(
        second_mask, positive, verts, tris, ortho, raster_size
    )
    direct = float(first_positive["iou"] + second_negative["iou"])
    swapped = float(first_negative["iou"] + second_positive["iou"])
    if direct >= swapped:
        directions = {first_name: positive, second_name: negative}
        selected = "direct"
    else:
        directions = {first_name: negative, second_name: positive}
        selected = "swapped"
    return directions, {
        "first": first_name,
        "second": second_name,
        "direct_score": round(direct, 6),
        "swapped_score": round(swapped, 6),
        "score_margin": round(abs(direct - swapped), 6),
        "selected": selected,
    }


def build_bundle(
    *,
    mesh_path: Path,
    view_paths: dict[str, Path],
    output_npz: Path,
    registered_dir: Path,
    metadata_path: Path,
    report_path: Path,
    raster_size: int,
    view_size: int,
    min_registration_iou: float,
    min_visible_union_percent: float,
    generated_confidence: float,
    front_source_type: str,
) -> dict[str, Any]:
    positions, _, uv, tris = read_glb(mesh_path)
    if uv is None:
        raise RuntimeError("projection mesh has no UV coordinates")
    positions = positions.astype(np.float64)
    centre = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
    verts = positions - centre
    ortho = float((verts.max(axis=0) - verts.min(axis=0)).max())
    if ortho <= 0.0:
        raise RuntimeError("projection mesh has invalid bounds")

    edge1 = verts[tris[:, 1]] - verts[tris[:, 0]]
    edge2 = verts[tris[:, 2]] - verts[tris[:, 0]]
    face_normals = np.cross(edge1, edge2)
    face_normals /= np.maximum(
        np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12
    )

    images: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for name in VIEW_NAMES:
        path = view_paths[name]
        if not path.is_file():
            raise RuntimeError(f"required {name} view is missing: {path}")
        images[name], masks[name] = _load_image(path)

    front_pair, front_selection = _choose_pair(
        "front",
        "back",
        masks["front"],
        masks["back"],
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        verts,
        tris,
        ortho,
        raster_size,
    )
    side_pair, side_selection = _choose_pair(
        "right",
        "left",
        masks["right"],
        masks["left"],
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        verts,
        tris,
        ortho,
        raster_size,
    )
    directions = {**front_pair, **side_pair}

    registered_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "verts": verts.astype(np.float32),
        "tris": tris.astype(np.int32),
        "uvs": uv[tris].astype(np.float32),
        "normals": face_normals.astype(np.float32),
        "view_names": np.array(VIEW_NAMES),
        "ortho_scale": np.float32(ortho),
    }
    view_locations: list[np.ndarray] = []
    per_view: dict[str, Any] = {}
    visible_union = np.zeros(len(tris), dtype=bool)
    failures: list[str] = []

    for name in VIEW_NAMES:
        direction = directions[name]
        screen, depth = project(verts, direction, ortho)
        visible, silhouette = rasterise(screen, depth, tris, raster_size)
        source_box = subject_bbox(masks[name])
        initial_box = subject_bbox(silhouette)
        mesh_box, affine_iou = refine_box(
            masks[name],
            silhouette,
            source_box,
            initial_box,
            raster_size,
        )
        view, alpha, dense = dense_mask_refine(
            images[name],
            masks[name],
            source_box,
            mesh_box,
            silhouette,
            view_size,
            raster_size,
        )
        final_iou = float(
            dense["dense_iou"] if dense.get("accepted") else dense["affine_iou"]
        )
        output_view = registered_dir / f"{name}.png"
        if not cv2.imwrite(str(output_view), np.dstack([view, alpha])):
            raise RuntimeError(f"could not write registered view {output_view}")
        arrays[f"vis_{name}"] = visible
        view_locations.append(direction * (ortho * 3.0))
        visible_union |= visible
        if final_iou < min_registration_iou:
            failures.append(f"MULTIVIEW_REGISTRATION_FAILED_{name.upper()}")
        per_view[name] = {
            "input": str(view_paths[name]),
            "registered": str(output_view),
            "direction": [float(value) for value in direction],
            "affine_iou": round(float(affine_iou), 6),
            "dense_registration": dense,
            "final_registration_iou": round(final_iou, 6),
            "visible_triangles": int(visible.sum()),
            "visible_percent": round(float(visible.mean() * 100.0), 3),
        }

    visible_union_percent = float(visible_union.mean() * 100.0)
    if visible_union_percent < min_visible_union_percent:
        failures.append("MULTIVIEW_VISIBLE_UNION_TOO_LOW")
    arrays["view_locs"] = np.asarray(view_locations, dtype=np.float32)

    metadata = {
        "policy": {"semantic_projection": ["real", "generated"]},
        "front_direction": "+z"
        if directions["front"][2] > 0
        else "-z",
        "views": [
            {
                "view": name,
                "source_type": front_source_type
                if name == "front"
                else "generated",
                "confidence": 1.0 if name == "front" and front_source_type == "real"
                else float(generated_confidence),
            }
            for name in VIEW_NAMES
        ],
    }

    report = {
        "passed": not failures,
        "mesh": str(mesh_path),
        "mesh_triangles": int(len(tris)),
        "view_order": list(VIEW_NAMES),
        "front_back_assignment": front_selection,
        "right_left_assignment": side_selection,
        "visible_union_triangles": int(visible_union.sum()),
        "visible_union_percent": round(visible_union_percent, 3),
        "minimum_registration_iou": float(min_registration_iou),
        "minimum_visible_union_percent": float(min_visible_union_percent),
        "views": per_view,
        "failure_codes": failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if failures:
        return report
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--front", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--back", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--registered-views-dir", required=True)
    parser.add_argument("--view-metadata", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--raster-size", type=int, default=1024)
    parser.add_argument("--view-size", type=int, default=1536)
    parser.add_argument(
        "--min-registration-iou",
        type=float,
        default=DEFAULT_MIN_REGISTRATION_IOU,
    )
    parser.add_argument(
        "--min-visible-union-percent",
        type=float,
        default=DEFAULT_MIN_VISIBLE_UNION_PERCENT,
    )
    parser.add_argument("--generated-confidence", type=float, default=0.78)
    parser.add_argument(
        "--front-source-type",
        choices=("real", "generated"),
        default="generated",
    )
    args = parser.parse_args()

    report = build_bundle(
        mesh_path=Path(args.mesh),
        view_paths={
            "front": Path(args.front),
            "right": Path(args.right),
            "back": Path(args.back),
            "left": Path(args.left),
        },
        output_npz=Path(args.output_npz),
        registered_dir=Path(args.registered_views_dir),
        metadata_path=Path(args.view_metadata),
        report_path=Path(args.report),
        raster_size=args.raster_size,
        view_size=args.view_size,
        min_registration_iou=args.min_registration_iou,
        min_visible_union_percent=args.min_visible_union_percent,
        generated_confidence=args.generated_confidence,
        front_source_type=args.front_source_type,
    )
    print(
        "REGISTERED_MULTIVIEW_BUNDLE "
        f"passed={report['passed']} "
        f"visible_union={report['visible_union_percent']:.3f}% "
        f"failures={report['failure_codes']}",
        flush=True,
    )
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
