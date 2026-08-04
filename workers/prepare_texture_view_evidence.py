"""Convert the projection mesh cache into per-view evidence records.

The view builder already emits exact frontmost triangle-ID buffers. This adapter makes those
buffers consumable by the generic evidence/classification stages without re-rendering a camera or
changing camera semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def prepare(npz_path: Path, views_dir: Path, output_dir: Path) -> dict:
    data = np.load(npz_path, allow_pickle=False)
    names = [str(x) for x in data["view_names"]]
    normals = np.asarray(data["normals"], np.float32)
    locations = np.asarray(data["view_locs"], np.float32)
    manifest = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(names):
        face = np.asarray(data[f"face_id_{name}"], np.int32)
        # ``vis_<view>`` in the projection cache is a per-triangle mask; the exact pixel
        # evidence mask is the frontmost triangle-ID buffer itself.
        visible = face >= 0
        image = cv2.imread(str(views_dir / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"VIEW_IMAGE_MISSING:{name}")
        alpha = (image[:, :, 3].astype(np.float32) / 255.0
                 if image.ndim == 3 and image.shape[2] >= 4
                 else np.ones(face.shape, np.float32))
        if alpha.shape != face.shape:
            raise RuntimeError(f"VIEW_IMAGE_DIMENSION_MISMATCH:{name}")
        direction = locations[index] / max(float(np.linalg.norm(locations[index])), 1e-12)
        facing = np.zeros(face.shape, np.float32)
        valid = face >= 0
        facing[valid] = normals[face[valid]] @ direction
        path = output_dir / f"view_evidence_{name}.npz"
        np.savez_compressed(path, triangle_id=face, barycentric=np.zeros((*face.shape, 3), np.float32),
                            depth=np.asarray(data[f"vis_{name}"], np.float32), visible=visible,
                            triangle_visibility=visible, exact_triangle_id_match=valid,
                            normal_facing=facing, projected_triangle_area=np.zeros(face.shape, np.float32),
                            source_alpha=alpha, camera_hash=hashlib.sha256(direction.tobytes()).hexdigest(),
                            mesh_hash="projection_cache", semantic_source_class="ORIGINAL_NONFACE")
        manifest.append({"view_name": name, "path": str(path), "source_class": "ORIGINAL_NONFACE",
                         "camera_hash": hashlib.sha256(direction.tobytes()).hexdigest(),
                         "mesh_hash": "projection_cache"})
    return {"schema": "view_evidence_manifest_v2", "views": manifest,
            "triangle_count": int(len(normals)), "backend": "existing_exact_triangle_id_buffer"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-npz", required=True)
    parser.add_argument("--views-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    manifest = prepare(Path(args.projection_npz), Path(args.views_dir), Path(args.output_dir))
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
