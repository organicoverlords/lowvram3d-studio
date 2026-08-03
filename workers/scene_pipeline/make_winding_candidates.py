"""Make winding-reversed copies of already generated CPU candidates."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import trimesh
from trimesh.visual.texture import TextureVisuals

from workers.scene_pipeline.projection import source_uv
from workers.scene_pipeline.source_mesh_repair import ROOT, vertex_normals


def main() -> None:
    points = np.load(ROOT / "points.npy").astype(np.float32)
    mask = np.load(ROOT / "mask.npy").astype(bool)
    source = cv2.cvtColor(cv2.imread(str(ROOT / "source_rgb_512.png")), cv2.COLOR_BGR2RGB)
    valid = mask & np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    pixels = np.flatnonzero(valid.ravel())
    vertices = points.reshape(-1, 3)[pixels]
    uvs = source_uv(512, 384).reshape(-1, 2)[pixels]
    visual = TextureVisuals(uv=uvs.astype(np.float32), image=Image.fromarray(source, mode="RGB"))
    for name in ("adaptive_conservative", "adaptive_balanced", "adaptive_coverage"):
        faces = np.load(ROOT / f"{name}_faces.npy")[:, [0, 2, 1]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=vertex_normals(vertices, faces), visual=visual, process=False)
        out = ROOT / f"{name}_winding.glb"
        mesh.export(out, file_type="glb")
        np.save(ROOT / f"{name}_winding_faces.npy", faces)
        print(out)


if __name__ == "__main__":
    main()
