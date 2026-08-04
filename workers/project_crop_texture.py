"""Give a generated mesh the appearance of the crop it was generated from.

Mini Turbo returns geometry only, so every generated asset renders as untextured
grey. The pixels it was conditioned on are still on disk next to it, and the
mesh is aligned to that view by construction -- an image-to-3D model produces
its subject facing the conditioning camera -- so the crop can be projected
straight back onto the mesh.

Projection is along the mesh's own -Z (its front), mapping the front-facing
bounding box to the crop. That keeps UVs in *object* space, which matters
because a scatter region reuses one mesh across every instance it placed: a
world-space projection would need one baked copy per instance.

What this is and is not: the front of the object gets its real appearance, and
the sides stretch along the projection axis. That is inherent to texturing from
a single view and is exactly the artefact `offaxis_stability` exists to catch.
It is not a claim that the back of the barn was observed.

    py -3.12 workers/project_crop_texture.py --glb in.glb --crop crop.png \
        --output textured.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Fraction of the bounding box to pad the projection by, so vertices exactly on
# the silhouette do not land on the crop's very edge pixel.
EDGE_PAD = 0.01


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glb", required=True)
    parser.add_argument("--crop", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh
    from PIL import Image

    source = Path(args.glb).resolve()
    crop_path = Path(args.crop).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    scene = trimesh.load(str(source), process=False)
    geometries = (list(scene.geometry.values())
                  if hasattr(scene, "geometry") else [scene])
    mesh = (trimesh.util.concatenate(geometries) if len(geometries) > 1
            else geometries[0])

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    span = np.maximum(high - low, 1e-9)
    pad = span * EDGE_PAD
    low, high = low - pad, high + pad
    span = high - low

    # Object-space front projection: mesh +X is the image's right and mesh +Y is
    # its up, so u runs with X and v runs *against* Y -- glTF's v origin is the
    # top of the texture, not the bottom.
    u = (vertices[:, 0] - low[0]) / span[0]
    v = 1.0 - (vertices[:, 1] - low[1]) / span[1]
    uv = np.clip(np.stack([u, v], axis=-1), 0.0, 1.0)

    # The crop is matted; flatten it onto its own average colour rather than
    # onto white, or every silhouette edge is fringed with a bright halo where
    # the projection spills past the subject.
    crop = Image.open(crop_path).convert("RGBA")
    pixels = np.asarray(crop, dtype=np.float64)
    alpha = pixels[:, :, 3:4] / 255.0
    covered = alpha[:, :, 0] > 0.5
    fill = (pixels[covered][:, :3].mean(axis=0) if covered.any()
            else np.array([128.0, 128.0, 128.0]))
    flattened = pixels[:, :, :3] * alpha + fill[None, None, :] * (1.0 - alpha)
    texture = Image.fromarray(flattened.astype(np.uint8), mode="RGB")

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=texture,
        metallicFactor=0.0,
        roughnessFactor=0.9,
    )
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    mesh.export(str(output))

    receipt = {
        "schema_version": "crop_projection_v1",
        "classification": "PROVEN",
        "glb": str(source),
        "crop": str(crop_path),
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "projection": "object_space_front_minus_z",
        "vertices": int(len(vertices)),
        "triangles": int(len(mesh.faces)),
        "texture_size": list(texture.size),
        "mask_coverage": round(float(covered.mean()), 4),
        "fill_rgb": [round(float(c), 1) for c in fill],
        # Stated so nothing downstream reads a textured mesh as an observed one.
        "observed_from": "single view; sides stretch along the projection axis",
    }
    receipt_path = (Path(args.receipt) if args.receipt
                    else output.with_suffix(".projection.json"))
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
