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

What this is and is not: faces that actually point at the conditioning camera
get their real appearance. Everything else -- the back, and anything grazing
enough that a few pixels would be smeared down its whole length -- is filled
with the crop's average colour instead. A flat surface is not a claim about what
was there; a stretched one is, and it is a false one.

Coverage is reported, so a mesh whose appearance is mostly invention says so.

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

# A face is textured from the crop only if it actually faces the conditioning
# camera. Below this, the projection is smearing a few pixels along the whole
# length of a surface the camera never saw properly -- the visible streaking on
# the barn's sides -- and a flat colour is a more honest and better-looking
# answer than a stretched one. cos 78 degrees: generous, because a roof pitch
# is steeply angled and still genuinely observed.
FACING_MIN = 0.2
# Width in pixels of the flat-colour strip appended to the texture, which every
# unobserved face is mapped into.
FILL_STRIP_PX = 8


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

    # Only faces pointing at the conditioning camera get the crop. The rest --
    # the back, and anything grazing enough that a handful of pixels would be
    # stretched down its whole length -- are given a flat fill instead, by
    # duplicating their vertices and mapping the copies into a strip appended to
    # the texture. Duplicating only these faces keeps the welding done at
    # decimation: the barn goes from 74,556 vertices to about 110,000, not back
    # to three per triangle.
    faces = np.asarray(mesh.faces)
    observed = mesh.face_normals[:, 2] > FACING_MIN
    unobserved = np.flatnonzero(~observed)
    if len(unobserved):
        used = np.unique(faces[unobserved])
        remap = np.full(len(vertices), -1, dtype=np.int64)
        remap[used] = np.arange(len(used)) + len(vertices)
        vertices = np.vstack([vertices, vertices[used]])
        uv = np.vstack([uv, np.zeros((len(used), 2))])
        faces = faces.copy()
        faces[unobserved] = remap[faces[unobserved]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

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
    # Append the flat-colour strip the unobserved faces are mapped into, and
    # rescale the observed UVs so they still address the crop itself.
    strip = np.repeat(fill[None, None, :], FILL_STRIP_PX, axis=1)
    strip = np.repeat(strip, flattened.shape[0], axis=0)
    flattened = np.concatenate([flattened, strip], axis=1)
    crop_fraction = (flattened.shape[1] - FILL_STRIP_PX) / flattened.shape[1]
    uv[:, 0] *= crop_fraction
    if len(unobserved):
        uv[len(uv) - len(used):] = [1.0 - 0.5 / flattened.shape[1], 0.5]
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
        "facing_min": FACING_MIN,
        "textured_face_count": int(observed.sum()),
        "flat_filled_face_count": int((~observed).sum()),
        "textured_face_fraction": round(float(observed.mean()), 4),
        "vertices": int(len(vertices)),
        "triangles": int(len(mesh.faces)),
        "texture_size": list(texture.size),
        "mask_coverage": round(float(covered.mean()), 4),
        "fill_rgb": [round(float(c), 1) for c in fill],
        # Stated so nothing downstream reads a textured mesh as an observed one.
        "observed_from": (
            "single view; faces below the facing threshold are flat-filled "
            "rather than smeared, so textured_face_fraction is the share of "
            "this mesh's appearance that was actually observed"),
    }
    receipt_path = (Path(args.receipt) if args.receipt
                    else output.with_suffix(".projection.json"))
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
