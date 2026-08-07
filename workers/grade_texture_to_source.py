"""Correct a generated texture's exposure and white balance against its source.

TRELLIS stage 6 produces a texture whose spatial assignment is correct -- the
bow carries the arched facade, the stern the cargo grid, which is exactly what
Hunyuan3D-Paint gets wrong on this subject -- but whose colour is pale and
nearly neutral where the concept art is dark tarred timber with warm lit
windows.

That is the same failure `workers/match_view_colour.py` documents for
MV-Adapter, in a different generator: the six views came back at
[124.5, 123.7, 121.4], three channels within three levels of each other, while
the conditioning photograph was [47.5, 44.1, 38.1]. A generator ignoring the
reference's colour and falling back on its own prior. The fix that worked there
is the smallest one that can be right, and it ports directly.

A single per-channel gain in LINEAR light, fitted on the source's foreground
against the texture's used texels. Linear because exposure and white balance are
multiplicative on radiance and not on sRGB code values; fitting a gain on gamma
-encoded numbers bends the midtones and crushes the ends. Gain only -- no
per-pixel transfer, no histogram matching -- because anything richer invents
local colour the generator did not observe, which is the class of error this
project has spent the most time undoing.

Only texels the mesh actually uses are fitted. An atlas is mostly empty, and
including its background pulls the fit toward whatever the padding happens to be.

    py workers/grade_texture_to_source.py --mesh in.glb --source matte.png \
       --out graded.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

#: sRGB transfer, exact rather than the 2.2 approximation, because the toe
#: matters for a subject this dark.
def to_linear(srgb: np.ndarray) -> np.ndarray:
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def to_srgb(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, linear * 12.92,
                    1.055 * linear ** (1 / 2.4) - 0.055)


def used_texels(mesh, size: tuple[int, int]) -> np.ndarray:
    """Boolean mask of atlas pixels covered by at least one UV vertex.

    Vertex UVs rather than rasterised triangles: cheaper, and the fit only
    needs a representative sample, not exact coverage.
    """
    uv = np.asarray(mesh.visual.uv, dtype=np.float64)
    x = np.clip((uv[:, 0] % 1.0) * (size[0] - 1), 0, size[0] - 1).astype(int)
    y = np.clip((1.0 - uv[:, 1] % 1.0) * (size[1] - 1), 0, size[1] - 1).astype(int)
    mask = np.zeros((size[1], size[0]), dtype=bool)
    mask[y, x] = True
    return mask


def run(mesh_path: Path, source_path: Path, out_path: Path,
        dilate: int = 2) -> dict:
    import trimesh
    from scipy import ndimage

    scene = trimesh.load(mesh_path, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    texture = mesh.visual.material.baseColorTexture.convert("RGB")
    pixels = np.asarray(texture).astype(np.float64) / 255.0

    mask = used_texels(mesh, texture.size)
    if dilate:
        mask = ndimage.binary_dilation(mask, iterations=dilate)

    source = Image.open(source_path).convert("RGBA")
    source_rgb = np.asarray(source)[..., :3].astype(np.float64) / 255.0
    foreground = np.asarray(source)[..., 3] > 16

    texture_mean = to_linear(pixels[mask]).mean(axis=0)
    source_mean = to_linear(source_rgb[foreground]).mean(axis=0)
    gain = source_mean / np.maximum(texture_mean, 1e-6)

    graded = to_srgb(to_linear(pixels) * gain)
    mesh.visual.material.baseColorTexture = Image.fromarray(
        (graded * 255.0 + 0.5).astype(np.uint8))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)

    return {
        "schema": "lowvram3d_texture_grade_v1",
        "mesh_in": str(mesh_path),
        "source": str(source_path),
        "mesh_out": str(out_path),
        "atlas_size": list(texture.size),
        "used_texel_fraction": round(float(mask.mean()), 4),
        "texture_linear_mean": [round(v, 5) for v in texture_mean],
        "source_linear_mean": [round(v, 5) for v in source_mean],
        "gain": [round(v, 4) for v in gain],
        # Neutrality of the input, the tell that a generator fell back on its
        # own prior rather than reading the reference.
        "input_channel_spread": round(
            float(texture_mean.max() / max(texture_mean.min(), 1e-6)), 4),
        "note": ("single per-channel gain in linear light; no per-pixel "
                 "transfer, so no local colour is invented"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dilate", type=int, default=2)
    args = parser.parse_args(argv)

    result = run(args.mesh, args.source, args.out, args.dilate)
    args.out.with_suffix(".grade.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
