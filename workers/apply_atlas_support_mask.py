"""Apply a measured direct/conservative atlas mask to an existing diagnostic GLB."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from atlas_raster import rasterise
from fast_texture_projection import bind_texture
from mesh_io import read_glb


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glb", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--support-owner", type=Path, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    positions, normals, uv, tris = read_glb(args.mesh)
    direct, _weights = rasterise(uv, tris, args.resolution)
    support = np.load(args.support_owner)
    if support.shape != direct.shape:
        raise RuntimeError("SUPPORT_OWNER_SHAPE_MISMATCH")
    mask = np.zeros(len(tris), dtype=bool)
    ids = np.concatenate([
        direct[direct >= 0].astype(np.int64, copy=False),
        support[support >= 0].astype(np.int64, copy=False),
    ])
    if ids.size:
        mask[np.unique(ids)] = True
    args.output_glb.parent.mkdir(parents=True, exist_ok=True)
    bind_texture(args.input_glb, args.output_glb, args.atlas.read_bytes(), mask)
    receipt = {
        "schema": "panda_atlas_support_fixed_candidate_v1",
        "classification": "DIAGNOSTIC_ONLY_NOT_PRODUCTION_READY",
        "source_glb": str(args.input_glb),
        "source_glb_sha256": sha256(args.input_glb),
        "atlas_png": str(args.atlas),
        "atlas_png_sha256": sha256(args.atlas),
        "output_glb": str(args.output_glb),
        "output_glb_sha256": sha256(args.output_glb),
        "triangle_count": int(len(tris)),
        "atlas_bound_triangles": int(mask.sum()),
        "neutral_fallback_triangles": int((~mask).sum()),
        "direct_texels": int((direct >= 0).sum()),
        "conservative_support_texels": int((support >= 0).sum()),
        "geometry_source_unchanged": "texture binding only",
        "promotion_authorized": False,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
