"""Fill unresolved UV triangles from aggregate region palettes, never source-image pixels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from atlas_raster import rasterise
from lowvram3d.texture_provenance import EvidenceState, FrequencyAuthority, create_empty_atlas_provenance, load_npz, save_npz


def complete(npz_path: Path, basecolor: Path, triangle_provenance: Path,
             regions_path: Path, output: Path, output_provenance: Path) -> dict:
    mesh = np.load(npz_path, allow_pickle=False)
    uv = np.asarray(mesh["uvs"], np.float32)
    uv_vertices = uv.reshape(-1, 2)
    triangles = np.arange(len(uv_vertices), dtype=np.int32).reshape(-1, 3)
    owner, _weights = rasterise(uv_vertices, triangles, int(cv2.imread(str(basecolor)).shape[0]))
    image = cv2.cvtColor(cv2.imread(str(basecolor), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    tri = load_npz(triangle_provenance)
    regions = np.load(regions_path).astype(np.int32)
    direct = np.asarray(tri["evidence_state"]) == EvidenceState.DIRECT_OBSERVED
    if len(regions) != len(uv):
        raise RuntimeError("SURFACE_REGION_SHAPE_MISMATCH")
    colours = np.zeros((len(uv), 3), np.float32)
    counts = np.zeros(len(uv), np.int32)
    ids = owner.reshape(-1)
    pixels = image.reshape(-1, 3)
    valid = (ids >= 0) & direct[np.clip(ids, 0, len(direct) - 1)]
    np.add.at(colours, ids[valid], pixels[valid])
    np.add.at(counts, ids[valid], 1)
    colours /= np.maximum(counts[:, None], 1)
    out = image.copy()
    filled = np.zeros(owner.shape, bool)
    for region in np.unique(regions):
        donor = direct & (regions == region) & (counts > 0)
        targets = (~direct) & (regions == region)
        if not targets.any():
            continue
        palette = np.median(colours[donor], axis=0) if donor.any() else np.array([96, 96, 96], np.float32)
        target_pixels = np.isin(owner, np.flatnonzero(targets))
        out[target_pixels] = np.clip(palette, 0, 255).astype(np.uint8)
        filled[target_pixels] = True
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    atlas = create_empty_atlas_provenance(out.shape[0], out.shape[1])
    atlas["triangle_id"] = owner
    atlas["evidence_state"][owner >= 0] = np.uint8(EvidenceState.PROCEDURAL_COMPLETION)
    atlas["frequency_authority"][owner >= 0] = np.uint8(FrequencyAuthority.LOW_ONLY)
    atlas["completion_method"][owner >= 0] = "region_material_palette"
    direct_pixels = (owner >= 0) & direct[np.clip(owner, 0, len(direct) - 1)]
    atlas["evidence_state"][direct_pixels] = np.uint8(EvidenceState.DIRECT_OBSERVED)
    atlas["frequency_authority"][direct_pixels] = np.uint8(FrequencyAuthority.FULL)
    atlas["completion_method"][direct_pixels] = "direct_projection"
    save_npz(output_provenance, atlas)
    return {"schema": "unobserved_atlas_completion_v1", "output": str(output),
            "direct_pixels_preserved": int(direct_pixels.sum()),
            "procedural_pixels": int(filled.sum()), "raw_rgb_donor_pixels": 0,
            "frequency_authority": "LOW_ONLY_FOR_COMPLETION"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-npz", required=True)
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--triangle-provenance", required=True)
    parser.add_argument("--regions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-provenance", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = complete(Path(args.projection_npz), Path(args.basecolor),
                      Path(args.triangle_provenance), Path(args.regions),
                      Path(args.output), Path(args.output_provenance))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
