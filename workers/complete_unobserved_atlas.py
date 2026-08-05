"""Complete only genuinely unobserved atlas texels, preserving per-texel provenance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lowvram3d.texture_provenance import (
    EvidenceState, FrequencyAuthority, Lineage, SourceClass, load_npz, save_npz,
)


def _oriented(value: np.ndarray) -> np.ndarray:
    """Convert one internal raster array to canonical glTF row orientation exactly once."""
    return np.flip(np.asarray(value), axis=0).copy()


def _load_projection_atlas(triangle_provenance: Path, explicit_path: Path | None = None) -> dict[str, np.ndarray]:
    path = explicit_path or triangle_provenance.with_name("atlas_provenance.npz")
    if not path.exists():
        raise RuntimeError("PER_TEXEL_PROVENANCE_MISSING")
    atlas = load_npz(path)
    required = {
        "triangle_id", "direct_observed_texel_mask", "atlas_occupied_mask",
        "source_view", "source_pixel", "barycentric", "visibility", "facing",
        "face_id_match", "source_mask_valid", "confidence", "evidence_state",
    }
    missing = sorted(required.difference(atlas))
    if missing:
        raise RuntimeError("PER_TEXEL_PROVENANCE_FIELDS_MISSING:" + ",".join(missing))
    return {name: _oriented(value) if np.asarray(value).ndim >= 2 else value
            for name, value in atlas.items()}


def complete(projection_npz: Path, basecolor: Path, triangle_provenance: Path,
             regions_path: Path, output: Path, output_provenance: Path,
             atlas_provenance: Path | None = None) -> dict:
    image_bgr = cv2.imread(str(basecolor), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError("BASECOLOR_UNREADABLE")
    out = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    atlas = _load_projection_atlas(triangle_provenance, atlas_provenance)
    owner = np.asarray(atlas["triangle_id"], np.int32)
    occupied = np.asarray(atlas["atlas_occupied_mask"], bool)
    direct = np.asarray(atlas["direct_observed_texel_mask"], bool)
    gap = np.asarray(atlas.get("visible_source_gap_mask", np.zeros_like(direct)), bool)
    generated = np.asarray(atlas.get("generated_observed_mask", np.zeros_like(direct)), bool)
    known = (direct | gap | generated) & occupied
    if out.shape[:2] != owner.shape:
        raise RuntimeError("ATLAS_PROVENANCE_DIMENSION_MISMATCH")

    regions = np.asarray(np.load(regions_path), np.int32)
    if owner.size and (regions.size == 0 or int(owner[owner >= 0].max(initial=-1)) >= len(regions)):
        raise RuntimeError("SURFACE_REGION_SHAPE_MISMATCH")
    safe_owner = np.clip(owner, 0, max(len(regions) - 1, 0))
    region_atlas = np.full(owner.shape, -1, np.int32)
    valid_owner = occupied & (owner >= 0)
    region_atlas[valid_owner] = regions[safe_owner[valid_owner]]
    targets = occupied & ~known

    # Aggregate only already-observed pixels. No source-image pixel is copied to an
    # unobserved target, and the fallback remains low-frequency and provenance-explicit.
    nonblack_known = known & np.any(out != 0, axis=2)
    global_palette = (np.mean(out[nonblack_known], axis=0).astype(np.float32)
                      if nonblack_known.any() else np.array([96.0, 96.0, 96.0], np.float32))
    for region in np.unique(region_atlas[targets]):
        if int(region) < 0:
            continue
        donors = nonblack_known & (region_atlas == int(region))
        wanted = targets & (region_atlas == int(region))
        if not wanted.any():
            continue
        palette = np.mean(out[donors], axis=0).astype(np.float32) if donors.any() else global_palette
        out[wanted] = np.clip(palette, 0, 255).astype(np.uint8)
    if targets.any() & ~(np.isin(region_atlas, np.unique(region_atlas[known]))).any():
        out[targets & (region_atlas < 0)] = np.clip(global_palette, 0, 255).astype(np.uint8)

    # Mutate the existing atlas provenance in place. Direct and visible-gap fields are not
    # recreated from triangle state and therefore cannot be promoted across a partial triangle.
    before = {key: np.array(atlas[key], copy=True) for key in (
        "triangle_id", "source_view", "source_pixel", "barycentric", "confidence",
        "visibility", "facing", "face_id_match", "source_mask_valid", "evidence_state",
    )}
    atlas["atlas_occupied_mask"] = occupied
    atlas["uv_occupied_mask"] = occupied
    atlas["direct_observed_texel_mask"] = direct
    atlas["direct_observed"] = direct
    atlas["visible_source_gap_mask"] = gap
    atlas["visible_source_gap"] = gap
    atlas["generated_observed_mask"] = generated
    atlas["generated_observed"] = generated
    atlas["unobserved_surface_mask"] = targets
    atlas["unobserved_surface"] = targets
    atlas["procedural_completion_mask"] = targets
    atlas["procedural_completion"] = targets
    atlas["material_prior_mask"] = np.zeros_like(targets)
    atlas["material_prior"] = np.zeros_like(targets)
    atlas["unresolved_mask"] = np.zeros_like(targets)
    atlas["unresolved"] = np.zeros_like(targets)
    flat = targets
    atlas["evidence_state"][flat] = np.uint8(EvidenceState.PROCEDURAL_COMPLETION)
    atlas["source_class"][flat] = np.uint8(SourceClass.COMPONENT_PRIOR)
    atlas["lineage"][flat] = np.uint16(Lineage.COMPONENT_PRIOR)
    atlas["lineage_bits"][flat] = np.uint16(Lineage.COMPONENT_PRIOR)
    atlas["source_view"][flat] = -1
    atlas["primary_view"][flat] = -1
    atlas["source_pixel"][flat] = -1
    atlas["barycentric"][flat] = 0.0
    atlas["visibility"][flat] = False
    atlas["facing"][flat] = 0.0
    atlas["face_id_match"][flat] = False
    atlas["source_mask_valid"][flat] = False
    atlas["confidence"][flat] = 0.0
    atlas["frequency_authority"][flat] = np.uint8(FrequencyAuthority.LOW_ONLY)
    atlas["completion_method"][flat] = "region_material_palette"
    atlas["primary_surface_region"][flat] = region_atlas[flat]

    preserved = all(np.array_equal(before[key][direct], atlas[key][direct]) for key in before)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    save_npz(output_provenance, atlas)
    export_fields = {
        "atlas_owner_triangle": "triangle_id", "uv_occupied_mask": "uv_occupied_mask",
        "direct_observed_texel_mask": "direct_observed_texel_mask",
        "visible_source_gap_mask": "visible_source_gap_mask",
        "unobserved_surface_mask": "unobserved_surface_mask",
        "procedural_completion_mask": "procedural_completion_mask",
        "unresolved_mask": "unresolved_mask", "direct_visibility": "visibility",
        "direct_face_id_match": "face_id_match", "direct_source_view": "source_view",
        "direct_source_pixel": "source_pixel", "direct_source_mask_valid": "source_mask_valid",
        "direct_triangle_id": "triangle_id",
    }
    for name, field in export_fields.items():
        value = atlas.get(field)
        if value is not None:
            np.save(output_provenance.with_name(name + ".npy"), value)
    return {
        "schema": "unobserved_atlas_completion_v2",
        "direct_pixels_preserved": int(direct.sum()),
        "visible_gap_pixels_preserved": int(gap.sum()),
        "procedural_pixels": int(targets.sum()),
        "raw_rgb_donor_pixels": 0,
        "direct_provenance_preserved_after_completion": bool(preserved),
        "unknown_source_class_on_occupied_texels": int(np.count_nonzero(
            occupied & (atlas["source_class"] == SourceClass.UNKNOWN))),
        "completion_frequency_authority": "LOW_ONLY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection-npz", required=True)
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--triangle-provenance", required=True)
    parser.add_argument("--atlas-provenance", default="")
    parser.add_argument("--regions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-provenance", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = complete(Path(args.projection_npz), Path(args.basecolor),
                      Path(args.triangle_provenance), Path(args.regions),
                      Path(args.output), Path(args.output_provenance),
                      Path(args.atlas_provenance) if args.atlas_provenance else None)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
