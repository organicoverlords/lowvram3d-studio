"""Build the observation cache once and emit a bounded grid of fusion variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from fast_texture_projection import bind_texture, immutable_buffer_hashes
from multiview_texture_fusion import (
    build_cache,
    fuse,
    protected_weights,
    sha256_bytes,
    sha256_file,
)
from multiview_texture_projection import push_pull_fill


def write_maps(output_dir: Path, result: dict, cache: dict, observed_map, synthesized_map,
               confidence_map, multiview_map) -> dict:
    size = cache["atlas_size"]
    seam = np.zeros((size, size), bool)
    ownership = result["ownership_map"]
    owned = cache["owned"]
    for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
        seam |= owned & (np.roll(ownership, shift, axis=axis) != ownership) & (
            np.roll(owned, shift, axis=axis))
    ratio_map = np.zeros((size, size), np.float32)
    ratio_map[owned] = np.clip(result["ratio_value"] / 4.0, 0, 1)
    files = {
        "ownership_map.png": ((ownership + 1) * 40).astype(np.uint8),
        "raw_ownership_map.png": ((result["raw_ownership_map"] + 1) * 40).astype(np.uint8),
        "ownership_confidence_map.png": (np.clip(confidence_map, 0, 1) * 255).astype(np.uint8),
        "leader_runner_ratio_map.png": (ratio_map * 255).astype(np.uint8),
        "protected_region_ownership_map.png": (result["protected_map"] * 255).astype(np.uint8),
        "observed_coverage_mask.png": (observed_map * 255).astype(np.uint8),
        "synthesized_coverage_mask.png": (synthesized_map * 255).astype(np.uint8),
        "multiview_coverage_mask.png": (multiview_map * 255).astype(np.uint8),
        "seam_map.png": (seam * 255).astype(np.uint8),
    }
    for name, array in files.items():
        Image.fromarray(array, "L").save(output_dir / name)
    return {name: str(output_dir / name) for name in files}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--views-receipt", required=True)
    parser.add_argument("--variants", required=True, help="JSON list of variant definitions")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--region-config", default=None)
    parser.add_argument("--atlas-size", type=int, default=2048)
    parser.add_argument("--depth-tolerance", type=float, default=0.010)
    parser.add_argument("--min-facing-cosine", type=float, default=0.20)
    parser.add_argument("--detail-radius", type=int, default=3)
    parser.add_argument("--edge-bleed", type=int, default=12)
    args = parser.parse_args()

    mesh = Path(args.mesh)
    bundle = Path(args.bundle)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(Path(args.views_receipt).read_text(encoding="utf-8"))
    variants = json.loads(Path(args.variants).read_text(encoding="utf-8"))

    cache = build_cache(mesh, bundle, receipt, args.atlas_size, args.depth_tolerance,
                        args.min_facing_cosine, args.detail_radius)
    regions = protected_weights(cache, Path(args.region_config) if args.region_config else None)
    owned = cache["owned"]
    owned_count = int(owned.sum())
    baseline_hashes = immutable_buffer_hashes(mesh)

    face_diagnostics = {}
    for name, record in regions.items():
        inside = record["weight"] > 0.5
        slot = record["owner_slot"]
        face_diagnostics[name] = {
            "texels_in_region": int(inside.sum()),
            "owner_semantic": record["owner_semantic"],
            "owner_valid_fraction": float((cache["confidence"][slot][inside] > 0).mean())
            if inside.any() else 0.0,
            "owner_confidence_percentiles": {
                str(q): float(np.percentile(cache["confidence"][slot][inside], q))
                for q in (10, 25, 50, 75, 90)} if inside.any() else {},
            "mean_observations": float(
                (cache["confidence"][:, inside] > 0).sum(axis=0).mean()) if inside.any() else 0.0,
        }

    results = {}
    for variant in variants:
        name = str(variant["name"])
        target = output_root / name
        target.mkdir(parents=True, exist_ok=True)
        result = fuse(
            cache, regions,
            mode=str(variant["mode"]),
            ratio=float(variant.get("ratio", 1.30)),
            margin=float(variant.get("margin", 0.10)),
            grazing_cosine=float(variant.get("grazing_cosine", 0.40)),
            detail_ratio=float(variant.get("detail_ratio", 1.60)),
            protected_min_confidence=float(variant.get("protected_min_confidence", 0.0)),
            colour_compatibility=float(variant.get("colour_compatibility", 60.0)),
            regularise_radius=int(variant.get("regularise_radius", 1)))

        size = cache["atlas_size"]
        atlas = np.zeros((size, size, 3), np.float64)
        atlas[owned] = result["colour"]
        filled = np.zeros((size, size), bool)
        filled[owned] = result["observed"]
        atlas, synthesized, resolved = push_pull_fill(atlas, filled)
        synthesized_on_surface = synthesized & owned
        unresolved = owned & ~resolved
        bled, _mask, _ = push_pull_fill(atlas, resolved, rounds=args.edge_bleed)
        atlas_path = target / "panda_multiview_basecolor.png"
        Image.fromarray(np.clip(bled, 0, 255).astype(np.uint8)).save(atlas_path)

        confidence_map = np.zeros((size, size), np.float32)
        confidence_map[owned] = result["leader_confidence"]
        multiview_map = np.zeros((size, size), bool)
        multiview_map[owned] = result["observation_count"] >= 2
        maps = write_maps(target, result, cache, filled, synthesized_on_surface,
                          confidence_map, multiview_map)

        glb_path = target / "tactical_red_panda_scout_textured.glb"
        atlas_bytes = atlas_path.read_bytes()
        bind_texture(mesh, glb_path, atlas_bytes)
        after = immutable_buffer_hashes(glb_path)

        def percent(count: int) -> float:
            return round(100.0 * count / owned_count, 4)

        results[name] = {
            "settings": variant,
            "atlas": str(atlas_path),
            "atlas_sha256": sha256_bytes(atlas_bytes),
            "glb": str(glb_path),
            "glb_sha256": sha256_file(glb_path),
            "geometry_buffers_unchanged": after == baseline_hashes,
            "geometry_buffers": after,
            "winner_take_all_fraction": result["winner_take_all_fraction"],
            "ownership_regularisation": result["regularisation"],
            "coverage": {
                "atlas_texels_with_geometry": owned_count,
                "directly_observed_percent": percent(int(result["observed"].sum())),
                "multiview_observed_percent": percent(
                    int((result["observation_count"] >= 2).sum())),
                "blended_from_multiple_views_percent": percent(
                    int((result["blend_count"] >= 2).sum())),
                "synthesized_percent": percent(int(synthesized_on_surface.sum())),
                "unresolved_percent": percent(int(unresolved.sum())),
            },
            "ownership_share": {
                cache["semantics"][slot]: percent(int((result["ownership"] == slot).sum()))
                for slot in range(len(cache["semantics"]))},
            "maps": maps,
        }
        print(f"FUSION_VARIANT {name} observed={results[name]['coverage']['directly_observed_percent']}%"
              f" synth={results[name]['coverage']['synthesized_percent']}%"
              f" wta={result['winner_take_all_fraction']:.3f}", flush=True)

    report = {
        "schema": "fusion_variant_grid_v1",
        "mesh": str(mesh),
        "mesh_sha256": sha256_file(mesh),
        "bundle": str(bundle),
        "region_config": args.region_config,
        "cache_settings": {
            "atlas_size": args.atlas_size, "depth_tolerance": args.depth_tolerance,
            "min_facing_cosine": args.min_facing_cosine, "detail_radius": args.detail_radius,
            "edge_bleed": args.edge_bleed},
        "protected_region_diagnostics": face_diagnostics,
        "per_view": cache["diagnostics"],
        "variants": results,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"FUSION_GRID_DONE variants={len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
