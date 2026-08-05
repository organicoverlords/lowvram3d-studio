"""Geometry-preservation receipt and vision-aligned bar gate for the local bar repair.

Phase 3 asks whether anything outside the repair moved; phase 4 asks whether the bar is
gone from the views where it was visible.  Both are answered against the rendered
triangle-ID buffers rather than against a repainted silhouette, so a pixel that changes
is attributed to a specific face rather than to a rasteriser difference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mesh_io import read_glb

# Raw camera index -> the label the proven camera contract assigns to it.  Never derive
# these from a positional tuple: the raw bundle names indices 0-3 "horizontal_*".
RAW_INDEX_COUNT = 6


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def bundle_views(bundle: Path) -> list[dict]:
    contract = json.loads((bundle / "camera_contract.json").read_text(encoding="utf-8"))
    return contract["views"]


def direction_key(view: dict) -> tuple[int, int, int]:
    """Round the camera direction to a lattice key so two bundles can be paired.

    Indices cannot be paired directly: a relabelled bundle renumbers its views, so raw
    index 0 in one bundle and index 0 in another need not be the same camera.  The
    camera direction is the only identity that survives relabelling.
    """
    return tuple(int(round(float(component))) for component in view["camera_direction"])


def file_prefix(view: dict) -> str:
    """On-disk array prefix, which stops being the semantic label once a bundle is relabelled."""
    return str(view.get("control_file_prefix") or view["semantic_name"])


def load_view(bundle: Path, prefix: str):
    ids = np.load(bundle / f"{prefix}_triangle_ids.npy")
    mask = np.asarray(Image.open(bundle / f"{prefix}_mask.png").convert("L")) > 127
    return ids, mask


def overlay(base: np.ndarray, colour, image: Image.Image) -> Image.Image:
    pixels = np.asarray(image.convert("RGB")).copy()
    pixels[base] = colour
    return Image.fromarray(pixels)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-glb", required=True)
    parser.add_argument("--repaired-glb", required=True)
    parser.add_argument("--repair-report", required=True)
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--repaired-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    canonical = Path(args.canonical_glb)
    repaired = Path(args.repaired_glb)
    source_bundle = Path(args.source_bundle)
    repaired_bundle = Path(args.repaired_bundle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repair = json.loads(Path(args.repair_report).read_text(encoding="utf-8"))

    removed = set(repair["removed_face_ids"]) | set(repair["dropped_orphan_face_ids"])
    bar_faces = np.asarray(sorted(set(repair["removed_face_ids"])), dtype=np.int64)
    fragment_faces = np.asarray(sorted(set(repair["dropped_orphan_face_ids"])), dtype=np.int64)

    # ---------------------------------------------------------------- preservation
    src_pos, src_nrm, src_uv, src_tris = read_glb(canonical)
    out_pos, out_nrm, out_uv, out_tris = read_glb(repaired)
    keep = np.ones(len(src_tris), dtype=bool)
    keep[np.asarray(sorted(removed), dtype=np.int64)] = False
    kept_face_ids = np.where(keep)[0]
    expected = src_tris[keep]
    preservation = {
        "canonical_sha256": sha256_file(canonical),
        "repaired_sha256": sha256_file(repaired),
        "positions_sha256_before": sha256_array(src_pos),
        "positions_sha256_after": sha256_array(out_pos),
        "normals_sha256_before": sha256_array(src_nrm),
        "normals_sha256_after": sha256_array(out_nrm),
        "uv_sha256_before": sha256_array(src_uv),
        "uv_sha256_after": sha256_array(out_uv),
        "unaffected_triangles_sha256_before": sha256_array(expected),
        "unaffected_triangles_sha256_after": sha256_array(out_tris),
        "unaffected_triangles_identical": bool(np.array_equal(expected, out_tris)),
        "vertex_count_before": int(len(src_pos)),
        "vertex_count_after": int(len(out_pos)),
        "triangle_count_before": int(len(src_tris)),
        "triangle_count_after": int(len(out_tris)),
        "bounds_before": {"min": src_pos.min(axis=0).tolist(), "max": src_pos.max(axis=0).tolist()},
        "bounds_after": {"min": out_pos[np.unique(out_tris)].min(axis=0).tolist(),
                         "max": out_pos[np.unique(out_tris)].max(axis=0).tolist()},
        "no_global_resampling": bool(len(src_pos) == len(out_pos)),
        "no_global_smoothing": sha256_array(src_nrm) == sha256_array(out_nrm),
    }

    # ---------------------------------------------------------------- vision gate
    source_views = {direction_key(view): view for view in bundle_views(source_bundle)}
    repaired_views = {direction_key(view): view for view in bundle_views(repaired_bundle)}
    if (len(source_views) != RAW_INDEX_COUNT or len(repaired_views) != RAW_INDEX_COUNT
            or set(source_views) != set(repaired_views)):
        raise RuntimeError("BAR_PROOF_CAMERA_DIRECTION_MISMATCH")

    views, gate_failures = {}, []
    pairing = []
    for key in sorted(source_views):
        source_view, repaired_view = source_views[key], repaired_views[key]
        index = int(repaired_view["index"])
        source_map = {index: file_prefix(source_view)}
        repaired_map = {index: file_prefix(repaired_view)}
        pairing.append({
            "camera_direction": list(key),
            "source_raw_index": int(source_view["index"]),
            "source_prefix": source_map[index],
            "repaired_raw_index": index,
            "repaired_prefix": repaired_map[index],
            "azimuth_deg": repaired_view["azimuth_deg"],
            "elevation_deg": repaired_view["elevation_deg"],
        })
        src_ids, src_mask = load_view(source_bundle, source_map[index])
        rep_ids, rep_mask = load_view(repaired_bundle, repaired_map[index])
        bar_pixels = int(np.isin(src_ids, bar_faces).sum())
        fragment_pixels = int(np.isin(src_ids, fragment_faces).sum())
        lost = src_mask & ~rep_mask
        gained = rep_mask & ~src_mask
        removed_visible = np.isin(src_ids, np.concatenate((bar_faces, fragment_faces)))
        unexplained = int((lost & ~removed_visible).sum())
        record = {
            "raw_index": index,
            "source_prefix": source_map[index],
            "repaired_prefix": repaired_map[index],
            "semantic_label": str(repaired_view.get("proven_semantic") or repaired_map[index]),
            "source_foreground_pixels": int(src_mask.sum()),
            "repaired_foreground_pixels": int(rep_mask.sum()),
            "bar_pixels_in_source": bar_pixels,
            "fragment_pixels_in_source": fragment_pixels,
            "pixels_lost": int(lost.sum()),
            "pixels_gained": int(gained.sum()),
            "pixels_lost_not_attributable_to_removed_faces": unexplained,
            # Removal renumbers faces, so a repaired ID must be mapped back through the
            # keep mask before it can be compared with the removed set.
            "removed_faces_still_visible_in_repaired": int(np.isin(
                kept_face_ids[rep_ids[rep_ids >= 0]],
                np.concatenate((bar_faces, fragment_faces))).sum()),
        }
        if record["pixels_gained"]:
            gate_failures.append(f"index_{index}_gained_pixels")
        if unexplained:
            gate_failures.append(f"index_{index}_unexplained_loss")
        views[str(index)] = record

        base = Image.fromarray(np.stack([src_mask.astype(np.uint8) * 60] * 3, axis=-1))
        image = overlay(removed_visible, (255, 40, 40), base)
        image = overlay(lost & ~removed_visible, (255, 220, 0), image)
        image = overlay(gained, (0, 160, 255), image)
        image.resize((image.width * 2, image.height * 2), Image.NEAREST).save(
            output_dir / f"raw{index}_{source_map[index]}_silhouette_delta.png")

        ids_visible = rep_ids.copy()
        colour = np.zeros(ids_visible.shape + (3,), dtype=np.uint8)
        finite = ids_visible >= 0
        colour[..., 0] = np.where(finite, (ids_visible * 37) % 256, 0)
        colour[..., 1] = np.where(finite, (ids_visible * 91) % 256, 0)
        colour[..., 2] = np.where(finite, (ids_visible * 151) % 256, 0)
        Image.fromarray(colour).save(output_dir / f"raw{index}_{repaired_map[index]}_face_ids.png")

        edge = finite & ~np.pad(finite, 1, mode="constant")[2:, 1:-1] \
            | finite & ~np.pad(finite, 1, mode="constant")[:-2, 1:-1] \
            | finite & ~np.pad(finite, 1, mode="constant")[1:-1, 2:] \
            | finite & ~np.pad(finite, 1, mode="constant")[1:-1, :-2]
        boundary = np.zeros(finite.shape + (3,), dtype=np.uint8)
        boundary[finite] = (40, 40, 46)
        boundary[edge] = (255, 255, 255)
        Image.fromarray(boundary).save(
            output_dir / f"raw{index}_{repaired_map[index]}_boundary_overlay.png")

        # Face-level close-up of the repaired region, in the camera frame the controls
        # were built in, so the crop box is derived from the removed faces themselves.
        if removed_visible.any():
            ys, xs = np.nonzero(removed_visible)
            pad = 24
            box = (max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0),
                   min(int(xs.max()) + pad + 1, ids_visible.shape[1]),
                   min(int(ys.max()) + pad + 1, ids_visible.shape[0]))
            record["repair_region_crop_xyxy"] = list(box)
            panels = []
            for ids_array, tint in ((src_ids, (255, 40, 40)), (rep_ids, None)):
                face = ids_array[box[1]:box[3], box[0]:box[2]]
                seen = face >= 0
                panel = np.zeros(face.shape + (3,), dtype=np.uint8)
                panel[seen] = np.stack([(face[seen] * 53) % 200 + 40,
                                        (face[seen] * 97) % 200 + 40,
                                        (face[seen] * 151) % 200 + 40], axis=-1)
                if tint is not None:
                    panel[removed_visible[box[1]:box[3], box[0]:box[2]]] = tint
                panels.append(panel)
            joined = np.concatenate(
                (panels[0], np.full((panels[0].shape[0], 4, 3), 255, np.uint8), panels[1]), axis=1)
            crop = Image.fromarray(joined)
            crop.resize((crop.width * 6, crop.height * 6), Image.NEAREST).save(
                output_dir / f"raw{index}_{source_map[index]}_repair_region_faces.png")

    report = {
        "schema": "bar_repair_proof_v1",
        "classification": ("PANDA_PRODUCTION_MESH_BAR_REPAIR_PROVEN" if not gate_failures
                           else "PANDA_PRODUCTION_MESH_BAR_REPAIR_REJECTED"),
        "canonical_glb": str(canonical),
        "repaired_glb": str(repaired),
        "source_bundle": str(source_bundle),
        "repaired_bundle": str(repaired_bundle),
        "removed_face_ids": sorted(removed),
        "bar_face_ids": bar_faces.tolist(),
        "orphan_fragment_face_ids": fragment_faces.tolist(),
        "geometry_preservation": preservation,
        "camera_pairing_by_direction": pairing,
        "views_by_raw_index": views,
        "gate_failures": gate_failures,
        "gates": {
            "no_pixels_gained_in_any_view": not any(
                v["pixels_gained"] for v in views.values()),
            "every_lost_pixel_attributable_to_removed_faces": not any(
                v["pixels_lost_not_attributable_to_removed_faces"] for v in views.values()),
            "no_removed_face_survives_in_repaired": not any(
                v["removed_faces_still_visible_in_repaired"] for v in views.values()),
            "unaffected_triangles_identical": preservation["unaffected_triangles_identical"],
            "attributes_byte_identical": (
                preservation["positions_sha256_before"] == preservation["positions_sha256_after"]
                and preservation["normals_sha256_before"] == preservation["normals_sha256_after"]
                and preservation["uv_sha256_before"] == preservation["uv_sha256_after"]),
        },
    }
    report["gates"]["all_passed"] = all(report["gates"].values())
    if not report["gates"]["all_passed"]:
        report["classification"] = "PANDA_PRODUCTION_MESH_BAR_REPAIR_REJECTED"
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"BAR_REPAIR_PROOF {report['classification']}", flush=True)
    return 0 if report["gates"]["all_passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
