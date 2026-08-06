"""Drop unsupported detached shards into a derived GLB.

For pre-LOD generator cleanup, the established conservative component policy is retained. For a
post-LOD mesh, face count alone is never evidence of debris: decimation can reduce a valid cord,
leaf or pendant to one triangle. Post-LOD removal therefore requires all of:

* detached from the dominant welded component;
* tiny by face count and world-space diagonal;
* high/outboard relative to the subject;
* unsupported by the original source silhouette in both mirrored and non-mirrored registration.
When a ticket-01 anchor receipt is supplied, a removal candidate intersecting a registered anchor,
or dropping an anchor below its seed-support floor, fails the cleanup receipt before promotion.

Surviving positions and UV coordinates are carried byte-identically.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from mesh_io import read_glb, triangle_components, write_glb
from lowvram3d.thin_feature_anchors import anchor_receipt_sha256, parse_anchor_receipt
from source_support import component_position, component_support, load_support_context

WELD = 4e-4
SOURCE_SUPPORT_MIN = 0.18
OUTBOARD_MIN = 0.14
VERY_HIGH_MIN = 0.78
ANCHOR_SEED_TOLERANCE = 0.08


def _receipt_reference_bounds(receipt: dict) -> tuple[np.ndarray, np.ndarray]:
    """Decode the immutable clean-source normalization frame from an anchor receipt."""
    frame = ((receipt.get("discovery") or {}).get("normalization_frame") or {})
    low = np.asarray(frame["bounds_min"], np.float64)
    high = np.asarray(frame["bounds_max"], np.float64)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("anchor receipt normalization frame bounds are malformed")
    if np.any(low > high):
        raise ValueError("anchor receipt normalization frame bounds are inverted")
    diagonal = float(frame.get("diagonal", 0.0))
    if not np.isfinite(diagonal) or diagonal <= 0.0:
        raise ValueError("anchor receipt normalization frame diagonal is malformed")
    measured = float(np.linalg.norm(high - low))
    if not np.isclose(measured, diagonal, rtol=1.0e-7, atol=1.0e-7):
        raise ValueError("anchor receipt normalization frame diagonal does not match bounds")
    center = np.asarray(frame.get("center", ()), np.float64)
    if center.shape != (3,) or not np.isfinite(center).all() or not np.allclose(center, (low + high) * 0.5):
        raise ValueError("anchor receipt normalization frame center does not match bounds")
    return low, high


def _anchor_state(
    positions: np.ndarray,
    anchors: list[dict],
    reference_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Measure anchor seed retention in the immutable clean-source normalized frame."""
    if not anchors:
        present_ids: list[str] = []
        missing_ids: list[str] = []
        records: list[dict] = []
    else:
        low, high = reference_bounds or (positions.min(axis=0), positions.max(axis=0))
        diagonal = max(float(np.linalg.norm(high - low)), 1.0e-12)
        center = (low + high) * 0.5
        normalized = (positions - center) / diagonal
        present_ids = []
        missing_ids = []
        records = []
        for anchor in anchors:
            anchor_id = str(anchor["anchor_id"])
            seeds = anchor.get("seeds") or []
            retained = 0
            for seed in seeds:
                target = np.asarray(seed, np.float64)
                if target.shape != (3,) or not len(normalized):
                    continue
                if float(np.min(np.linalg.norm(normalized - target, axis=1))) <= ANCHOR_SEED_TOLERANCE:
                    retained += 1
            ratio = retained / max(len(seeds), 1)
            floor = float((anchor.get("survival_floor") or {}).get(
                "exclusive_pixel_retention_ratio", 0.0
            ))
            supported_views = list(anchor.get("supported_views") or [])
            under_floor = [view for view in supported_views if ratio < floor]
            present = bool(seeds) and retained == len(seeds) and not under_floor
            (present_ids if present else missing_ids).append(anchor_id)
            records.append({
                "anchor_id": anchor_id,
                "retained_seeds": retained,
                "seed_count": len(seeds),
                "support_ratio": round(float(ratio), 8),
                "minimum_support_floor": floor,
                "under_floor_views": under_floor,
                "present": present,
            })
    present_ids.sort()
    missing_ids.sort()
    payload = {
        "all_ids": sorted(present_ids + missing_ids),
        "present_ids": present_ids,
        "missing_ids": missing_ids,
        "records": records,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        **payload,
        "all_present": not missing_ids,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _anchor_intersections(
    positions: np.ndarray,
    vertices: np.ndarray,
    anchors: list[dict],
    reference_bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[str]:
    """Return anchors whose registered region intersects a removal candidate."""
    if not anchors or not len(vertices):
        return []
    low, high = reference_bounds or (positions.min(axis=0), positions.max(axis=0))
    diagonal = max(float(np.linalg.norm(high - low)), 1.0e-12)
    center = (low + high) * 0.5
    normalized = (vertices - center) / diagonal
    hits: list[str] = []
    for anchor in anchors:
        region = anchor.get("bounds_normalized") or {}
        region_low = np.asarray(region.get("min", ()), np.float64)
        region_high = np.asarray(region.get("max", ()), np.float64)
        if region_low.shape != (3,) or region_high.shape != (3,):
            continue
        # Receipt bounds are normalized floats quantized to a deterministic precision; a small
        # normalized pad covers that precision and expected LOD drift.
        pad = ANCHOR_SEED_TOLERANCE
        candidate_low, candidate_high = normalized.min(axis=0), normalized.max(axis=0)
        overlaps = np.all(candidate_high >= region_low - pad) and np.all(
            candidate_low <= region_high + pad
        )
        if overlaps:
            hits.append(str(anchor["anchor_id"]))
            continue
        seeds = np.asarray(anchor.get("seeds") or [], np.float64)
        if seeds.ndim == 2 and seeds.shape[1:] == (3,):
            if np.any(np.min(np.linalg.norm(normalized[:, None, :] - seeds[None, :, :], axis=2), axis=0)
                      <= pad):
                hits.append(str(anchor["anchor_id"]))
    return sorted(set(hits))


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--height-min", type=float, default=0.70)
    parser.add_argument("--max-triangles", type=int, default=20)
    parser.add_argument("--max-diagonal-fraction", type=float, default=0.062)
    parser.add_argument("--anchor-receipt")
    parser.add_argument("--source-hash", "--source-mesh-sha256", dest="source_hash")
    args = parser.parse_args()

    input_path = Path(args.input)
    positions, normals, uv, tris = read_glb(input_path)
    positions = positions.astype(np.float64)
    component, _ = triangle_components(positions, tris, WELD)
    context = load_support_context(input_path, positions)

    anchors: list[dict] = []
    anchor_receipt: dict | None = None
    anchor_receipt_hash = None
    anchor_source_hash = args.source_hash
    if args.anchor_receipt:
        receipt_path = Path(args.anchor_receipt)
        try:
            receipt_bytes = receipt_path.read_bytes()
            if not args.source_hash:
                raise ValueError("--source-hash is required with --anchor-receipt")
            anchor_receipt = parse_anchor_receipt(
                receipt_bytes,
                expected_source_mesh_sha256=args.source_hash,
            )
            anchors = list(anchor_receipt.get("anchors") or [])
            anchor_receipt_hash = anchor_receipt_sha256(anchor_receipt)
            anchor_source_hash = anchor_receipt["source_mesh_sha256"]
        except (OSError, ValueError) as exc:
            report = {
                "input": str(input_path),
                "output": str(args.output),
                "status": "failed",
                "policy": "anchor_receipt_validation",
                "source_aware": context is not None,
                "failure_codes": ["ANCHOR_RECEIPT_INVALID"],
                "detail": str(exc),
                "anchor_receipt": str(receipt_path),
                "source_hash": args.source_hash,
            }
            _write_report(Path(args.report), report)
            raise RuntimeError(str(exc)) from exc

    anchor_reference_bounds = (
        _receipt_reference_bounds(anchor_receipt)
        if anchor_receipt is not None
        else (positions.min(axis=0), positions.max(axis=0))
    )
    anchor_before = _anchor_state(positions, anchors, anchor_reference_bounds)
    anchor_failure_reasons: list[str] = []
    if anchors and anchor_before["missing_ids"]:
        anchor_failure_reasons.append("ANCHOR_MISSING_BEFORE_CLEANUP")

    low, high = positions.min(axis=0), positions.max(axis=0)
    scene_diagonal = float(np.linalg.norm(high - low))
    max_diagonal = scene_diagonal * args.max_diagonal_fraction
    legacy_span = max(float(high[1] - low[1]), 1e-9)

    sizes = np.bincount(component)
    body = int(np.argmax(sizes))
    removed: list[dict] = []
    kept: list[dict] = []
    drop = np.zeros(len(tris), bool)

    for index, size_value in enumerate(sizes):
        members = component == index
        count = int(size_value)
        vertices = positions[np.unique(tris[members])]
        diagonal = float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
        record: dict = {
            "component": index,
            "triangles": count,
            "diagonal": round(diagonal, 6),
        }
        if index == body:
            record["verdict"] = "kept: dominant attached body"
            kept.append(record)
            continue

        tiny = count <= args.max_triangles and diagonal <= max_diagonal
        intersecting_anchors = _anchor_intersections(
            positions, vertices, anchors, anchor_reference_bounds
        )
        if intersecting_anchors:
            record["anchor_intersections"] = intersecting_anchors
        if context is not None:
            position = component_position(context, vertices)
            support = component_support(context, positions, tris, members)
            height = float(position["height_mean"])
            lateral = float(position["lateral_mean"])
            source_score = float(support["support"])
            high_or_outboard = height >= args.height_min and (
                lateral >= OUTBOARD_MIN or height >= VERY_HIGH_MIN
            )
            unsupported = source_score < SOURCE_SUPPORT_MIN
            record.update({
                "height_mean": round(height, 5),
                "lateral_mean": round(lateral, 5),
                "source_support": support,
            })
            remove_candidate = tiny and high_or_outboard and unsupported
            if remove_candidate and intersecting_anchors:
                anchor_failure_reasons.append(
                    "ANCHOR_INTERSECTION:" + ",".join(intersecting_anchors)
                )
            remove = remove_candidate and not intersecting_anchors
            if remove:
                record["verdict"] = (
                    "removed: detached tiny high/outboard component lacks source-silhouette support"
                )
            else:
                reasons = []
                if not tiny:
                    reasons.append("not microscopically small")
                if not high_or_outboard:
                    reasons.append("not high/outboard")
                if not unsupported:
                    reasons.append("supported by source silhouette")
                record["verdict"] = "kept: " + ", ".join(reasons or ["ambiguous"])
        else:
            # Legacy pre-LOD cleanup. It runs before source-aware decimation and is preserved to
            # avoid changing previously proven high-master cleanup semantics.
            height = float(((vertices[:, 1] - low[1]) / legacy_span).mean())
            record["height_mean"] = round(height, 5)
            remove_candidate = count <= 1 or (
                height >= args.height_min
                and count <= args.max_triangles
                and diagonal <= max_diagonal
            )
            if remove_candidate and intersecting_anchors:
                anchor_failure_reasons.append(
                    "ANCHOR_INTERSECTION:" + ",".join(intersecting_anchors)
                )
            remove = remove_candidate and not intersecting_anchors
            if remove:
                record["verdict"] = (
                    "removed: legacy pre-LOD detached singleton/shard policy"
                )
            else:
                record["verdict"] = "kept: legacy pre-LOD policy found no shard evidence"

        if remove:
            drop |= members
            removed.append(record)
        else:
            kept.append(record)

    survivors = tris[~drop]
    if not len(survivors):
        raise RuntimeError("debris policy would remove the entire mesh")
    after_positions = positions[np.unique(survivors)]
    anchor_after = _anchor_state(after_positions, anchors, anchor_reference_bounds)
    removed_anchor_ids = sorted(
        set(anchor_before["present_ids"]) - set(anchor_after["present_ids"])
    )
    if removed_anchor_ids:
        anchor_failure_reasons.append(
            "ANCHOR_REMOVED_BY_CLEANUP:" + ",".join(removed_anchor_ids)
        )

    anchor_gate = {
        "source_mesh_sha256": anchor_source_hash,
        "receipt_sha256": anchor_receipt_hash,
        "before": anchor_before,
        "after": anchor_after,
        "before_ids": anchor_before["all_ids"],
        "after_ids": anchor_after["all_ids"],
        "anchor_ids_before": anchor_before["present_ids"],
        "anchor_ids_after": anchor_after["present_ids"],
        "anchor_set_before": anchor_before["present_ids"],
        "anchor_set_after": anchor_after["present_ids"],
        "before_sha256": anchor_before["sha256"],
        "after_sha256": anchor_after["sha256"],
        "anchor_hash_before": anchor_before["sha256"],
        "anchor_hash_after": anchor_after["sha256"],
        "removed_ids": removed_anchor_ids,
        "failure_reasons": sorted(set(anchor_failure_reasons)),
        "passed": not anchor_failure_reasons,
    }
    if anchor_failure_reasons:
        report = {
            "input": str(input_path),
            "output": str(args.output),
            "status": "failed",
            "policy": "source_supported_post_lod" if context is not None else "legacy_pre_lod",
            "source_aware": context is not None,
            "anchor_gate": anchor_gate,
            "failure_codes": ["ANCHOR_GATE_FAILED"],
            "detail": "; ".join(sorted(set(anchor_failure_reasons))),
            "source_path": None if context is None else str(context.source_path),
            "height_min": args.height_min,
            "outboard_min": OUTBOARD_MIN,
            "very_high_min": VERY_HIGH_MIN,
            "source_support_min": SOURCE_SUPPORT_MIN,
            "max_triangles": args.max_triangles,
            "max_diagonal": round(max_diagonal, 6),
            "components_total": int(len(sizes)),
            "triangles_before": int(len(tris)),
            "triangles_after": int(len(survivors)),
            "triangles_removed": int(drop.sum()),
            "triangles_removed_percent": round(float(drop.sum() / len(tris) * 100), 6),
            "removed": sorted(removed, key=lambda row: -row["triangles"]),
            "kept_non_body": sorted(
                [row for row in kept if row["component"] != body],
                key=lambda row: -row["triangles"],
            ),
        }
        _write_report(Path(args.report), report)
        raise RuntimeError(report["detail"])

    used = np.unique(survivors)
    remap = np.full(len(positions), -1, np.int64)
    remap[used] = np.arange(len(used))
    kept_uv = uv[used] if uv is not None else None
    output_path = Path(args.output)
    write_glb(output_path, positions[used], normals[used], kept_uv, remap[survivors])

    check_positions, _, check_uv, _ = read_glb(output_path)
    uv_identical = None if kept_uv is None else bool(np.array_equal(check_uv, kept_uv))
    positions_identical = bool(np.array_equal(check_positions, positions[used]))
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "status": "passed",
        "policy": "source_supported_post_lod" if context is not None else "legacy_pre_lod",
        "source_aware": context is not None,
        "source_path": None if context is None else str(context.source_path),
        "height_min": args.height_min,
        "outboard_min": OUTBOARD_MIN,
        "very_high_min": VERY_HIGH_MIN,
        "source_support_min": SOURCE_SUPPORT_MIN,
        "max_triangles": args.max_triangles,
        "max_diagonal": round(max_diagonal, 6),
        "components_total": int(len(sizes)),
        "components_removed": len(removed),
        "triangles_before": int(len(tris)),
        "triangles_after": int(len(survivors)),
        "triangles_removed": int(drop.sum()),
        "triangles_removed_percent": round(float(drop.sum() / len(tris) * 100), 6),
        "uv_bit_identical": uv_identical,
        "positions_bit_identical": positions_identical,
        "anchor_gate": anchor_gate,
        "removed": sorted(removed, key=lambda row: -row["triangles"]),
        "kept_non_body": sorted(
            [row for row in kept if row["component"] != body],
            key=lambda row: -row["triangles"],
        ),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"DEBRIS_STRIP policy={report['policy']} removed={len(removed)} components/"
        f"{int(drop.sum())} triangles ({report['triangles_removed_percent']}%) "
        f"kept_detached={len(report['kept_non_body'])} uv_identical={uv_identical}",
        flush=True,
    )


if __name__ == "__main__":
    main()
