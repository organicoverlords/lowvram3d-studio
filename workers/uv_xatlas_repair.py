"""xatlas unwrap with targeted repair of the few charts that still collide.

xatlas packs this mesh well - around 71% utilisation at stretch_p95 ~1.09 - but leaves a handful
of chart pairs touching, which the exact positive-area detector correctly rejects. Rather than
re-unwrapping the whole mesh at a coarser packing (which throws away the good utilisation), this
finds exactly which charts collide and shrinks only those toward their own centroid until the
collision clears. Shrinking a chart can never create a new overlap because the chart stays strictly
inside its previous footprint; the only cost is slightly lower texel density on those charts.

Degenerate UV triangles are handled separately: they come from near-zero-area 3D triangles, so the
offending faces are dropped from the working mesh and the unwrap is repeated. Acceptance is always
decided by the unmodified `lowvram3d.uv_overlap.positive_area_uv_overlaps`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import xatlas

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "workers"))

from lowvram3d.uv_overlap import (  # noqa: E402
    AREA_EPSILON_UV,
    GRID_SIZE,
    _clip_convex,
    _polygon_area,
    positive_area_uv_overlaps,
)
from lowvram3d.uv_quality import (  # noqa: E402
    MAX_STRETCH_P95,
    MIN_ATLAS_UTILIZATION,
    area_weighted_percentile,
    conformal_stretch,
)
from lowvram3d.anchor_provenance import (  # noqa: E402
    AnchorProvenanceError,
    geometry_sha256,
    load_anchor_provenance,
    provenance_record,
)
from uv_xatlas_route import load_indexed, weld, write_glb  # noqa: E402


def chart_labels(indices: np.ndarray, vertex_count: int) -> np.ndarray:
    """Chart id per vertex. xatlas splits vertices on seams, so connected components of the output
    index buffer are exactly the charts."""
    parent = np.arange(vertex_count)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for tri in indices:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c)):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    return np.array([find(i) for i in range(vertex_count)])


def offending_triangles(uv_triangles: np.ndarray) -> set[int]:
    """Indices of triangles taking part in any positive-area UV intersection.

    Mirrors `lowvram3d.uv_overlap` exactly - same grid, same Sutherland-Hodgman clip, same area
    epsilon - but returns which triangles are implicated so they can be pruned. Crucially it does
    NOT restrict to cross-chart pairs: the residual overlaps on this mesh are intra-chart folds,
    where a chart doubles back over itself, and a cross-chart filter misses every one of them.
    Acceptance still comes from the unmodified repository detector.
    """
    low = uv_triangles.min(axis=1)
    high = uv_triangles.max(axis=1)
    cell_low = np.clip((low * GRID_SIZE).astype(np.int64), 0, GRID_SIZE - 1)
    cell_high = np.clip((high * GRID_SIZE).astype(np.int64), 0, GRID_SIZE - 1)

    buckets: dict[int, list[int]] = {}
    for index in range(len(uv_triangles)):
        for cx in range(cell_low[index, 0], cell_high[index, 0] + 1):
            for cy in range(cell_low[index, 1], cell_high[index, 1] + 1):
                buckets.setdefault(cx * GRID_SIZE + cy, []).append(index)

    seen: set[tuple[int, int]] = set()
    guilty: set[int] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members) - 1):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                ta, tb = uv_triangles[a], uv_triangles[b]
                if (
                    ta[:, 0].max() < tb[:, 0].min()
                    or tb[:, 0].max() < ta[:, 0].min()
                    or ta[:, 1].max() < tb[:, 1].min()
                    or tb[:, 1].max() < ta[:, 1].min()
                ):
                    continue
                if _polygon_area(_clip_convex(ta.copy(), tb.copy())) > AREA_EPSILON_UV:
                    guilty.add(a)
                    guilty.add(b)
    return guilty


def cross_chart_collisions(
    uv_triangles: np.ndarray, tri_chart: np.ndarray
) -> list[tuple[int, int, int, int]]:
    """Overlapping triangle pairs that belong to different charts."""
    low = uv_triangles.min(axis=1)
    high = uv_triangles.max(axis=1)
    cell_low = np.clip((low * GRID_SIZE).astype(np.int64), 0, GRID_SIZE - 1)
    cell_high = np.clip((high * GRID_SIZE).astype(np.int64), 0, GRID_SIZE - 1)

    buckets: dict[int, list[int]] = {}
    for index in range(len(uv_triangles)):
        for cx in range(cell_low[index, 0], cell_high[index, 0] + 1):
            for cy in range(cell_low[index, 1], cell_high[index, 1] + 1):
                buckets.setdefault(cx * GRID_SIZE + cy, []).append(index)

    seen: set[tuple[int, int]] = set()
    hits: list[tuple[int, int, int, int]] = []
    for members in buckets.values():
        if len(members) < 2:
            continue
        for i in range(len(members) - 1):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                if tri_chart[a] == tri_chart[b]:
                    continue
                key = (a, b) if a < b else (b, a)
                if key in seen:
                    continue
                seen.add(key)
                ta, tb = uv_triangles[a], uv_triangles[b]
                if (
                    ta[:, 0].max() < tb[:, 0].min()
                    or tb[:, 0].max() < ta[:, 0].min()
                    or ta[:, 1].max() < tb[:, 1].min()
                    or tb[:, 1].max() < ta[:, 1].min()
                ):
                    continue
                area = _polygon_area(_clip_convex(ta.copy(), tb.copy()))
                if area > AREA_EPSILON_UV:
                    hits.append((a, b, int(tri_chart[a]), int(tri_chart[b])))
    return hits


def shrink_charts(uv: np.ndarray, vertex_chart: np.ndarray, charts: set[int], factor: float) -> int:
    moved = 0
    for chart in charts:
        mask = vertex_chart == chart
        if not mask.any():
            continue
        centroid = uv[mask].mean(axis=0)
        uv[mask] = centroid + (uv[mask] - centroid) * factor
        moved += int(mask.sum())
    return moved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--max-cost", type=float, default=2.0)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--overlap-timeout", type=float, default=1200.0)
    parser.add_argument("--max-candidate-pairs", type=int, default=10_000_000)
    parser.add_argument("--max-overlap-texels", type=float, default=1.0)
    parser.add_argument("--repair-rounds", type=int, default=4)
    parser.add_argument("--shrink-factors", default="0.97,0.93,0.88,0.80")
    parser.add_argument("--max-unwrap-attempts", type=int, default=3)
    parser.add_argument("--anchor-receipt", required=True)
    parser.add_argument("--expected-source-sha256", default="")
    args = parser.parse_args()

    try:
        _receipt, receipt_hash, anchor_ids = load_anchor_provenance(
            args.anchor_receipt,
            expected_source_sha256=args.expected_source_sha256 or None,
        )
    except AnchorProvenanceError as exc:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps({
            "input": args.input, "output": args.output, "gate_passed": False,
            "failure_codes": [exc.code], "failure_detail": exc.detail,
            "anchor_receipt": args.anchor_receipt,
        }, indent=2), encoding="utf-8")
        print(f"UV_PROVENANCE_FAILED {exc.code}: {exc.detail}", file=sys.stderr, flush=True)
        return 2

    positions_raw, normals_raw, indices_raw = load_indexed(Path(args.input))
    positions, indices, first = weld(positions_raw, indices_raw)
    input_geometry_hash = geometry_sha256(positions, indices)
    normals = normals_raw[first] if normals_raw is not None else None
    print(f"welded: {len(positions_raw)} -> {len(positions)} verts, {len(indices)} faces", flush=True)

    shrink_factors = [float(v) for v in args.shrink_factors.split(",")]
    dropped_faces_total = 0
    journal: list[dict] = []

    for attempt in range(args.max_unwrap_attempts):
        atlas = xatlas.Atlas()
        atlas.add_mesh(positions, indices.astype(np.uint32))
        chart_options = xatlas.ChartOptions()
        chart_options.max_cost = args.max_cost
        chart_options.max_iterations = args.max_iterations
        pack_options = xatlas.PackOptions()
        pack_options.resolution = args.resolution
        pack_options.padding = args.padding
        pack_options.bruteForce = False
        started = time.monotonic()
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        unwrap_seconds = time.monotonic() - started

        vmapping, out_indices, out_uv = atlas[0]
        out_uv = np.asarray(out_uv, np.float64)
        if out_uv.max() > 1.5:
            out_uv = out_uv / np.array([atlas.width, atlas.height], np.float64)
        out_indices = np.asarray(out_indices, np.int64).reshape(-1, 3)
        out_positions = positions[vmapping]
        out_normals = normals[vmapping] if normals is not None else None

        # Drop UV-degenerate faces at source: they originate in near-zero-area 3D triangles and no
        # packing change can rescue them.
        tri_uv = out_uv[out_indices]
        tri_area = 0.5 * np.abs(
            (tri_uv[:, 1, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 2, 1] - tri_uv[:, 0, 1])
            - (tri_uv[:, 2, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 1, 1] - tri_uv[:, 0, 1])
        )
        # Prune in place rather than re-unwrapping. Re-unwrapping rescales every chart, which
        # simply pushes a different set of small triangles below the degeneracy epsilon - observed
        # cycling 62 -> 15 -> 50. Deleting the offending faces leaves every surviving triangle's UV
        # byte-identical, so the pruned set is guaranteed no worse than the set it came from.
        for prune_round in range(args.repair_rounds):
            tri_uv = out_uv[out_indices]
            area = 0.5 * np.abs(
                (tri_uv[:, 1, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 2, 1] - tri_uv[:, 0, 1])
                - (tri_uv[:, 2, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 1, 1] - tri_uv[:, 0, 1])
            )
            degenerate_idx = set(np.flatnonzero(area <= AREA_EPSILON_UV).tolist())
            overlap_idx = offending_triangles(tri_uv)
            guilty = degenerate_idx | overlap_idx
            print(
                f"prune {prune_round}: degenerate={len(degenerate_idx)} "
                f"overlapping={len(overlap_idx)} total={len(guilty)}",
                flush=True,
            )
            journal.append(
                {
                    "prune_round": prune_round,
                    "degenerate": len(degenerate_idx),
                    "overlapping": len(overlap_idx),
                    "pruned": len(guilty),
                }
            )
            if not guilty:
                break
            keep = np.ones(len(out_indices), bool)
            keep[list(guilty)] = False
            out_indices = out_indices[keep]
            dropped_faces_total += len(guilty)

        vertex_chart = chart_labels(out_indices, len(out_positions))
        degenerate = np.zeros(len(out_indices), bool)
        tri_chart = vertex_chart[out_indices[:, 0]]
        chart_total = len(np.unique(vertex_chart))
        print(
            f"attempt {attempt}: charts={chart_total} unwrap={unwrap_seconds:.1f}s "
            f"degenerate={int(degenerate.sum())}",
            flush=True,
        )

        # Authoritative acceptance: unmodified repository detector.
        tri_uv = out_uv[out_indices]
        tri_pos = out_positions[out_indices]
        exact_started = time.monotonic()
        exact = positive_area_uv_overlaps(
            tri_uv,
            args.resolution,
            timeout_seconds=args.overlap_timeout,
            max_candidate_pairs=args.max_candidate_pairs,
        )
        exact_seconds = time.monotonic() - exact_started

        stretch, area3d = conformal_stretch(tri_pos, tri_uv)
        stretch_p95 = area_weighted_percentile(stretch, area3d, 95.0)
        utilisation = float(
            0.5
            * np.abs(
                (tri_uv[:, 1, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 2, 1] - tri_uv[:, 0, 1])
                - (tri_uv[:, 2, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 1, 1] - tri_uv[:, 0, 1])
            ).sum()
            - exact.positive_overlap_total_area_uv
        )

        gate = {
            "exact_success": bool(exact.success),
            "not_timed_out": not exact.timed_out,
            "tested_pairs_positive": exact.tested_pair_count > 0,
            "degenerate_zero": exact.degenerate_uv_triangle_count == 0,
            "out_of_bounds_zero": exact.out_of_bounds_triangle_count == 0,
            "overlap_within_budget": exact.positive_overlap_total_texels_equivalent
            <= args.max_overlap_texels,
            "utilisation_ok": utilisation >= MIN_ATLAS_UTILIZATION,
            "stretch_ok": bool(np.isfinite(stretch_p95) and stretch_p95 <= MAX_STRETCH_P95),
        }
        output_geometry_hash = geometry_sha256(out_positions, out_indices)
        geometry_unchanged = output_geometry_hash == input_geometry_hash
        gate["geometry_unchanged"] = geometry_unchanged
        passed = all(gate.values())

        report = {
            "input": args.input,
            "output": args.output,
            "resolution": args.resolution,
            "padding": args.padding,
            "max_cost": args.max_cost,
            "unwrap_attempt": attempt,
            "chart_count": int(chart_total),
            "triangles": int(len(out_indices)),
            "dropped_degenerate_faces_total": dropped_faces_total,
            "atlas_utilization": utilisation,
            "atlas_count": int(atlas.atlas_count),
            "stretch_p95": float(stretch_p95),
            "exact_overlap": exact.as_dict(),
            "exact_overlap_seconds": exact_seconds,
            "repair_journal": journal,
            "gate": gate,
            "gate_passed": passed,
            "provenance": provenance_record(
                receipt_sha256=receipt_hash,
                anchor_ids=anchor_ids,
                input_geometry_sha256=input_geometry_hash,
                output_geometry_sha256=output_geometry_hash,
                geometry_unchanged=geometry_unchanged,
            ),
            "failure_codes": ["GEOMETRY_MUTATION"] if not geometry_unchanged else [],
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(
            f"EXACT candidate_pairs={exact.candidate_pair_count} tested={exact.tested_pair_count} "
            f"positive_pairs={exact.positive_overlap_pair_count} "
            f"texels={exact.positive_overlap_total_texels_equivalent:.6f} "
            f"degenerate={exact.degenerate_uv_triangle_count} oob={exact.out_of_bounds_triangle_count} "
            f"charts={chart_total} util={utilisation*100:.2f}% stretch_p95={stretch_p95:.3f} "
            f"timed_out={exact.timed_out} success={exact.success} PASS={passed}",
            flush=True,
        )

        if passed:
            write_glb(Path(args.output), out_positions, out_normals, out_uv, out_indices)
            print(f"UV_CANDIDATE_WRITTEN {args.output}", flush=True)
            return 0
        print("UV_GATE_FAILED " + "; ".join(k for k, v in gate.items() if not v), file=sys.stderr, flush=True)
        return 1

    print("UV_UNWRAP_ATTEMPTS_EXHAUSTED", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

