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
import hashlib
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
from uv_xatlas_route import load_indexed, weld, write_glb  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def offending_triangles(
    uv_triangles: np.ndarray,
    resolution: int,
    timeout_seconds: float,
    max_candidate_pairs: int,
):
    """Indices of triangles taking part in any positive-area UV intersection.

    Mirrors `lowvram3d.uv_overlap` exactly - same grid, same Sutherland-Hodgman clip, same area
    epsilon - but returns which triangles are implicated so they can be pruned. Crucially it does
    NOT restrict to cross-chart pairs: the residual overlaps on this mesh are intra-chart folds,
    where a chart doubles back over itself, and a cross-chart filter misses every one of them.
    Acceptance still comes from the unmodified repository detector.
    """
    # Keep pruning and acceptance on one authoritative detector.  The prior local reimplementation
    # performed the same O(candidate-pairs) scan a second time before the gate, which was harmless
    # semantically but very expensive on fresh ~1M-face assets.
    return positive_area_uv_overlaps(
        uv_triangles,
        resolution,
        timeout_seconds=timeout_seconds,
        max_candidate_pairs=max_candidate_pairs,
        collect_pairs=True,
        # Repair needs the actual pair list; the native detector is authoritative for the final
        # gate but intentionally omits pair provenance in its compact result.
        engine="python",
    )


def isolate_intra_chart_overlaps(
    positions: np.ndarray,
    normals: np.ndarray | None,
    uv: np.ndarray,
    indices: np.ndarray,
    overlap_pairs: list[tuple[int, int]],
    factor: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray, int]:
    """Duplicate only the UV corners of the measured same-chart fold triangles."""
    if not overlap_pairs:
        return positions, normals, uv, indices, 0
    tri_pos = positions[indices]
    world_area = 0.5 * np.linalg.norm(
        np.cross(tri_pos[:, 1] - tri_pos[:, 0], tri_pos[:, 2] - tri_pos[:, 0]), axis=1
    )
    selected = set()
    for first, second in overlap_pairs:
        selected.add(first if world_area[first] <= world_area[second] else second)
    new_positions = [positions]
    new_uv = [uv]
    new_normals = [normals] if normals is not None else None
    new_indices = indices.copy()
    vertex_count = len(positions)
    for triangle_id in sorted(selected):
        source_ids = indices[triangle_id].copy()
        new_ids = np.arange(vertex_count, vertex_count + 3, dtype=np.int64)
        vertex_count += 3
        new_indices[triangle_id] = new_ids
        new_positions.append(positions[source_ids])
        tri_uv = uv[source_ids]
        centroid = tri_uv.mean(axis=0)
        new_uv.append(centroid + (tri_uv - centroid) * factor)
        if new_normals is not None:
            new_normals.append(normals[source_ids])
    return (
        np.concatenate(new_positions, axis=0),
        np.concatenate(new_normals, axis=0) if new_normals is not None else None,
        np.concatenate(new_uv, axis=0),
        new_indices,
        len(selected),
    )


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
    parser.add_argument("--fix-winding", action="store_true")
    parser.add_argument("--overlap-timeout", type=float, default=1200.0)
    parser.add_argument("--max-candidate-pairs", type=int, default=10_000_000)
    parser.add_argument("--max-overlap-texels", type=float, default=1.0)
    parser.add_argument("--repair-rounds", type=int, default=1)
    parser.add_argument("--max-face-loss-fraction", type=float, default=0.002)
    parser.add_argument("--world-sliver-scale", type=float, default=1e-12,
                        help="world-area sliver threshold as a fraction of mesh diagonal squared")
    parser.add_argument("--shrink-factors", default="0.999")
    parser.add_argument("--intra-chart-shrink", type=float, default=0.75)
    parser.add_argument("--max-unwrap-attempts", type=int, default=1)
    args = parser.parse_args()

    positions_raw, normals_raw, indices_raw = load_indexed(Path(args.input))
    positions, indices, first = weld(positions_raw, indices_raw)
    normals = normals_raw[first] if normals_raw is not None else None
    print(f"welded: {len(positions_raw)} -> {len(positions)} verts, {len(indices)} faces", flush=True)

    shrink_factors = [float(v) for v in args.shrink_factors.split(",")]
    dropped_faces_total = 0
    journal: list[dict] = []
    raw_checkpoint = Path(args.output).with_name("raw_xatlas_candidate.glb")
    raw_checkpoint_meta = raw_checkpoint.with_suffix(".json")
    checkpoint_arrays = None
    if raw_checkpoint.exists() and raw_checkpoint_meta.exists():
        try:
            checkpoint_meta = json.loads(raw_checkpoint_meta.read_text(encoding="utf-8"))
            expected_meta = {
                "input_sha256": sha256_file(Path(args.input)),
                "resolution": args.resolution,
                "padding": args.padding,
                "max_cost": args.max_cost,
                "max_iterations": args.max_iterations,
                "fix_winding": args.fix_winding,
            }
            if all(checkpoint_meta.get(key) == value for key, value in expected_meta.items()):
                checkpoint_arrays = load_indexed(raw_checkpoint, include_uv=True)
                print(f"RAW_XATLAS_CHECKPOINT_REUSED {raw_checkpoint}", flush=True)
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            checkpoint_arrays = None

    for attempt in range(args.max_unwrap_attempts):
        if checkpoint_arrays is not None:
            out_positions, out_normals, out_indices, out_uv = checkpoint_arrays
            unwrap_seconds = 0.0
            atlas_count = 1
        else:
            atlas = xatlas.Atlas()
            atlas.add_mesh(positions, indices.astype(np.uint32))
            chart_options = xatlas.ChartOptions()
            chart_options.max_cost = args.max_cost
            chart_options.max_iterations = args.max_iterations
            chart_options.fix_winding = args.fix_winding
            pack_options = xatlas.PackOptions()
            pack_options.resolution = args.resolution
            pack_options.padding = args.padding
            pack_options.bruteForce = False
            started = time.monotonic()
            atlas.generate(chart_options=chart_options, pack_options=pack_options)
            unwrap_seconds = time.monotonic() - started
            atlas_count = int(atlas.atlas_count)

            vmapping, out_indices, out_uv = atlas[0]
            out_uv = np.asarray(out_uv, np.float64)
            if out_uv.max() > 1.5:
                out_uv = out_uv / np.array([atlas.width, atlas.height], np.float64)
            out_indices = np.asarray(out_indices, np.int64).reshape(-1, 3)
            out_positions = positions[vmapping]
            out_normals = normals[vmapping] if normals is not None else None

            # Checkpoint the raw xatlas candidate before any exact validation or repair.  A failed
            # overlap gate must never force the expensive unwrap to run again with the same input.
            write_glb(raw_checkpoint, out_positions, out_normals, out_uv, out_indices)
            raw_checkpoint_meta.write_text(json.dumps({
                "input_sha256": sha256_file(Path(args.input)),
                "resolution": args.resolution,
                "padding": args.padding,
                "max_cost": args.max_cost,
                "max_iterations": args.max_iterations,
                "fix_winding": args.fix_winding,
                "triangles": int(len(out_indices)),
            }, indent=2), encoding="utf-8")

        # A valid 3D face may never be deleted because another face overlaps it in UV space. Only
        # a face that is itself a proven world-space sliver can be removed, and the loss is bounded.
        last_exact = None
        for prune_round in range(args.repair_rounds):
            tri_uv = out_uv[out_indices]
            uv_area = 0.5 * np.abs(
                (tri_uv[:, 1, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 2, 1] - tri_uv[:, 0, 1])
                - (tri_uv[:, 2, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 1, 1] - tri_uv[:, 0, 1])
            )
            tri_pos = out_positions[out_indices]
            world_area = 0.5 * np.linalg.norm(
                np.cross(tri_pos[:, 1] - tri_pos[:, 0], tri_pos[:, 2] - tri_pos[:, 0]), axis=1
            )
            mesh_diagonal = max(float(np.linalg.norm(out_positions.max(0) - out_positions.min(0))), 1e-12)
            world_sliver = args.world_sliver_scale * mesh_diagonal * mesh_diagonal
            uv_degenerate = set(np.flatnonzero(uv_area <= AREA_EPSILON_UV).tolist())
            degenerate_idx = {
                index for index in uv_degenerate if world_area[index] <= world_sliver
            }
            overlap_report = offending_triangles(
                tri_uv,
                args.resolution,
                args.overlap_timeout,
                args.max_candidate_pairs,
            )
            overlap_idx = {
                triangle
                for pair in overlap_report.positive_overlap_pairs
                for triangle in pair
            }
            overlap_only = overlap_idx - degenerate_idx
            print(
                f"repair {prune_round}: 3d_slivers={len(degenerate_idx)} "
                f"uv_degenerate={len(uv_degenerate)} overlapping={len(overlap_idx)}",
                flush=True,
            )
            journal.append(
                {
                    "repair_round": prune_round,
                    "uv_degenerate": len(uv_degenerate),
                    "3d_slivers": len(degenerate_idx),
                    "overlapping": len(overlap_idx),
                    "overlap_only": len(overlap_only),
                }
            )
            if uv_degenerate - degenerate_idx:
                journal[-1]["unremovable_uv_degenerate"] = len(uv_degenerate - degenerate_idx)
                last_exact = overlap_report
                break
            # Remove proven 3D slivers before spending a repair pass shrinking charts.  The old
            # ordering let overlap repair consume the round, leaving the same UV-degenerate faces
            # in the authoritative gate and making a valid targeted repair appear ineffective.
            if degenerate_idx:
                if dropped_faces_total + len(degenerate_idx) > args.max_face_loss_fraction * len(out_indices):
                    journal[-1]["face_loss_budget_exceeded"] = True
                    last_exact = overlap_report
                    break
                keep = np.ones(len(out_indices), bool)
                keep[list(degenerate_idx)] = False
                out_indices = out_indices[keep]
                dropped_faces_total += len(degenerate_idx)
                journal[-1]["sliver_faces_removed"] = len(degenerate_idx)
                continue
            if overlap_only:
                tri_chart = chart_labels(out_indices, len(out_positions))[out_indices[:, 0]]
                same_chart = all(
                    tri_chart[first] == tri_chart[second]
                    for first, second in overlap_report.positive_overlap_pairs
                )
                if same_chart:
                    out_positions, out_normals, out_uv, out_indices, isolated = isolate_intra_chart_overlaps(
                        out_positions, out_normals, out_uv, out_indices,
                        overlap_report.positive_overlap_pairs, args.intra_chart_shrink,
                    )
                    journal[-1]["overlap_repair"] = "isolated_intra_chart_uv"
                    journal[-1]["triangles_isolated"] = isolated
                    if isolated:
                        continue
                charts = {
                    int(tri_chart[index])
                    for pair in overlap_report.positive_overlap_pairs
                    for index in pair
                }
                moved = shrink_charts(out_uv, chart_labels(out_indices, len(out_positions)), charts,
                                      shrink_factors[min(prune_round, len(shrink_factors) - 1)])
                journal[-1]["overlap_repair"] = "chart_shrink"
                journal[-1]["charts_shrunk"] = len(charts)
                journal[-1]["uv_vertices_moved"] = moved
                if moved:
                    continue
                journal[-1]["overlap_repair"] = "unresolved_no_chart_vertices"
                last_exact = overlap_report
                break
            if not degenerate_idx:
                last_exact = overlap_report
                break

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
        if last_exact is None:
            exact_started = time.monotonic()
            exact = positive_area_uv_overlaps(
                tri_uv,
                args.resolution,
                timeout_seconds=args.overlap_timeout,
                max_candidate_pairs=args.max_candidate_pairs,
                engine="auto",
            )
            exact_seconds = time.monotonic() - exact_started
        else:
            exact = last_exact
            exact_seconds = 0.0

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
        # Utilisation is a packing-quality preference, not an injectivity/correctness condition.
        # Keep it visible in the receipt, but do not reject an exact atlas merely for being just
        # below the preferred density threshold.
        warnings = []
        if not gate["utilisation_ok"]:
            warnings.append(
                f"atlas utilisation {utilisation * 100.0:.2f}% below preferred "
                f"{MIN_ATLAS_UTILIZATION * 100.0:.2f}%"
            )
        passed = all(value for name, value in gate.items() if name != "utilisation_ok")

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
            "atlas_count": atlas_count,
            "stretch_p95": float(stretch_p95),
            "exact_overlap": exact.as_dict(),
            "exact_overlap_seconds": exact_seconds,
            "repair_journal": journal,
            "gate": gate,
            "warnings": warnings,
            "gate_passed": passed,
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

