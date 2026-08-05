"""Deterministic bounded MVS-style triangle view assignment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesh_io import read_glb


def adjacency(positions, tris):
    edge_map = {}
    for tid, tri in enumerate(tris):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_map.setdefault(tuple(sorted((int(a), int(b)))), []).append(tid)
    pairs = []
    for ids in edge_map.values():
        if len(ids) == 2:
            pairs.append(tuple(ids))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--evidence-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protected-triangles", default="")
    parser.add_argument("--rear-triangles", default="")
    parser.add_argument("--passes", type=int, default=5)
    args = parser.parse_args()
    positions, normals, _uv, tris = read_glb(Path(args.mesh))
    tris = tris.astype(np.int64)
    manifest = json.loads(Path(args.evidence_manifest).read_text(encoding="utf-8"))
    records = manifest.get("views", [])
    ntri = len(tris)
    unary = np.full((ntri, len(records)), 100.0, np.float64)
    classes = []
    for vid, item in enumerate(records):
        classes.append(str(item.get("semantic_source_class", "UNKNOWN")))
        with np.load(item["path"], allow_pickle=False) as e:
            ids = e["triangle_id"]
            valid = ids >= 0
            face = ids[valid]
            counts = np.bincount(face, minlength=ntri)
            facing = np.asarray(e["normal_facing"])[valid]
            alpha = np.asarray(e["source_alpha"])[valid]
            score = np.zeros(ntri, np.float64)
            np.add.at(score, face, np.clip(facing, 0.0, 1.0) * alpha)
            support = score / np.maximum(counts, 1)
            unary[:, vid] = 4.0 - np.clip(support, 0.0, 1.0) * 3.0 - np.minimum(counts, 64) / 64.0
    protected = np.load(args.protected_triangles).astype(bool) if args.protected_triangles else np.zeros(ntri, bool)
    rear = np.load(args.rear_triangles).astype(bool) if args.rear_triangles else np.zeros(ntri, bool)
    if protected.shape != (ntri,) or rear.shape != (ntri,):
        raise RuntimeError("TRIANGLE_REGION_MASK_SHAPE_MISMATCH")
    for vid, cls in enumerate(classes):
        if protected[...].any() and cls not in {"ORIGINAL_FACE", "FACE_REFINEMENT"}:
            unary[protected, vid] += 50.0
        if rear[...].any() and cls in {"ORIGINAL_FACE", "FACE_REFINEMENT", "GENERATED_FRONT"}:
            unary[rear, vid] += 50.0
    labels = np.argmin(unary, axis=1).astype(np.int32)
    edges = adjacency(positions, tris)
    normals = np.asarray(normals, np.float64)
    lambda_pair = 0.15
    energy_before = float(unary[np.arange(ntri), labels].sum())
    changes = []
    for _ in range(max(1, min(args.passes, 5))):
        changed = 0
        for tid in range(ntri):
            candidates = np.arange(len(records), dtype=np.int32)
            best_label, best_cost = int(labels[tid]), float("inf")
            neighbours = []
            for a, b in edges:
                if a == tid: neighbours.append(b)
                elif b == tid: neighbours.append(a)
            for label in candidates.tolist():
                cost = float(unary[tid, label])
                for other in neighbours:
                    if labels[other] != label:
                        dot = max(0.0, float(normals[tris[tid]].mean(0) @ normals[tris[other]].mean(0)))
                        cost += lambda_pair * (1.0 - dot)
                if cost < best_cost or (cost == best_cost and label < best_label):
                    best_label, best_cost = label, cost
            if best_label != labels[tid]:
                labels[tid] = best_label
                changed += 1
        changes.append(changed)
        if changed == 0:
            break
    energy_after = float(unary[np.arange(ntri), labels].sum())
    confidence = (np.partition(unary, 1, axis=1)[:, 1] - unary[np.arange(ntri), labels]).astype(np.float32)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "primary_view_per_triangle.npy", labels)
    np.save(out / "triangle_label_confidence.npy", confidence)
    report = {"schema": "surface_view_assignment_v1", "views": classes, "triangle_count": ntri, "energy_before": energy_before, "energy_after": energy_after, "icm_pass_changes": changes, "deterministic": True, "hard_exclusions": {"protected": int(protected.sum()), "rear": int(rear.sum())}}
    (out / "assignment_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"VIEW_ASSIGNMENT triangles={ntri} energy={energy_before:.3f}->{energy_after:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
