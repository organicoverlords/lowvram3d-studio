"""STAGE 4: delete only REMOVE_OUTBOARD_DEBRIS components, then validate hard gates."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys

import numpy as np
import open3d as o3d

CACHE = sys.argv[1]
DECISIONS = sys.argv[2]
SOURCE_GLB = sys.argv[3]
OUTDIR = sys.argv[4]

os.makedirs(OUTDIR, exist_ok=True)
decisions = json.load(open(DECISIONS, encoding="utf-8"))
remove_ids = {d["component_id"] for d in decisions["components"] if d["action"] == "REMOVE_OUTBOARD_DEBRIS"}
main_id = decisions["main_component_id"]
assert main_id not in remove_ids, "main component must never be removed"

data = np.load(CACHE)
vertices = data["vertices"].astype(np.float64)
triangles = data["triangles"].astype(np.int32)
labels = data["labels"]

preserved = os.path.join(OUTDIR, "source_preserved.glb")
shutil.copy2(SOURCE_GLB, preserved)
source_digest_before = hashlib.sha256(open(SOURCE_GLB, "rb").read()).hexdigest()


def build(mask: np.ndarray) -> o3d.geometry.TriangleMesh:
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles[mask]),
    )
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


keep_mask = ~np.isin(labels, list(remove_ids))
removed_mask = ~keep_mask
kept_mesh = build(keep_mask)
removed_mesh = build(removed_mask)

candidate = os.path.join(OUTDIR, "debris_clean_candidate.glb")
removed_path = os.path.join(OUTDIR, "removed_components.glb")
o3d.io.write_triangle_mesh(candidate, kept_mesh)
o3d.io.write_triangle_mesh(removed_path, removed_mesh)


def topology(mesh_triangles: np.ndarray) -> dict:
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(mesh_triangles),
    )
    mesh.remove_unreferenced_vertices()
    boundary = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=False)).shape[0]
    non_manifold = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=True)).shape[0]
    return {"faces": int(mesh_triangles.shape[0]), "boundary_edges": int(boundary),
            "non_manifold_edges": int(non_manifold)}


before = topology(triangles)
after = topology(triangles[keep_mask])
main_before = int((labels == main_id).sum())
main_after = int((labels[keep_mask] == main_id).sum())

errors = []
if main_after != main_before:
    errors.append(f"main component face count changed {main_before} -> {main_after}")
if after["boundary_edges"] > before["boundary_edges"]:
    errors.append(f"boundary edges increased {before['boundary_edges']} -> {after['boundary_edges']}")
if after["non_manifold_edges"] > before["non_manifold_edges"]:
    errors.append(
        f"non-manifold edges increased {before['non_manifold_edges']} -> {after['non_manifold_edges']}"
    )
kept_ids = {int(v) for v in np.unique(labels[keep_mask])}
for decision in decisions["components"]:
    if decision["action"].startswith("KEEP") and decision["component_id"] not in kept_ids:
        errors.append(f"KEEP component {decision['component_id']} missing after deletion")
source_digest_after = hashlib.sha256(open(SOURCE_GLB, "rb").read()).hexdigest()
if source_digest_after != source_digest_before:
    errors.append("source mesh was modified in place")

debris_faces = int(removed_mask.sum())
report = {
    "removed_component_ids": sorted(remove_ids),
    "components_before": int(labels.max()) + 1,
    "components_after": len(kept_ids),
    "verified_debris_removed_faces": debris_faces,
    "verified_debris_removed_faces_percent": round(debris_faces / before["faces"] * 100, 4),
    "main_surface_removed_faces": 0,
    "main_surface_removed_faces_percent": 0.0,
    "main_component_faces_before": main_before,
    "main_component_faces_after": main_after,
    "topology_before": before,
    "topology_after": after,
    "source_glb_sha256": source_digest_after,
    "source_unmodified": source_digest_after == source_digest_before,
    "candidate_glb": candidate,
    "removed_glb": removed_path,
    "errors": errors,
    "success": not errors,
}
with open(os.path.join(OUTDIR, "debris_removal_report.json"), "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
print("STAGE4_REMOVE " + json.dumps({k: report[k] for k in
      ("removed_component_ids", "verified_debris_removed_faces_percent",
       "main_component_faces_before", "main_component_faces_after", "success")}))
print(f"topology before={before} after={after}")
if errors:
    print("ERRORS: " + "; ".join(errors))
