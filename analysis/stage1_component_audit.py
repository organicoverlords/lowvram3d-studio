"""STAGE 1: deterministic connected-component audit via Open3D clustering.

Runs once and caches every geometric metric so later classification never recomputes it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import open3d as o3d
import trimesh

MESH_PATH = sys.argv[1]
OUT_NPZ = sys.argv[2]
OUT_JSON = sys.argv[3]
WELD_DISTANCE = 4e-4

scene = trimesh.load(MESH_PATH, process=False)
if isinstance(scene, trimesh.Scene):
    mesh = trimesh.util.concatenate([g for g in scene.geometry.values()])
else:
    mesh = scene
mesh.merge_vertices(merge_tex=True, merge_norm=True)

vertices = np.asarray(mesh.vertices, np.float64)
triangles = np.asarray(mesh.faces, np.int32)

o3d_mesh = o3d.geometry.TriangleMesh(
    o3d.utility.Vector3dVector(vertices),
    o3d.utility.Vector3iVector(triangles),
)
o3d_mesh.merge_close_vertices(WELD_DISTANCE)
o3d_mesh.remove_duplicated_triangles()
o3d_mesh.remove_degenerate_triangles()
o3d_mesh.compute_vertex_normals()

labels_v, counts_v, areas_v = o3d_mesh.cluster_connected_triangles()
labels = np.asarray(labels_v)
counts = np.asarray(counts_v)
areas = np.asarray(areas_v)

welded_vertices = np.asarray(o3d_mesh.vertices)
welded_triangles = np.asarray(o3d_mesh.triangles)
model_low = welded_vertices.min(axis=0)
model_high = welded_vertices.max(axis=0)
model_diag = float(np.linalg.norm(model_high - model_low))

main_id = int(np.argmax(counts))
main_faces = welded_triangles[labels == main_id]
main_points = welded_vertices[np.unique(main_faces)]
main_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(main_points))
main_tree = o3d.geometry.KDTreeFlann(main_cloud)

records = []
for component in range(len(counts)):
    selector = labels == component
    face_subset = welded_triangles[selector]
    vertex_ids = np.unique(face_subset)
    points = welded_vertices[vertex_ids]
    low, high = points.min(axis=0), points.max(axis=0)
    extent = high - low
    bbox_diagonal = float(np.linalg.norm(extent))
    centroid = points.mean(axis=0)
    area = float(areas[component])
    sorted_extent = np.sort(extent)[::-1]
    elongation = float(sorted_extent[0] / max(sorted_extent[2], 1e-9))
    compactness = float(area / max(bbox_diagonal ** 2, 1e-12))

    if component == main_id:
        nearest = 0.0
    else:
        distances = []
        step = max(1, len(points) // 64)
        for point in points[::step]:
            _, _, squared = main_tree.search_knn_vector_3d(point, 1)
            distances.append(float(np.sqrt(squared[0])))
        nearest = float(min(distances)) if distances else float("inf")

    sub = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(welded_vertices),
        o3d.utility.Vector3iVector(face_subset),
    )
    sub.remove_unreferenced_vertices()
    boundary_edges = int(np.asarray(sub.get_non_manifold_edges(allow_boundary_edges=False)).shape[0])
    watertight = bool(sub.is_watertight())

    records.append({
        "component_id": component,
        "is_main": component == main_id,
        "faces": int(counts[component]),
        "surface_area": area,
        "bbox_diagonal": bbox_diagonal,
        "bbox_extent": [float(v) for v in extent],
        "centroid_normalized": [float(v) for v in (centroid - model_low) / np.maximum(model_high - model_low, 1e-9)],
        "centroid_world": [float(v) for v in centroid],
        "nearest_surface_distance_to_main": nearest,
        "distance_as_model_ratio": float(nearest / max(model_diag, 1e-9)),
        "compactness": compactness,
        "elongation": elongation,
        "boundary_edge_count": boundary_edges,
        "watertight": watertight,
    })

records.sort(key=lambda r: -r["faces"])
digest = hashlib.sha256(open(MESH_PATH, "rb").read()).hexdigest()
report = {
    "input_mesh": MESH_PATH,
    "input_mesh_sha256": digest,
    "component_signature_version": 1,
    "weld_distance": WELD_DISTANCE,
    "model_diagonal": model_diag,
    "total_faces": int(welded_triangles.shape[0]),
    "component_count": int(len(counts)),
    "main_component_id": main_id,
    "components": records,
}
os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
with open(OUT_JSON, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

np.savez_compressed(
    OUT_NPZ,
    vertices=welded_vertices.astype(np.float32),
    triangles=welded_triangles.astype(np.int32),
    labels=labels.astype(np.int32),
    counts=counts.astype(np.int64),
    areas=areas.astype(np.float64),
    main_id=np.int32(main_id),
    model_low=model_low.astype(np.float32),
    model_high=model_high.astype(np.float32),
)
print(
    f"STAGE1_AUDIT faces={welded_triangles.shape[0]} components={len(counts)} "
    f"main_id={main_id} main_faces={counts[main_id]} sha={digest[:12]}",
    flush=True,
)
