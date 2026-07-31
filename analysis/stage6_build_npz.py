"""STAGE 6a: rebuild the topology-indexed raster NPZ from the newly UV-unwrapped mesh.

Nothing is reused from the previous job: vertices, triangles, UVs, normals and per-view visibility
are all derived from this mesh. Coordinates are converted from the glTF Y-up convention into the
Blender Z-up world the projector's cameras are expressed in.
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
OUT_REPORT = sys.argv[3]

VIEWS = {
    "front": (0.0, -3.0, 0.0),
    "right": (3.0, 0.0, 0.0),
    "back": (0.0, 3.0, 0.0),
    "left": (-3.0, 0.0, 0.0),
}
ORTHO_SCALE = 2.6

loaded = trimesh.load(MESH_PATH, process=False)
mesh = trimesh.util.concatenate(list(loaded.geometry.values())) if isinstance(loaded, trimesh.Scene) else loaded

raw = np.asarray(mesh.vertices, np.float64)
# glTF Y-up -> Blender Z-up, matching every camera in the raster route.
verts = np.column_stack((raw[:, 0], -raw[:, 2], raw[:, 1])).astype(np.float32)
tris = np.asarray(mesh.faces, np.int32)
uv_per_vertex = np.asarray(mesh.visual.uv, np.float32)
uvs = uv_per_vertex[tris].astype(np.float32)

edge1 = verts[tris[:, 1]] - verts[tris[:, 0]]
edge2 = verts[tris[:, 2]] - verts[tris[:, 0]]
face_normals = np.cross(edge1, edge2)
lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
normals = (face_normals / np.maximum(lengths, 1e-20)).astype(np.float32)

centroids = verts[tris].mean(axis=1)

scene = o3d.t.geometry.RaycastingScene()
scene.add_triangles(
    o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(verts.astype(np.float32)),
        o3d.core.Tensor(tris.astype(np.uint32)),
    )
)

visibility = {}
for name, location in VIEWS.items():
    camera = np.asarray(location, np.float32)
    origins = centroids + normals * 1e-3
    direction = camera[None, :] - origins
    distance = np.linalg.norm(direction, axis=1, keepdims=True)
    unit = direction / np.maximum(distance, 1e-20)
    rays = o3d.core.Tensor(np.hstack([origins, unit]).astype(np.float32))
    hits = scene.cast_rays(rays)
    t_hit = hits["t_hit"].numpy()
    visibility[name] = (t_hit >= distance[:, 0] - 1e-4) | ~np.isfinite(t_hit)
    print(f"visibility {name}: {int(visibility[name].sum())}/{len(tris)}", flush=True)

os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
np.savez_compressed(
    OUT_NPZ,
    verts=verts,
    tris=tris,
    uvs=uvs,
    normals=normals,
    view_names=np.array(list(VIEWS)),
    view_locs=np.asarray(list(VIEWS.values()), np.float32),
    ortho_scale=np.float32(ORTHO_SCALE),
    **{f"vis_{name}": visibility[name] for name in VIEWS},
)

report = {
    "source_mesh": MESH_PATH,
    "source_mesh_sha256": hashlib.sha256(open(MESH_PATH, "rb").read()).hexdigest(),
    "npz_sha256": hashlib.sha256(open(OUT_NPZ, "rb").read()).hexdigest(),
    "vertices": int(len(verts)),
    "triangles": int(len(tris)),
    "uv_shape": list(uvs.shape),
    "normals_shape": list(normals.shape),
    "uv_min": [float(v) for v in uv_per_vertex.min(axis=0)],
    "uv_max": [float(v) for v in uv_per_vertex.max(axis=0)],
    "visible_triangles": {name: int(mask.sum()) for name, mask in visibility.items()},
    "axis_convention": "converted glTF Y-up to Blender Z-up as (x, -z, y)",
    "reused_from_previous_job": False,
}
with open(OUT_REPORT, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
print(f"STAGE6_NPZ verts={len(verts)} tris={len(tris)} npz_sha={report['npz_sha256'][:12]}", flush=True)
