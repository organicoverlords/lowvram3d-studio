"""Minimal, generic GLB read/write for the pipeline workers.

Reads POSITION, TEXCOORD_0 and indices from every primitive in the first mesh, and synthesises
NORMAL when the file does not carry one. Material-split GLBs are common after texture export;
reading only primitive zero silently dropped the neutral/rear geometry from texture completion.

Deliberately not trimesh: trimesh exposes ColorVisuals for a mesh with vertex colours and no
material and silently drops TEXCOORD_0, which is how a UV-bearing atlas mesh came back with no UVs.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

COMPONENT_SIZE = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
COMPONENT_DTYPE = {5120: "<i1", 5121: "<u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _node_local_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        values = np.asarray(node["matrix"], dtype=np.float64)
        if values.size != 16:
            raise RuntimeError("GLTF_NODE_MATRIX_INVALID")
        return values.reshape(4, 4).T
    translation = np.asarray(node.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    scale = np.asarray(node.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    rotation = np.asarray(node.get("rotation", [0.0, 0.0, 0.0, 1.0]), dtype=np.float64)
    if translation.shape != (3,) or scale.shape != (3,) or rotation.shape != (4,):
        raise RuntimeError("GLTF_NODE_TRS_INVALID")
    x, y, z, w = rotation
    rotation_matrix = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ], dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _scene_mesh_transforms(meta: dict) -> tuple[dict[int, np.ndarray], dict]:
    nodes = meta.get("nodes", [])
    scenes = meta.get("scenes", [])
    if meta.get("skins"):
        raise RuntimeError("GLTF_SKINS_UNSUPPORTED")
    if meta.get("animations"):
        raise RuntimeError("GLTF_ANIMATIONS_UNSUPPORTED")
    if not nodes:
        meshes = meta.get("meshes", [])
        transforms = {index: np.eye(4, dtype=np.float64) for index in range(len(meshes))}
        return transforms, {"default_scene": None, "mesh_nodes": [], "transforms": [], "identity_all": True}
    scene_index = int(meta.get("scene", 0))
    if scene_index < 0 or scene_index >= len(scenes):
        raise RuntimeError("GLTF_DEFAULT_SCENE_INVALID")
    parents: dict[int, int] = {}
    for parent_index, node in enumerate(nodes):
        for child in node.get("children", []):
            child_index = int(child)
            if child_index < 0 or child_index >= len(nodes) or child_index in parents:
                raise RuntimeError("GLTF_NODE_HIERARCHY_INVALID")
            parents[child_index] = parent_index
    transforms: dict[int, np.ndarray] = {}
    mesh_nodes: list[dict] = []
    visited: set[int] = set()

    def visit(index: int, parent_global: np.ndarray) -> None:
        if index in visited:
            raise RuntimeError("GLTF_NODE_REUSED_OR_CYCLIC")
        visited.add(index)
        node = nodes[index]
        global_matrix = parent_global @ _node_local_matrix(node)
        if "mesh" in node:
            mesh_index = int(node["mesh"])
            if mesh_index in transforms:
                raise RuntimeError("GLTF_MULTIPLE_MESH_INSTANCES_UNSUPPORTED")
            transforms[mesh_index] = global_matrix
            mesh_nodes.append({"node_index": index, "mesh_index": mesh_index, "global_matrix": global_matrix})
        for child in node.get("children", []):
            visit(int(child), global_matrix)

    for root in scenes[scene_index].get("nodes", []):
        visit(int(root), np.eye(4, dtype=np.float64))
    if len(transforms) != len(meta.get("meshes", [])):
        raise RuntimeError("GLTF_MESH_NOT_REACHABLE_FROM_DEFAULT_SCENE")
    report_transforms: list[dict] = []
    for record in mesh_nodes:
        matrix = record["global_matrix"]
        linear = matrix[:3, :3]
        scales = np.linalg.norm(linear, axis=0)
        if np.any(scales <= 1e-12) or not np.allclose(scales, scales[0], rtol=1e-5, atol=1e-7):
            raise RuntimeError(f"GLTF_NON_UNIFORM_OR_ZERO_SCALE:{record['node_index']}:{scales.tolist()}")
        normalized = linear / scales[0]
        if not np.allclose(normalized.T @ normalized, np.eye(3), atol=1e-5):
            raise RuntimeError(f"GLTF_SHEAR_UNSUPPORTED:{record['node_index']}")
        determinant = float(np.linalg.det(linear))
        if determinant < 0:
            raise RuntimeError(f"GLTF_NEGATIVE_SCALE_UNSUPPORTED:{record['node_index']}:{determinant}")
        report_transforms.append({
            "node_index": int(record["node_index"]),
            "mesh_index": int(record["mesh_index"]),
            "global_matrix": matrix.tolist(),
            "identity": bool(np.allclose(matrix, np.eye(4), atol=1e-7)),
            "uniform_scale": float(scales[0]),
            "determinant": determinant,
        })
    return transforms, {
        "default_scene": scene_index,
        "mesh_nodes": report_transforms,
        "identity_all": all(item["identity"] for item in report_transforms),
    }


def _chunks(data: bytes):
    offset, meta, binary = 12, None, None
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        payload = data[offset + 8: offset + 8 + length]
        if kind == 0x4E4F534A:
            meta = json.loads(payload)
        elif kind == 0x004E4942:
            binary = payload
        offset += 8 + length
    if meta is None or binary is None:
        raise RuntimeError("not a self-contained binary GLB")
    return meta, binary


def face_normals(positions: np.ndarray, tris: np.ndarray) -> np.ndarray:
    edge1 = positions[tris[:, 1]] - positions[tris[:, 0]]
    edge2 = positions[tris[:, 2]] - positions[tris[:, 0]]
    normals = np.cross(edge1, edge2)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def vertex_normals(positions: np.ndarray, tris: np.ndarray) -> np.ndarray:
    accumulated = np.zeros_like(positions, dtype=np.float64)
    # Upstream accumulates raw cross products, so larger faces contribute
    # proportionally more than smaller faces before the vertex normalization.
    vertices = positions.astype(np.float64)
    per_face = np.cross(vertices[tris[:, 1]] - vertices[tris[:, 0]], vertices[tris[:, 2]] - vertices[tris[:, 0]])
    for corner in range(3):
        np.add.at(accumulated, tris[:, corner], per_face)
    lengths = np.linalg.norm(accumulated, axis=1, keepdims=True)
    # An isolated vertex with no incident area gets an arbitrary but finite normal.
    accumulated[lengths[:, 0] < 1e-12] = (0.0, 0.0, 1.0)
    return (accumulated / np.maximum(lengths, 1e-12)).astype(np.float32)


def triangle_components(positions: np.ndarray, tris: np.ndarray, weld: float = 4e-4):
    """Connected component id per triangle, over position-welded vertices.

    Runs the component search on the VERTEX graph, not on a triangle-triangle adjacency matrix.
    Forming `incidence @ incidence.T` looks natural and is what the first version did, but it
    materialises every pair of triangles sharing a vertex - on a million-triangle mesh that asked
    for 1.85 TiB and killed the stage. The vertex graph has 3 edges per triangle and gives the same
    partition, because two triangles sharing a welded vertex are connected by definition.

    Returns (component_per_triangle, welded_vertex_index).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    welded = np.unique(np.round(np.asarray(positions, np.float64) / weld).astype(np.int64),
                       axis=0, return_inverse=True)[1]
    corners = welded[tris]
    rows = np.concatenate([corners[:, 0], corners[:, 1], corners[:, 2]])
    cols = np.concatenate([corners[:, 1], corners[:, 2], corners[:, 0]])
    size = int(welded.max()) + 1
    graph = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)), shape=(size, size)).tocsr()
    labels = connected_components(graph, directed=False)[1]
    return labels[corners[:, 0]], welded


def read_glb(path: Path, *, return_normal_source: bool = False, return_scene_report: bool = False):
    """Return (positions, normals, uv, indices). normals and uv may be synthesised or None."""
    meta, binary = _chunks(Path(path).read_bytes())
    mesh_transforms, scene_report = _scene_mesh_transforms(meta)

    def accessor(index):
        acc = meta["accessors"][index]
        view = meta["bufferViews"][acc["bufferView"]]
        count, width = acc["count"], TYPE_COUNT[acc["type"]]
        item = COMPONENT_SIZE[acc["componentType"]] * width
        stride = view.get("byteStride") or item
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        raw = np.frombuffer(binary, np.uint8, count=stride * (count - 1) + item, offset=start)
        if stride != item:
            raw = np.lib.stride_tricks.as_strided(raw, (count, item), (stride, 1)).copy()
        return raw.reshape(-1).view(COMPONENT_DTYPE[acc["componentType"]]).reshape(count, width)

    position_parts, normal_parts, uv_parts, index_parts = [], [], [], []
    has_normals = True
    has_uvs = True
    vertex_offset = 0
    for mesh_index, mesh in enumerate(meta["meshes"]):
        transform = mesh_transforms.get(mesh_index, np.eye(4, dtype=np.float64))
        linear = transform[:3, :3]
        for primitive in mesh["primitives"]:
            attributes = primitive["attributes"]
            positions_raw = accessor(attributes["POSITION"]).astype(np.float64)
            homogeneous = np.concatenate([positions_raw, np.ones((len(positions_raw), 1))], axis=1)
            positions_part = (homogeneous @ transform.T)[:, :3].astype(np.float32)
            position_parts.append(positions_part)
            indices_part = accessor(primitive["indices"]).reshape(-1, 3).astype(np.int64)
            index_parts.append(indices_part + vertex_offset)
            if "NORMAL" in attributes:
                normals_raw = accessor(attributes["NORMAL"]).astype(np.float64)
                normal_matrix = np.linalg.inv(linear)
                transformed_normals = normals_raw @ normal_matrix
                transformed_normals /= np.maximum(np.linalg.norm(transformed_normals, axis=1, keepdims=True), 1e-12)
                normal_parts.append(transformed_normals.astype(np.float32))
            else:
                has_normals = False
            if "TEXCOORD_0" in attributes:
                uv_parts.append(accessor(attributes["TEXCOORD_0"]).astype(np.float32))
            else:
                has_uvs = False
            vertex_offset += len(positions_part)
    if not position_parts:
        raise RuntimeError("GLB contains no mesh primitives")
    positions = np.concatenate(position_parts, axis=0)
    tris = np.concatenate(index_parts, axis=0)
    normal_source = "EMBEDDED_GLTF" if has_normals else "AREA_WEIGHTED_RECOMPUTED"
    normals = (np.concatenate(normal_parts, axis=0)
               if has_normals else vertex_normals(positions, tris))
    uv = np.concatenate(uv_parts, axis=0) if has_uvs else None
    if return_normal_source and return_scene_report:
        return positions, normals, uv, tris, normal_source, scene_report
    if return_scene_report:
        return positions, normals, uv, tris, scene_report
    if return_normal_source:
        return positions, normals, uv, tris, normal_source
    return positions, normals, uv, tris


def write_glb(path: Path, positions, normals, uv, indices) -> None:
    """Write a single-primitive GLB. UV is optional; everything else is required."""
    arrays = [(np.ascontiguousarray(positions, np.float32), "VEC3", 34962),
              (np.ascontiguousarray(normals, np.float32), "VEC3", 34962)]
    if uv is not None:
        arrays.append((np.ascontiguousarray(uv, np.float32), "VEC2", 34962))
    arrays.append((np.ascontiguousarray(indices, np.uint32).reshape(-1), "SCALAR", 34963))

    blobs, accessors, views, offset = [], [], [], 0
    for array, kind, target in arrays:
        raw = array.tobytes()
        pad = (-len(raw)) % 4
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": len(raw), "target": target})
        entry = {"bufferView": len(views) - 1,
                 "componentType": 5125 if kind == "SCALAR" else 5126,
                 "count": int(array.shape[0]), "type": kind}
        if not accessors:
            entry["min"] = [float(v) for v in array.min(axis=0)]
            entry["max"] = [float(v) for v in array.max(axis=0)]
        accessors.append(entry)
        blobs.append(raw + b"\x00" * pad)
        offset += len(raw) + pad

    attributes = {"POSITION": 0, "NORMAL": 1}
    if uv is not None:
        attributes["TEXCOORD_0"] = 2
    binary = b"".join(blobs)
    meta = {
        "asset": {"version": "2.0", "generator": "lowvram3d-pipeline-v2"},
        "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"name": "geometry_0", "primitives": [
            {"attributes": attributes, "indices": len(accessors) - 1}]}],
        "buffers": [{"byteLength": len(binary)}], "bufferViews": views, "accessors": accessors,
    }
    chunk = json.dumps(meta, separators=(",", ":")).encode()
    chunk += b" " * ((-len(chunk)) % 4)
    total = 12 + 8 + len(chunk) + 8 + len(binary)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(
        b"glTF" + struct.pack("<II", 2, total)
        + struct.pack("<II", len(chunk), 0x4E4F534A) + chunk
        + struct.pack("<II", len(binary), 0x004E4942) + binary
    )
