"""Minimal, generic GLB read/write for the pipeline workers.

Reads POSITION, TEXCOORD_0 and indices from the first primitive, and synthesises NORMAL when the
file does not carry one - several generators emit position-and-index-only meshes, and assuming the
attribute exists made the shared readers work on exactly one asset.

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
    per_face = face_normals(positions.astype(np.float64), tris)
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


def read_glb(path: Path):
    """Return (positions, normals, uv, indices). normals and uv may be synthesised or None."""
    meta, binary = _chunks(Path(path).read_bytes())

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

    primitive = meta["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    positions = accessor(attributes["POSITION"]).astype(np.float32)
    tris = accessor(primitive["indices"]).reshape(-1, 3).astype(np.int64)
    normals = (accessor(attributes["NORMAL"]).astype(np.float32)
               if "NORMAL" in attributes else vertex_normals(positions, tris))
    uv = accessor(attributes["TEXCOORD_0"]).astype(np.float32) if "TEXCOORD_0" in attributes else None
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
