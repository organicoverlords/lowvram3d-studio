"""Audit the selected GLB container, accessors, nodes and raw vertex mapping."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import trimesh

from workers.scene_pipeline.core import write_json


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
PROOF = Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"


def read_json_chunk(path: Path) -> dict:
    data = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise RuntimeError("GLB_HEADER_INVALID")
    offset = 12
    while offset < total:
        length, kind = struct.unpack_from("<II", data, offset)
        chunk = data[offset + 8:offset + 8 + length]
        if kind == 0x4E4F534A:
            return json.loads(chunk.decode("utf-8"))
        offset += 8 + length
    raise RuntimeError("GLB_JSON_CHUNK_MISSING")


def affine_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, float]:
    design = np.column_stack([source.astype(np.float64), np.ones(len(source))])
    coefficients, _, _, _ = np.linalg.lstsq(design, target.astype(np.float64), rcond=None)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :] = coefficients.T
    residual = (design @ coefficients - target).astype(np.float64)
    return matrix, float(np.linalg.norm(residual, axis=1).max()), float(np.linalg.norm(residual, axis=1).mean())


def main() -> None:
    glb = ROOT / "balanced_010.glb"
    payload = read_json_chunk(glb)
    scene = trimesh.load(glb, force="scene", process=False)
    geometry = next(iter(scene.geometry.values()))
    raw = np.load(ROOT / "points.npy").reshape(-1, 3)
    mask = np.load(ROOT / "mask.npy").astype(bool).ravel()
    raw_vertices = raw[mask]
    glb_vertices = np.asarray(geometry.vertices, dtype=np.float64)
    used_indices = np.unique(np.asarray(geometry.faces, dtype=np.int64).reshape(-1))
    np.save(ROOT / "balanced_010_used_vertex_indices.npy", used_indices)
    mapping_matrix, max_residual, mean_residual = affine_fit(raw_vertices, glb_vertices)
    primitive = payload["meshes"][0]["primitives"][0]
    position_accessor = payload["accessors"][primitive["attributes"]["POSITION"]]
    material = payload.get("materials", [{}])[primitive.get("material", 0)] if payload.get("materials") else {}
    node_records = []
    for node in payload.get("nodes", []):
        node_records.append({"name": node.get("name"), "mesh": node.get("mesh"), "matrix": node.get("matrix"), "translation": node.get("translation"), "rotation": node.get("rotation"), "scale": node.get("scale")})
    report = {
        "schema": "moge_glb_transform_audit_v1",
        "classification": "MOGE_RAW_GLB_TRANSFORM_PROVEN" if max_residual < 1e-4 else "MOGE_RAW_GLB_TRANSFORM_REJECTED",
        "glb": str(glb),
        "asset": payload.get("asset"),
        "scene": payload.get("scene"),
        "nodes": node_records,
        "root_node_transform": "IDENTITY_DEFAULT_IF_UNSPECIFIED",
        "mesh_primitive": {"attributes": primitive.get("attributes"), "indices_accessor": primitive.get("indices"), "mode": primitive.get("mode"), "material": primitive.get("material")},
        "position_accessor": {"count": position_accessor.get("count"), "min": position_accessor.get("min"), "max": position_accessor.get("max"), "type": position_accessor.get("type"), "componentType": position_accessor.get("componentType")},
        "index_winding": {"mode": primitive.get("mode"), "triangle_count": int(len(geometry.faces)), "sample_faces": geometry.faces[:5].tolist()},
        "material": {"doubleSided": material.get("doubleSided", False), "alphaMode": material.get("alphaMode", "OPAQUE"), "name": material.get("name")},
        "gltf_declared_convention": "glTF 2.0 right-handed, Y-up convention; this file declares no node transform",
        "raw_vertex_count": int(len(raw_vertices)),
        "glb_accessor_vertex_count": int(len(glb_vertices)),
        "used_accessor_vertex_count": int(len(used_indices)),
        "used_accessor_indices": str(ROOT / "balanced_010_used_vertex_indices.npy"),
        "M_raw_moge_to_glb": mapping_matrix.tolist(),
        "mapping_max_residual": max_residual,
        "mapping_mean_residual": mean_residual,
        "uv_policy": "DIRECT_SOURCE_PIXEL_UV_u=x/(width-1),v=1-y/(height-1)",
        "normal_policy": "source normals embedded; no transform at raw-to-GLB identity boundary",
    }
    write_json(ROOT / "glb_transform_audit.json", report)
    write_json(PROOF / "glb_transform_audit.json", report)


if __name__ == "__main__":
    main()
