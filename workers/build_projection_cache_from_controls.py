"""Build a generic projection cache from an existing registered control bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesh_io import read_glb


def build(mesh: Path, contract_path: Path, controls: Path, output: Path) -> dict:
    positions, _normals, uv, triangles = read_glb(mesh)
    if uv is None:
        raise RuntimeError("TEXTURE_UV_MISSING")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    views = sorted(contract["views"], key=lambda item: int(item["index"]))
    uv_triangles = np.asarray(uv[triangles], np.float32)
    edge_a = positions[triangles[:, 1]] - positions[triangles[:, 0]]
    edge_b = positions[triangles[:, 2]] - positions[triangles[:, 0]]
    normals = np.cross(edge_a, edge_b)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    arrays = {"verts": positions.astype(np.float32), "tris": triangles.astype(np.int32),
              "uvs": uv_triangles, "normals": normals.astype(np.float32),
              "view_names": np.asarray([str(v["semantic_name"]) for v in views])}
    locations = []
    for view in views:
        semantic = str(view["semantic_name"])
        locations.append(np.asarray(view["camera_position"], np.float32))
        ids = np.load(controls / f"{semantic}_triangle_ids.npy").astype(np.int32)
        arrays[f"face_id_{semantic}"] = ids
        visible_ids = np.unique(ids[ids >= 0])
        visible = np.zeros(len(triangles), bool)
        visible[visible_ids[visible_ids < len(triangles)]] = True
        arrays[f"vis_{semantic}"] = visible
    arrays["view_locs"] = np.asarray(locations, np.float32)
    arrays["ortho_scale"] = np.float32(float(contract.get("ortho_scale", 2.0)))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return {"schema": "projection_cache_from_controls_v1", "mesh": str(mesh),
            "views": [str(v["semantic_name"]) for v in views],
            "triangle_count": int(len(triangles)), "exact_triangle_id_buffers": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--camera-contract", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    report = build(Path(args.mesh), Path(args.camera_contract), Path(args.controls), Path(args.output))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
