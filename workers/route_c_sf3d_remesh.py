"""Bounded remeshing of already-generated SF3D geometry, with no new inference.

SF3D's remesh options normally sit inside `run_image`, which means asking for them re-runs the
network and then bakes a texture. But `Mesh.triangle_remesh` and `Mesh.quad_remesh` are pure
geometry operations, so an existing GLB can be wrapped back into an SF3D `Mesh` and remeshed
directly. That is what this does: same official code path, no second inference pass, and no
texturing.

Targets are expressed as a fraction of the input vertex count and kept conservative. Aggressive
decimation on a hull this size collapses the deck ledges into the sides, and the point of the test
is topology quality, not the smallest triangle count that still loads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="already-generated SF3D .glb")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sf3d-root", default=r"C:\AI\StableFast3D")
    parser.add_argument("--triangle-vertex-fraction", type=float, default=0.6)
    parser.add_argument("--quad-vertex-fraction", type=float, default=0.6)
    parser.add_argument("--triangle-remesh-steps", type=int, default=10)
    args = parser.parse_args()

    sys.path.insert(0, args.sf3d_root)

    import numpy as np
    import torch
    import trimesh

    from sf3d.models.mesh import Mesh

    source = Path(args.input)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    preserved = root / "input_preserved.glb"
    preserved.write_bytes(source.read_bytes())

    original = trimesh.load(str(source), force="mesh", process=False)
    vertices = np.asarray(original.vertices, dtype=np.float64)
    faces = np.asarray(original.faces, dtype=np.int64)

    def build() -> Mesh:
        return Mesh(torch.from_numpy(vertices.copy()).float(),
                    torch.from_numpy(faces.copy()).long())

    def stats(mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> dict:
        a = mesh_vertices[mesh_faces[:, 0]]
        b = mesh_vertices[mesh_faces[:, 1]]
        c = mesh_vertices[mesh_faces[:, 2]]
        lengths = np.stack((np.linalg.norm(b - a, axis=1),
                            np.linalg.norm(c - b, axis=1),
                            np.linalg.norm(a - c, axis=1)), axis=1)
        aspect = lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-12)
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        return {
            "vertices": int(len(mesh_vertices)),
            "triangles": int(len(mesh_faces)),
            "degenerate_triangles": int(np.count_nonzero(areas <= 1e-12)),
            "extreme_aspect_triangles": int(np.count_nonzero(aspect >= 50.0)),
            "max_triangle_aspect": round(float(aspect.max()), 1),
            "median_edge_length": round(float(np.median(lengths)), 6),
            "surface_area": round(float(areas.sum()), 6),
            "extent": [round(float(v), 6) for v in (mesh_vertices.max(axis=0) - mesh_vertices.min(axis=0))],
        }

    report = {
        "schema": "sf3d_bounded_remesh_v1",
        "input": str(source),
        "input_sha256": sha256(source),
        "input_preserved_copy": str(preserved),
        "inference_rerun": False,
        "texturing": False,
        "original": stats(vertices, faces),
        "variants": {},
    }

    target_triangle = max(64, int(len(vertices) * args.triangle_vertex_fraction))
    target_quad = max(64, int(len(vertices) * args.quad_vertex_fraction))

    for name, runner, target in (
        ("triangle_remesh", "triangle", target_triangle),
        ("quad_remesh", "quad", target_quad),
    ):
        entry = {"requested_target_vertex_count": int(target), "success": False}
        try:
            mesh = build()
            started = time.time()
            if runner == "triangle":
                result = mesh.triangle_remesh(triangle_vertex_count=int(target),
                                              triangle_remesh_steps=args.triangle_remesh_steps)
            else:
                result = mesh.quad_remesh(quad_vertex_count=int(target))
            entry["seconds"] = round(time.time() - started, 1)

            out_vertices = result.v_pos.detach().cpu().numpy().astype(np.float64)
            out_faces = result.t_pos_idx.detach().cpu().numpy().astype(np.int64)
            if len(out_faces) == 0:
                raise RuntimeError("remesh produced no faces")
            destination = root / f"sf3d_{name}.glb"
            trimesh.Trimesh(vertices=out_vertices, faces=out_faces, process=False).export(str(destination))
            entry.update({
                "success": True,
                "output": str(destination),
                "output_sha256": sha256(destination),
                "after": stats(out_vertices, out_faces),
            })
        except Exception as error:  # noqa: BLE001 - a backend that is not viable must be recorded
            entry["error"] = f"{type(error).__name__}: {error}"[:800]
        report["variants"][name] = entry
        print(f"SF3D_REMESH {name} success={entry['success']} "
              f"{entry.get('after', {}).get('triangles', entry.get('error', ''))}", flush=True)

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"SF3D_REMESH_DONE report={args.report}", flush=True)


if __name__ == "__main__":
    main()
