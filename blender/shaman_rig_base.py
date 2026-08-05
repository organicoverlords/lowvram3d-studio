"""Build the welded rig base from the canonical corner-split shaman GLB.

The canonical source stores one vertex per triangle corner (3.00 vertices per
triangle), so edges are effectively unshared and no skinning method that relies
on mesh connectivity can work on it. This stage welds at a fixed distance,
gates the result against the source, and exports the rig base that every later
milestone builds on.

The canonical source file is opened read-only and never written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import bmesh
import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import argv_after_double_dash, export_glb, reset_scene  # noqa: E402


WELD_DISTANCE = 1e-4

# Gate tolerances, fixed before the run.
MAX_BOUNDS_DELTA = 1e-4              # model units
MAX_VOLUME_DELTA_RATIO = 0.005       # 0.5 %
MIN_DOMINANT_COMPONENT_RATIO = 0.95

# Face loss is gated against measured collapsible slivers rather than an
# arbitrary ratio: a distance weld can only remove a face if that face has an
# edge shorter than the weld distance. Anything beyond that count means the
# weld ate real surface and must fail closed.
COLLAPSIBLE_FACE_MARGIN = 1.02


def mesh_points(obj) -> np.ndarray:
    buffer = np.empty(len(obj.data.vertices) * 3, dtype=np.float32)
    obj.data.vertices.foreach_get("co", buffer)
    points = buffer.reshape(-1, 3)
    matrix = np.array(obj.matrix_world.to_4x4())
    homogeneous = np.hstack([points, np.ones((points.shape[0], 1), dtype=np.float32)])
    return (matrix @ homogeneous.T).T[:, :3]


def bmesh_volume(obj) -> float:
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    bmesh.ops.triangulate(mesh, faces=mesh.faces)
    volume = mesh.calc_volume(signed=False)
    mesh.free()
    return float(volume)


def component_stats(obj) -> dict:
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    mesh.verts.ensure_lookup_table()

    total = len(mesh.verts)
    seen = np.zeros(total, dtype=bool)
    components = 0
    largest = 0
    for index in range(total):
        if seen[index]:
            continue
        components += 1
        size = 0
        stack = [mesh.verts[index]]
        seen[index] = True
        while stack:
            vertex = stack.pop()
            size += 1
            for edge in vertex.link_edges:
                other = edge.other_vert(vertex)
                if other is not None and not seen[other.index]:
                    seen[other.index] = True
                    stack.append(other)
        largest = max(largest, size)

    boundary = sum(1 for edge in mesh.edges if len(edge.link_faces) == 1)
    non_manifold = sum(1 for edge in mesh.edges if len(edge.link_faces) > 2)
    mesh.free()
    return {
        "vertices": total,
        "connected_components": components,
        "largest_component_vertices": largest,
        "largest_component_ratio": (largest / total) if total else 0.0,
        "boundary_edges": boundary,
        "non_manifold_edges": non_manifold,
    }


def collapsible_face_count(obj, distance: float) -> dict:
    """Faces a distance weld is allowed to remove: those with a sub-distance edge."""

    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    short_edges = 0
    collapsible = set()
    for edge in mesh.edges:
        if edge.calc_length() < distance:
            short_edges += 1
            for face in edge.link_faces:
                collapsible.add(face.index)
    total_faces = len(mesh.faces)
    mesh.free()
    return {
        "weld_distance": distance,
        "short_edges": short_edges,
        "collapsible_faces": len(collapsible),
        "total_faces": total_faces,
    }


def weld(obj, distance: float) -> dict:
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    before = {"verts": len(mesh.verts), "edges": len(mesh.edges), "faces": len(mesh.faces)}
    bmesh.ops.remove_doubles(mesh, verts=mesh.verts, dist=distance)
    after = {"verts": len(mesh.verts), "edges": len(mesh.edges), "faces": len(mesh.faces)}
    mesh.to_mesh(obj.data)
    mesh.free()
    obj.data.update()
    return {"before": before, "after": after, "distance": distance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--weld-distance", type=float, default=WELD_DISTANCE)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    bpy.ops.import_scene.gltf(filepath=args.input)

    meshes = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if len(meshes) != 1:
        raise RuntimeError(f"rig base expects exactly one mesh, found {len(meshes)}")
    obj = meshes[0]
    obj.name = "shaman_rig_base"
    obj.data.name = "shaman_rig_base_mesh"

    source_points = mesh_points(obj)
    source = {
        "vertices": len(obj.data.vertices),
        "polygons": len(obj.data.polygons),
        "triangles": sum(max(len(p.vertices) - 2, 0) for p in obj.data.polygons),
        "volume": bmesh_volume(obj),
        "bounds_min": [float(v) for v in source_points.min(axis=0)],
        "bounds_max": [float(v) for v in source_points.max(axis=0)],
    }

    collapsible = collapsible_face_count(obj, args.weld_distance)
    weld_result = weld(obj, args.weld_distance)

    base_points = mesh_points(obj)
    components = component_stats(obj)
    base = {
        "vertices": len(obj.data.vertices),
        "polygons": len(obj.data.polygons),
        "triangles": sum(max(len(p.vertices) - 2, 0) for p in obj.data.polygons),
        "volume": bmesh_volume(obj),
        "bounds_min": [float(v) for v in base_points.min(axis=0)],
        "bounds_max": [float(v) for v in base_points.max(axis=0)],
    }

    lost_faces = source["polygons"] - base["polygons"]
    face_loss = lost_faces / max(source["polygons"], 1)
    allowed_face_loss = int(collapsible["collapsible_faces"] * COLLAPSIBLE_FACE_MARGIN)
    volume_delta = abs(base["volume"] - source["volume"]) / max(source["volume"], 1e-9)
    bounds_delta = max(
        max(abs(a - b) for a, b in zip(source["bounds_min"], base["bounds_min"])),
        max(abs(a - b) for a, b in zip(source["bounds_max"], base["bounds_max"])),
    )

    failures = []
    if lost_faces > allowed_face_loss:
        failures.append("RIG_BASE_FACE_LOSS_EXCEEDS_COLLAPSIBLE_SLIVERS")
    if lost_faces < 0:
        failures.append("RIG_BASE_FACE_COUNT_INCREASED")
    if volume_delta > MAX_VOLUME_DELTA_RATIO:
        failures.append("RIG_BASE_VOLUME_DELTA_EXCEEDED")
    if bounds_delta > MAX_BOUNDS_DELTA:
        failures.append("RIG_BASE_BOUNDS_MOVED")
    if components["largest_component_ratio"] < MIN_DOMINANT_COMPONENT_RATIO:
        failures.append("RIG_BASE_NO_DOMINANT_COMPONENT")
    if not np.isfinite(base_points).all():
        failures.append("RIG_BASE_NONFINITE_VERTICES")
    if base["vertices"] >= source["vertices"]:
        failures.append("RIG_BASE_WELD_DID_NOT_SHARE_VERTICES")

    gates = {
        "lost_faces": int(lost_faces),
        "collapsible_faces": collapsible["collapsible_faces"],
        "allowed_face_loss": allowed_face_loss,
        "face_loss_ratio": float(face_loss),
        "volume_delta_ratio": float(volume_delta),
        "volume_delta_limit": MAX_VOLUME_DELTA_RATIO,
        "bounds_delta": float(bounds_delta),
        "bounds_delta_limit": MAX_BOUNDS_DELTA,
        "dominant_component_ratio": components["largest_component_ratio"],
        "dominant_component_limit": MIN_DOMINANT_COMPONENT_RATIO,
        "vertices_shared": bool(base["vertices"] < source["vertices"]),
        "vertex_reduction_ratio": float(
            1.0 - base["vertices"] / max(source["vertices"], 1)
        ),
    }

    report = {
        "stage": "RIG_BASE",
        "passed": not failures,
        "failures": failures,
        "canonical_source": args.input,
        "canonical_source_modified": False,
        "weld": weld_result,
        "collapsible_analysis": collapsible,
        "source": source,
        "rig_base": base,
        "components": components,
        "gates": gates,
        "method": "bmesh_remove_doubles_fixed_distance",
        "note": (
            "Topology invariance is enforced from this rig base onward. The "
            "corner-split canonical buffer is not preserved through skinning "
            "and this stage is where that is recorded."
        ),
    }

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if failures:
        print("RIG_BASE_FAILED=" + ",".join(failures), flush=True)
        raise SystemExit(2)

    Path(args.output_glb).parent.mkdir(parents=True, exist_ok=True)
    export_glb(args.output_glb)
    bpy.ops.wm.save_as_mainfile(filepath=args.output_blend)

    print(f"RIG_BASE_VERTICES={base['vertices']}", flush=True)
    print(f"RIG_BASE_FACES={base['polygons']}", flush=True)
    print(f"RIG_BASE_COMPONENTS={components['connected_components']}", flush=True)
    print(f"RIG_BASE_DOMINANT_RATIO={components['largest_component_ratio']:.6f}", flush=True)
    print(f"RIG_BASE_VOLUME_DELTA={volume_delta:.8f}", flush=True)
    print(f"RIG_BASE_GLB={args.output_glb}", flush=True)
    print("RIG_BASE_PASSED=true", flush=True)


if __name__ == "__main__":
    main()
