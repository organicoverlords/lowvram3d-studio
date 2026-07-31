"""Fresh-process geometry/material validation for a raster-export candidate GLB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import bmesh
import bpy

from common import argv_after_double_dash, import_mesh, reset_scene, save_json

WELD_DISTANCE = 4e-4


def component_count(bm: bmesh.types.BMesh) -> int:
    bm.faces.ensure_lookup_table()
    seen: set[int] = set()
    count = 0
    for face in bm.faces:
        if face.index in seen:
            continue
        count += 1
        stack = [face]
        seen.add(face.index)
        while stack:
            current = stack.pop()
            for edge in current.edges:
                for neighbour in edge.link_faces:
                    if neighbour.index not in seen:
                        seen.add(neighbour.index)
                        stack.append(neighbour)
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--cleanup-report", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv_after_double_dash())

    cleanup = json.loads(Path(args.cleanup_report).read_text(encoding="utf-8-sig"))
    reset_scene()
    objects = import_mesh(args.input)

    faces = 0
    components = 0
    boundary_edges = 0
    non_manifold_edges = 0
    uv_layers = 0
    object_reports = []
    for obj in objects:
        uv_layers += len(obj.data.uv_layers)
        bm = bmesh.new()
        try:
            bm.from_mesh(obj.data)
            bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=WELD_DISTANCE)
            bmesh.ops.triangulate(bm, faces=list(bm.faces))
            obj_faces = len(bm.faces)
            obj_components = component_count(bm)
            obj_boundary = sum(1 for edge in bm.edges if len(edge.link_faces) == 1)
            obj_non_manifold = sum(1 for edge in bm.edges if not edge.is_manifold)
            faces += obj_faces
            components += obj_components
            boundary_edges += obj_boundary
            non_manifold_edges += obj_non_manifold
            object_reports.append(
                {
                    "object": obj.name,
                    "faces": obj_faces,
                    "components": obj_components,
                    "boundary_edges": obj_boundary,
                    "non_manifold_edges": obj_non_manifold,
                }
            )
        finally:
            bm.free()

    materials = {slot.material for obj in objects for slot in obj.material_slots if slot.material}
    image_nodes = []
    for material in materials:
        if not material.use_nodes:
            continue
        for node in material.node_tree.nodes:
            if node.type == "TEX_IMAGE" and node.image:
                image_nodes.append(
                    {
                        "material": material.name,
                        "image": node.image.name,
                        "size": list(node.image.size),
                        "packed": bool(node.image.packed_file),
                        "has_data": bool(node.image.has_data),
                    }
                )

    expected_faces = int(cleanup.get("faces_after", -1))
    expected_components = int(cleanup.get("components_kept", -1))
    expected_boundary = int(cleanup.get("boundary_edges_after", -1))
    errors = []
    if not objects:
        errors.append("no mesh objects after fresh GLB import")
    if faces != expected_faces:
        errors.append(f"face count changed across GLB export: expected {expected_faces}, got {faces}")
    if expected_components >= 0 and components > expected_components + 2:
        errors.append(
            f"connected components increased across GLB export: expected <= {expected_components + 2}, got {components}"
        )
    boundary_tolerance = max(8, int(max(expected_boundary, 0) * 0.02))
    if expected_boundary >= 0 and boundary_edges > expected_boundary + boundary_tolerance:
        errors.append(
            f"boundary edges increased across GLB export: expected <= {expected_boundary + boundary_tolerance}, got {boundary_edges}"
        )
    if uv_layers <= 0:
        errors.append("no UV layers after fresh GLB import")
    if not materials:
        errors.append("no materials after fresh GLB import")
    if not image_nodes:
        errors.append("no image texture nodes after fresh GLB import")
    if any(not node["has_data"] for node in image_nodes):
        errors.append("one or more GLB images did not resolve")

    report = {
        "success": not errors,
        "validator_version": 2,
        "input": str(Path(args.input)),
        "cleanup_report": str(Path(args.cleanup_report)),
        "faces": faces,
        "components": components,
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
        "uv_layers": uv_layers,
        "material_count": len(materials),
        "image_nodes": image_nodes,
        "objects": object_reports,
        "errors": errors,
    }
    save_json(args.report, report)
    if errors:
        raise RuntimeError(f"Fresh GLB geometry validation failed: {errors}")
    print(
        f"GEOMETRY_QUALITY_VALID faces={faces} components={components} "
        f"boundary={boundary_edges} materials={len(materials)} images={len(image_nodes)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
