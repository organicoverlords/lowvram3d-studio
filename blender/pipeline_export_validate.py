"""Final export and fresh-import validation for Pipeline V2."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import bpy
import sys
import numpy as np

from common import argv_after_double_dash, export_glb, import_mesh, reset_scene, save_json, world_bounds
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
from lowvram3d.anchor_provenance import AnchorProvenanceError, geometry_sha256, load_anchor_provenance, provenance_record


def finite_vector(vector) -> bool:
    return all(math.isfinite(float(value)) for value in vector)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--anchor-receipt", required=True)
    parser.add_argument("--expected-source-sha256", default="")
    args = parser.parse_args(argv_after_double_dash())

    try:
        _receipt, receipt_hash, anchor_ids = load_anchor_provenance(
            args.anchor_receipt, expected_source_sha256=args.expected_source_sha256 or None
        )
    except AnchorProvenanceError as exc:
        save_json(args.report, {"passed": False, "failures": [exc.detail], "failure_codes": [exc.code]})
        raise SystemExit(2)

    reset_scene()
    meshes = import_mesh(args.input)
    if not meshes:
        raise RuntimeError(f"no mesh imported from {args.input}")
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    def geometry_hash(objects):
        vertices, triangles, base = [], [], 0
        for item in objects:
            vertices.extend([list(item.matrix_world @ vertex.co) for vertex in item.data.vertices])
            verts = [base + int(v) for v in polygon.vertices]
            triangles.extend([[verts[0], verts[i], verts[i + 1]] for i in range(1, len(verts) - 1)])
            base += len(item.data.vertices)
        return geometry_sha256(np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64))
    input_geometry_hash = geometry_hash(meshes)
    failures = []
    for obj in meshes:
        if any(float(value) < 0.0 for value in obj.scale):
            failures.append(f"{obj.name}: negative object scale")
        if not finite_vector(obj.scale) or not finite_vector(obj.location):
            failures.append(f"{obj.name}: non-finite transform")
        if len(obj.data.polygons) <= 0:
            failures.append(f"{obj.name}: zero triangles")

    minimum, maximum = world_bounds(meshes)
    extent = maximum - minimum
    if not finite_vector(minimum) or not finite_vector(maximum) or extent.length <= 1e-6:
        failures.append("invalid or zero bounds")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not failures:
        export_glb(str(output), selected_only=False)
    fbx = output.with_suffix(".fbx")
    if not failures:
        bpy.ops.export_scene.fbx(
            filepath=str(fbx), use_selection=False, add_leaf_bones=False,
            bake_anim=bool(armatures), apply_unit_scale=True, axis_forward="-Y", axis_up="Z",
        )

    # Fresh re-import catches missing textures, invalid skin references and state that existed only
    # in the export scene.
    imported_stats = {}
    output_geometry_hash = None
    if not failures:
        reset_scene()
        imported = import_mesh(str(output))
        imported_armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
        if not imported:
            failures.append("fresh GLB import produced no mesh")
        else:
            output_geometry_hash = geometry_hash(imported)
            lo2, hi2 = world_bounds(imported)
            imported_stats = {
                "mesh_objects": len(imported),
                "armatures": len(imported_armatures),
                "triangles": sum(len(obj.data.polygons) for obj in imported),
                "bounds_min": [float(v) for v in lo2],
                "bounds_max": [float(v) for v in hi2],
                "material_slots": sum(len(obj.material_slots) for obj in imported),
                "uv_layers": sum(len(obj.data.uv_layers) for obj in imported),
            }
            if imported_stats["triangles"] <= 0:
                failures.append("fresh GLB import has zero triangles")
            if imported_stats["uv_layers"] <= 0:
                failures.append("fresh GLB import has no UV layer")
            if output_geometry_hash != input_geometry_hash:
                failures.append("fresh GLB import geometry hash differs from accepted input")

    failure_codes = []
    if output_geometry_hash not in (None, input_geometry_hash):
        failure_codes.append("GEOMETRY_MUTATION")
    if any("UV" in failure or "texture" in failure.lower() or "material" in failure.lower() for failure in failures):
        failure_codes.append("TEXTURE_ONLY")
    report = {
        "input": args.input,
        "output_glb": str(output),
        "output_fbx": str(fbx),
        "passed": not failures,
        "failures": failures,
        "source_mesh_objects": len(meshes),
        "source_armatures": len(armatures),
        "source_bounds_min": [float(v) for v in minimum],
        "source_bounds_max": [float(v) for v in maximum],
        "source_extent": [float(v) for v in extent],
        "fresh_import": imported_stats,
        "provenance": provenance_record(
            receipt_sha256=receipt_hash, anchor_ids=anchor_ids,
            input_geometry_sha256=input_geometry_hash,
            output_geometry_sha256=output_geometry_hash or "",
            geometry_unchanged=output_geometry_hash == input_geometry_hash,
        ),
        "failure_codes": sorted(set(failure_codes)),
        "glb_bytes": output.stat().st_size if output.exists() else 0,
        "fbx_bytes": fbx.stat().st_size if fbx.exists() else 0,
        "unreal_orientation": {"forward": "-Y", "up": "Z"},
    }
    save_json(args.report, report)
    print(f"EXPORT_VALIDATE passed={report['passed']} failures={failures}", flush=True)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
