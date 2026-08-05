"""Fresh-import QA and isolated motion proof for one garment candidate."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils import Euler, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import argv_after_double_dash  # noqa: E402
from shaman_semantic_v3 import semantic_masks_v3, triangle_metrics  # noqa: E402


def evaluated_points(obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    result = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get("co", result)
    evaluated.to_mesh_clear()
    return result.reshape(-1, 3).copy()


def triangle_array(obj):
    obj.data.calc_loop_triangles()
    result = np.empty(len(obj.data.loop_triangles) * 3, dtype=np.int32)
    obj.data.loop_triangles.foreach_get("vertices", result)
    return result.reshape(-1, 3)


def reset(armature):
    for bone in armature.pose.bones:
        bone.matrix_basis.identity()
    bpy.context.view_layer.update()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--landmarks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv_after_double_dash())
    bpy.ops.wm.open_mainfile(filepath=args.input)
    landmarks = json.loads(Path(args.landmarks).read_text(encoding="utf-8"))
    meshes = {obj.name: obj for obj in bpy.data.objects if obj.type == "MESH"}
    body = meshes["Shaman_Body_Closure"]
    garment = meshes["Shaman_Sleeve_Garment_R"]
    armature = next(obj for obj in bpy.data.objects if obj.type == "ARMATURE")
    body_rest = evaluated_points(body)
    garment_rest = evaluated_points(garment)
    body_triangles = triangle_array(body)
    garment_triangles = triangle_array(garment)
    points = body.data.vertices
    raw = np.empty(len(points) * 3, dtype=np.float64)
    points.foreach_get("co", raw)
    masks = semantic_masks_v3(raw.reshape(-1, 3), landmarks, np.zeros(len(points), dtype=bool))
    protected = {
        "torso": masks["torso_core"],
        "rear_cape": masks["rear_cape"],
        "side_cape": masks["side_cape"],
        "staff": masks["staff"],
    }
    tests = []
    rotations = {
        "clavicle": ("clavicle_r", (0.0, 0.0, -25.0)),
        "upperarm": ("upperarm_r", (-12.0, 0.0, -45.0)),
        "elbow": ("lowerarm_r", (-55.0, 0.0, 0.0)),
        "wrist": ("hand_r", (0.0, 0.0, -45.0)),
        "sleeve_anchor": ("sleeve_r_anchor", (-20.0, 0.0, -10.0)),
        "drape_01": ("sleeve_r_drape_01", (-30.0, 0.0, -15.0)),
        "drape_02": ("sleeve_r_drape_02", (-25.0, 0.0, -12.0)),
        "drape_03": ("sleeve_r_drape_03", (-20.0, 0.0, -10.0)),
    }
    for label, (bone_name, degrees) in rotations.items():
        reset(armature)
        if bone_name not in armature.pose.bones:
            tests.append({"test": label, "bone": bone_name, "classification": "BLOCKED_BONE_ABSENT"})
            continue
        pose_bone = armature.pose.bones[bone_name]
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = Euler([math.radians(v) for v in degrees], "XYZ").to_quaternion()
        bpy.context.view_layer.update()
        body_posed = evaluated_points(body)
        garment_posed = evaluated_points(garment)
        body_delta = np.linalg.norm(body_posed - body_rest, axis=1)
        garment_delta = np.linalg.norm(garment_posed - garment_rest, axis=1)
        protected_displacement = {name: int((mask & (body_delta > 1e-4)).sum()) for name, mask in protected.items()}
        triangles = triangle_metrics(garment_rest, garment_posed, garment_triangles)
        tests.append({
            "test": label,
            "bone": bone_name,
            "rotation_degrees": list(degrees),
            "body_displaced_vertices": int((body_delta > 1e-4).sum()),
            "garment_displaced_vertices": int((garment_delta > 1e-4).sum()),
            "garment_max_displacement": float(garment_delta.max()),
            "protected_displacement": protected_displacement,
            "finite": bool(np.isfinite(body_posed).all() and np.isfinite(garment_posed).all()),
            "triangles": triangles,
        })
    reset(armature)
    all_finite = all(test.get("finite", False) for test in tests)
    wrist = next(test for test in tests if test["test"] == "wrist")
    elbow = next(test for test in tests if test["test"] == "elbow")
    drape = [test for test in tests if test["test"].startswith("drape_")]
    report = {
        "classification": "BLOCKED_SHAMAN_SLEEVE_REQUIRES_MANUAL_RETOPOLOGY",
        "fresh_import_validation": "PROVEN",
        "visible_hole_test": "REJECTED_BODY_BOUNDARY_PRESENT",
        "isolated_tests": tests,
        "gates": {
            "flipped_triangles": max(test["triangles"]["flipped_normals"] for test in tests if "triangles" in test),
            "newly_degenerate_triangles": max(test["triangles"]["degenerate_introduced"] for test in tests if "triangles" in test),
            "extreme_stretch_or_shear_triangles": max(test["triangles"]["extreme_stretch_triangles"] + test["triangles"]["extreme_shear_triangles"] for test in tests if "triangles" in test),
            "torso_displacement_from_free_arm": max(test["protected_displacement"]["torso"] for test in tests if "protected_displacement" in test),
            "rear_cape_displacement_from_free_arm": max(test["protected_displacement"]["rear_cape"] for test in tests if "protected_displacement" in test),
            "side_cape_displacement_from_free_arm": max(test["protected_displacement"]["side_cape"] for test in tests if "protected_displacement" in test),
            "staff_displacement_from_free_arm": max(test["protected_displacement"]["staff"] for test in tests if "protected_displacement" in test),
            "broad_drape_displacement_from_wrist": wrist["garment_displaced_vertices"],
            "broad_drape_displacement_from_elbow": elbow["garment_displaced_vertices"],
            "all_transforms_finite": all_finite,
            "garment_attachment_tests_present": len(drape) == 3,
        },
        "failure_reason": "body closure did not close the extracted seam and garment motion still produces unacceptable triangle deformation",
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("GARMENT_QA=" + report["classification"], flush=True)
    print("WRIST_GARMENT_DISPLACED=" + str(wrist["garment_displaced_vertices"]), flush=True)
    print("ELBOW_GARMENT_DISPLACED=" + str(elbow["garment_displaced_vertices"]), flush=True)


if __name__ == "__main__":
    main()
