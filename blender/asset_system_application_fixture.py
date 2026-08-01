"""Deterministic Blender fixture for the asset-system application worker."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import bpy

from apply_asset_system_plan import apply_asset_system
from common import argv_after_double_dash, export_glb, reset_scene, save_json


def add_box(vertices, faces, center, size):
    cx, cy, cz = center
    sx, sy, sz = (value * 0.5 for value in size)
    start_vertex = len(vertices)
    start_face = len(faces)
    vertices.extend([
        (cx - sx, cy - sy, cz - sz),
        (cx + sx, cy - sy, cz - sz),
        (cx + sx, cy + sy, cz - sz),
        (cx - sx, cy + sy, cz - sz),
        (cx - sx, cy - sy, cz + sz),
        (cx + sx, cy - sy, cz + sz),
        (cx + sx, cy + sy, cz + sz),
        (cx - sx, cy + sy, cz + sz),
    ])
    quads = [
        (0, 1, 2, 3), (4, 7, 6, 5),
        (0, 4, 5, 1), (1, 5, 6, 2),
        (2, 6, 7, 3), (4, 0, 3, 7),
    ]
    for a, b, c, d in quads:
        faces.append((start_vertex + a, start_vertex + b, start_vertex + c))
        faces.append((start_vertex + a, start_vertex + c, start_vertex + d))
    return {
        "vertex_indices": list(range(start_vertex, start_vertex + 8)),
        "face_indices": list(range(start_face, start_face + 12)),
    }


def build_fixture(path: Path) -> dict:
    reset_scene()
    vertices = []
    faces = []
    regions = {
        "torso": add_box(vertices, faces, (0.0, 0.0, 1.15), (0.55, 0.36, 1.10)),
        "arm_l": add_box(vertices, faces, (0.73, 0.0, 1.48), (0.92, 0.22, 0.22)),
        "arm_r": add_box(vertices, faces, (-0.73, 0.0, 1.48), (0.92, 0.22, 0.22)),
        "head": add_box(vertices, faces, (0.0, 0.0, 1.92), (0.38, 0.34, 0.42)),
        "weapon": add_box(vertices, faces, (1.25, 0.0, 0.95), (0.16, 0.16, 1.45)),
    }
    mesh = bpy.data.meshes.new("FixtureMesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("FixtureMesh", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    export_glb(str(path), selected_only=False)
    return regions


def region(identifier, label, data, *, handling, mesh_state="fused"):
    return {
        "id": identifier,
        "label": label,
        "category": label,
        "confidence": 0.99,
        "mesh_state": mesh_state,
        "vertex_indices": [],
        "face_indices": data["face_indices"],
        "metadata": {},
        "metrics": {},
        "views": {},
        "handling": handling,
    }


def decision(identifier, label, handling):
    return {
        "part_id": identifier,
        "label": label,
        "handling": handling,
        "allowed": True,
        "confidence": 0.99,
        "reason_codes": ["FIXTURE_EXPLICIT"],
        "gates": {},
    }


def pose_material_plan(regions):
    parts = [
        region("torso", "torso", regions["torso"], handling="vertex_group_only"),
        region("arm_l", "left_arm", regions["arm_l"], handling="vertex_group_only"),
        region("arm_r", "right_arm", regions["arm_r"], handling="vertex_group_only"),
        region("weapon", "weapon", regions["weapon"], handling="protected_fused_region"),
    ]
    return {
        "schema_version": 1,
        "status": "ready",
        "profile": "humanoid_complex_accessories",
        "parts": {
            "status": "planned",
            "parts": parts,
            "decisions": [
                decision("torso", "torso", "vertex_group_only"),
                decision("arm_l", "left_arm", "vertex_group_only"),
                decision("arm_r", "right_arm", "vertex_group_only"),
                decision("weapon", "weapon", "protected_fused_region"),
            ],
        },
        "pose_prep": {
            "eligible": True,
            "action": "apply_a_pose",
            "reason_codes": ["POSE_FIXTURE_EXPLICIT"],
            "specification": {
                "arm_from_torso_degrees": 40.0,
                "elbow_bend_degrees": 8.0,
            },
        },
        "materials": {
            "status": "planned",
            "features": [
                {
                    "id": "robe_cloth",
                    "category": "cloth",
                    "confidence": 0.99,
                    "metadata": {"face_indices": regions["torso"]["face_indices"]},
                },
                {
                    "id": "head_hair",
                    "category": "hair",
                    "confidence": 0.99,
                    "metadata": {"face_indices": regions["head"]["face_indices"]},
                },
            ],
            "decisions": [
                {
                    "feature_id": "robe_cloth",
                    "category": "cloth",
                    "status": "enabled",
                    "material_family": "OpaqueLit",
                    "enabled": True,
                    "parameters": {"roughness_floor": 0.58, "sheen_weight": 0.18},
                },
                {
                    "feature_id": "head_hair",
                    "category": "hair",
                    "status": "enabled",
                    "material_family": "MaskedSoft",
                    "enabled": True,
                    "parameters": {"use_cards": True, "dense_groom": False},
                },
            ],
            "soft_surface_budget": {
                "mode": "standard_low_vram",
                "lod_card_limits": {"0": 2000, "1": 900, "2": 300, "3": 0},
                "dense_groom_allowed": False,
            },
        },
    }


def hard_split_plan(regions):
    weapon = region(
        "weapon",
        "weapon",
        regions["weapon"],
        handling="hard_split",
        mesh_state="separate",
    )
    return {
        "schema_version": 1,
        "status": "ready",
        "profile": "humanoid_complex_accessories",
        "parts": {
            "status": "planned",
            "parts": [weapon],
            "decisions": [decision("weapon", "weapon", "hard_split")],
        },
        "pose_prep": {
            "eligible": False,
            "action": "preserve_source_pose",
            "reason_codes": ["FIXTURE_SPLIT_ONLY"],
        },
        "materials": {
            "status": "ordinary_pbr_fallback",
            "features": [],
            "decisions": [],
            "soft_surface_budget": {
                "mode": "standard_low_vram",
                "lod_card_limits": {"0": 2000, "1": 900, "2": 300, "3": 0},
                "dense_groom_allowed": False,
            },
        },
    }


def run_case(root: Path, name: str, plan: dict, input_glb: Path) -> dict:
    case = root / name
    case.mkdir(parents=True, exist_ok=True)
    plan_path = case / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    report_path = case / "application_report.json"
    result = apply_asset_system(
        input_path=input_glb,
        plan_path=plan_path,
        output_glb=case / "output.glb",
        output_blend=case / "output.blend",
        source_pose_copy=case / "source_pose.glb",
        report_path=report_path,
    )
    if not result["passed"]:
        raise RuntimeError(f"fixture case {name} failed: {result['failures']}")
    if result["input_sha256_before"] != result["source_pose_copy_sha256"]:
        raise RuntimeError(f"fixture case {name} did not preserve source bytes")
    if not result["fresh_import_validated"]:
        raise RuntimeError(f"fixture case {name} failed fresh import")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv_after_double_dash())
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    input_glb = root / "fixture_input.glb"
    regions = build_fixture(input_glb)

    pose_result = run_case(root, "pose_materials", pose_material_plan(regions), input_glb)
    if not pose_result["pose_prep"]["applied"]:
        raise RuntimeError("A-pose fixture did not deform the arm regions")
    if not pose_result["pose_prep"]["arm_angles_valid"]:
        raise RuntimeError("A-pose fixture arm-angle gate failed")
    if len(pose_result["materials"]["created"]) != 2:
        raise RuntimeError("material fixture did not create two material families")
    if len(pose_result["materials"]["soft_surface_regions"]) != 1:
        raise RuntimeError("soft-surface fixture did not create one root group")

    split_result = run_case(root, "hard_split", hard_split_plan(regions), input_glb)
    if len(split_result["parts"]["hard_splits"]) != 1:
        raise RuntimeError("hard-split fixture did not produce exactly one prop object")
    if split_result["candidate"]["faces"] != split_result["before"]["faces"]:
        raise RuntimeError("hard-split fixture lost or duplicated faces")

    summary = {
        "passed": True,
        "input": str(input_glb),
        "pose_materials": {
            "groups": len(pose_result["parts"]["vertex_groups"]),
            "pose_applied": pose_result["pose_prep"]["applied"],
            "arm_angles_valid": pose_result["pose_prep"]["arm_angles_valid"],
            "materials": len(pose_result["materials"]["created"]),
            "soft_surface_regions": len(pose_result["materials"]["soft_surface_regions"]),
            "fresh_import_validated": pose_result["fresh_import_validated"],
        },
        "hard_split": {
            "objects_before": split_result["before"]["objects"],
            "objects_after": split_result["candidate"]["objects"],
            "splits": len(split_result["parts"]["hard_splits"]),
            "faces_preserved": split_result["candidate"]["faces"] == split_result["before"]["faces"],
            "fresh_import_validated": split_result["fresh_import_validated"],
        },
    }
    save_json(str(root / "fixture_report.json"), summary)
    print(
        "ASSET_SYSTEM_BLENDER_FIXTURE "
        f"passed={summary['passed']} "
        f"pose={summary['pose_materials']['pose_applied']} "
        f"materials={summary['pose_materials']['materials']} "
        f"splits={summary['hard_split']['splits']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
