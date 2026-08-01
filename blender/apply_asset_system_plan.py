"""Apply a deterministic PARTS / POSE_PREP / MATERIALS plan in Blender.

The policy layer decides what is allowed; this worker performs only those explicit
operations. Missing indices, ambiguous objects, unsafe pose state, or failed fresh
import validation stop the candidate. Source input bytes are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import bpy
from mathutils import Quaternion, Vector

from common import (
    argv_after_double_dash,
    export_glb,
    extended_mesh_stats,
    import_mesh,
    reset_scene,
    save_json,
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return cleaned[:63] or "region"


def _mesh_by_name(meshes: list[bpy.types.Object], metadata: dict[str, Any]) -> bpy.types.Object | None:
    requested = str(metadata.get("object_name", "")).strip()
    if requested:
        return next((obj for obj in meshes if obj.name == requested), None)
    return meshes[0] if len(meshes) == 1 else None


def _validated_indices(values: Any, limit: int, field: str) -> list[int]:
    result = sorted({int(value) for value in (values or [])})
    if any(index < 0 or index >= limit for index in result):
        raise ValueError(f"{field} contains an out-of-range index for limit {limit}")
    return result


def _part_vertices(obj: bpy.types.Object, part: dict[str, Any]) -> list[int]:
    vertices = set(_validated_indices(part.get("vertex_indices"), len(obj.data.vertices), "vertex_indices"))
    faces = _validated_indices(part.get("face_indices"), len(obj.data.polygons), "face_indices")
    for face_index in faces:
        vertices.update(obj.data.polygons[face_index].vertices)
    return sorted(vertices)


def _create_vertex_group(obj: bpy.types.Object, name: str, vertices: list[int]) -> str:
    group_name = f"part::{_safe_name(name)}"
    existing = obj.vertex_groups.get(group_name)
    group = existing or obj.vertex_groups.new(name=group_name)
    if vertices:
        group.add(vertices, 1.0, "REPLACE")
    return group.name


def _split_faces(obj: bpy.types.Object, face_indices: list[int], part_id: str) -> bpy.types.Object:
    if not face_indices:
        raise ValueError(f"hard split {part_id!r} has no explicit face indices")
    before_faces = sum(len(item.data.polygons) for item in bpy.context.scene.objects if item.type == "MESH")
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.object.mode_set(mode="OBJECT")
    for index in face_indices:
        obj.data.polygons[index].select = True
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.separate(type="SELECTED")
    bpy.ops.object.mode_set(mode="OBJECT")
    selected = [item for item in bpy.context.selected_objects if item.type == "MESH" and item != obj]
    if len(selected) != 1:
        raise RuntimeError(f"hard split {part_id!r} produced {len(selected)} candidate objects")
    result = selected[0]
    result.name = f"part_{_safe_name(part_id)}"
    after_faces = sum(len(item.data.polygons) for item in bpy.context.scene.objects if item.type == "MESH")
    if after_faces != before_faces:
        raise RuntimeError(f"hard split changed face count {before_faces} -> {after_faces}")
    return result


def apply_parts(meshes: list[bpy.types.Object], plan: dict[str, Any]) -> dict[str, Any]:
    parts_plan = plan.get("parts") or {}
    parts = {str(item.get("id")): item for item in parts_plan.get("parts") or []}
    decisions = parts_plan.get("decisions") or []
    report: dict[str, Any] = {
        "status": parts_plan.get("status", "missing"),
        "vertex_groups": [],
        "hard_splits": [],
        "preserved": [],
        "failures": [],
    }
    for decision in decisions:
        part_id = str(decision.get("part_id", ""))
        part = parts.get(part_id)
        if not part:
            report["failures"].append(f"PART_DECISION_WITHOUT_REGION:{part_id}")
            continue
        obj = _mesh_by_name(meshes, part.get("metadata") or {})
        if obj is None:
            report["failures"].append(f"PART_OBJECT_AMBIGUOUS:{part_id}")
            continue
        try:
            vertices = _part_vertices(obj, part)
            handling = str(decision.get("handling", "leave_unknown"))
            if handling != "leave_unknown" and not vertices:
                raise ValueError("no explicit vertices or faces")
            if vertices:
                group_name = _create_vertex_group(obj, part_id, vertices)
                report["vertex_groups"].append({
                    "part_id": part_id,
                    "label": part.get("label"),
                    "group": group_name,
                    "vertices": len(vertices),
                    "handling": handling,
                })
            if handling == "hard_split":
                faces = _validated_indices(part.get("face_indices"), len(obj.data.polygons), "face_indices")
                split = _split_faces(obj, faces, part_id)
                report["hard_splits"].append({"part_id": part_id, "object": split.name})
                meshes.append(split)
            else:
                report["preserved"].append({"part_id": part_id, "handling": handling})
        except Exception as exc:  # Blender candidate must fail closed with exact region evidence.
            report["failures"].append(f"PART_APPLICATION_FAILED:{part_id}:{exc}")
    return report


def _group_vertices(obj: bpy.types.Object, group_name: str) -> list[int]:
    group = obj.vertex_groups.get(group_name)
    if group is None:
        return []
    result: list[int] = []
    for vertex in obj.data.vertices:
        if any(item.group == group.index and item.weight > 0.5 for item in vertex.groups):
            result.append(vertex.index)
    return result


def _centroid(obj: bpy.types.Object, indices: list[int]) -> Vector:
    if not indices:
        raise ValueError("cannot compute centroid of an empty region")
    return sum((obj.data.vertices[index].co for index in indices), Vector()) / len(indices)


def _smoothstep(value: float) -> float:
    t = min(1.0, max(0.0, value))
    return t * t * (3.0 - 2.0 * t)


def _rotate_region_to_a_pose(
    obj: bpy.types.Object,
    arm_indices: list[int],
    torso_indices: list[int],
    *,
    side: str,
    arm_degrees: float,
    elbow_degrees: float,
) -> dict[str, Any]:
    torso_center = _centroid(obj, torso_indices)
    arm_points = [obj.data.vertices[index].co.copy() for index in arm_indices]
    pivot_index = min(arm_indices, key=lambda index: (obj.data.vertices[index].co - torso_center).length)
    pivot = obj.data.vertices[pivot_index].co.copy()
    distances = sorted(((point - pivot).length, point) for point in arm_points)
    distal_points = [point for _, point in distances[max(0, int(len(distances) * 0.80)):]]
    distal = sum(distal_points, Vector()) / max(len(distal_points), 1)
    current = distal - pivot
    length = current.length
    if length <= 1e-6:
        raise ValueError("arm region has zero measurable length")
    sign = 1.0 if side == "left" else -1.0
    radians = math.radians(float(arm_degrees))
    target = Vector((sign * math.cos(radians), 0.0, -math.sin(radians)))
    target.normalize()
    current_normalized = current.normalized()
    rotation = current_normalized.rotation_difference(target)
    identity = Quaternion((1.0, 0.0, 0.0, 0.0))
    blend_length = max(length * 0.25, 1e-6)
    for index in arm_indices:
        vertex = obj.data.vertices[index]
        radial = vertex.co - pivot
        weight = _smoothstep(radial.length / blend_length)
        blended = identity.slerp(rotation, weight)
        vertex.co = pivot + blended @ radial

    elbow = pivot + target * (length * 0.52)
    bend = Quaternion(Vector((0.0, 1.0, 0.0)), math.radians(float(elbow_degrees)) * -sign)
    for index in arm_indices:
        vertex = obj.data.vertices[index]
        longitudinal = (vertex.co - pivot).dot(target) / max(length, 1e-6)
        weight = _smoothstep((longitudinal - 0.50) / 0.32)
        if weight <= 0.0:
            continue
        vertex.co = elbow + identity.slerp(bend, weight) @ (vertex.co - elbow)
    obj.data.update()

    new_distal = _centroid(obj, [index for index in arm_indices if (obj.data.vertices[index].co - pivot).length >= length * 0.75])
    direction = (new_distal - pivot).normalized()
    horizontal = math.degrees(math.atan2(abs(direction.z), max(abs(direction.x), 1e-8)))
    return {
        "side": side,
        "vertices": len(arm_indices),
        "pivot": list(pivot),
        "target_degrees_below_horizontal": arm_degrees,
        "measured_degrees_below_horizontal": horizontal,
        "elbow_bend_degrees": elbow_degrees,
    }


def apply_pose(meshes: list[bpy.types.Object], plan: dict[str, Any]) -> dict[str, Any]:
    pose = plan.get("pose_prep") or {}
    action = str(pose.get("action", "preserve_source_pose"))
    report: dict[str, Any] = {"action": action, "applied": False, "arms": [], "failures": []}
    if action != "apply_a_pose":
        report["reason_codes"] = pose.get("reason_codes") or []
        return report
    if len(meshes) != 1:
        report["failures"].append("POSE_REQUIRES_SINGLE_INDEXED_MESH")
        return report
    obj = meshes[0]
    torso = _group_vertices(obj, "part::torso")
    left = _group_vertices(obj, "part::arm_l") or _group_vertices(obj, "part::left_arm")
    right = _group_vertices(obj, "part::arm_r") or _group_vertices(obj, "part::right_arm")
    if not torso or not left or not right:
        report["failures"].append("POSE_REQUIRED_VERTEX_GROUPS_MISSING")
        return report
    specification = pose.get("specification") or {}
    arm_degrees = float(specification.get("arm_from_torso_degrees", 40.0))
    elbow_degrees = float(specification.get("elbow_bend_degrees", 8.0))
    try:
        report["arms"].append(_rotate_region_to_a_pose(
            obj, left, torso, side="left", arm_degrees=arm_degrees, elbow_degrees=elbow_degrees,
        ))
        report["arms"].append(_rotate_region_to_a_pose(
            obj, right, torso, side="right", arm_degrees=arm_degrees, elbow_degrees=elbow_degrees,
        ))
    except Exception as exc:
        report["failures"].append(f"POSE_APPLICATION_FAILED:{exc}")
        return report
    report["applied"] = True
    report["arm_angles_valid"] = all(
        32.5 <= float(item["measured_degrees_below_horizontal"]) <= 47.5
        for item in report["arms"]
    )
    if not report["arm_angles_valid"]:
        report["failures"].append("POSE_ARM_ANGLE_OUT_OF_RANGE")
    return report


def _principled_input(node: bpy.types.Node, *names: str):
    for name in names:
        socket = node.inputs.get(name)
        if socket is not None:
            return socket
    return None


def _set_input(node: bpy.types.Node, value: Any, *names: str) -> None:
    socket = _principled_input(node, *names)
    if socket is not None:
        socket.default_value = value


def _make_material(name: str, family: str, category: str, parameters: dict[str, Any]) -> bpy.types.Material:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    material.node_tree.links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    _set_input(principled, float(parameters.get("roughness_floor", parameters.get("roughness_min", 0.5))), "Roughness")
    _set_input(principled, float(parameters.get("metallic", parameters.get("metallic_max", 0.0))), "Metallic")
    _set_input(principled, float(parameters.get("coat_weight_max", 0.0)), "Coat Weight", "Clearcoat")
    _set_input(principled, float(parameters.get("sheen_weight", 0.0)), "Sheen Weight", "Sheen")
    _set_input(principled, float(parameters.get("subsurface_weight_max", 0.0)), "Subsurface Weight", "Subsurface")
    if family == "MaskedSoft":
        _set_input(principled, 1.0, "Alpha")
        try:
            material.surface_render_method = "DITHERED"
        except Exception:
            try:
                material.blend_method = "HASHED"
            except Exception:
                pass
    if family == "TranslucentSpecial":
        _set_input(principled, float(parameters.get("transmission_weight", 1.0)), "Transmission Weight", "Transmission")
        _set_input(principled, float(parameters.get("ior", 1.45)), "IOR")
    if category == "emissive":
        _set_input(principled, (1.0, 0.5, 0.1, 1.0), "Emission Color", "Emission")
        _set_input(principled, float(parameters.get("emission_strength_default", 2.0)), "Emission Strength")
    material["lowvram3d_family"] = family
    material["lowvram3d_category"] = category
    return material


def apply_materials(meshes: list[bpy.types.Object], plan: dict[str, Any]) -> dict[str, Any]:
    materials = plan.get("materials") or {}
    features = {str(item.get("id")): item for item in materials.get("features") or []}
    report: dict[str, Any] = {
        "status": materials.get("status", "missing"),
        "created": [],
        "assigned_faces": 0,
        "soft_surface_regions": [],
        "failures": [],
    }
    if not meshes:
        report["failures"].append("MATERIAL_NO_MESH")
        return report
    for decision in materials.get("decisions") or []:
        if not decision.get("enabled"):
            continue
        feature_id = str(decision.get("feature_id", ""))
        feature = features.get(feature_id)
        if not feature:
            report["failures"].append(f"MATERIAL_DECISION_WITHOUT_FEATURE:{feature_id}")
            continue
        metadata = feature.get("metadata") or {}
        obj = _mesh_by_name(meshes, metadata)
        if obj is None:
            report["failures"].append(f"MATERIAL_OBJECT_AMBIGUOUS:{feature_id}")
            continue
        try:
            face_indices = _validated_indices(metadata.get("face_indices"), len(obj.data.polygons), "material face_indices")
            if not face_indices:
                raise ValueError("enabled material feature has no explicit face indices")
            family = str(decision.get("material_family", "OpaqueLit"))
            category = str(decision.get("category", "unknown"))
            material = _make_material(
                f"M_{_safe_name(feature_id)}_{family}",
                family,
                category,
                decision.get("parameters") or {},
            )
            obj.data.materials.append(material)
            slot_index = len(obj.data.materials) - 1
            for face_index in face_indices:
                obj.data.polygons[face_index].material_index = slot_index
            report["assigned_faces"] += len(face_indices)
            report["created"].append({
                "feature_id": feature_id,
                "category": category,
                "family": family,
                "material": material.name,
                "faces": len(face_indices),
            })
            if family == "MaskedSoft":
                group_vertices = sorted({vertex for face_index in face_indices for vertex in obj.data.polygons[face_index].vertices})
                group = _create_vertex_group(obj, f"soft::{feature_id}", group_vertices)
                report["soft_surface_regions"].append({
                    "feature_id": feature_id,
                    "root_group": group,
                    "card_budget": materials.get("soft_surface_budget"),
                })
        except Exception as exc:
            report["failures"].append(f"MATERIAL_APPLICATION_FAILED:{feature_id}:{exc}")
    return report


def apply_asset_system(
    *,
    input_path: str | Path,
    plan_path: str | Path,
    output_glb: str | Path,
    output_blend: str | Path,
    source_pose_copy: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    input_path = Path(input_path)
    output_glb = Path(output_glb)
    output_blend = Path(output_blend)
    source_pose_copy = Path(source_pose_copy)
    input_hash_before = sha256(input_path)
    source_pose_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path, source_pose_copy)
    if sha256(source_pose_copy) != input_hash_before:
        raise RuntimeError("source-pose copy hash mismatch")

    plan = load_json(plan_path)
    if plan.get("status") != "ready":
        raise RuntimeError(f"asset-system plan is not ready: {plan.get('status')}")

    reset_scene()
    meshes = import_mesh(str(input_path))
    if not meshes:
        raise RuntimeError("input produced no mesh objects")
    before = extended_mesh_stats(meshes)
    parts_report = apply_parts(meshes, plan)
    active_meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    pose_report = apply_pose(active_meshes, plan)
    material_report = apply_materials(active_meshes, plan)
    candidate = extended_mesh_stats(active_meshes)

    failures = list(parts_report["failures"]) + list(pose_report["failures"]) + list(material_report["failures"])
    if candidate["faces"] != before["faces"]:
        failures.append(f"APPLICATION_FACE_COUNT_CHANGED:{before['faces']}->{candidate['faces']}")
    if not candidate.get("finite_bounds"):
        failures.append("APPLICATION_NONFINITE_BOUNDS")

    input_hash_after = sha256(input_path)
    if input_hash_after != input_hash_before:
        failures.append("SOURCE_INPUT_HASH_CHANGED")

    output_blend.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    export_glb(str(output_glb), selected_only=False)

    reset_scene()
    fresh_meshes = import_mesh(str(output_glb))
    fresh = extended_mesh_stats(fresh_meshes)
    fresh_valid = bool(fresh_meshes and fresh.get("finite_bounds") and fresh["faces"] == candidate["faces"])
    if not fresh_valid:
        failures.append("FRESH_IMPORT_VALIDATION_FAILED")

    report = {
        "passed": not failures,
        "input": str(input_path),
        "input_sha256_before": input_hash_before,
        "input_sha256_after": input_hash_after,
        "source_pose_copy": str(source_pose_copy),
        "source_pose_copy_sha256": sha256(source_pose_copy),
        "output_glb": str(output_glb),
        "output_glb_sha256": sha256(output_glb),
        "output_blend": str(output_blend),
        "before": before,
        "candidate": candidate,
        "fresh_import": fresh,
        "fresh_import_validated": fresh_valid,
        "parts": parts_report,
        "pose_prep": pose_report,
        "materials": material_report,
        "failures": failures,
    }
    save_json(str(report_path), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--source-pose-copy", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv_after_double_dash())
    report = apply_asset_system(
        input_path=args.input,
        plan_path=args.plan,
        output_glb=args.output_glb,
        output_blend=args.output_blend,
        source_pose_copy=args.source_pose_copy,
        report_path=args.report,
    )
    print(
        "ASSET_SYSTEM_APPLICATION "
        f"passed={report['passed']} "
        f"groups={len(report['parts']['vertex_groups'])} "
        f"splits={len(report['parts']['hard_splits'])} "
        f"pose={report['pose_prep']['action']} "
        f"materials={len(report['materials']['created'])}",
        flush=True,
    )
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
