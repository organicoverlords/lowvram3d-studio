"""Prepare a SceneSpec in Blender without generating or editing mesh geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector

from lowvram3d.scene_preparation import build_scene_preparation_plan


def _argv_after_double_dash() -> list[str]:
    if "--" not in sys.argv:
        return []
    return sys.argv[sys.argv.index("--") + 1 :]


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _collection(name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.get(name)
    if collection is None:
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
    return collection


def _move_to_collection(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for current in list(obj.users_collection):
        current.objects.unlink(obj)
    if obj.name not in collection.objects:
        collection.objects.link(obj)


def _apply_transform(obj: bpy.types.Object, transform: dict[str, Any]) -> None:
    obj.location = tuple(float(value) for value in transform["translation_m"])
    obj.rotation_euler = tuple(math.radians(float(value)) for value in transform["rotation_deg_xyz"])
    obj.scale = tuple(float(value) for value in transform["scale_xyz"])


def _asset_root(task: dict[str, Any], collection: bpy.types.Collection) -> bpy.types.Object:
    root = bpy.data.objects.new(f"SCENE_ASSET_{task['asset_id']}", None)
    collection.objects.link(root)
    root.empty_display_type = "PLAIN_AXES"
    root.empty_display_size = 1.0
    root["scene_spec_asset_id"] = task["asset_id"]
    root["scene_spec_representation"] = task["representation"]
    root["scene_spec_semantic_class"] = task["semantic_class"]
    root["scene_spec_collision"] = task["collision"]
    root["scene_spec_navigation"] = task["navigation"]
    root["scene_spec_lod_policy"] = task["lod_policy"]
    root["scene_spec_action"] = task["action"]
    root["scene_spec_tags_json"] = json.dumps(task.get("tags", []), sort_keys=True)
    return root


def _import_task(task: dict[str, Any], collection: bpy.types.Collection) -> dict[str, Any]:
    source = Path(task["source_uri"])
    if not source.exists():
        raise FileNotFoundError(f"source asset missing: {source}")
    if source.suffix.lower() not in {".glb", ".gltf"}:
        raise ValueError(f"unsupported source format: {source.suffix}")

    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(source))
    imported = [obj for obj in bpy.data.objects if obj not in before]
    if not imported:
        raise RuntimeError(f"Blender imported no objects from {source}")

    root = _asset_root(task, collection)
    imported_set = set(imported)
    for obj in imported:
        _move_to_collection(obj, collection)
    for obj in imported:
        if obj.parent not in imported_set:
            world = obj.matrix_world.copy()
            obj.parent = root
            obj.matrix_world = world
    _apply_transform(root, task["transform"])

    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    polygons = sum(len(obj.data.polygons) for obj in mesh_objects)
    vertices = sum(len(obj.data.vertices) for obj in mesh_objects)
    expected = int(task.get("triangle_budget") or 0)
    if expected and polygons != expected:
        raise RuntimeError(
            f"triangle count mismatch for {task['asset_id']}: imported={polygons}, expected={expected}"
        )

    return {
        "asset_id": task["asset_id"],
        "action": "import_gltf",
        "source_uri": str(source),
        "source_sha256": _sha256_file(source),
        "imported_object_count": len(imported),
        "mesh_object_count": len(mesh_objects),
        "vertices": vertices,
        "triangles": polygons,
        "expected_triangles": expected,
        "triangle_contract_preserved": not expected or polygons == expected,
        "mesh_edit_operations": 0,
        "collection": collection.name,
        "root_object": root.name,
    }


def _placeholder_task(task: dict[str, Any], collection: bpy.types.Collection) -> dict[str, Any]:
    root = _asset_root(task, collection)
    _apply_transform(root, task["transform"])
    root["scene_spec_placeholder"] = True
    return {
        "asset_id": task["asset_id"],
        "action": "placeholder",
        "classification": "NOT_BUILT_REFERENCE_ONLY",
        "collection": collection.name,
        "root_object": root.name,
        "mesh_edit_operations": 0,
    }


def _create_spline(task: dict[str, Any], collection: bpy.types.Collection) -> dict[str, Any]:
    curve_data = bpy.data.curves.new(f"SCENE_SPLINE_{task['spline_id']}", type="CURVE")
    curve_data.dimensions = "3D"
    curve_data.resolution_u = 1
    spline = curve_data.splines.new(type="POLY")
    points = task["points_m"]
    spline.points.add(len(points) - 1)
    for index, point in enumerate(points):
        spline.points[index].co = (float(point[0]), float(point[1]), float(point[2]), 1.0)
    obj = bpy.data.objects.new(f"SCENE_SPLINE_{task['spline_id']}", curve_data)
    collection.objects.link(obj)
    obj["scene_spec_spline_id"] = task["spline_id"]
    obj["scene_spec_kind"] = task["kind"]
    obj["scene_spec_width_m"] = float(task["width_m"])
    obj["scene_spec_exclusion_radius_m"] = float(task.get("exclusion_radius_m", 0.0))
    if task.get("grammar_id"):
        obj["scene_spec_grammar_id"] = task["grammar_id"]
    obj["scene_spec_tags_json"] = json.dumps(task.get("tags", []), sort_keys=True)
    return {
        "spline_id": task["spline_id"],
        "point_count": len(points),
        "collection": collection.name,
        "object": obj.name,
        "geometry_role": "REFERENCE_CURVE_ONLY",
    }


def _create_camera(camera: dict[str, Any], collection: bpy.types.Collection) -> dict[str, Any]:
    data = bpy.data.cameras.new("SCENE_SOURCE_CAMERA")
    data.type = "PERSP" if camera["projection"] == "perspective" else "ORTHO"
    if data.type == "PERSP":
        data.angle = math.radians(float(camera["field_of_view_deg"]))
    else:
        data.ortho_scale = float(camera["ortho_scale_m"])
    data.clip_start = float(camera["near_m"])
    data.clip_end = float(camera["far_m"])
    obj = bpy.data.objects.new("SCENE_SOURCE_CAMERA", data)
    collection.objects.link(obj)
    obj.location = tuple(float(value) for value in camera["position_m"])
    target = Vector(tuple(float(value) for value in camera["look_at_m"]))
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()
    obj["scene_spec_camera_id"] = camera["id"]
    obj["scene_spec_projection"] = camera["projection"]
    obj["scene_spec_position_m_json"] = json.dumps(camera["position_m"])
    obj["scene_spec_look_at_m_json"] = json.dumps(camera["look_at_m"])
    bpy.context.scene.camera = obj
    return {
        "object": obj.name,
        "camera_id": camera["id"],
        "projection": camera["projection"],
        "position_m": camera["position_m"],
        "look_at_m": camera["look_at_m"],
        "near_m": camera["near_m"],
        "far_m": camera["far_m"],
        "field_of_view_deg": camera.get("field_of_view_deg"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a SceneSpec in Blender")
    parser.add_argument("--scene-spec", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args(_argv_after_double_dash())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    asset_receipts: list[dict[str, Any]] = []
    spline_receipts: list[dict[str, Any]] = []
    camera_receipt: dict[str, Any] | None = None

    try:
        spec = _read_json(args.scene_spec)
        plan = build_scene_preparation_plan(spec)
        if plan["classification"] != "PROVEN":
            raise ValueError(f"preparation plan rejected: {plan.get('errors')}")

        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene.unit_settings.system = "METRIC"
        scene.unit_settings.scale_length = 1.0
        scene["scene_spec_id"] = spec["scene_id"]
        scene["scene_spec_sha256"] = plan["scene_spec_sha256"]
        scene["scene_spec_geometry_generation"] = False

        collections = {name: _collection(name) for name in plan["collections"]}
        camera_receipt = _create_camera(plan["camera"], collections["SCENE_REFERENCE_ONLY"])

        for task in plan["asset_tasks"]:
            collection = collections[task["collection"]]
            if task["action"] == "import_gltf":
                asset_receipts.append(_import_task(task, collection))
            else:
                asset_receipts.append(_placeholder_task(task, collection))

        for task in plan["spline_tasks"]:
            spline_receipts.append(
                _create_spline(task, collections[task["collection"]])
            )

        blend_path = output_dir / f"{spec['scene_id']}_prepared.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        classification = "PROVEN"
    except Exception as exc:  # Blender must always emit a receipt.
        classification = "REJECTED"
        errors.append(f"{type(exc).__name__}: {exc}")
        blend_path = output_dir / "scene_preparation_failed.blend"
        try:
            bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        except Exception as save_exc:
            errors.append(f"save failed: {type(save_exc).__name__}: {save_exc}")

    receipt = {
        "schema_version": "blender_scene_preparation_receipt_v1",
        "classification": classification,
        "scene_spec": str(Path(args.scene_spec).resolve()),
        "blend": str(blend_path),
        "collections": sorted(
            name for name in bpy.data.collections.keys() if name.startswith("SCENE_")
        ),
        "camera": camera_receipt,
        "assets": asset_receipts,
        "splines": spline_receipts,
        "errors": errors,
        "mesh_geometry_generated": False,
        "mesh_edit_operations": 0,
        "gpu_work_started": False,
        "neural_work_started": False,
        "unreal_work_started": False,
        "placeholder_assets_are_not_promoted": True,
    }
    report_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    print(f"BLENDER_SCENE_PREPARATION={classification}")
    return 0 if classification == "PROVEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
