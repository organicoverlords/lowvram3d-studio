"""Build a Blender proof scene from image-world surface artifacts.

The scene is diagnostic, not a final game asset. Observed and procedurally
completed regions remain distinguishable, and export promotion remains blocked
until semantic terrain separation and visual review are complete.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import traceback

import bpy
import numpy as np

from lowvram3d.image_world.blender_mesh import build_terrain_mesh_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", required=True, help="Surface-projection output directory")
    parser.add_argument("--blend-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--render-output", required=True)
    parser.add_argument("--glb-output")
    parser.add_argument("--horizontal-size", type=float, default=1000.0)
    parser.add_argument("--vertical-scale", type=float, default=250.0)
    parser.add_argument("--maximum-resolution", type=int, default=257)
    parser.add_argument("--render-size", type=int, default=768)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projection = Path(args.projection).resolve()
    arrays = projection / "arrays"
    blend_output = Path(args.blend_output).resolve()
    report_output = Path(args.report_output).resolve()
    render_output = Path(args.render_output).resolve()
    glb_output = None if not args.glb_output else Path(args.glb_output).resolve()

    report = {
        "status": "FAILED",
        "classification": "DIAGNOSTIC_SURFACE_SCENE_NOT_TERRAIN_PROOF",
        "promotion_allowed": False,
        "projection": str(projection),
        "blend_output": str(blend_output),
        "render_output": str(render_output),
        "glb_output": None if glb_output is None else str(glb_output),
        "warnings": [
            "Semantic terrain masks are still required.",
            "Horizontal structures and water may remain in this surface baseline.",
            "The lighthouse landmark is represented only by a placement socket.",
        ],
        "errors": [],
    }

    try:
        height = _load_required(arrays / "completed-height.npy", np.float32)
        observed = _load_required(arrays / "observed-mask.npy", np.uint8).astype(bool)
        generated = _load_required(arrays / "generated-mask.npy", np.uint8).astype(bool)
        confidence = _load_required(arrays / "confidence.npy", np.float32)
        mesh_data = build_terrain_mesh_data(
            height,
            observed,
            generated,
            confidence,
            horizontal_size=args.horizontal_size,
            vertical_scale=args.vertical_scale,
            maximum_resolution=args.maximum_resolution,
        )

        _reset_scene()
        collections = _create_collections()
        terrain = _create_terrain_object(mesh_data, collections["terrain"])
        _create_landmark_socket(mesh_data, collections["residuals"])
        _create_source_camera_debug(collections["source"])
        camera = _create_proof_camera(mesh_data, collections["debug"])
        _create_lighting(collections["debug"])
        _configure_world()
        _configure_render(camera, render_output, args.render_size)

        blend_output.parent.mkdir(parents=True, exist_ok=True)
        render_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_output))
        bpy.ops.render.render(write_still=True)
        if not render_output.is_file() or render_output.stat().st_size <= 0:
            raise RuntimeError("Blender proof render was not written")

        if glb_output is not None:
            glb_output.parent.mkdir(parents=True, exist_ok=True)
            _select_only(terrain)
            bpy.ops.export_scene.gltf(
                filepath=str(glb_output),
                export_format="GLB",
                use_selection=True,
                export_apply=True,
            )
            if not glb_output.is_file() or glb_output.stat().st_size <= 0:
                raise RuntimeError("Diagnostic GLB was not written")

        report.update(
            {
                "status": "BLENDER_DIAGNOSTIC_SCENE_BUILT",
                "mesh": {
                    "vertices": int(mesh_data.vertices.shape[0]),
                    "quad_faces": int(mesh_data.faces.shape[0]),
                    "source_rows": mesh_data.source_rows,
                    "source_cols": mesh_data.source_cols,
                    "mesh_rows": mesh_data.mesh_rows,
                    "mesh_cols": mesh_data.mesh_cols,
                    "observed_vertex_fraction": float(mesh_data.observed.mean()),
                    "generated_vertex_fraction": float(mesh_data.generated.mean()),
                    "height_minimum": mesh_data.minimum_height,
                    "height_maximum": mesh_data.maximum_height,
                    "horizontal_size": mesh_data.horizontal_size,
                    "vertical_scale": mesh_data.vertical_scale,
                },
                "collections": sorted(collections),
                "proof_render_present": True,
                "blend_present": True,
                "diagnostic_glb_present": bool(glb_output and glb_output.is_file()),
                "blender_version": bpy.app.version_string,
            }
        )
        report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("IMAGE_WORLD_BLENDER_DIAGNOSTIC_SCENE_BUILT")
        return 0
    except Exception as exc:
        report["errors"] = [f"{type(exc).__name__}: {exc}", traceback.format_exc()]
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(report["errors"][0], file=sys.stderr)
        return 2


def _load_required(path: Path, dtype) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.asarray(np.load(path, allow_pickle=False), dtype=dtype)


def _reset_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.name != "Collection":
            bpy.data.collections.remove(collection)
    root = bpy.context.scene.collection
    default = bpy.data.collections.get("Collection")
    if default is not None:
        root.children.unlink(default)
        bpy.data.collections.remove(default)


def _create_collections() -> dict[str, bpy.types.Collection]:
    names = {
        "source": "IMAGE_WORLD_SOURCE",
        "terrain": "IMAGE_WORLD_TERRAIN",
        "residuals": "IMAGE_WORLD_RESIDUALS",
        "debug": "IMAGE_WORLD_DEBUG",
    }
    result = {}
    for key, name in names.items():
        collection = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(collection)
        result[key] = collection
    return result


def _material(name: str, base_color: tuple[float, float, float, float], roughness: float) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = base_color
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    principled.inputs["Base Color"].default_value = base_color
    principled.inputs["Roughness"].default_value = roughness
    return material


def _create_terrain_object(mesh_data, collection: bpy.types.Collection) -> bpy.types.Object:
    mesh = bpy.data.meshes.new("Lighthouse_Surface_Diagnostic_Mesh")
    mesh.from_pydata(mesh_data.vertices.tolist(), [], mesh_data.faces.tolist())
    mesh.update(calc_edges=True)
    terrain = bpy.data.objects.new("Lighthouse_Surface_Diagnostic", mesh)
    collection.objects.link(terrain)

    observed_material = _material("Observed_Surface", (0.08, 0.32, 0.12, 1.0), 0.82)
    generated_material = _material("Generated_Surface", (0.38, 0.09, 0.42, 1.0), 0.72)
    uncertain_material = _material("Low_Confidence_Surface", (0.55, 0.24, 0.04, 1.0), 0.70)
    mesh.materials.append(observed_material)
    mesh.materials.append(generated_material)
    mesh.materials.append(uncertain_material)

    confidence_attribute = mesh.attributes.new("source_confidence", "FLOAT", "POINT")
    observed_attribute = mesh.attributes.new("source_observed", "BOOLEAN", "POINT")
    generated_attribute = mesh.attributes.new("source_generated", "BOOLEAN", "POINT")
    for index, value in enumerate(mesh_data.confidence):
        confidence_attribute.data[index].value = float(value)
        observed_attribute.data[index].value = bool(mesh_data.observed[index])
        generated_attribute.data[index].value = bool(mesh_data.generated[index])

    for polygon in mesh.polygons:
        indices = np.asarray(polygon.vertices, dtype=np.int64)
        observed_fraction = float(mesh_data.observed[indices].mean())
        confidence = float(mesh_data.confidence[indices].mean())
        if confidence < 0.15:
            polygon.material_index = 2
        elif observed_fraction >= 0.5:
            polygon.material_index = 0
        else:
            polygon.material_index = 1

    terrain["image_world_classification"] = "DIAGNOSTIC_SURFACE_NOT_TERRAIN_PROOF"
    terrain["promotion_allowed"] = False
    terrain["observed_vertex_fraction"] = float(mesh_data.observed.mean())
    terrain["generated_vertex_fraction"] = float(mesh_data.generated.mean())
    return terrain


def _create_landmark_socket(mesh_data, collection: bpy.types.Collection) -> None:
    socket = bpy.data.objects.new("LANDMARK_SOCKET_LighthouseFortress", None)
    socket.empty_display_type = "CIRCLE"
    socket.empty_display_size = mesh_data.horizontal_size * 0.035
    socket.location = (
        mesh_data.horizontal_size * 0.20,
        -mesh_data.horizontal_size * 0.14,
        mesh_data.vertical_scale * 0.62,
    )
    socket["asset_required"] = "fantasy_lighthouse_fortress"
    socket["placement_status"] = "UNVERIFIED_PLACEHOLDER"
    collection.objects.link(socket)


def _create_source_camera_debug(collection: bpy.types.Collection) -> None:
    marker = bpy.data.objects.new("SOURCE_CAMERA_RECOVERY_PENDING", None)
    marker.empty_display_type = "PLAIN_AXES"
    marker.empty_display_size = 25.0
    marker["camera_recovery_proven"] = False
    collection.objects.link(marker)


def _create_proof_camera(mesh_data, collection: bpy.types.Collection) -> bpy.types.Object:
    camera_data = bpy.data.cameras.new("ImageWorld_Proof_Camera")
    camera = bpy.data.objects.new("ImageWorld_Proof_Camera", camera_data)
    collection.objects.link(camera)
    distance = mesh_data.horizontal_size * 1.05
    camera.location = (distance * 0.80, -distance * 0.90, distance * 0.68)
    camera.data.lens = 52.0
    camera.data.sensor_width = 36.0
    _look_at(camera, (0.0, 0.0, mesh_data.vertical_scale * 0.20))
    return camera


def _create_lighting(collection: bpy.types.Collection) -> None:
    sun_data = bpy.data.lights.new("ImageWorld_Key", "SUN")
    sun_data.energy = 3.0
    sun_data.angle = math.radians(18.0)
    sun = bpy.data.objects.new("ImageWorld_Key", sun_data)
    sun.rotation_euler = (math.radians(28.0), math.radians(-18.0), math.radians(-35.0))
    collection.objects.link(sun)

    area_data = bpy.data.lights.new("ImageWorld_Fill", "AREA")
    area_data.energy = 950.0
    area_data.shape = "DISK"
    area_data.size = 500.0
    area = bpy.data.objects.new("ImageWorld_Fill", area_data)
    area.location = (-350.0, 220.0, 420.0)
    _look_at(area, (0.0, 0.0, 80.0))
    collection.objects.link(area)


def _configure_world() -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("ImageWorld_Diagnostic_World")
    bpy.context.scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.025, 0.035, 0.055, 1.0)
    background.inputs["Strength"].default_value = 0.28


def _configure_render(camera: bpy.types.Object, path: Path, render_size: int) -> None:
    scene = bpy.context.scene
    scene.camera = camera
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = render_size
    scene.render.resolution_y = render_size
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(path)
    scene.render.film_transparent = False
    scene.render.use_file_extension = True


def _look_at(obj: bpy.types.Object, target: tuple[float, float, float]) -> None:
    direction = np.asarray(target, dtype=np.float64) - np.asarray(obj.location, dtype=np.float64)
    obj.rotation_euler = direction_to_track_quaternion(direction)


def direction_to_track_quaternion(direction: np.ndarray):
    from mathutils import Vector

    vector = Vector(tuple(float(value) for value in direction))
    if vector.length <= 1e-8:
        raise ValueError("look-at direction is zero")
    return vector.to_track_quat("-Z", "Y").to_euler()


def _select_only(obj: bpy.types.Object) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


if __name__ == "__main__":
    sys.exit(main())
