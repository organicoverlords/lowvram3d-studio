from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import bpy
from mathutils import Vector


def argv_after_double_dash() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path: str) -> list[bpy.types.Object]:
    suffix = Path(path).suffix.lower()
    if suffix in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    elif suffix == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif suffix == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    elif suffix == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif suffix == ".ply":
        bpy.ops.wm.ply_import(filepath=path)
    else:
        raise ValueError(f"Unsupported mesh format: {suffix}")
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def export_glb(path: str, selected_only: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=selected_only,
        export_apply=True,
        export_materials="EXPORT",
        export_animations=True,
        # Keep these explicit: a rigged export must carry the armature,
        # skinning modifiers, and vertex-group weights rather than relying on
        # exporter defaults that have changed between Blender releases.
        export_skins=True,
    )


def export_fbx(path: str, selected_only: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=selected_only,
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_ALL",
        add_leaf_bones=False,
        bake_anim=True,
        path_mode="COPY",
        embed_textures=True,
    )


def mesh_objects() -> list[bpy.types.Object]:
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def select_only(objects: Iterable[bpy.types.Object]) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    objects = list(objects)
    for obj in objects:
        obj.select_set(True)
    if objects:
        bpy.context.view_layer.objects.active = objects[0]


def join_meshes(objects: list[bpy.types.Object]) -> bpy.types.Object:
    if not objects:
        raise RuntimeError("No mesh objects")
    if len(objects) == 1:
        return objects[0]
    select_only(objects)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def apply_transforms(objects: Iterable[bpy.types.Object]) -> None:
    for obj in objects:
        select_only([obj])
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)


def world_bounds(objects: Iterable[bpy.types.Object]) -> tuple[Vector, Vector]:
    corners: list[Vector] = []
    for obj in objects:
        corners.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    if not corners:
        return Vector((0, 0, 0)), Vector((0, 0, 0))
    minimum = Vector((min(v.x for v in corners), min(v.y for v in corners), min(v.z for v in corners)))
    maximum = Vector((max(v.x for v in corners), max(v.y for v in corners), max(v.z for v in corners)))
    return minimum, maximum


def normalize_scene(objects: list[bpy.types.Object], target_size: float = 2.0) -> dict:
    minimum, maximum = world_bounds(objects)
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    longest = max(extent.x, extent.y, extent.z, 1e-6)
    scale = target_size / longest
    for obj in objects:
        obj.location -= center
        obj.scale *= scale
    apply_transforms(objects)
    minimum2, maximum2 = world_bounds(objects)
    return {
        "source_min": list(minimum),
        "source_max": list(maximum),
        "source_center": list(center),
        "normalized_min": list(minimum2),
        "normalized_max": list(maximum2),
        "scale": scale,
    }


def denormalize_scene(objects: list[bpy.types.Object], normalization: dict) -> None:
    """Invert :func:`normalize_scene` after camera-space processing.

    normalize_scene applies a uniform scale and translation to mesh data and
    resets object transforms. Reversing those operations on each vertex keeps
    UVs, vertex colors and normals intact while restoring game-world scale.
    """
    scale = float(normalization.get("scale", 1.0))
    if abs(scale) < 1e-12:
        raise ValueError("Cannot denormalize a scene with a zero scale")
    center_values = normalization.get("source_center")
    if center_values is None:
        minimum = Vector(normalization.get("source_min", (0.0, 0.0, 0.0)))
        maximum = Vector(normalization.get("source_max", (0.0, 0.0, 0.0)))
        center = (minimum + maximum) * 0.5
    else:
        center = Vector(center_values)
    inverse = 1.0 / scale
    for obj in objects:
        for vertex in obj.data.vertices:
            vertex.co = vertex.co * inverse + center
        obj.data.update()


def mesh_stats(objects: Iterable[bpy.types.Object]) -> dict:
    objects = list(objects)
    vertices = sum(len(obj.data.vertices) for obj in objects)
    faces = sum(len(obj.data.polygons) for obj in objects)
    edges = sum(len(obj.data.edges) for obj in objects)
    uv_layers = sum(len(obj.data.uv_layers) for obj in objects)
    materials = sum(len(obj.data.materials) for obj in objects)
    armatures = len([obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"])
    actions = len(bpy.data.actions)
    minimum, maximum = world_bounds(objects)
    finite = all(math.isfinite(value) for value in (*minimum, *maximum))
    return {
        "objects": len(objects),
        "vertices": vertices,
        "faces": faces,
        "edges": edges,
        "uv_layers": uv_layers,
        "materials": materials,
        "armatures": armatures,
        "actions": actions,
        "bounds_min": list(minimum),
        "bounds_max": list(maximum),
        "finite_bounds": finite,
    }


def save_json(path: str, data: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2), encoding="utf-8")


def look_at(obj: bpy.types.Object, point: Vector) -> None:
    direction = point - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def create_camera(name: str, location: tuple[float, float, float], ortho_scale: float = 2.6) -> bpy.types.Object:
    camera_data = bpy.data.cameras.new(name)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ortho_scale
    camera = bpy.data.objects.new(name, camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = location
    look_at(camera, Vector((0, 0, 0)))
    return camera


def enable_cycles_gpu(samples: int = 32) -> str:
    """Point Cycles at the GPU and cap bake samples.

    Two defaults make CPU baking pathologically slow: the device is CPU, and Cycles
    inherits the scene default of 4096 samples. Selected-to-active bakes converge at
    16-32, and data passes (normal, diffuse colour) are effectively deterministic, so
    4096 buys nothing but hours. Falls back to CPU when no GPU backend is usable.
    """
    import os

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(os.environ.get("LOWVRAM3D_BAKE_SAMPLES", str(samples)))
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.use_denoising = False
    scene.cycles.max_bounces = 2
    scene.cycles.device = "CPU"

    if os.environ.get("LOWVRAM3D_CYCLES_CPU") == "1":
        return "CPU"
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
    except (KeyError, AttributeError):
        return "CPU"
    for backend in ("OPTIX", "CUDA", "HIP", "ONEAPI"):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
        except Exception:
            continue
        accelerators = [d for d in prefs.devices if d.type == backend]
        if not accelerators:
            continue
        for device in prefs.devices:
            device.use = device.type == backend
        scene.cycles.device = "GPU"
        return backend
    return "CPU"


def preferred_render_engine() -> str:
    """Pick a real-time engine that exists in this Blender build.

    The EEVEE identifier is not stable across releases: 4.2 exposes BLENDER_EEVEE_NEXT,
    while 4.1 and 5.x expose BLENDER_EEVEE. Hard-coding either one makes every render
    stage fail on the other, so resolve it against the enum the build actually offers.
    """
    available = {item.identifier for item in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        if candidate in available:
            return candidate
    return next(iter(available))


def configure_render(width: int, height: int, transparent: bool = True) -> None:
    scene = bpy.context.scene
    scene.render.engine = preferred_render_engine()
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = transparent
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except Exception:
        pass


def triangle_count(obj: bpy.types.Object) -> int:
    obj.data.calc_loop_triangles()
    return len(obj.data.loop_triangles)


def object_dimensions(obj: bpy.types.Object) -> tuple[float, float, float]:
    values = sorted((abs(float(obj.dimensions.x)), abs(float(obj.dimensions.y)), abs(float(obj.dimensions.z))))
    return values[0], values[1], values[2]


def object_center(obj: bpy.types.Object) -> Vector:
    minimum, maximum = world_bounds([obj])
    return (minimum + maximum) * 0.5


def shade_smooth(objects: Iterable[bpy.types.Object]) -> int:
    """Share vertex normals so the mesh can survive a glTF round trip.

    glTF emits one vertex per face wherever normals are not shared, so a flat-shaded
    mesh (what marching-cubes generators produce) re-imports as disconnected triangle
    soup no matter how clean it was on export. Smooth normals also make a far better
    high-poly bake source: baking from flat shading yields faceted normal maps.
    """
    smoothed = 0
    for obj in objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        for polygon in mesh.polygons:
            if not polygon.use_smooth:
                polygon.use_smooth = True
                smoothed += 1
        mesh.update()
    return smoothed


def weld_vertices(objects: Iterable[bpy.types.Object], distance: float = 1e-5) -> dict[str, int]:
    """Merge coincident vertices so imported geometry regains connected topology.

    glTF stores one vertex per face-corner wherever normals or UVs are split, so a
    flat-shaded mesh round-tripped through GLB re-imports as fully disconnected
    triangles: every edge non-manifold, one loose component per face. Loose-part
    splitting, decimation, unwrapping, baking and skinning all degrade badly on that
    input, so the weld happens once, at ingest, before anything else reads the mesh.
    """
    import bmesh

    before = 0
    after = 0
    for obj in objects:
        if obj.type != "MESH":
            continue
        bm = bmesh.new()
        try:
            bm.from_mesh(obj.data)
            before += len(bm.verts)
            bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=distance)
            after += len(bm.verts)
            bm.to_mesh(obj.data)
        finally:
            bm.free()
        obj.data.update()
    return {"vertices_before": before, "vertices_after": after, "vertices_merged": before - after}


def non_manifold_edge_count(obj: bpy.types.Object) -> int:
    import bmesh

    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        return sum(1 for edge in bm.edges if not edge.is_manifold)
    finally:
        bm.free()


def welded_topology_stats(objects: Iterable[bpy.types.Object], distance: float = 4e-4) -> dict[str, int]:
    """Measure connected topology after reconstructing glTF seam-split vertices on a copy."""
    import bmesh

    faces = 0
    components = 0
    boundary_edges = 0
    non_manifold_edges = 0
    for obj in objects:
        if obj.type != "MESH":
            continue
        bm = bmesh.new()
        try:
            bm.from_mesh(obj.data)
            bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=distance)
            bmesh.ops.triangulate(bm, faces=list(bm.faces))
            bm.faces.ensure_lookup_table()
            seen: set[int] = set()
            for face in bm.faces:
                if face.index in seen:
                    continue
                components += 1
                stack = [face]
                seen.add(face.index)
                while stack:
                    current = stack.pop()
                    for edge in current.edges:
                        for neighbour in edge.link_faces:
                            if neighbour.index not in seen:
                                seen.add(neighbour.index)
                                stack.append(neighbour)
            faces += len(bm.faces)
            boundary_edges += sum(1 for edge in bm.edges if len(edge.link_faces) == 1)
            non_manifold_edges += sum(1 for edge in bm.edges if not edge.is_manifold)
        finally:
            bm.free()
    return {
        "faces": faces,
        "components": components,
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
    }


def material_texture_paths(objects: Iterable[bpy.types.Object]) -> list[str]:
    paths: set[str] = set()
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if not material or not material.use_nodes:
                continue
            for node in material.node_tree.nodes:
                if node.type != "TEX_IMAGE" or not node.image:
                    continue
                image = node.image
                if image.packed_file:
                    paths.add(f"packed:{image.name}")
                elif image.filepath:
                    paths.add(str(Path(bpy.path.abspath(image.filepath))))
    return sorted(paths)


def extended_mesh_stats(objects: Iterable[bpy.types.Object]) -> dict:
    items = list(objects)
    stats = mesh_stats(items)
    stats.update(
        {
            "triangles": sum(triangle_count(obj) for obj in items),
            "non_manifold_edges": sum(non_manifold_edge_count(obj) for obj in items),
            "texture_paths": material_texture_paths(items),
            "object_names": [obj.name for obj in items],
        }
    )
    return stats
