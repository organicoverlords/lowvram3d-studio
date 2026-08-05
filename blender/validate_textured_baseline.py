"""Independent proof that a textured GLB is real, run in a FRESH Blender process.

Deliberately imports nothing from the producing run: it reopens the exported file from disk and
re-derives every claim. A structurally perfect GLB carrying an all-black or constant atlas passes
naive checks, so the texture is inspected as pixels -- finite, non-constant, not fully transparent,
and carrying genuine variation -- not merely as a present image datablock.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path  # noqa: F401  (json is used for the transform dump below)

import bpy
import numpy as np

from common import argv_after_double_dash, import_mesh, reset_scene, save_json


def texture_statistics(image: bpy.types.Image) -> dict:
    pixels = np.array(image.pixels[:], dtype=np.float64)
    if pixels.size == 0:
        return {"readable": False}
    rgba = pixels.reshape(-1, 4)
    rgb, alpha = rgba[:, :3], rgba[:, 3]
    # Quantise to 8-bit to count distinct colours the way an exported PNG would store them.
    quantised = np.unique(np.round(rgb * 255).astype(np.int16), axis=0)
    return {
        "readable": True,
        "resolution": [int(image.size[0]), int(image.size[1])],
        "packed": bool(image.packed_file is not None),
        "source": image.source,
        "all_finite": bool(np.isfinite(pixels).all()),
        "nan_count": int(np.isnan(pixels).sum()),
        "rgb_min": round(float(rgb.min()), 6),
        "rgb_max": round(float(rgb.max()), 6),
        "rgb_mean": round(float(rgb.mean()), 6),
        "rgb_std": round(float(rgb.std()), 6),
        "unique_colours_8bit": int(len(quantised)),
        "alpha_min": round(float(alpha.min()), 6),
        "alpha_mean": round(float(alpha.mean()), 6),
        "nonblack_pixel_percent": round(float((rgb.max(axis=1) > 0.02).mean() * 100), 4),
    }


def drives_base_colour(node: bpy.types.Node, max_depth: int = 8) -> bool:
    """Whether this image node reaches a Principled 'Base Color' input, directly or indirectly.

    A direct link is not the only valid wiring: the glTF importer reconstructs an exported
    base-colour texture as TEX_IMAGE -> Mix -> Principled 'Base Color' whenever the mesh also
    carries a colour attribute, so requiring adjacency would reject a perfectly textured file.
    """
    seen: set[str] = set()
    frontier = [(node, 0)]
    while frontier:
        current, depth = frontier.pop()
        if depth > max_depth or current.name in seen:
            continue
        seen.add(current.name)
        for output_socket in current.outputs:
            for link in output_socket.links:
                target, socket = link.to_node, link.to_socket
                if target.type == "BSDF_PRINCIPLED" and socket.name == "Base Color":
                    return True
                frontier.append((target, depth + 1))
    return False


def scene_bounds(objects: list) -> tuple[np.ndarray, np.ndarray]:
    from mathutils import Vector

    lo = np.array([1e30, 1e30, 1e30])
    hi = -lo
    for obj in objects:
        for corner in obj.bound_box:
            world = np.array(obj.matrix_world @ Vector(corner))
            lo = np.minimum(lo, world)
            hi = np.maximum(hi, world)
    return lo, hi


def transform_diagnostics(objects: list) -> dict:
    """Report whether the asset stands upright, and in which space any rotation lives.

    A preview can look sideways for two very different reasons: the camera is wrong, or the
    exported object carries a non-identity rotation. Distinguishing them requires comparing the
    object-space extents (obj.dimensions, which ignore object rotation) against the world-space
    bounding box, so both are reported rather than inferred from one another.
    """
    lo, hi = scene_bounds(objects)
    world_dimensions = hi - lo
    world_up_axis = int(np.argmax(world_dimensions))

    per_object = []
    for obj in objects:
        local = np.array(obj.dimensions)
        rotation = np.degrees(np.array(obj.rotation_euler))
        per_object.append({
            "object": obj.name,
            "rotation_mode": obj.rotation_mode,
            "rotation_euler_degrees": [round(float(v), 4) for v in rotation],
            "scale": [round(float(v), 6) for v in obj.scale],
            "location": [round(float(v), 6) for v in obj.location],
            "dimensions_object_space": [round(float(v), 6) for v in local],
            "object_space_up_axis": "XYZ"[int(np.argmax(local))],
            "has_identity_rotation": bool(np.allclose(rotation, 0.0, atol=1e-3)),
            "has_uniform_scale": bool(np.allclose(np.array(obj.scale), obj.scale[0], atol=1e-6)),
            "polygons": len(obj.data.polygons),
            "triangles": sum(max(len(p.vertices) - 2, 0) for p in obj.data.polygons),
        })

    return {
        "world_bounding_box_min": [round(float(v), 6) for v in lo],
        "world_bounding_box_max": [round(float(v), 6) for v in hi],
        "world_dimensions": [round(float(v), 6) for v in world_dimensions],
        "detected_up_axis": "XYZ"[world_up_axis],
        "detected_up_axis_index": world_up_axis,
        # Blender's world convention is Z-up. Anything else means the delivered transform, not the
        # camera, is what makes the asset look like it is lying down.
        "upright_in_blender_world": world_up_axis == 2,
        "all_objects_identity_rotation": all(o["has_identity_rotation"] for o in per_object),
        "objects": per_object,
    }


def render_preview(output: Path, objects: list, azimuth_degrees: float, samples: int = 32) -> None:
    scene = bpy.context.scene
    # The EEVEE identifier moved between releases (BLENDER_EEVEE -> BLENDER_EEVEE_NEXT -> back);
    # pick whichever this build actually offers rather than pinning a version-specific name.
    available = scene.render.bl_rna.properties["engine"].enum_items.keys()
    for candidate in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE", "BLENDER_WORKBENCH"):
        if candidate in available:
            scene.render.engine = candidate
            break
    try:
        scene.eevee.taa_render_samples = samples
    except AttributeError:
        pass
    scene.render.resolution_x = 640
    scene.render.resolution_y = 640
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = "PNG"

    from mathutils import Matrix, Vector

    lo, hi = scene_bounds(objects)
    centre = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) / 2.0
    dimensions = hi - lo

    # Orient from the bounding box rather than assuming a convention. Meshes reach this stage from
    # marching cubes via a glTF round trip, and the up axis is not reliably Z; picking the longest
    # extent as up and viewing along the shallowest keeps a standing figure upright and face-on.
    up_axis = int(np.argmax(dimensions))
    horizontal = [axis for axis in range(3) if axis != up_axis]
    # The shallower horizontal axis is the subject's depth, so azimuth 0 looks at the broad side.
    depth_axis = min(horizontal, key=lambda axis: dimensions[axis])
    side_axis = next(axis for axis in horizontal if axis != depth_axis)

    up_hint = np.zeros(3)
    up_hint[up_axis] = 1.0
    forward = np.zeros(3)
    forward[depth_axis] = -1.0
    sideways = np.zeros(3)
    sideways[side_axis] = 1.0

    angle = np.radians(azimuth_degrees)
    offset = forward * np.cos(angle) + sideways * np.sin(angle)
    position = centre + offset * radius * 3.0
    back = Vector((position - centre).tolist()).normalized()
    right = Vector(up_hint.tolist()).cross(back).normalized()
    up = back.cross(right).normalized()

    camera_data = bpy.data.cameras.new("PreviewCam")
    camera = bpy.data.objects.new("PreviewCam", camera_data)
    scene.collection.objects.link(camera)
    camera.matrix_world = Matrix((
        (right.x, up.x, back.x, position[0]),
        (right.y, up.y, back.y, position[1]),
        (right.z, up.z, back.z, position[2]),
        (0.0, 0.0, 0.0, 1.0),
    ))
    camera_data.lens = 50
    scene.camera = camera

    world = bpy.data.worlds.new("PreviewWorld")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 2.0
    scene.world = world

    key_data = bpy.data.lights.new("Key", type="AREA")
    key_data.energy = 400.0
    key_data.size = radius * 4.0
    key = bpy.data.objects.new("Key", key_data)
    key.location = (centre[0] + radius * 2, centre[1] - radius * 2.5, centre[2] + radius * 2)
    key.rotation_euler = (np.radians(55), 0.0, np.radians(40))
    scene.collection.objects.link(key)

    output.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--preview", default="", help="Front preview path; siblings are derived.")
    parser.add_argument("--preview-dir", default="",
                        help="Render preview_front/preview_three_quarter/preview_side here.")
    args = parser.parse_args(argv_after_double_dash())

    glb = Path(args.glb)
    if not glb.is_file() or glb.stat().st_size == 0:
        raise SystemExit(f"VALIDATION_FAILED: missing or empty GLB: {glb}")

    reset_scene()
    objects = import_mesh(str(glb))
    if not objects:
        raise SystemExit("VALIDATION_FAILED: fresh reimport produced no mesh objects")

    checks: dict[str, bool] = {"reimport_succeeded": True}
    meshes = []
    images: dict[str, dict] = {}

    for obj in objects:
        uv_layers = [layer.name for layer in obj.data.uv_layers]
        materials = [slot.name for slot in obj.data.materials if slot]
        image_nodes = []
        for material in obj.data.materials:
            if not material or not material.use_nodes:
                continue
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE" and node.image:
                    linked_to_base_colour = drives_base_colour(node)
                    image_nodes.append({
                        "image": node.image.name,
                        "linked_to_base_color": linked_to_base_colour,
                    })
                    images[node.image.name] = texture_statistics(node.image)
        meshes.append({
            "object": obj.name,
            "vertices": len(obj.data.vertices),
            "polygons": len(obj.data.polygons),
            "uv_layers": uv_layers,
            "materials": materials,
            "image_nodes": image_nodes,
        })

    checks["mesh_has_polygons"] = all(m["polygons"] > 0 for m in meshes)
    checks["mesh_has_uvs"] = all(len(m["uv_layers"]) > 0 for m in meshes)
    checks["material_present"] = all(len(m["materials"]) > 0 for m in meshes)
    checks["base_color_texture_linked"] = any(
        node["linked_to_base_color"] for m in meshes for node in m["image_nodes"]
    )
    checks["texture_packed"] = any(stat.get("packed") for stat in images.values())

    usable = [s for s in images.values() if s.get("readable")]
    checks["texture_readable"] = bool(usable)
    checks["texture_all_finite"] = all(s["all_finite"] for s in usable) if usable else False
    checks["texture_not_black"] = any(s["nonblack_pixel_percent"] > 5.0 for s in usable)
    checks["texture_not_constant"] = any(s["unique_colours_8bit"] > 256 for s in usable)
    checks["texture_not_transparent"] = any(s["alpha_mean"] > 0.1 for s in usable)
    checks["texture_has_variation"] = any(s["rgb_std"] > 0.01 for s in usable)

    transforms = transform_diagnostics(objects)

    previews: dict[str, str] = {}
    targets: list[tuple[str, float, Path]] = []
    if args.preview_dir:
        directory = Path(args.preview_dir)
        targets = [
            ("front", 0.0, directory / "preview_front.png"),
            ("three_quarter", 40.0, directory / "preview_three_quarter.png"),
            ("side", 90.0, directory / "preview_side.png"),
        ]
    elif args.preview:
        targets = [("front", 0.0, Path(args.preview))]

    for label, azimuth, path in targets:
        reset_scene()
        objects = import_mesh(str(glb))
        render_preview(path, objects, azimuth)
        if path.is_file() and path.stat().st_size > 0:
            previews[label] = str(path)
    if targets:
        checks["previews_rendered"] = len(previews) == len(targets)

    passed = all(checks.values())
    save_json(args.report, {
        "success": passed,
        "glb": str(glb),
        "glb_bytes": glb.stat().st_size,
        "checks": checks,
        "transforms": transforms,
        "meshes": meshes,
        "textures": images,
        "previews": previews,
    })
    print(json.dumps(transforms, indent=2), flush=True)

    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}", flush=True)
    if not passed:
        raise SystemExit(f"VALIDATION_FAILED: {[k for k, v in checks.items() if not v]}")
    print("VALIDATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
