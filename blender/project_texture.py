from __future__ import annotations

import argparse
import math
from pathlib import Path

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector

from common import (
    enable_cycles_gpu,
    argv_after_double_dash,
    configure_render,
    create_camera,
    denormalize_scene,
    export_glb,
    extended_mesh_stats,
    import_mesh,
    normalize_scene,
    reset_scene,
    save_json,
    select_only,
)

VIEWS = {
    "front": (0.0, -3.0, 0.0),
    "right": (3.0, 0.0, 0.0),
    "back": (0.0, 3.0, 0.0),
    "left": (-3.0, 0.0, 0.0),
    "top": (0.0, 0.0, 3.0),
    "bottom": (0.0, 0.0, -3.0),
}


def load_image(path: Path) -> bpy.types.Image:
    return bpy.data.images.load(str(path), check_existing=True)


def sample(image: bpy.types.Image, u: float, v: float) -> tuple[float, float, float, float]:
    width, height = image.size
    if width <= 0 or height <= 0:
        return (0.5, 0.5, 0.5, 0.0)
    x = min(width - 1, max(0, int(u * (width - 1))))
    y = min(height - 1, max(0, int(v * (height - 1))))
    index = (y * width + x) * 4
    pixels = image.pixels
    return tuple(float(pixels[index + channel]) for channel in range(4))


def camera_score(camera: bpy.types.Object, position: Vector, normal: Vector) -> float:
    direction = (camera.location - position).normalized()
    return max(0.0, float(normal.dot(direction)))


def project_vertex(scene: bpy.types.Scene, camera: bpy.types.Object, world_position: Vector) -> tuple[float, float] | None:
    ndc = world_to_camera_view(scene, camera, world_position)
    if ndc.z < 0 or not (0.0 <= ndc.x <= 1.0 and 0.0 <= ndc.y <= 1.0):
        return None
    return float(ndc.x), float(ndc.y)


def border_weight(u: float, v: float) -> float:
    distance = max(0.0, min(u, v, 1.0 - u, 1.0 - v))
    return max(0.05, min(1.0, distance / 0.08))


def visible_from_camera(
    scene: bpy.types.Scene,
    depsgraph,
    camera: bpy.types.Object,
    surface_point: Vector,
    expected: bpy.types.Object,
) -> bool:
    vector = surface_point - camera.location
    distance = vector.length
    if distance <= 1e-8:
        return False
    direction = vector.normalized()
    try:
        hit, location, _normal, _index, hit_object, _matrix = scene.ray_cast(
            depsgraph,
            camera.location,
            direction,
            distance=distance + 0.01,
        )
    except TypeError:
        hit, location, _normal, _index, hit_object, _matrix = scene.ray_cast(
            depsgraph,
            camera.location,
            direction,
        )
    if not hit or hit_object is None:
        return False
    original = getattr(hit_object, "original", hit_object)
    expected_original = getattr(expected, "original", expected)
    return original == expected_original and (location - surface_point).length <= 0.04


def weighted_color(samples: list[tuple[tuple[float, float, float, float], float]]) -> tuple[float, float, float, float]:
    if not samples:
        return (0.5, 0.5, 0.5, 1.0)
    total = sum(weight for _color, weight in samples)
    if total <= 1e-12:
        return samples[0][0]
    return tuple(sum(color[index] * weight for color, weight in samples) / total for index in range(4))


def create_color_attribute(
    obj: bpy.types.Object,
    images: dict[str, bpy.types.Image],
    source_front: bpy.types.Image | None,
) -> tuple[str, dict[str, int]]:
    mesh = obj.data
    name = "ProjectedColor"
    existing = mesh.color_attributes.get(name)
    if existing:
        mesh.color_attributes.remove(existing)
    colors = mesh.color_attributes.new(name=name, type="FLOAT_COLOR", domain="CORNER")
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    cameras = {view: bpy.data.objects[f"Camera_{view}"] for view in VIEWS}
    stats = {"polygons": len(mesh.polygons), "visible_samples": 0, "fallback_polygons": 0}

    for polygon in mesh.polygons:
        center = obj.matrix_world @ polygon.center
        normal = (obj.matrix_world.to_3x3() @ polygon.normal).normalized()
        ranked = sorted(
            cameras.items(),
            key=lambda item: camera_score(item[1], center, normal),
            reverse=True,
        )
        visible = []
        for view_name, camera in ranked[:4]:
            score = camera_score(camera, center, normal)
            if score <= 0.01:
                continue
            if visible_from_camera(scene, depsgraph, camera, center, obj):
                visible.append((view_name, camera, score))
            if len(visible) >= 2:
                break
        if not visible:
            stats["fallback_polygons"] += 1
            visible = [(view_name, camera, camera_score(camera, center, normal)) for view_name, camera in ranked[:2]]

        for loop_index in polygon.loop_indices:
            vertex = mesh.vertices[mesh.loops[loop_index].vertex_index]
            world_position = obj.matrix_world @ vertex.co
            candidates: list[tuple[tuple[float, float, float, float], float]] = []
            for view_name, camera, score in visible:
                uv = project_vertex(scene, camera, world_position)
                if uv is None:
                    continue
                generated = images.get(view_name)
                source = source_front if view_name == "front" and source_front else None
                if source is not None:
                    color = sample(source, uv[0], uv[1])
                    if color[3] > 0.04:
                        weight = (max(score, 0.01) ** 2) * border_weight(*uv) * color[3] * 3.0
                        candidates.append((color, weight))
                if generated is not None:
                    color = sample(generated, uv[0], uv[1])
                    if color[3] > 0.04:
                        weight = (max(score, 0.01) ** 2) * border_weight(*uv) * color[3]
                        candidates.append((color, weight))
            chosen = weighted_color(candidates)
            if candidates:
                stats["visible_samples"] += 1
            colors.data[loop_index].color = chosen
    return name, stats


def ensure_shared_uv(objects: list[bpy.types.Object]) -> None:
    missing = [obj for obj in objects if not obj.data.uv_layers]
    if not missing:
        return
    for obj in objects:
        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name="UVMap")
    select_only(objects)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.02, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode="OBJECT")


def make_projected_material(image: bpy.types.Image) -> tuple[bpy.types.Material, bpy.types.Node, bpy.types.Node]:
    material = bpy.data.materials.new("ProjectedPBR")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    vertex_color = nodes.new("ShaderNodeVertexColor")
    vertex_color.layer_name = "ProjectedColor"
    target = nodes.new("ShaderNodeTexImage")
    target.image = image
    target.select = True
    nodes.active = target
    emission = nodes.new("ShaderNodeEmission")
    links.new(vertex_color.outputs["Color"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material, principled, output


def bake_shared_basecolor(
    objects: list[bpy.types.Object],
    texture_path: Path,
    size: int,
    padding: int,
) -> bpy.types.Image:
    scene = bpy.context.scene
    enable_cycles_gpu()
    scene.render.bake.use_selected_to_active = False
    scene.render.bake.margin = padding
    image = bpy.data.images.new("BaseColorBake", width=size, height=size, alpha=True)
    image.generated_color = (0.0, 0.0, 0.0, 0.0)
    image.filepath_raw = str(texture_path)
    image.file_format = "PNG"
    material, principled, output = make_projected_material(image)
    first = True
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
        target = next(node for node in material.node_tree.nodes if node.type == "TEX_IMAGE")
        target.select = True
        material.node_tree.nodes.active = target
        select_only([obj])
        bpy.context.view_layer.objects.active = obj
        scene.render.bake.use_clear = first
        bpy.ops.object.bake(type="EMIT")
        first = False
    image.save()

    links = material.node_tree.links
    if output.inputs["Surface"].is_linked:
        links.remove(output.inputs["Surface"].links[0])
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    texture = material.node_tree.nodes.new("ShaderNodeTexImage")
    texture.image = image
    links.new(texture.outputs["Color"], principled.inputs["Base Color"])
    return image


def image_metrics(image: bpy.types.Image, max_samples: int = 150_000) -> dict[str, float | int]:
    pixel_count = int(image.size[0] * image.size[1])
    stride = max(1, pixel_count // max_samples)
    samples = 0
    covered = 0
    luminance_sum = 0.0
    luminance_sq = 0.0
    for pixel_index in range(0, pixel_count, stride):
        offset = pixel_index * 4
        r, g, b, a = (float(image.pixels[offset + channel]) for channel in range(4))
        samples += 1
        if a > 0.02:
            covered += 1
        luminance = (r + g + b) / 3.0
        luminance_sum += luminance
        luminance_sq += luminance * luminance
    mean = luminance_sum / max(samples, 1)
    return {
        "samples": samples,
        "coverage": covered / max(samples, 1),
        "luminance_mean": mean,
        "luminance_variance": max(0.0, luminance_sq / max(samples, 1) - mean * mean),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--views-dir", required=True)
    parser.add_argument("--source-image", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--texture", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--padding-px", type=int, default=16)
    args = parser.parse_args(argv_after_double_dash())

    reset_scene()
    objects = import_mesh(args.input)
    if not objects:
        raise RuntimeError("No mesh imported")
    ensure_shared_uv(objects)
    normalization = normalize_scene(objects)
    configure_render(512, 512)
    for name, location in VIEWS.items():
        create_camera(f"Camera_{name}", location)

    view_dir = Path(args.views_dir)
    images: dict[str, bpy.types.Image] = {}
    for name in VIEWS:
        candidates = [view_dir / f"{name}.png", view_dir / f"{name}_color.png"]
        match = next((path for path in candidates if path.is_file()), None)
        if match:
            images[name] = load_image(match)
    source = load_image(Path(args.source_image)) if args.source_image and Path(args.source_image).is_file() else None
    if not images and source is None:
        raise RuntimeError("No projection images were supplied")

    projection_stats = {}
    for obj in objects:
        _attribute, stats = create_color_attribute(obj, images, source)
        projection_stats[obj.name] = stats
    texture_path = Path(args.texture)
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    image = bake_shared_basecolor(objects, texture_path, args.size, args.padding_px)
    metrics = image_metrics(image)
    if metrics["coverage"] < 0.005:
        raise RuntimeError("Projected base-colour atlas has insufficient coverage")

    denormalize_scene(objects, normalization)
    select_only(objects)
    export_glb(args.output, selected_only=True)
    if args.report:
        save_json(
            args.report,
            {
                "success": True,
                "backend": "visibility_weighted_multiview_projection_cpu",
                "source_authority": bool(source),
                "views": sorted(images),
                "objects": projection_stats,
                "texture": str(texture_path),
                "texture_size": args.size,
                "padding_px": args.padding_px,
                "image_metrics": metrics,
                "normalization": normalization,
                "output_stats": extended_mesh_stats(objects),
            },
        )


if __name__ == "__main__":
    main()
