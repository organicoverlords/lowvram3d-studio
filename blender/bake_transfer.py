from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import bpy

from common import (
    argv_after_double_dash,
    enable_cycles_gpu,
    export_glb,
    extended_mesh_stats,
    import_mesh,
    reset_scene,
    save_json,
    select_only,
    world_bounds,
)

MAP_SPECS = {
    "basecolor": {"color": (0.0, 0.0, 0.0, 0.0), "colorspace": "sRGB"},
    "normal": {"color": (0.5, 0.5, 1.0, 0.0), "colorspace": "Non-Color"},
    "roughness": {"color": (0.6, 0.6, 0.6, 1.0), "colorspace": "Non-Color"},
    "metallic": {"color": (0.0, 0.0, 0.0, 1.0), "colorspace": "Non-Color"},
    "ao": {"color": (1.0, 1.0, 1.0, 0.0), "colorspace": "Non-Color"},
}


def import_new(path: str, prefix: str) -> list[bpy.types.Object]:
    before = set(bpy.context.scene.objects)
    import_mesh(path)
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj not in before]
    for index, obj in enumerate(objects):
        obj.name = f"{prefix}_{index:03d}_{obj.name}"
    return objects


def new_image(name: str, path: Path, size: int) -> bpy.types.Image:
    spec = MAP_SPECS[name]
    image = bpy.data.images.new(
        name=f"LowVRAM_{name}", width=size, height=size, alpha=True, float_buffer=False,
    )
    image.generated_color = spec["color"]
    image.filepath_raw = str(path)
    image.file_format = "PNG"
    try:
        image.colorspace_settings.name = spec["colorspace"]
    except Exception:
        pass
    return image


def external_image(name: str, source: Path, destination: Path, size: int) -> bpy.types.Image:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    image = bpy.data.images.load(str(destination), check_existing=False)
    if tuple(image.size) != (size, size):
        image.scale(size, size)
    image.filepath_raw = str(destination)
    image.file_format = "PNG"
    try:
        image.colorspace_settings.name = MAP_SPECS[name]["colorspace"]
    except Exception:
        pass
    image.save()
    return image


def make_target_material(images: dict[str, bpy.types.Image]):
    material = bpy.data.materials.new("LowVRAM_GamePBR")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    principled = nodes.new("ShaderNodeBsdfPrincipled")
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    texture_nodes = {}
    for name, image in images.items():
        node = nodes.new("ShaderNodeTexImage")
        node.name = f"BAKE_{name}"
        node.label = name
        node.image = image
        texture_nodes[name] = node
    links.new(texture_nodes["basecolor"].outputs["Color"], principled.inputs["Base Color"])
    links.new(texture_nodes["roughness"].outputs["Color"], principled.inputs["Roughness"])
    links.new(texture_nodes["metallic"].outputs["Color"], principled.inputs["Metallic"])
    normal_node = nodes.new("ShaderNodeNormalMap")
    links.new(texture_nodes["normal"].outputs["Color"], normal_node.inputs["Color"])
    links.new(normal_node.outputs["Normal"], principled.inputs["Normal"])
    return material, texture_nodes


def assign_target_material(objects: list[bpy.types.Object], material) -> None:
    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)


def active_image(material, node) -> None:
    for candidate in material.node_tree.nodes:
        candidate.select = False
    node.select = True
    material.node_tree.nodes.active = node


def configure_cycles(padding: int) -> None:
    scene = bpy.context.scene
    backend = enable_cycles_gpu()
    print(f"cycles backend: {backend} samples={scene.cycles.samples}", flush=True)
    scene.render.bake.margin = padding
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.use_pass_color = True
    try:
        scene.render.bake.target = "IMAGE_TEXTURES"
    except Exception:
        pass


def clear_image(image: bpy.types.Image, color) -> None:
    image.generated_color = color
    pixels = list(color) * (image.size[0] * image.size[1])
    image.pixels.foreach_set(pixels)
    image.update()


def sample_image(image: bpy.types.Image, max_samples: int = 200_000) -> dict:
    pixels = image.pixels
    pixel_count = image.size[0] * image.size[1]
    stride = max(1, pixel_count // max_samples)
    covered = 0
    luminance_sum = 0.0
    luminance_sq = 0.0
    samples = 0
    for pixel_index in range(0, pixel_count, stride):
        offset = pixel_index * 4
        r, g, b, a = (float(pixels[offset + channel]) for channel in range(4))
        samples += 1
        if a > 0.01:
            covered += 1
        luminance = (r + g + b) / 3.0
        luminance_sum += luminance
        luminance_sq += luminance * luminance
    mean = luminance_sum / max(samples, 1)
    variance = max(0.0, luminance_sq / max(samples, 1) - mean * mean)
    return {
        "samples": samples,
        "coverage": covered / max(samples, 1),
        "luminance_mean": mean,
        "luminance_variance": variance,
    }


def bake_selected(
    high_objects: list[bpy.types.Object],
    low_objects: list[bpy.types.Object],
    material,
    image_node,
    bake_type: str,
    ray_distance: float,
) -> None:
    scene = bpy.context.scene
    scene.render.bake.use_selected_to_active = True
    scene.render.bake.cage_extrusion = ray_distance
    scene.render.bake.max_ray_distance = ray_distance * 2.0
    first = True
    for low in low_objects:
        active_image(material, image_node)
        select_only([*high_objects, low])
        bpy.context.view_layer.objects.active = low
        scene.render.bake.use_clear = first
        bpy.ops.object.bake(type=bake_type)
        first = False


def bake_low_only(low_objects, material, image_node, bake_type: str) -> None:
    scene = bpy.context.scene
    scene.render.bake.use_selected_to_active = False
    first = True
    for low in low_objects:
        active_image(material, image_node)
        select_only([low])
        bpy.context.view_layer.objects.active = low
        scene.render.bake.use_clear = first
        bpy.ops.object.bake(type=bake_type)
        first = False


def principled_input(material, names: tuple[str, ...]):
    if not material or not material.use_nodes:
        return None
    principled = next((node for node in material.node_tree.nodes if node.type == "BSDF_PRINCIPLED"), None)
    if not principled:
        return None
    for name in names:
        if name in principled.inputs:
            return principled.inputs[name]
    return None


def prepare_emission_materials(high_objects, input_names: tuple[str, ...], fallback: float) -> list[tuple]:
    states = []
    materials = {
        slot.material for obj in high_objects for slot in obj.material_slots
        if slot.material and slot.material.use_nodes
    }
    for material in materials:
        nodes = material.node_tree.nodes
        links = material.node_tree.links
        output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
        if output is None:
            output = nodes.new("ShaderNodeOutputMaterial")
        original = output.inputs["Surface"].links[0].from_socket if output.inputs["Surface"].is_linked else None
        if output.inputs["Surface"].is_linked:
            links.remove(output.inputs["Surface"].links[0])
        emission = nodes.new("ShaderNodeEmission")
        source = principled_input(material, input_names)
        if source and source.is_linked:
            links.new(source.links[0].from_socket, emission.inputs["Color"])
        elif source:
            value = source.default_value
            if hasattr(value, "__len__"):
                color = tuple(float(v) for v in value[:3]) + (1.0,)
            else:
                scalar = float(value)
                color = (scalar, scalar, scalar, 1.0)
            emission.inputs["Color"].default_value = color
        else:
            emission.inputs["Color"].default_value = (fallback, fallback, fallback, 1.0)
        links.new(emission.outputs["Emission"], output.inputs["Surface"])
        states.append((material, output, original, emission))
    return states


def restore_materials(states) -> None:
    for material, output, original, emission in states:
        links = material.node_tree.links
        if output.inputs["Surface"].is_linked:
            links.remove(output.inputs["Surface"].links[0])
        if original:
            links.new(original, output.inputs["Surface"])
        material.node_tree.nodes.remove(emission)


def bake_with_retry(high_objects, low_objects, material, node, image, bake_type, extent, clear_color):
    attempts = []
    best = None
    for fraction in (0.003, 0.01, 0.03, 0.07):
        clear_image(image, clear_color)
        distance = max(extent * fraction, 1e-5)
        try:
            bake_selected(high_objects, low_objects, material, node, bake_type, distance)
            metrics = sample_image(image)
            attempt = {"distance": distance, "metrics": metrics, "error": None}
            attempts.append(attempt)
            if best is None or metrics["coverage"] > best["metrics"]["coverage"]:
                image.save()
                best = attempt
            if metrics["coverage"] >= 0.02:
                return best, attempts
        except Exception as exc:
            attempts.append({"distance": distance, "metrics": None, "error": str(exc)})
    return best, attempts


def save_constant_map(image: bpy.types.Image, color) -> dict:
    clear_image(image, color)
    image.save()
    return sample_image(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high", required=True)
    parser.add_argument("--low", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maps-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--basecolor-image", default="")
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--padding-px", type=int, default=8)
    args = parser.parse_args(argv_after_double_dash())

    maps_dir = Path(args.maps_dir)
    maps_dir.mkdir(parents=True, exist_ok=True)
    reset_scene()
    high_objects = import_new(args.high, "HIGH")
    low_objects = import_new(args.low, "LOW")
    if not high_objects or not low_objects:
        raise RuntimeError("High-poly and low-poly meshes are both required for transfer baking")

    supplied_basecolor = Path(args.basecolor_image) if args.basecolor_image else None
    images: dict[str, bpy.types.Image] = {}
    for name in MAP_SPECS:
        destination = maps_dir / f"{name}.png"
        if name == "basecolor" and supplied_basecolor and supplied_basecolor.is_file():
            images[name] = external_image(name, supplied_basecolor, destination, args.size)
        else:
            images[name] = new_image(name, destination, args.size)

    material, nodes = make_target_material(images)
    assign_target_material(low_objects, material)
    configure_cycles(args.padding_px)
    minimum, maximum = world_bounds(high_objects)
    extent = max(maximum - minimum)
    report = {
        "attempts": {}, "maps": {}, "errors": [], "warnings": [],
        "map_sources": {},
    }

    if supplied_basecolor and supplied_basecolor.is_file():
        metrics = sample_image(images["basecolor"])
        report["attempts"]["basecolor"] = [{
            "source": str(supplied_basecolor), "metrics": metrics, "error": None,
        }]
        report["map_sources"]["basecolor"] = "projected_source_and_multiview_atlas"
        if metrics["coverage"] < 0.005:
            report["errors"].append("Supplied projected base-colour atlas has insufficient coverage")
    else:
        best, attempts = bake_with_retry(
            high_objects, low_objects, material, nodes["basecolor"], images["basecolor"],
            "DIFFUSE", extent, MAP_SPECS["basecolor"]["color"],
        )
        report["attempts"]["basecolor"] = attempts
        report["map_sources"]["basecolor"] = "high_poly_material_transfer"
        if not best or best["metrics"]["coverage"] < 0.01:
            report["errors"].append("Base-colour transfer produced insufficient UV coverage")
        images["basecolor"].save()

    best, attempts = bake_with_retry(
        high_objects, low_objects, material, nodes["normal"], images["normal"],
        "NORMAL", extent, MAP_SPECS["normal"]["color"],
    )
    report["attempts"]["normal"] = attempts
    report["map_sources"]["normal"] = "high_poly_geometry_transfer"
    if not best or best["metrics"]["coverage"] < 0.01:
        report["errors"].append("Normal transfer produced insufficient UV coverage")
    images["normal"].save()

    for name, input_names, fallback in (
        ("roughness", ("Roughness",), 0.6),
        ("metallic", ("Metallic",), 0.0),
    ):
        states = prepare_emission_materials(high_objects, input_names, fallback)
        if not states:
            metrics = save_constant_map(images[name], MAP_SPECS[name]["color"])
            report["attempts"][name] = [{"source": "constant_default", "metrics": metrics, "error": None}]
            report["map_sources"][name] = "explicit_default_no_source_material"
            report["warnings"].append(
                f"No source {name} material was available; exported an explicit deterministic default map."
            )
            continue
        try:
            best, attempts = bake_with_retry(
                high_objects, low_objects, material, nodes[name], images[name],
                "EMIT", extent, MAP_SPECS[name]["color"],
            )
            report["attempts"][name] = attempts
            report["map_sources"][name] = "high_poly_material_transfer"
            if not best or best["metrics"]["coverage"] < 0.01:
                report["errors"].append(f"{name.title()} transfer produced insufficient UV coverage")
        finally:
            restore_materials(states)
        images[name].save()

    try:
        clear_image(images["ao"], MAP_SPECS["ao"]["color"])
        bake_low_only(low_objects, material, nodes["ao"], "AO")
        images["ao"].save()
        report["map_sources"]["ao"] = "low_poly_geometry_bake"
    except Exception as exc:
        report["errors"].append(f"AO bake failed: {exc}")

    for name, image in images.items():
        metrics = sample_image(image)
        report["maps"][name] = {
            "path": image.filepath_raw,
            "width": int(image.size[0]),
            "height": int(image.size[1]),
            "metrics": metrics,
            "source": report["map_sources"].get(name, "unknown"),
        }

    select_only(low_objects)
    export_glb(args.output, selected_only=True)
    report.update({
        "success": not report["errors"],
        "backend": "blender_cycles_selected_to_active_cpu",
        "high_stats": extended_mesh_stats(high_objects),
        "low_stats": extended_mesh_stats(low_objects),
        "output": args.output,
        "cpu_baking": True,
        "selected_to_active": True,
        "padding_px": args.padding_px,
        "external_basecolor": str(supplied_basecolor) if supplied_basecolor else None,
    })
    save_json(args.report, report)
    if report["errors"]:
        raise RuntimeError("; ".join(report["errors"]))


if __name__ == "__main__":
    main()
