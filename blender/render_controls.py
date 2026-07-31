from __future__ import annotations

import argparse
from pathlib import Path

import bpy

from common import (
    argv_after_double_dash,
    configure_render,
    create_camera,
    import_mesh,
    mesh_objects,
    normalize_scene,
    reset_scene,
    save_json,
)

VIEWS = {
    "front": (0.0, -3.0, 0.0),
    "right": (3.0, 0.0, 0.0),
    "back": (0.0, 3.0, 0.0),
    "left": (-3.0, 0.0, 0.0),
    "top": (0.0, 0.0, 3.0),
    "bottom": (0.0, 0.0, -3.0),
}


def generated_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("PositionControl")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    tex = nodes.new("ShaderNodeTexCoord")
    links.new(tex.outputs["Generated"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def normal_material() -> bpy.types.Material:
    mat = bpy.data.materials.new("NormalControl")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    emission = nodes.new("ShaderNodeEmission")
    geometry = nodes.new("ShaderNodeNewGeometry")
    multiply = nodes.new("ShaderNodeVectorMath")
    multiply.operation = "SCALE"
    multiply.inputs[3].default_value = 0.5
    add = nodes.new("ShaderNodeVectorMath")
    add.operation = "ADD"
    add.inputs[1].default_value = (0.5, 0.5, 0.5)
    links.new(geometry.outputs["Normal"], multiply.inputs[0])
    links.new(multiply.outputs["Vector"], add.inputs[0])
    links.new(add.outputs["Vector"], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return mat


def assign_material(material: bpy.types.Material) -> None:
    for obj in mesh_objects():
        obj.data.materials.clear()
        obj.data.materials.append(material)


def render_pass(output_dir: Path, pass_name: str, material: bpy.types.Material) -> None:
    assign_material(material)
    for view_name, location in VIEWS.items():
        camera = bpy.data.objects.get(f"Camera_{view_name}")
        if camera is None:
            camera = create_camera(f"Camera_{view_name}", location)
        bpy.context.scene.camera = camera
        path = output_dir / f"{view_name}_{pass_name}.png"
        bpy.context.scene.render.filepath = str(path)
        bpy.ops.render.render(write_still=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args(argv_after_double_dash())

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reset_scene()
    objects = import_mesh(args.input)
    normalization = normalize_scene(objects)
    configure_render(args.size, args.size)
    for name, location in VIEWS.items():
        create_camera(f"Camera_{name}", location)
    render_pass(output_dir, "position", generated_material())
    render_pass(output_dir, "normal", normal_material())
    save_json(args.metadata, {
        "views": [{"name": name, "location": location} for name, location in VIEWS.items()],
        "size": args.size,
        "normalization": normalization,
    })


if __name__ == "__main__":
    main()
