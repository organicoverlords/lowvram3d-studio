"""Render the high-poly PLY master that TRELLIS writes beside each GLB.

    blender.exe --background --factory-startup --python tools/render_highpoly_ply.py -- \
        --ply evidence/compare/fattree/fattree_t512.ply --out highpoly.png --views 4

The GLB in each asset directory is the decimated deliverable -- 148,852 faces
for the fat tree. The PLY beside it is what the generator actually decoded:
7,635,867 vertices and 23,278,541 faces carrying per-vertex colour. That is the
mesh worth looking at when the question is how much detail the decode found, as
opposed to how much survived the face budget.

Vertex colour, not a texture. The PLY has no UVs, so the colour has to be wired
through a Color Attribute node; a Principled BSDF left at its default renders
23 million faces of flat grey and looks like a decimation bug that isn't there.

Memory is the real constraint. 23M faces is several gigabytes in Blender, which
on a 15 GB machine means nothing else heavy can be running -- a paint job beside
this will be reclaimed by the OS with no traceback. The loaded face count is
printed so a silent partial import cannot pass as a successful render.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--size", type=int, default=1400)
    parser.add_argument("--samples", type=int, default=32)
    args = parser.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    source = Path(args.ply).resolve()
    print(f"importing {source} ({source.stat().st_size / 1e6:.0f} MB)", flush=True)
    # Blender 4.x+ renamed the PLY operator; try the new one, fall back.
    if hasattr(bpy.ops.wm, "ply_import"):
        bpy.ops.wm.ply_import(filepath=str(source))
    else:
        bpy.ops.import_mesh.ply(filepath=str(source))

    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise SystemExit("PLY_ABORT: nothing imported")
    faces = sum(len(o.data.polygons) for o in meshes)
    verts = sum(len(o.data.vertices) for o in meshes)
    print(f"loaded {verts} vertices, {faces} faces", flush=True)

    material = bpy.data.materials.new("vertex_colour")
    material.use_nodes = True
    tree = material.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.85
    for obj in meshes:
        layers = list(obj.data.color_attributes)
        if layers:
            node = tree.nodes.new("ShaderNodeVertexColor")
            node.layer_name = layers[0].name
            tree.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
            print(f"wired colour attribute '{layers[0].name}'", flush=True)
        obj.data.materials.clear()
        obj.data.materials.append(material)

    scene = bpy.context.scene
    available = [i.identifier for i in
                 bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    scene.render.engine = next((e for e in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")
                                if e in available), available[0])
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples
    # Standard, not AgX: a catalogue render should show what was shaded.
    try:
        scene.view_settings.view_transform = "Standard"
    except (AttributeError, TypeError):
        pass

    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 1.3
    scene.world = world
    for angle, energy in ((35.0, 4.0), (215.0, 1.8)):
        sun = bpy.data.objects.new(f"sun{angle:.0f}",
                                   bpy.data.lights.new(f"sun{angle:.0f}", type="SUN"))
        sun.data.energy = energy
        sun.rotation_euler = (math.radians(58), 0.0, math.radians(angle))
        scene.collection.objects.link(sun)

    corners = [obj.matrix_world @ Vector(c) for obj in meshes for c in obj.bound_box]
    low = Vector((min(c.x for c in corners), min(c.y for c in corners),
                  min(c.z for c in corners)))
    high = Vector((max(c.x for c in corners), max(c.y for c in corners),
                   max(c.z for c in corners)))
    centre = (low + high) * 0.5
    radius = max((high - low).x, (high - low).y, (high - low).z) * 0.5

    camera_data = bpy.data.cameras.new("cam")
    camera_data.lens = 55
    camera_data.clip_end = radius * 50
    camera = bpy.data.objects.new("cam", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    scene.render.resolution_x = args.size
    scene.render.resolution_y = args.size
    scene.render.image_settings.file_format = "PNG"

    destination = Path(args.out).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for index in range(args.views):
        theta = math.radians(index * 360.0 / args.views + 25.0)
        distance = radius * 3.1
        camera.location = (centre.x + distance * math.sin(theta),
                           centre.y - distance * math.cos(theta),
                           centre.z + radius * 0.85)
        direction = centre - Vector(camera.location)
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        tile = destination.parent / f"{destination.stem}_v{index}.png"
        scene.render.filepath = str(tile)
        bpy.ops.render.render(write_still=True)
        written.append(tile)
        print(f"  view {index} written", flush=True)

    # Stitch into one strip so the whole thing arrives as a single image.
    images = [bpy.data.images.load(str(t)) for t in written]
    width, height = images[0].size
    total = width * len(images)
    strip = bpy.data.images.new("strip", total, height, alpha=False)
    buffer = [0.0] * (total * height * 4)
    for index, image in enumerate(images):
        pixels = list(image.pixels)
        offset = index * width
        for row in range(height):
            src = row * width * 4
            dst = (row * total + offset) * 4
            buffer[dst:dst + width * 4] = pixels[src:src + width * 4]
    strip.pixels = buffer
    strip.filepath_raw = str(destination)
    strip.file_format = "PNG"
    strip.save()
    for tile in written:
        tile.unlink(missing_ok=True)
    print(f"wrote {destination}  ({faces} faces at full resolution)", flush=True)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
