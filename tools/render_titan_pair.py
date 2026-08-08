"""Render the two moss titans in the grass field, framed on the pair.

    blender.exe --background evidence/deliverables/blender/ALL_ASSETS_inspection.blend \
        --python tools/render_titan_pair.py -- --out lineup_titans.png

The lineup camera holds all 666 units of the row, which puts a 1.3 m frog at a
few pixels and the titans at the edges. This one frames only the two moss
titans -- the same geometry from the same seed, one carrying TRELLIS's own
atlas and one carrying the vendor paint -- so the comparison the pair exists to
make is actually legible.

Everything else stays in shot behind them rather than being hidden: the row is
the scale reference that makes 120 m mean something.
"""

import argparse
import sys

import bpy
from mathutils import Vector


def bounds_of(objects):
    corners = [obj.matrix_world @ Vector(c)
               for obj in objects for c in obj.bound_box]
    low = Vector((min(c.x for c in corners), min(c.y for c in corners),
                  min(c.z for c in corners)))
    high = Vector((max(c.x for c in corners), max(c.y for c in corners),
                   max(c.z for c in corners)))
    return low, high


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--width", type=int, default=2800)
    parser.add_argument("--height", type=int, default=1400)
    args = parser.parse_args(argv)

    scene = bpy.context.scene

    titans = [o for o in scene.objects
              if o.type == "MESH" and o.name.startswith("mosstitan")]
    if not titans:
        raise SystemExit("RENDER_ABORT: no mosstitan meshes in this blend")
    print(f"framing {len(titans)} titan mesh(es): {[o.name for o in titans]}")

    low, high = bounds_of(titans)
    centre = (low + high) * 0.5
    width = high.x - low.x
    height = high.z - low.z

    camera_data = bpy.data.cameras.new("titan_pair")
    camera_data.lens = 35
    camera = bpy.data.objects.new("titan_pair", camera_data)
    # The pair spans ~370 units, and backing off by 0.95x that with a 42 mm
    # lens cropped both subjects at the frame edges. Distance is now derived
    # from the span and the lens rather than guessed: half the width, over the
    # tangent of half the horizontal field of view, plus a margin.
    import math as _math
    sensor = camera_data.sensor_width
    half_fov = _math.atan((sensor * 0.5) / camera_data.lens)
    distance = (width * 0.5) / _math.tan(half_fov) * 1.18
    camera.location = (centre.x, low.y - distance, height * 0.42)
    camera.rotation_euler = (1.5708, 0.0, 0.0)
    print(f"pair span {width:.1f}  camera back {distance:.1f}  lens {camera_data.lens}mm")
    scene.collection.objects.link(camera)
    scene.camera = camera

    available = [i.identifier for i in
                 bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    engine = next((e for e in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")
                   if e in available), available[0])
    scene.render.engine = engine
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples
    print(f"engine {engine}   samples {args.samples}")

    ground = [o for o in scene.objects if o.name == "ground_grass"]
    signs = [o for o in scene.objects if o.name.startswith("signpost_")]
    sky = scene.world and scene.world.use_nodes and any(
        n.bl_idname == "ShaderNodeTexSky" for n in scene.world.node_tree.nodes)
    print(f"ground {'yes' if ground else 'NO'}   signposts {len(signs)}   "
          f"sky {'yes' if sky else 'NO'}")

    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    # Absolute, because Blender resolves a bare relative path against the drive
    # root: "evidence/.../x.png" was written to C:\evidence\...\x.png, and the
    # script cheerfully printed "wrote" for a file that was not where it said.
    from pathlib import Path as _Path
    destination = _Path(args.out).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(destination)

    bpy.ops.render.render(write_still=True)
    if destination.is_file():
        print(f"wrote {destination}  {destination.stat().st_size / 1e6:.1f} MB")
    else:
        raise SystemExit(f"RENDER_ABORT: nothing at {destination}")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
