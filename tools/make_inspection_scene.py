"""One .blend holding every delivered asset in a row, ready to inspect.

Run inside Blender:
    blender.exe --background --factory-startup --python make_inspection_scene.py

Spacing is computed from each asset's own footprint rather than a fixed pitch.
These models differ by an order of magnitude in width -- a snail against a
castle -- so a constant gap either overlaps the large ones or strands the small
ones. Each asset is placed at the running edge plus a gap proportional to the
widest neighbour, which keeps them separated and readable at any zoom.

Every asset sits with its base on z=0 and its centre on y=0, so the row reads as
a lineup rather than a scatter, and relative heights are directly comparable --
which is the point of putting them side by side.

Scale comes from REAL_SIZES.json. Without it the comparison is worthless: the
meshes are normalised to a unit bounding box, so a whale stands shorter than a
snail and the lineup measures nothing but normalisation.
"""

import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).parent))
import real_sizes

REPO = Path(r"C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803")
SRC = REPO / "evidence" / "deliverables"
OUT = SRC / "blender" / "ALL_ASSETS_inspection.blend"

GAP = 0.35  # of the neighbouring asset's width


def bounds(objs):
    corners = [obj.matrix_world @ Vector(c)
               for obj in objs for c in obj.bound_box]
    lo = Vector((min(c.x for c in corners), min(c.y for c in corners),
                 min(c.z for c in corners)))
    hi = Vector((max(c.x for c in corners), max(c.y for c in corners),
                 max(c.z for c in corners)))
    return lo, hi


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    manifest = json.loads((SRC / "MANIFEST.json").read_text(encoding="utf-8"))
    assets = sorted(manifest["assets"], key=lambda a: a["name"])

    cursor = 0.0
    placed = []
    for asset in assets:
        glb = SRC / asset["name"]
        if not glb.exists():
            continue
        before = set(bpy.context.scene.objects)
        bpy.ops.import_scene.gltf(filepath=str(glb))
        new = [o for o in bpy.context.scene.objects
               if o not in before and o.type == "MESH"]
        if not new:
            continue

        for obj in new:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        for obj in new:
            obj.select_set(False)

        # Real size before placement, so the row compares subjects rather than
        # thirteen unit cubes. Without this a whale stands shorter than a snail.
        lo, hi = bounds(new)
        factor, metres, axis = real_sizes.scale_for(
            glb.stem, (hi.x - lo.x, hi.y - lo.y, hi.z - lo.z))
        for obj in new:
            obj.scale = (factor, factor, factor)
        bpy.context.view_layer.update()
        for obj in new:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        for obj in new:
            obj.select_set(False)

        lo, hi = bounds(new)
        width = hi.x - lo.x
        # Base on the floor, centred across the row, left edge at the cursor.
        shift = Vector((cursor - lo.x, -(lo.y + hi.y) / 2.0, -lo.z))
        for obj in new:
            obj.location += shift
        new[0].name = glb.stem

        placed.append({"name": glb.stem, "x_start": round(cursor, 3),
                       "width": round(width, 3), "height": round(hi.z - lo.z, 3),
                       "target_metres": metres, "target_axis": axis})
        cursor += width * (1.0 + GAP)

    scene = bpy.context.scene
    tallest = max((p["height"] for p in placed), default=1.0)
    span = cursor

    light_data = bpy.data.lights.new("key", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("key", light_data)
    light.rotation_euler = (0.9, 0.0, 0.7)
    scene.collection.objects.link(light)

    camera_data = bpy.data.cameras.new("inspect")
    camera_data.lens = 50
    camera = bpy.data.objects.new("inspect", camera_data)
    # Far enough back to hold the whole row, lifted to mid-height, looking level.
    camera.location = (span / 2.0, -span * 0.75, tallest * 0.55)
    camera.rotation_euler = (1.5708, 0.0, 0.0)
    scene.collection.objects.link(camera)
    scene.camera = camera

    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
    scene.world = world

    try:
        bpy.ops.file.pack_all()
    except RuntimeError as exc:
        print("pack warning", exc)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT), compress=True)

    for p in placed:
        print(f"  {p['name']:<46} x {p['x_start']:8.1f}  "
              f"{p['width']:7.1f} m wide  {p['height']:7.1f} m tall"
              f"   (target {p['target_metres']} m {p['target_axis']})")
    print(f"\n{len(placed)} assets, row spans {span:.2f} units")
    print("wrote", OUT, f"{OUT.stat().st_size/1e6:.1f} MB")


main()
