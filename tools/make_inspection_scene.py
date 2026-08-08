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
import math
import re
import sys
from datetime import datetime
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


def receipt_index():
    index = {}
    for path in (REPO / "evidence" / "compare").rglob("*.json"):
        if "rig_readiness" not in path.name and "rig_report" not in path.name:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        mesh = str(data.get("mesh", "")).replace("\\\\", "/")
        index[mesh] = data
        index[path.stem] = data
    return index


def metadata(asset, glb, readiness):
    stat = glb.stat()
    created = datetime.fromtimestamp(stat.st_ctime).astimezone()
    subject = asset["subject"]
    rig = readiness.get(glb.stem, readiness.get(subject, {}))
    if rig:
        rig_status = "readiness_ready" if rig.get("ready") else "readiness_not_ready"
    else:
        rig_status = "readiness_unknown"
    return {
        "asset_name": asset["name"],
        "subject": subject,
        "generation_route": asset.get("generator", "unknown"),
        "date": created.strftime("%Y-%m-%d %H:%M"),
        "date_source": "filesystem_creation",
        "file_size_bytes": stat.st_size,
        "source_path": asset.get("source", "unknown"),
        "rigging_status": rig_status,
        "animation_status": "unknown",
        "version": "v1",
        "version_source": "inspection_metadata_schema",
        "texture_or_atlas": asset.get("texture", "unknown"),
        "faces": asset.get("faces", "unknown"),
        "sort_time": stat.st_ctime,
    }


def placard_text(meta):
    size = f"{meta['file_size_bytes'] / 1e6:.1f} MB"
    return (f"{meta['subject']}\\n"
            f"{meta['generation_route']} | {meta['date']}\\n"
            f"{size} | {meta['faces']} faces\\n"
            f"tex {meta['texture_or_atlas']} | rig {meta['rigging_status']}\\n"
            f"anim {meta['animation_status']} | v {meta['version']}")


def add_placard(meta, location, width):
    curve = bpy.data.curves.new(f"placard_{meta['subject']}", "FONT")
    curve.body = placard_text(meta)
    curve.align_x = "CENTER"
    curve.size = max(min(width * 0.08, 0.8), 0.18)
    curve.extrude = 0.002
    obj = bpy.data.objects.new(f"placard_{meta['subject']}", curve)
    obj.location = (location.x + width / 2.0, location.y - 0.12, location.z)
    obj.rotation_euler = (math.radians(72), 0.0, 0.0)
    bpy.context.scene.collection.objects.link(obj)
    for key, value in meta.items():
        if key != "sort_time":
            obj[key] = value
    return obj


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    manifest = json.loads((SRC / "MANIFEST.json").read_text(encoding="utf-8"))
    readiness = {}
    for path in (REPO / "evidence" / "compare").rglob("*_rig_readiness.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        subject = path.parent.name.removesuffix("_new")
        readiness[subject] = data
        mesh_name = Path(str(data.get("mesh", ""))).stem
        if mesh_name:
            readiness[mesh_name] = data
    assets = []
    for asset in manifest["assets"]:
        glb = SRC / asset["name"]
        if glb.exists():
            assets.append((metadata(asset, glb, readiness), asset))
    assets.sort(key=lambda item: (item[0]["sort_time"], item[0]["asset_name"]))

    cursor = 0.0
    placed = []
    for meta, asset in assets:
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
        for key, value in meta.items():
            if key != "sort_time":
                new[0][key] = value
        add_placard(meta, Vector((cursor, 0.0, 0.0)), width)

        placed.append({"name": glb.stem, "x_start": round(cursor, 3),
                       "width": round(width, 3), "height": round(hi.z - lo.z, 3),
                       "target_metres": metres, "target_axis": axis,
                       "date": meta["date"], "rig": meta["rigging_status"]})
        cursor += max(width, width * (1.0 + GAP))

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

    # Global safety net for implicit-UV failure in Blender 5.2 EEVEE
    # (Material Preview): mirrors in-place fix applied to
    # evidence/deliverables/blender/ALL_ASSETS_inspection.blend
    # (13 mats wired TexCoord UV -> Image Texture.Vector, 198.1 MB,
    # Blender 5.2). bpy.ops.import_scene.gltf leaves Vector unlinked,
    # relying on implicit UV which EEVEE 5.2 ignores. Iterate all
    # materials after the import loop and before pack_all().
    for mat in bpy.data.materials:
        if not mat.use_nodes or mat.node_tree is None:
            continue
        # Reuse existing TexCoord if present, else create one per material
        tc = next((n for n in mat.node_tree.nodes if n.bl_idname == "ShaderNodeTexCoord"), None)
        for node in mat.node_tree.nodes:
            if node.bl_idname == "ShaderNodeTexImage" and not node.inputs["Vector"].is_linked:
                if tc is None:
                    tc = mat.node_tree.nodes.new("ShaderNodeTexCoord")
                mat.node_tree.links.new(tc.outputs["UV"], node.inputs["Vector"])

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
