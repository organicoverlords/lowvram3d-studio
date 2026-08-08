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
    # Real newlines. These were written "\\n" -- a literal backslash and an n --
    # so every placard rendered as one run-on line with the escapes visible on
    # the board. A Blender FONT body takes actual line breaks; nothing in the
    # build logs would ever have flagged this, because the text was set fine.
    return (f"{meta['subject']}\n"
            f"{meta['generation_route']} | {meta['date']}\n"
            f"{size} | {meta['faces']} faces\n"
            f"tex {meta['texture_or_atlas']} | rig {meta['rigging_status']}\n"
            f"anim {meta['animation_status']} | v {meta['version']}")


def flat_material(name, colour, roughness=0.8):
    """One material per name, reused. Signposts share a post and a board."""
    existing = bpy.data.materials.get(name)
    if existing is not None:
        return existing
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = colour
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def add_placard(meta, location, width):
    """A standing signpost -- post, board, and text carved on the board.

    Everything is proportional to the asset it labels, not a fixed size. The
    lineup runs from a 1.3 m frog to a 120 m titan; a signpost sized for one is
    either a toothpick or a billboard beside the other. Scaling with the asset
    keeps every label the same apparent size relative to its own subject, which
    is what makes the row readable at a single zoom.

    The board faces -Y because that is where the camera stands, and it tilts
    back a few degrees so it catches the sky light instead of rendering as a
    flat dark rectangle edge-on to the sun.
    """
    board_w = max(min(width * 0.62, 24.0), 1.4)
    board_h = board_w * 0.52
    post_h = max(min(width * 0.42, 16.0), 1.1)
    post_r = max(board_w * 0.022, 0.02)
    # Stand clear of the asset. These meshes are roughly as deep as they are
    # wide, so width is a fair proxy for how far forward the sign has to sit.
    stand_y = -(width * 0.38 + board_w * 0.5)
    centre_x = location.x + width / 2.0
    tilt = math.radians(-8.0)

    bpy.ops.mesh.primitive_cylinder_add(
        radius=post_r, depth=post_h, vertices=12,
        location=(centre_x, stand_y, post_h / 2.0))
    post = bpy.context.active_object
    post.name = f"signpost_{meta['subject']}"
    post.data.materials.append(flat_material("signpost_wood", (0.09, 0.055, 0.03, 1.0)))

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, 0.0))
    board = bpy.context.active_object
    board.name = f"signboard_{meta['subject']}"
    board.scale = (board_w, board_w * 0.035, board_h)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    board.location = (centre_x, stand_y, post_h + board_h * 0.42)
    board.rotation_euler = (tilt, 0.0, 0.0)
    board.data.materials.append(flat_material("signboard_face", (0.42, 0.31, 0.18, 1.0)))

    curve = bpy.data.curves.new(f"placard_{meta['subject']}", "FONT")
    curve.body = placard_text(meta)
    curve.align_x = "CENTER"
    curve.align_y = "CENTER"
    # Five lines have to fit the board height with margin, so the type size is
    # derived from the board rather than picked and hoped for.
    curve.size = board_h / 7.4
    curve.extrude = board_w * 0.002
    text = bpy.data.objects.new(f"placard_{meta['subject']}", curve)
    # Font objects lie in XY facing +Z; stand it up to face the camera at -Y,
    # then sit it just proud of the board's front face.
    text.rotation_euler = (math.radians(90.0) + tilt, 0.0, 0.0)
    text.location = (centre_x,
                     stand_y - board_w * 0.021 - board_h * 0.02,
                     post_h + board_h * 0.42)
    bpy.context.scene.collection.objects.link(text)
    text.data.materials.append(
        flat_material("signboard_text", (0.02, 0.02, 0.02, 1.0), roughness=0.6))

    # One rigid unit, so the sign can be moved or hidden as a whole.
    for child in (board, text):
        child.parent = post
        child.matrix_parent_inverse = post.matrix_world.inverted()

    for key, value in meta.items():
        if key != "sort_time":
            post[key] = value
            text[key] = value
    return post


def add_ground(span, depth):
    """A grass field under the row, sized to whatever the row turned out to be.

    Procedural rather than scattered geometry, and that is a choice worth
    stating. The lineup spans hundreds of metres and the tallest subject is a
    40 m titan; a particle or geometry-nodes grass field covering that at a
    blade density that reads as grass is millions of instances, on a machine
    with 15 GB of RAM that already renders on the CPU to stay out of the GPU's
    way. A layered-noise material costs nothing, holds up at the distance this
    camera actually views from, and does not put the lineup at risk.

    Two noise scales, because one reads as plastic: a coarse one for the patchy
    lighter/darker drift a real field has, a fine one driving bump so the
    surface catches the sun instead of returning flat colour.
    """
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(span / 2.0, 0.0, 0.0))
    ground = bpy.context.active_object
    ground.name = "ground_grass"
    # Generous margin: the row is centred, and the field has to reach past the
    # camera's pull-back distance or the horizon shows the plane's own edge.
    ground.scale = (span * 3.0 + 40.0, span * 3.0 + 40.0, 1.0)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    mat = bpy.data.materials.new("grass_field")
    mat.use_nodes = True
    tree = mat.node_tree
    bsdf = tree.nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.92
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.18

    coordinate = tree.nodes.new("ShaderNodeTexCoord")

    # Noise Scale is a multiplier on the incoming coordinate, and this plane's
    # scale has been applied, so its object coordinates run to +-1000 units.
    # The first version used Scale 1.6 and 240 -- values that suit a mesh about
    # a metre across. Here they drove the noise to coordinates in the hundreds
    # of thousands, far past any feature size that survives to a pixel, so both
    # layers averaged to flat colour and the field rendered as green paint.
    #
    # Sized to the subject instead: patches of roughly 80 units across a field
    # holding 120 m creatures, and a blade layer at roughly 2 units, which is
    # the smallest thing still worth more than a pixel from this camera.
    patch = tree.nodes.new("ShaderNodeTexNoise")
    patch.inputs["Scale"].default_value = 0.012
    patch.inputs["Detail"].default_value = 6.0
    patch.inputs["Roughness"].default_value = 0.62
    tree.links.new(coordinate.outputs["Object"], patch.inputs["Vector"])

    ramp = tree.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.36
    ramp.color_ramp.elements[0].color = (0.018, 0.055, 0.010, 1.0)
    ramp.color_ramp.elements[1].position = 0.64
    ramp.color_ramp.elements[1].color = (0.105, 0.215, 0.035, 1.0)
    tree.links.new(patch.outputs["Fac"], ramp.inputs["Fac"])
    tree.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    # Clumping between the patch drift and the blades, so the field does not
    # read as one uniform frequency at every distance.
    clump = tree.nodes.new("ShaderNodeTexNoise")
    clump.inputs["Scale"].default_value = 0.09
    clump.inputs["Detail"].default_value = 8.0
    tree.links.new(coordinate.outputs["Object"], clump.inputs["Vector"])

    blades = tree.nodes.new("ShaderNodeTexNoise")
    blades.inputs["Scale"].default_value = 0.55
    blades.inputs["Detail"].default_value = 10.0
    blades.inputs["Roughness"].default_value = 0.75
    tree.links.new(coordinate.outputs["Object"], blades.inputs["Vector"])

    mix = tree.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "OVERLAY"
    mix.inputs["Fac"].default_value = 0.65
    tree.links.new(clump.outputs["Fac"], mix.inputs["Color1"])
    tree.links.new(blades.outputs["Fac"], mix.inputs["Color2"])

    bump = tree.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.85
    bump.inputs["Distance"].default_value = 0.4
    tree.links.new(mix.outputs["Color"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    ground.data.materials.append(mat)
    return ground


def add_sky(scene, sun):
    """A physical sky, with the sun light pointed the same way as the sun in it.

    If these two disagree the shadows fall one way and the bright part of the
    sky sits another, which reads as wrong without being obviously identifiable
    as wrong. So the elevation and rotation are set once here and the sun lamp
    is rotated to match rather than kept at its old hand-picked angle.
    """
    # 22 degrees put the sun low and behind a camera that looks level at the
    # horizon, so the frame filled with the dullest band of the dome and the
    # sky rendered grey. Higher and swung round to the side gives the camera
    # the blue part of the sky and rakes the light across the subjects instead
    # of flattening them from behind.
    elevation = math.radians(38.0)
    rotation = math.radians(58.0)

    world = bpy.data.worlds.new("sky")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs[1].default_value = 1.0
    try:
        sky = world.node_tree.nodes.new("ShaderNodeTexSky")
        # Blender renamed these between versions: what was NISHITA is
        # MULTIPLE_SCATTERING in 5.2, and hardcoding either one fails on the
        # other with an enum error that says nothing useful. Ask the property
        # what it accepts and take the best available.
        available = [item.identifier for item in
                     sky.bl_rna.properties["sky_type"].enum_items]
        for candidate in ("MULTIPLE_SCATTERING", "NISHITA",
                          "SINGLE_SCATTERING", "HOSEK_WILKIE", "PREETHAM"):
            if candidate in available:
                sky.sky_type = candidate
                print(f"sky_type {candidate}  (available: {', '.join(available)})")
                break
        for attribute, value in (("sun_elevation", elevation),
                                 ("sun_rotation", rotation),
                                 ("sun_intensity", 1.0),
                                 # A little haze reads as air rather than as a
                                 # flat gradient; ozone deepens the blue.
                                 ("dust_density", 0.9),
                                 ("air_density", 1.0),
                                 ("ozone_density", 2.2),
                                 ("sun_disc", True)):
            if hasattr(sky, attribute):
                setattr(sky, attribute, value)
        world.node_tree.links.new(sky.outputs["Color"], background.inputs[0])
    except (RuntimeError, AttributeError, TypeError, KeyError) as exc:
        # A missing Sky Texture is survivable -- a plain blue keeps the scene
        # usable -- but it must not pass silently as "sky added".
        print("SKY_FALLBACK", exc)
        background.inputs[0].default_value = (0.30, 0.48, 0.78, 1.0)
    scene.world = world

    # Blender's sun points down -Z at rotation zero; tilt it up to the sky's
    # elevation and swing it round to the sky's compass bearing.
    sun.rotation_euler = (math.pi / 2.0 - elevation, 0.0, rotation)
    return world


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

    add_sky(scene, light)
    add_ground(span, tallest)

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
