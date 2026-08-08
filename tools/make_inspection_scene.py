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
from mathutils import Euler, Vector

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
    # ctime is metadata-change time on Windows and is not a reliable creation
    # order after hard-linking/copying deliverables. Prefer the manifest's
    # authoritative source timestamp, then fall back to the deliverable.
    source_value = asset.get("source")
    source_path = REPO / source_value if source_value else None
    source_stat = source_path.stat() if source_path and source_path.is_file() else stat
    created = datetime.fromtimestamp(source_stat.st_ctime).astimezone()
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


def wire_imported_material_uvs(materials):
    """Make imported image textures use the mesh UV map explicitly.

    Blender 5.2 EEVEE Material Preview does not reliably evaluate the
    implicit UV source left by the glTF importer.  An explicit TexCoord UV
    link keeps packed textures visible after the scene is rebuilt.
    """
    image_nodes = 0
    linked_nodes = 0
    for mat in materials:
        if not mat or not mat.use_nodes or mat.node_tree is None:
            continue
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        texcoord = next(
            (node for node in nodes if node.bl_idname == "ShaderNodeTexCoord"),
            None,
        )
        for image in (node for node in nodes
                      if node.bl_idname == "ShaderNodeTexImage"):
            image_nodes += 1
            vector = image.inputs.get("Vector")
            if vector is None:
                continue
            # Relink even if the importer supplied a different coordinate
            # source: every imported image must explicitly use mesh UVs.
            for link in list(vector.links):
                links.remove(link)
            if texcoord is None:
                texcoord = nodes.new("ShaderNodeTexCoord")
            links.new(texcoord.outputs["UV"], vector)
            if vector.is_linked:
                linked_nodes += 1
    return image_nodes, linked_nodes


# Not called. Kept because the measurement behind it is worth not repeating:
# TRELLIS's own atlases average RGB [35.6 37.2 21.1] against the vendor paint's
# [74.8 73.6 29.4], so under a bright sky a prepaint asset reflects mostly sky
# ambient and reads slate-blue beside its painted twin. That is exposure, not
# hue -- a first attempt treated it as saturation and made everything lurid
# without touching the cast.
CONTRAST = 0.12
"""Contrast lift for the ORIGINAL (prepaint) atlases only.

TRELLIS's own atlases are flatter than the vendor paint's, so a small contrast
lift makes them read better in the showroom. It is applied to nothing else.

**The painted assets are never touched.** They are the output of a stage whose
quality is the thing under evaluation, and quietly grading them would make that
comparison meaningless -- the whole reason both variants are in the scene is to
see the paint stage's effect unaltered.

Non-destructive either way: this edits the material in this .blend, not the
atlas on disk, so the deliverable an engine loads is unchanged.
"""


def grade_base_colour(mat) -> bool:
    """Add a small contrast lift between base-colour image and shader."""
    tree = mat.node_tree
    bsdf = next((n for n in tree.nodes
                 if n.bl_idname == "ShaderNodeBsdfPrincipled"), None)
    if bsdf is None:
        return False
    socket = bsdf.inputs.get("Base Color")
    if socket is None or not socket.is_linked:
        return False
    source = socket.links[0].from_node
    if source.bl_idname != "ShaderNodeTexImage":
        return False

    contrast = tree.nodes.new("ShaderNodeBrightContrast")
    contrast.inputs["Contrast"].default_value = CONTRAST
    tree.links.new(source.outputs["Color"], contrast.inputs["Color"])
    tree.links.new(contrast.outputs["Color"], socket)
    return True


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
    # Was width*0.62 capped at 24: on the castle that produced a 14.9-unit
    # board against 24 units of castle, so the label was the subject. A sign is
    # legible at a fraction of the asset it labels, and stops being a sign when
    # it competes with it.
    board_w = max(min(width * 0.30, 6.0), 0.9)
    board_h = board_w * 0.52
    post_h = max(min(width * 0.30, 5.0), 0.8)
    post_r = max(board_w * 0.022, 0.02)
    # Stand clear of the asset. These meshes are roughly as deep as they are
    # wide, so width is a fair proxy for how far forward the sign has to sit.
    # Relative to the row the asset sits in, not to y=0. The lineup wraps into
    # several rows so the whole set can be framed at once, and a signpost pinned
    # to y=0 would stand in front of a different row's asset.
    stand_y = location.y - (width * 0.38 + board_w * 0.5)
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
    # Wide spread, because break-up at this viewing distance is carried by
    # colour and not by relief. The camera stands ~110 units back, where a 4 cm
    # blade is far under a pixel and averages to flat no matter how it is lit --
    # so the earlier fix of pushing the blade layer to a realistic 4 cm was
    # right for a close-up and wrong for this shot. The band that actually
    # resolves here is roughly 0.3 to 5 units, and that is where the contrast
    # has to live.
    ramp.color_ramp.elements[0].position = 0.30
    ramp.color_ramp.elements[0].color = (0.012, 0.038, 0.006, 1.0)
    ramp.color_ramp.elements[1].position = 0.72
    ramp.color_ramp.elements[1].color = (0.155, 0.290, 0.048, 1.0)
    tree.links.new(patch.outputs["Fac"], ramp.inputs["Fac"])

    # Clumping between the patch drift and the blades, so the field does not
    # read as one uniform frequency at every distance.
    # Blades at the size of actual blades. The previous layer sat at Scale 0.55,
    # which on this plane means features roughly 1.8 units across -- so each
    # "blade" was nearly two metres of grass, and the field read as paint no
    # matter how the strength was tuned. Noise Scale is a coordinate multiplier,
    # so feature size is about 1/Scale in object units: 4 cm blades want Scale
    # near 25, not near 0.5. That is a factor of fifty, which is why nudging the
    # old value never worked.
    #
    # Four octaves, each doing one job, because a single frequency reads as
    # either a bald lawn up close or a flat colour at distance:
    #   drift   0.02  ~50 units   where the field is greener or drier
    #   clumps  0.5   ~2 units    tussocks
    #   tufts   4.0   ~25 cm      where blades gather
    #   blades  26.0  ~4 cm       the blades themselves
    clump = tree.nodes.new("ShaderNodeTexNoise")
    clump.inputs["Scale"].default_value = 1.6
    clump.inputs["Detail"].default_value = 8.0
    clump.inputs["Roughness"].default_value = 0.6
    tree.links.new(coordinate.outputs["Object"], clump.inputs["Vector"])

    tufts = tree.nodes.new("ShaderNodeTexNoise")
    tufts.inputs["Scale"].default_value = 0.45
    tufts.inputs["Detail"].default_value = 6.0
    tufts.inputs["Roughness"].default_value = 0.7
    tree.links.new(coordinate.outputs["Object"], tufts.inputs["Vector"])

    # Voronoi rather than noise for the blades. Noise is smooth everywhere, and
    # grass is not: it is a field of separate slivers with hard edges between
    # them, which is exactly what Voronoi distance-to-cell gives and what makes
    # the bump read as blades instead of as crumpled fabric.
    blades = tree.nodes.new("ShaderNodeTexVoronoi")
    blades.feature = "F1"
    blades.inputs["Scale"].default_value = 26.0
    if "Randomness" in blades.inputs:
        blades.inputs["Randomness"].default_value = 1.0
    tree.links.new(coordinate.outputs["Object"], blades.inputs["Vector"])

    fine = tree.nodes.new("ShaderNodeMixRGB")
    fine.blend_type = "OVERLAY"
    fine.inputs["Fac"].default_value = 0.75
    tree.links.new(tufts.outputs["Fac"], fine.inputs["Color1"])
    tree.links.new(blades.outputs["Distance"], fine.inputs["Color2"])

    mix = tree.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "OVERLAY"
    mix.inputs["Fac"].default_value = 0.6
    tree.links.new(clump.outputs["Fac"], mix.inputs["Color1"])
    tree.links.new(fine.outputs["Color"], mix.inputs["Color2"])

    # Tint by the clump layer as well as bump by it, so a tussock is both raised
    # and a slightly different green. Colour variation at the tussock scale is
    # most of what separates a lawn from a field.
    tint = tree.nodes.new("ShaderNodeMixRGB")
    tint.blend_type = "MIX"
    tint.inputs["Fac"].default_value = 0.55
    tint.inputs["Color2"].default_value = (0.115, 0.155, 0.022, 1.0)
    tree.links.new(ramp.outputs["Color"], tint.inputs["Color1"])
    tree.links.new(tint.outputs["Color"], bsdf.inputs["Base Color"])

    tree.links.new(clump.outputs["Fac"], tint.inputs["Fac"])

    bump = tree.nodes.new("ShaderNodeBump")
    # Distance is in object units, so it has to be blade-sized too: 0.4 was a
    # 40 cm displacement on a 4 cm feature, which flattens into noise.
    bump.inputs["Strength"].default_value = 1.0
    bump.inputs["Distance"].default_value = 0.22
    tree.links.new(mix.outputs["Color"], bump.inputs["Height"])
    tree.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

    # Grass scatters rather than reflects; a smooth ground plane under a bright
    # sky reads as painted plastic regardless of its colour.
    bsdf.inputs["Roughness"].default_value = 0.95
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.18

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
    # Chronological. The row is a record of what was generated and when, so the
    # order carries information that a size sort would destroy -- and with
    # display scales compressed (see DISPLAY_EXPONENT) the readability problem
    # that a size sort was solving no longer exists.
    assets.sort(key=lambda item: (item[0]["sort_time"], item[0]["asset_name"]))

    cursor = 0.0
    row_y = 0.0
    row_depth = 0.0
    row_index = 0
    span = 0.0
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

        imported_materials = {
            slot.material
            for obj in new
            for slot in obj.material_slots
            if slot.material is not None
        }
        image_nodes, linked_nodes = wire_imported_material_uvs(imported_materials)
        print(f"{glb.name}: explicit UV links {linked_nodes}/{image_nodes}")

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
        # Compressed for this scene only; REAL_SIZES.json and the FBX export
        # keep the true metres.
        shown = display_metres(metres)
        factor *= shown / metres if metres > 0 else 1.0
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
        depth = hi.y - lo.y

        # Wrap into rows instead of one long line. At true scale the set runs
        # from a 1.3 m frog to a 120 m titan, so a single row spanned 670 units:
        # framing all of it put the frog under a pixel, and framing the frog lost
        # everything else. Wrapping keeps the footprint roughly square, which is
        # what lets one viewport show every asset at a usable size.
        #
        # Greedy rather than balanced, because a width is only known after the
        # asset has been imported and scaled, and importing twice to plan the
        # layout would double the most expensive part of this script.
        if cursor > 0.0 and cursor + width > ROW_TARGET:
            row_y -= row_depth * (1.0 + GAP) + row_depth * 0.35
            cursor = 0.0
            row_depth = 0.0
            row_index += 1

        # Base on the floor, centred across its own row, left edge at the cursor.
        shift = Vector((cursor - lo.x, row_y - (lo.y + hi.y) / 2.0, -lo.z))
        for obj in new:
            obj.location += shift
        new[0].name = glb.stem
        for key, value in meta.items():
            if key != "sort_time":
                new[0][key] = value
        add_placard(meta, Vector((cursor, row_y, 0.0)), width)

        placed.append({"name": glb.stem, "x_start": round(cursor, 3),
                       "row": row_index, "y": round(row_y, 3),
                       "width": round(width, 3), "height": round(hi.z - lo.z, 3),
                       "target_metres": metres, "target_axis": axis,
                       "date": meta["date"], "rig": meta["rigging_status"]})
        cursor += max(width, width * (1.0 + GAP))
        row_depth = max(row_depth, depth)
        span = max(span, cursor)

    scene = bpy.context.scene
    tallest = max((p["height"] for p in placed), default=1.0)
    depth_total = abs(row_y) + row_depth

    light_data = bpy.data.lights.new("key", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("key", light_data)
    light.rotation_euler = (0.9, 0.0, 0.7)
    scene.collection.objects.link(light)

    # Frame on the bounds the assets ACTUALLY occupy, not on the counters used
    # to place them. The first version of this derived the camera from `span`
    # and the row offsets, and put the camera inside the moss titan: those
    # counters describe where each asset's left edge was written, not how far
    # the geometry then extended in depth and height around it.
    subjects = [o for o in scene.objects
                if o.type == "MESH" and o.name != "ground_grass"]
    low, high = bounds(subjects)
    centre = (low + high) * 0.5
    extent_x = high.x - low.x
    extent_y = high.y - low.y
    extent_z = high.z - low.z
    reach = math.sqrt(extent_x ** 2 + extent_y ** 2 + extent_z ** 2)

    camera_data = bpy.data.cameras.new("inspect")
    camera_data.lens = 50
    # The default far clip is far too near for a scene hundreds of units across,
    # and it silently deletes the back of the layout rather than warning.
    camera_data.clip_start = max(0.1, reach * 0.0005)
    camera_data.clip_end = reach * 8.0

    # Pull back until both the width and the height fit the frame, from the
    # lens and sensor rather than from a guessed multiplier. Depth counts too:
    # a row at the back is further from the camera than the centre is.
    half_h_fov = math.atan((camera_data.sensor_width * 0.5) / camera_data.lens)
    aspect = 2600.0 / 1400.0
    half_v_fov = math.atan(math.tan(half_h_fov) / aspect)
    need_x = (extent_x * 0.5) / math.tan(half_h_fov)
    need_z = (extent_z * 0.5) / math.tan(half_v_fov)
    distance = max(need_x, need_z) * 1.25 + extent_y * 0.5

    camera = bpy.data.objects.new("inspect", camera_data)
    # Looking DOWN at the layout, not across it. At ground level the front row
    # simply hides the rest: this set is 457 units deep and its tallest members
    # are 120 m, so a level camera renders two moss titans and eighteen things
    # behind them. Elevating puts every row on its own band of the frame.
    #
    # 40 degrees is the compromise. Shallower and the titans keep occluding;
    # steeper and the assets are seen from above, where a standing character
    # reads as a pair of shoulders and the height comparison the lineup exists
    # for stops being visible at all.
    elevation = math.radians(40.0)
    camera.location = (centre.x,
                       centre.y - distance * math.cos(elevation),
                       centre.z + distance * math.sin(elevation))
    direction = centre - Vector(camera.location)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(camera)
    scene.camera = camera
    print(f"layout {extent_x:.1f} x {extent_y:.1f} x {extent_z:.1f}, "
          f"camera back {distance:.1f}, clip_end {camera_data.clip_end:.0f}")

    # Colour management, which nothing was setting, so the scene inherited
    # Blender 5's AgX default. AgX is a film-emulation transform: it rolls off
    # and desaturates highlights hard, which is right for a lit narrative shot
    # and wrong for a catalogue. Under it the whole showroom rendered as pale
    # sage -- the grass looked untextured and the assets looked washed, and no
    # amount of work on the ground material could have fixed either, because
    # the variation was being flattened after shading rather than never
    # generated.
    #
    # Standard is a straight sRGB transform: what the shader computed is what
    # lands in the file, which is the property an inspection render needs.
    try:
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = -0.15
    except (AttributeError, TypeError) as exc:
        print(f"colour management unchanged: {exc}")
    print(f"view transform {scene.view_settings.view_transform}, "
          f"exposure {scene.view_settings.exposure}")

    add_sky(scene, light)
    add_ground(max(extent_x, extent_y), tallest)
    frame_viewport(centre, reach)

    # Verify the invariant before packing: every imported image texture must
    # have an explicit TexCoord UV -> Image Texture.Vector link.  The links
    # are created immediately after each glTF import above, rather than as a
    # late repair that can be omitted by a future builder change.
    image_nodes = [
        node
        for mat in bpy.data.materials
        if mat.use_nodes and mat.node_tree is not None
        for node in mat.node_tree.nodes
        if node.bl_idname == "ShaderNodeTexImage"
    ]
    linked_nodes = [
        node for node in image_nodes
        if node.inputs.get("Vector") is not None
        and node.inputs["Vector"].is_linked
        and node.inputs["Vector"].links[0].from_node.bl_idname == "ShaderNodeTexCoord"
        and node.inputs["Vector"].links[0].from_socket.name == "UV"
    ]
    print(f"final explicit UV links {len(linked_nodes)}/{len(image_nodes)}")
    if len(linked_nodes) != len(image_nodes):
        raise RuntimeError(
            f"unlinked imported image textures: {len(image_nodes) - len(linked_nodes)}"
        )

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
