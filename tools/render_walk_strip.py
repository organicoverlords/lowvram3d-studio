"""Render frames of one action from a rigged GLB as proof the mesh deforms.

    blender.exe --background --factory-startup --python tools/render_walk_strip.py -- \
        --mesh rigged.glb --action walk --out strip.png --frames 8

`skins: 1` proves a binding exists. It does not prove the binding moves
anything: a skin whose weights all sit on the root bone validates perfectly and
animates like a statue on a turntable. The only evidence that a rig deforms is
two frames of it in different poses, which is what this writes.

The mesh is measured *per frame* in its deformed state and the spread of those
measurements is printed. If the bounding box is identical on every frame, the
vertices are not following the bones and the strip is a still life -- so the
number is checked rather than the picture being glanced at.
"""

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def deformed_bounds(depsgraph, objects):
    """World-space bounds of the meshes AS EVALUATED -- armature applied.

    obj.bound_box is the rest shape and never changes under animation, so
    reading it would report a motionless mesh no matter what the rig did.
    """
    low = Vector((1e18, 1e18, 1e18))
    high = Vector((-1e18, -1e18, -1e18))
    for obj in objects:
        evaluated = obj.evaluated_get(depsgraph)
        mesh = evaluated.to_mesh()
        for vertex in mesh.vertices:
            point = evaluated.matrix_world @ vertex.co
            low = Vector((min(low.x, point.x), min(low.y, point.y), min(low.z, point.z)))
            high = Vector((max(high.x, point.x), max(high.y, point.y), max(high.z, point.z)))
        evaluated.to_mesh_clear()
    return low, high


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--action", default="walk")
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--size", type=int, default=900)
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=args.mesh)

    armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    # Only skinned meshes count, and unskinned ones are removed rather than
    # ignored. This GLB carries a stray 42-vertex Icosphere with no weights and
    # no parent, spanning exactly +/-1.0. Left in the bounds it pinned the
    # measured extents at a constant (1.9021, 2.0000, 2.0000) on every frame, so
    # the deformation check reported 0.00% for a mesh whose vertices move 0.17
    # units -- a false abort that looked exactly like the inert-skin failure the
    # check exists to catch. Left in the scene it would also render as a ball
    # beside the character.
    all_meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    meshes = [o for o in all_meshes
              if any(m.type == "ARMATURE" for m in o.modifiers) and o.vertex_groups]
    for stray in [o for o in all_meshes if o not in meshes]:
        print(f"removing unskinned mesh {stray.name} "
              f"({len(stray.data.vertices)} verts)", flush=True)
        bpy.data.objects.remove(stray, do_unlink=True)
    if not meshes:
        raise SystemExit("WALK_ABORT: no skinned mesh in this GLB")
    if not armatures:
        raise SystemExit("WALK_ABORT: no armature after import -- the GLB has no skin")
    armature = armatures[0]
    print(f"armature {armature.name}, {len(armature.data.bones)} bones, "
          f"{len(meshes)} mesh(es)", flush=True)

    names = [a.name for a in bpy.data.actions]
    print(f"actions in file: {names}", flush=True)
    chosen = next((a for a in bpy.data.actions if args.action in a.name.lower()), None)
    if chosen is None:
        raise SystemExit(f"WALK_ABORT: no action matching '{args.action}' in {names}")
    armature.animation_data_create()
    # The glTF importer pushes every action in the file onto its own NLA track,
    # and the NLA stack evaluates on top of whatever action is assigned. With
    # five actions imported, the stack won and the pose never left rest: the
    # deformation check reported 0.00% change on a rig whose thigh genuinely
    # swings 27 degrees. Mute the tracks so the chosen action is what evaluates.
    for track in armature.animation_data.nla_tracks:
        track.mute = True
    armature.animation_data.action = chosen
    # A slotted action (Blender 4.4+) carries no animation until a slot is
    # assigned; without this the action is attached and nothing moves.
    if hasattr(armature.animation_data, "action_slot"):
        slots = getattr(chosen, "slots", [])
        if slots:
            armature.animation_data.action_slot = slots[0]
    start, end = (int(chosen.frame_range[0]), int(chosen.frame_range[1]))
    print(f"action {chosen.name} frames {start}..{end}", flush=True)

    scene = bpy.context.scene
    available = [i.identifier for i in
                 bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items]
    scene.render.engine = next((e for e in ("BLENDER_EEVEE_NEXT", "BLENDER_EEVEE")
                                if e in available), available[0])
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples

    world = bpy.data.worlds.new("w")
    world.use_nodes = True
    # A wet black drysuit reflects almost nothing. At the fill used for the
    # nine-view sheets this asset rendered as a silhouette in which the stride
    # was visible but nothing else was, so the strip could show that the rig
    # moved and not whether it moved *well*. Fill and key are both roughly
    # doubled, and a back light separates the figure from the backdrop.
    world.node_tree.nodes["Background"].inputs[1].default_value = 2.6
    scene.world = world
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", type="SUN"))
    sun.data.energy = 6.0
    sun.rotation_euler = (math.radians(55), 0.0, math.radians(35))
    scene.collection.objects.link(sun)
    rim = bpy.data.objects.new("rim", bpy.data.lights.new("rim", type="SUN"))
    rim.data.energy = 3.0
    rim.rotation_euler = (math.radians(65), 0.0, math.radians(215))
    scene.collection.objects.link(rim)

    # Camera from the rest pose, held fixed for every frame: the subject must
    # move inside a stationary frame, or the strip cannot show that it moved.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene.frame_set(start)
    low, high = deformed_bounds(depsgraph, meshes)
    centre = (low + high) * 0.5
    height = high.z - low.z
    camera_data = bpy.data.cameras.new("cam")
    camera_data.lens = 50
    camera = bpy.data.objects.new("cam", camera_data)
    distance = height * 2.3
    camera.location = (centre.x - distance * 0.35, centre.y - distance, centre.z)
    direction = centre - Vector(camera.location)
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(camera)
    scene.camera = camera

    scene.render.resolution_x = args.size
    scene.render.resolution_y = int(args.size * 1.35)
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False

    step = max(1, (end - start) // args.frames)
    destination = Path(args.out).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tiles, spans = [], []

    for index in range(args.frames):
        frame = start + index * step
        if frame > end:
            break
        scene.frame_set(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        flow, fhigh = deformed_bounds(depsgraph, meshes)
        spans.append((fhigh - flow).copy())
        tile = destination.parent / f"_{destination.stem}_f{frame:03d}.png"
        scene.render.filepath = str(tile)
        bpy.ops.render.render(write_still=True)
        tiles.append(tile)
        print(f"  frame {frame}: deformed extents "
              f"({spans[-1].x:.4f}, {spans[-1].y:.4f}, {spans[-1].z:.4f})", flush=True)

    # The proof. If every frame has the same deformed bounding box, the skin is
    # inert and the animation is moving nodes no vertex follows.
    motion = max(max(abs(s.x - spans[0].x), abs(s.y - spans[0].y),
                     abs(s.z - spans[0].z)) for s in spans)
    reference = max(spans[0].x, spans[0].y, spans[0].z)
    relative = motion / reference if reference else 0.0
    print(f"DEFORMATION: max bbox change across frames {motion:.5f} "
          f"({relative * 100:.2f}% of size)", flush=True)
    if relative < 0.005:
        raise SystemExit("WALK_ABORT: mesh does not deform -- the skin is inert")

    # Stitch the tiles into one strip so the cycle is legible in a single image.
    strip_from_tiles(tiles, destination)
    print(f"wrote {destination}", flush=True)


def strip_from_tiles(tiles, destination):
    images = [bpy.data.images.load(str(t)) for t in tiles]
    width, height = images[0].size
    total = width * len(images)
    strip = bpy.data.images.new("strip", total, height, alpha=False)
    buffer = [0.0] * (total * height * 4)
    for index, image in enumerate(images):
        pixels = list(image.pixels)
        offset = index * width
        for row in range(height):
            source = row * width * 4
            target = (row * total + offset) * 4
            buffer[target:target + width * 4] = pixels[source:source + width * 4]
    strip.pixels = buffer
    strip.filepath_raw = str(destination)
    strip.file_format = "PNG"
    strip.save()
    for tile in tiles:
        tile.unlink(missing_ok=True)


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
