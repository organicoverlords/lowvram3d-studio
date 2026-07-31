"""Render the proposed-REMOVE components in isolation (alone, and highlighted in context)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
MESH, REPORT, OUTDIR = Path(argv[0]), Path(argv[1]), Path(argv[2])
WELD_DISTANCE = 4e-4

report = json.loads(REPORT.read_text(encoding="utf-8"))
remove_keys = {(r["object"], r["component_id"]) for r in report["components"] if r["proposed_action"] == "REMOVE"}

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(MESH))
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]

highlight_mat = bpy.data.materials.new("Highlight")
highlight_mat.use_nodes = True
highlight_mat.node_tree.nodes.clear()
emit = highlight_mat.node_tree.nodes.new("ShaderNodeEmission")
emit.inputs["Color"].default_value = (1.0, 0.05, 0.9, 1.0)
out = highlight_mat.node_tree.nodes.new("ShaderNodeOutputMaterial")
highlight_mat.node_tree.links.new(emit.outputs["Emission"], out.inputs["Surface"])

base_mat = bpy.data.materials.new("Base")
base_mat.use_nodes = True
bsdf = base_mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = (0.55, 0.52, 0.45, 1.0)

isolate_objs = []
for obj in objs:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=WELD_DISTANCE)
    bm.faces.ensure_lookup_table()
    comp_id = {}
    comps = []
    for f in bm.faces:
        if f.index in comp_id:
            continue
        cid = len(comps)
        stack, members = [f], []
        comp_id[f.index] = cid
        while stack:
            cur = stack.pop()
            members.append(cur)
            for e in cur.edges:
                for nb in e.link_faces:
                    if nb.index not in comp_id:
                        comp_id[nb.index] = cid
                        stack.append(nb)
        comps.append(members)

    color_layer = bm.loops.layers.color.new("Hi") if bm.loops.layers.color.active is None else bm.loops.layers.color.active
    isolate_bm = bmesh.new()
    vmap = {}
    for ci, members in enumerate(comps):
        is_removable = (obj.name, ci) in remove_keys
        color = (1.0, 0.05, 0.9, 1.0) if is_removable else (0.55, 0.52, 0.45, 1.0)
        for f in members:
            for loop in f.loops:
                loop[color_layer] = color
            if is_removable:
                newverts = []
                for v in f.verts:
                    key = v.index
                    if key not in vmap:
                        vmap[key] = isolate_bm.verts.new(obj.matrix_world @ v.co)
                    newverts.append(vmap[key])
                try:
                    isolate_bm.faces.new(newverts)
                except ValueError:
                    pass
    bm.to_mesh(obj.data)
    bm.free()
    mat = bpy.data.materials.new(f"Hi_{obj.name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "Hi"
    em = nt.nodes.new("ShaderNodeEmission")
    o = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(vc.outputs["Color"], em.inputs["Color"])
    nt.links.new(em.outputs["Emission"], o.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)

    if len(isolate_bm.faces) > 0:
        mesh = bpy.data.meshes.new(f"iso_{obj.name}")
        isolate_bm.to_mesh(mesh)
        iso_obj = bpy.data.objects.new(f"iso_{obj.name}", mesh)
        bpy.context.scene.collection.objects.link(iso_obj)
        iso_obj.data.materials.append(highlight_mat)
        isolate_objs.append(iso_obj)
    isolate_bm.free()

scene = bpy.context.scene
engines = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
scene.render.resolution_x = scene.render.resolution_y = 640
scene.world = bpy.data.worlds.new("W")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.08, 0.08, 0.09, 1)
scene.view_settings.view_transform = "Standard"
cam_data = bpy.data.cameras.new("Cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 2.6
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

OUTDIR.mkdir(parents=True, exist_ok=True)

# in-context (main mesh + magenta highlight overlay)
for name, loc in (("front", (0.0, -3.0, 0.0)), ("threequarter", (3.0 * math.sin(math.radians(40)), -3.0 * math.cos(math.radians(40)), 0.6))):
    cam.location = Vector(loc)
    d = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(OUTDIR / f"incontext_{name}.png")
    bpy.ops.render.render(write_still=True)

# isolated only: hide the main objects
for obj in objs:
    obj.hide_render = True
for name, loc in (("front", (0.0, -3.0, 0.0)), ("threequarter", (3.0 * math.sin(math.radians(40)), -3.0 * math.cos(math.radians(40)), 0.6))):
    cam.location = Vector(loc)
    d = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(OUTDIR / f"isolated_only_{name}.png")
    bpy.ops.render.render(write_still=True)

print(f"ISOLATED_RENDER remove_count={len(remove_keys)}", flush=True)
