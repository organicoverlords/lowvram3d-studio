"""Measure-only component analysis: BVH nearest-surface distance to the single main body
component, no deletion, no transitive growth. Also renders a colour-coded debug contact sheet
so components can be reviewed visually before anything is removed.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:]
MESH, REPORT, RENDERDIR = Path(argv[0]), Path(argv[1]), Path(argv[2])
WELD_DISTANCE = 4e-4
ATTACH_DISTANCE_FRACTION = 0.006

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(MESH))
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]

scene_pts = [obj.matrix_world @ v.co for obj in objs for v in obj.data.vertices]
scene_lo = Vector(min(p[i] for p in scene_pts) for i in range(3))
scene_hi = Vector(max(p[i] for p in scene_pts) for i in range(3))
scene_diag = (scene_hi - scene_lo).length
attach_distance = scene_diag * ATTACH_DISTANCE_FRACTION
total_faces_all = 0

per_obj = []
all_components = []
for oi, obj in enumerate(objs):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=WELD_DISTANCE)
    bm.faces.ensure_lookup_table()
    total_faces_all += len(bm.faces)

    comp_id: dict[int, int] = {}
    comps: list[list] = []
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
    per_obj.append({"obj": obj, "bm": bm, "comps": comps})
    for ci, members in enumerate(comps):
        area = sum(f.calc_area() for f in members)
        verts_world = [obj.matrix_world @ v.co for f in members for v in f.verts]
        lo = Vector(min(p[i] for p in verts_world) for i in range(3))
        hi = Vector(max(p[i] for p in verts_world) for i in range(3))
        centroid = sum(verts_world, Vector((0, 0, 0))) / max(len(verts_world), 1)
        all_components.append({
            "obj_index": oi, "object": obj.name, "component_id": ci,
            "faces": len(members), "area": area, "bbox_diag": (hi - lo).length,
            "centroid": list(centroid), "world_verts": verts_world,
        })

body = max(all_components, key=lambda c: c["faces"])
body_verts_set = set(id(v) for v in body["world_verts"])
total_area_all = sum(c["area"] for c in all_components)

# BVH built ONLY from the main body -- single-hop distance, no transitive growth.
body_tris = []
vs = body["world_verts"]
for t in range(0, len(vs), 3):
    body_tris.append((t, t + 1, t + 2))
body_bvh = BVHTree.FromPolygons(vs, body_tris)

rows = []
for c in all_components:
    if c is body:
        continue
    sample = c["world_verts"][:: max(1, len(c["world_verts"]) // 40)] or c["world_verts"]
    dists = [body_bvh.find_nearest(p)[3] for p in sample]
    dists = [d for d in dists if d is not None]
    nearest = min(dists) if dists else float("inf")
    contact_ratio = sum(1 for d in dists if d <= attach_distance) / max(len(dists), 1)
    area_ratio = c["area"] / max(total_area_all, 1e-9)
    size_ratio = c["faces"] / max(total_faces_all, 1)

    small = area_ratio < 0.01 and size_ratio < 0.01 and c["bbox_diag"] < scene_diag * 0.05
    far = nearest > attach_distance * 3
    no_contact = contact_ratio < 0.05
    if small and far and no_contact:
        action = "REMOVE"
    elif small and (far or no_contact):
        action = "REVIEW"
    else:
        action = "KEEP"

    rows.append({
        "object": c["object"], "component_id": c["component_id"], "faces": c["faces"],
        "area_ratio": round(area_ratio, 6), "size_ratio": round(size_ratio, 6),
        "nearest_surface_distance": round(nearest, 5) if math.isfinite(nearest) else None,
        "distance_as_model_ratio": round(nearest / scene_diag, 5) if math.isfinite(nearest) else None,
        "contact_vertex_ratio": round(contact_ratio, 4),
        "proposed_action": action,
    })

rows.sort(key=lambda r: (r["proposed_action"] != "REMOVE", r["proposed_action"] != "REVIEW", -r["faces"]))
report = {
    "scene_diag": scene_diag, "attach_distance": attach_distance,
    "body_object": body["object"], "body_component_id": body["component_id"], "body_faces": body["faces"],
    "total_components": len(all_components), "components": rows,
}
REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
counts = {a: sum(1 for r in rows if r["proposed_action"] == a) for a in ("KEEP", "REVIEW", "REMOVE")}
print(f"MEASURE total={len(all_components)} {counts}", flush=True)

# ---- colour-coded debug render: each component gets a distinct bright vertex colour ----
random.seed(7)
palette_mat = {}
component_key_by_face = {}
for oi, info in enumerate(per_obj):
    obj, bm, comps = info["obj"], info["bm"], info["comps"]
    color_layer = bm.loops.layers.color.new("DebugColor") if bm.loops.layers.color.active is None else bm.loops.layers.color.active
    for ci, members in enumerate(comps):
        key = (obj.name, ci)
        if key == (body["object"], body["component_id"]):
            color = (0.55, 0.55, 0.55, 1.0)
        else:
            random.seed(hash(key) & 0xFFFF)
            color = (random.uniform(0.2, 1.0), random.uniform(0.2, 1.0), random.uniform(0.2, 1.0), 1.0)
        for f in members:
            for loop in f.loops:
                loop[color_layer] = color
    bm.to_mesh(obj.data)
    bm.free()
    mat = bpy.data.materials.new(f"Debug_{obj.name}")
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    vc = nt.nodes.new("ShaderNodeVertexColor")
    vc.layer_name = "DebugColor"
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(vc.outputs["Color"], emit.inputs["Color"])
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    obj.data.materials.clear()
    obj.data.materials.append(mat)

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

RENDERDIR.mkdir(parents=True, exist_ok=True)
for name, loc in (
    ("front", (0.0, -3.0, 0.0)),
    ("threequarter", (3.0 * math.sin(math.radians(40)), -3.0 * math.cos(math.radians(40)), 0.6)),
    ("side", (3.0, 0.0, 0.0)),
    ("back", (0.0, 3.0, 0.0)),
):
    cam.location = Vector(loc)
    d = Vector((0, 0, 0)) - cam.location
    cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = str(RENDERDIR / f"debug_components_{name}.png")
    bpy.ops.render.render(write_still=True)
    print(f"RENDERED {scene.render.filepath}", flush=True)
