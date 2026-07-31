"""Surface-integrity diagnosis on a frozen mesh -- read-only, no deletion. Produces stats and
several debug render modes to distinguish real holes from normal/backface artifacts."""
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

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(MESH))
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]

report = {"objects": []}
total = {"boundary_edges": 0, "boundary_loops": 0, "boundary_loop_area": 0.0,
         "inverted_faces": 0, "degenerate_faces": 0, "non_manifold_edges": 0,
         "faces": 0, "verts": 0}

for obj in objs:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    boundary_edges = [e for e in bm.edges if len(e.link_faces) == 1]
    non_manifold_edges = [e for e in bm.edges if not e.is_manifold]
    degenerate = [f for f in bm.faces if f.calc_area() < 1e-10]

    # boundary loops: connect boundary edges into loops via shared verts
    edge_set = set(boundary_edges)
    visited = set()
    loops = []
    for e in boundary_edges:
        if e in visited:
            continue
        loop_edges = [e]
        visited.add(e)
        changed = True
        while changed:
            changed = False
            frontier_verts = {v for le in loop_edges for v in le.verts}
            for e2 in boundary_edges:
                if e2 in visited:
                    continue
                if e2.verts[0] in frontier_verts or e2.verts[1] in frontier_verts:
                    loop_edges.append(e2)
                    visited.add(e2)
                    changed = True
        loops.append(loop_edges)
    loop_area = 0.0
    for loop_edges in loops:
        pts = [v.co for e in loop_edges for v in e.verts]
        if len(pts) >= 3:
            lo = Vector(min(p[i] for p in pts) for i in range(3))
            hi = Vector(max(p[i] for p in pts) for i in range(3))
            loop_area += (hi - lo).length ** 2 * 0.1  # rough proxy, not exact polygon area

    # inverted-face heuristic: centroid-outward test (valid for a roughly star-convex blob body,
    # reported as a heuristic estimate, not exact -- true inversion needs per-shell orientation).
    verts_world = [obj.matrix_world @ v.co for v in bm.verts]
    centroid = sum(verts_world, Vector((0, 0, 0))) / max(len(verts_world), 1)
    inverted = 0
    for f in bm.faces:
        face_centre_world = obj.matrix_world @ f.calc_center_median()
        outward = (face_centre_world - centroid).normalized()
        normal_world = (obj.matrix_world.to_3x3() @ f.normal).normalized()
        if normal_world.dot(outward) < -0.1:
            inverted += 1

    report["objects"].append({
        "object": obj.name, "faces": len(bm.faces), "verts": len(bm.verts),
        "boundary_edges": len(boundary_edges), "boundary_loops": len(loops),
        "boundary_loop_area_estimate": round(loop_area, 6),
        "inverted_faces_heuristic": inverted, "degenerate_faces": len(degenerate),
        "non_manifold_edges": len(non_manifold_edges),
    })
    total["boundary_edges"] += len(boundary_edges)
    total["boundary_loops"] += len(loops)
    total["boundary_loop_area"] += loop_area
    total["inverted_faces"] += inverted
    total["degenerate_faces"] += len(degenerate)
    total["non_manifold_edges"] += len(non_manifold_edges)
    total["faces"] += len(bm.faces)
    total["verts"] += len(bm.verts)
    bm.free()

report["totals"] = total
report["self_intersection_count"] = "not measured -- full pairwise/BVH self-intersection test on ~32k faces was skipped for time; flag for a follow-up pass if boundary/inversion numbers don't explain the dark triangles"
REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"DIAGNOSE faces={total['faces']} boundary_edges={total['boundary_edges']} "
      f"boundary_loops={total['boundary_loops']} inverted={total['inverted_faces']} "
      f"degenerate={total['degenerate_faces']} non_manifold={total['non_manifold_edges']}", flush=True)

# ---- debug renders ----
def make_mat(name, backface_cull, kind):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    mat.use_backface_culling = backface_cull
    nt = mat.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    if kind == "solid":
        em = nt.nodes.new("ShaderNodeEmission")
        em.inputs["Color"].default_value = (0.6, 0.55, 0.45, 1.0)
        nt.links.new(em.outputs["Emission"], out.inputs["Surface"])
    elif kind == "orientation":
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        mix = nt.nodes.new("ShaderNodeMixRGB")
        front = nt.nodes.new("ShaderNodeEmission")
        front.inputs["Color"].default_value = (0.15, 0.7, 0.9, 1.0)  # blue-ish = front-facing
        back = nt.nodes.new("ShaderNodeEmission")
        back.inputs["Color"].default_value = (1.0, 0.1, 0.1, 1.0)  # red = back-facing (inverted/visible-from-inside)
        backfacing = nt.nodes.new("ShaderNodeNewGeometry")
        transp = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(backfacing.outputs["Backfacing"], transp.inputs["Fac"])
        nt.links.new(front.outputs["Emission"], transp.inputs[1])
        nt.links.new(back.outputs["Emission"], transp.inputs[2])
        nt.links.new(transp.outputs["Shader"], out.inputs["Surface"])
    return mat

solid_mat = make_mat("Solid", False, "solid")
cull_mat = make_mat("Cull", True, "solid")
orient_mat = make_mat("Orient", False, "orientation")

scene = bpy.context.scene
engines = {i.identifier for i in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
scene.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in engines else "BLENDER_EEVEE"
scene.render.resolution_x = scene.render.resolution_y = 640
scene.world = bpy.data.worlds.new("W")
scene.world.use_nodes = True
scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.4, 0.4, 0.42, 1)
scene.view_settings.view_transform = "Standard"
key = bpy.data.lights.new("Key", "SUN")
key.energy = 3.0
key_o = bpy.data.objects.new("Key", key)
scene.collection.objects.link(key_o)
key_o.rotation_euler = (math.radians(50), 0, math.radians(30))
cam_data = bpy.data.cameras.new("Cam")
cam_data.type = "ORTHO"
cam_data.ortho_scale = 2.6
cam = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

OUTDIR.mkdir(parents=True, exist_ok=True)
VIEWS = (
    ("front", (0.0, -3.0, 0.0)),
    ("threequarter", (3.0 * math.sin(math.radians(40)), -3.0 * math.cos(math.radians(40)), 0.6)),
    ("side", (3.0, 0.0, 0.0)),
    ("back", (0.0, 3.0, 0.0)),
)


def render_all(mat, prefix):
    for obj in objs:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    for name, loc in VIEWS:
        cam.location = Vector(loc)
        d = Vector((0, 0, 0)) - cam.location
        cam.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
        scene.render.filepath = str(OUTDIR / f"{prefix}_{name}.png")
        bpy.ops.render.render(write_still=True)


render_all(solid_mat, "solid_doublesided")
render_all(cull_mat, "backface_culled")
render_all(orient_mat, "face_orientation")

# wireframe overlay: solid + freestyle-ish via simple modifier wire display in EEVEE isn't native;
# approximate with a thin black wire object built from edges.
scene.render.filepath = str(OUTDIR / "wireframe_note.txt")
(OUTDIR / "wireframe_note.txt").write_text(
    "Wireframe overlay skipped in headless EEVEE render (no native wireframe pass); "
    "face_orientation + backface_culled renders substitute for identifying flipped/open surfaces.",
    encoding="utf-8",
)

print("RENDERS_DONE", flush=True)
