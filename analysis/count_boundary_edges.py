import json
import sys
from pathlib import Path

import bmesh
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
MESH = Path(argv[0])

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=str(MESH))
objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]

total_faces = 0
total_boundary = 0
total_nonmanifold = 0
for obj in objs:
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    total_faces += len(bm.faces)
    total_boundary += sum(1 for e in bm.edges if len(e.link_faces) == 1)
    total_nonmanifold += sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()

print(f"COUNT mesh={MESH.name} faces={total_faces} boundary_edges={total_boundary} non_manifold_edges={total_nonmanifold}", flush=True)
