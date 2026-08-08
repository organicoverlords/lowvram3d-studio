"""Turn each delivered GLB into a .blend that opens by double-click.

Run inside Blender:
    blender.exe --background --factory-startup --python make_blend_files.py

A .blend is the only Blender file that opens without an import step. The GLB is
imported once here, the texture is packed into the file so it travels as one
artefact, and the result is saved. Opening it later is a file open, not a
pipeline.

Two adjustments are made on the way in, both about how the asset will actually
be used rather than how it was generated:

- glTF is Y-up by definition and Blender is Z-up. The importer already rotates
  for this, but it leaves the rotation on the object as an unapplied transform,
  which then propagates into anything exported downstream. It is applied here so
  the object's own axes are the ones the scene sees.
- The mesh is placed with its base at the origin. Generated meshes come out
  centred on their bounding box, so half the model sits below the floor.
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
OUT = SRC / "blender"


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SRC / "MANIFEST.json").read_text(encoding="utf-8"))
    written = []

    for asset in manifest["assets"]:
        glb = SRC / asset["name"]
        if not glb.exists():
            print("MISSING", glb.name)
            continue
        reset()
        bpy.ops.import_scene.gltf(filepath=str(glb))

        for mat in bpy.data.materials:
            if not mat.use_nodes or mat.node_tree is None:
                continue
            tc = next((n for n in mat.node_tree.nodes
                       if n.bl_idname == "ShaderNodeTexCoord"), None)
            for node in mat.node_tree.nodes:
                if node.bl_idname != "ShaderNodeTexImage":
                    continue
                vector = node.inputs.get("Vector")
                if vector is None:
                    continue
                # Do not trust the importer's implicit/generated coordinate
                # choice: Material Preview must use the mesh UV map explicitly.
                for link in list(vector.links):
                    mat.node_tree.links.remove(link)
                if tc is None:
                    tc = mat.node_tree.nodes.new("ShaderNodeTexCoord")
                mat.node_tree.links.new(tc.outputs["UV"], vector)

        objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if not objects:
            print("NO MESH", glb.name)
            continue

        for obj in objects:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
        # Bake the importer's Y-up to Z-up correction into the mesh data.
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

        # Apply the shared real-world scale before grounding the asset.
        corners = [obj.matrix_world @ Vector(c)
                   for obj in objects for c in obj.bound_box]
        lo = Vector((min(c.x for c in corners), min(c.y for c in corners),
                     min(c.z for c in corners)))
        hi = Vector((max(c.x for c in corners), max(c.y for c in corners),
                     max(c.z for c in corners)))
        factor, metres, axis = real_sizes.scale_for(
            glb.stem, (hi.x - lo.x, hi.y - lo.y, hi.z - lo.z))
        for obj in objects:
            obj.scale = (factor, factor, factor)
        bpy.context.view_layer.update()
        for obj in objects:
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        for obj in objects:
            obj.select_set(False)

        # Sit the asset on the floor rather than straddling it.
        lowest = min((obj.matrix_world @ v.co).z
                     for obj in objects for v in obj.data.vertices)
        for obj in objects:
            obj.location.z -= lowest

        obj = objects[0]
        obj.name = glb.stem
        if obj.data:
            obj.data.name = glb.stem

        # Pack textures so the .blend is self-contained. Without this it holds a
        # path back into evidence/deliverables and breaks the moment it moves.
        try:
            bpy.ops.file.pack_all()
        except RuntimeError as exc:
            print("pack warning", glb.name, exc)

        target = OUT / f"{glb.stem}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(target), compress=True)
        size = target.stat().st_size
        written.append({"blend": target.name, "from": glb.name, "bytes": size,
                        "meshes": len(objects)})
        print(f"{target.name:<50} {size/1e6:6.1f} MB  {len(objects)} mesh")

    (OUT / "MANIFEST.json").write_text(
        json.dumps({"schema": "lowvram3d_blend_deliverables_v1",
                    "note": ("Each .blend opens directly -- textures are packed "
                             "inside, Z-up applied, base sitting on the origin."),
                    "files": written}, indent=2), encoding="utf-8")
    print(f"\n{len(written)} .blend files in {OUT}")


main()
