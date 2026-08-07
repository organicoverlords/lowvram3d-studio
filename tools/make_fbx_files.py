"""Export each delivered GLB as an FBX with Unreal's axes and units.

Run inside Blender:
    blender.exe --background --factory-startup --python make_fbx_files.py

Unreal cannot open any file without importing it -- only a .uasset skips that,
and only a running editor can write one. So this produces the next best thing:
an FBX that lands correctly on drag-and-drop, with no rotation to fix and no
scale to type in.

What "correctly" means here, concretely:

- **Units.** Unreal's world unit is the centimetre, and the measured mapping for
  this project is x100. But x100 alone is not enough: these meshes are normalised
  to a unit bounding box, so one unit is not one metre -- it is whatever that
  subject happens to be. Applied flat, the whale imported at 106 cm and the snail
  at 81 cm. Each asset is therefore first scaled to its real size from
  REAL_SIZES.json, then x100. Both are baked into the mesh rather than left to
  the FBX exporter's unit options, which interact with the scene unit system
  differently between Blender versions -- baked geometry does not.

- **Axes.** glTF is Y-up, Blender and Unreal are both Z-up. The importer applies
  the Y-up correction as an object rotation, which is applied into the mesh here
  so the exported vertices are already in the target frame. The FBX is then
  written with Z up and -Y forward, which is Blender's own frame, so the
  exporter performs no further conversion.

- **Textures** are embedded in the FBX rather than referenced, so the file
  travels alone.

**This is unverified in Unreal.** No editor was running when it was written, so
the axis and scale claims rest on the measured mapping and on the exporter
settings below, not on an import that was observed to land correctly. Check one
asset before trusting the batch.
"""

import json
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).parent))
import real_sizes

REPO = Path(r"C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803")
SRC = REPO / "evidence" / "deliverables"
OUT = SRC / "unreal_fbx"

CM_PER_METRE = 100.0
# The x100 alone was never the whole story. It converts metres to Unreal's
# centimetres correctly, but the meshes are normalised to a unit bounding box,
# so "one unit" is not one metre -- it is whatever that subject happens to be.
# Applied flat, a whale imported at 106 cm and a snail at 81 cm. The real size
# comes from REAL_SIZES.json first; x100 turns it into Unreal units after.


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((SRC / "MANIFEST.json").read_text(encoding="utf-8"))
    written = []

    for asset in manifest["assets"]:
        glb = SRC / asset["name"]
        if not glb.exists():
            print("MISSING", glb.name)
            continue
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(glb))

        objects = [o for o in bpy.context.scene.objects if o.type == "MESH"]
        if not objects:
            print("NO MESH", glb.name)
            continue

        pre = objects[0].dimensions.copy()
        factor, metres, axis = real_sizes.scale_for(glb.stem, pre)
        unreal = factor * CM_PER_METRE
        for obj in objects:
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            obj.scale = (unreal, unreal, unreal)

        # Bake both the Y-up correction and the centimetre scale into the mesh,
        # so nothing downstream depends on an object-level transform surviving.
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)

        # Base on the origin. An asset centred on its bounding box spawns half
        # underground, which reads as a broken import.
        lowest = min((obj.matrix_world @ v.co).z
                     for obj in objects for v in obj.data.vertices)
        for obj in objects:
            obj.location.z -= lowest
        bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

        objects[0].name = glb.stem
        if objects[0].data:
            objects[0].data.name = glb.stem

        target = OUT / f"{glb.stem}.fbx"
        bpy.ops.export_scene.fbx(
            filepath=str(target),
            use_selection=False,
            apply_unit_scale=False,
            global_scale=1.0,
            apply_scale_options="FBX_SCALE_NONE",
            axis_forward="-Y",
            axis_up="Z",
            object_types={"MESH"},
            mesh_smooth_type="FACE",
            use_mesh_modifiers=True,
            path_mode="COPY",
            embed_textures=True,
            bake_space_transform=False,
        )
        size = target.stat().st_size
        dims = objects[0].dimensions
        written.append({
            "fbx": target.name, "from": glb.name, "bytes": size,
            "dimensions_cm": [round(d, 1) for d in dims],
            "target_metres": metres, "target_axis": axis,
        })
        print(f"{target.name:<52} {size/1e6:6.1f} MB  "
              f"{dims[0]/100:7.1f} x {dims[1]/100:7.1f} x {dims[2]/100:7.1f} m"
              f"   ({metres} m {axis})")

    (OUT / "MANIFEST.json").write_text(
        json.dumps({
            "schema": "lowvram3d_fbx_deliverables_v1",
            "target": "Unreal Engine",
            "units": "centimetres; scaled to REAL_SIZES.json then x100, applied to the mesh",
            "axes": "Z up, -Y forward, glTF Y-up correction baked into the mesh",
            "textures": "embedded in the FBX",
            "origin": "base of the mesh sits on the world origin",
            "unverified": ("No Unreal editor was running when these were "
                           "written. Axis and scale rest on the measured "
                           "mapping and the exporter settings, not on an "
                           "observed import. Check one before trusting all."),
            "files": written,
        }, indent=2), encoding="utf-8")
    print(f"\n{len(written)} FBX files in {OUT}")


main()
