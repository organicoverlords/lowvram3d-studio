"""Import the delivered FBX assets into the Showroom project and line them up.

Run this from inside the Unreal editor, with the Showroom project open:

    Window > Developer Tools > Output Log, switch the dropdown from Cmd to
    Python, then paste:

        exec(open(r"C:\\Users\\Lauri\\Desktop\\lowvram3d-scene-smoke-20260803\\tools\\unreal_showroom_import.py").read())

This route needs no plugin. It uses the editor's own Python API, which ships
with the Python Editor Script Plugin -- enable that in Edit > Plugins if the
Python option is missing from the Output Log dropdown. The alternative, driving
the editor over the UE_MCP_Bridge, would mean copying that plugin out of
UnrealAITest58 into Showroom and rebuilding a C++ project first.

The FBX files are already correct on import: Z-up, centimetres, real-world size
from REAL_SIZES.json, base sitting on the origin, textures embedded. So the
import settings below deliberately do NOT apply any further transform -- no
import rotation, no import uniform scale. Anything that "fixes" orientation here
would be double-correcting geometry that is already in Unreal's frame.
"""

import json
import os

import unreal

FBX_DIR = (r"C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803"
           r"\evidence\deliverables\unreal_fbx")
DEST = "/Game/LowVRAM3D"
LEVEL = "/Game/LowVRAM3D/Maps/AssetLineup"

GAP_FRACTION = 0.35  # of the neighbouring asset's width, as in the Blender row


def build_import_task(path):
    options = unreal.FbxImportUI()
    options.set_editor_property("import_mesh", True)
    options.set_editor_property("import_textures", True)
    options.set_editor_property("import_materials", True)
    options.set_editor_property("import_as_skeletal", False)
    options.set_editor_property("mesh_type_to_import",
                                unreal.FBXImportType.FBXIT_STATIC_MESH)

    static_mesh_data = options.static_mesh_import_data
    # No further transform. The FBX already carries Unreal's axes and units.
    static_mesh_data.set_editor_property("import_translation", unreal.Vector(0, 0, 0))
    static_mesh_data.set_editor_property("import_rotation", unreal.Rotator(0, 0, 0))
    static_mesh_data.set_editor_property("import_uniform_scale", 1.0)
    static_mesh_data.set_editor_property("combine_meshes", True)
    static_mesh_data.set_editor_property("generate_lightmap_u_vs", True)
    static_mesh_data.set_editor_property("auto_generate_collision", True)

    task = unreal.AssetImportTask()
    task.set_editor_property("filename", path)
    task.set_editor_property("destination_path", DEST)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    task.set_editor_property("options", options)
    return task


def main():
    files = sorted(f for f in os.listdir(FBX_DIR) if f.lower().endswith(".fbx"))
    if not files:
        unreal.log_error("no FBX files in %s" % FBX_DIR)
        return

    tasks = [build_import_task(os.path.join(FBX_DIR, f)) for f in files]
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks(tasks)

    imported = []
    for task in tasks:
        for path in task.get_editor_property("imported_object_paths") or []:
            asset = unreal.load_asset(path)
            if isinstance(asset, unreal.StaticMesh):
                imported.append(asset)
    unreal.log("imported %d static meshes" % len(imported))

    level_subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    level_subsystem.new_level(LEVEL)

    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    cursor = 0.0
    placed = []
    for mesh in sorted(imported, key=lambda m: m.get_name()):
        # Spacing from each asset's own footprint. These differ by more than an
        # order of magnitude -- a 1.3 m frog against a 40 m titan -- so a fixed
        # pitch either overlaps the large ones or strands the small ones.
        bounds = mesh.get_bounding_box()
        extent = bounds.max - bounds.min
        width = float(extent.x)

        actor = actor_subsystem.spawn_actor_from_object(
            mesh, unreal.Vector(cursor + width / 2.0, 0.0, 0.0))
        actor.set_actor_label(mesh.get_name())
        placed.append((mesh.get_name(), width / 100.0, float(extent.z) / 100.0))
        cursor += width * (1.0 + GAP_FRACTION)

    level_subsystem.save_current_level()

    for name, width_m, height_m in placed:
        unreal.log("  %-46s %7.1f m wide  %7.1f m tall" % (name, width_m, height_m))
    unreal.log("placed %d actors, row spans %.1f m" % (len(placed), cursor / 100.0))
    unreal.log("level saved to %s" % LEVEL)


main()
