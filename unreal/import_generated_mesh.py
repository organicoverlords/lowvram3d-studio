"""Import one generated GLB and report where it landed.

Kept separate from `build_structural_scene.py` on purpose. A Mini Turbo mesh is
routinely over a million triangles, and importing one takes longer than the
bridge's handler timeout -- the call raises while the editor carries on and
finishes the work. Doing that *inside* the scene build made a slow import look
like a failed one and dropped the mesh silently, so the scene came out as
primitives while every receipt still read PROVEN.

So: one import per call, and the caller treats a timeout as "ask the asset
registry whether it finished" rather than as a failure.

`imported_object_paths` is unreliable for the same family of reasons -- a cached
re-import reports nothing at all -- so the registry is the authority here.

Configure with a `MESH_IMPORT_REQUEST` global holding `glb` and `destination`.

    python -m uemcp python @unreal/import_generated_mesh.py --json
"""

import hashlib
import json
import os

import unreal

REQUEST = globals().get("MESH_IMPORT_REQUEST") or {}
GLB_PATH = REQUEST["glb"]
DESTINATION = REQUEST["destination"]
SOURCE_TAG = "lowvram3d.source_sha256"

report = {"schema_version": "generated_mesh_import_v1",
          "glb": GLB_PATH, "destination": DESTINATION}

if not os.path.isfile(GLB_PATH):
    raise RuntimeError("generated mesh not found: " + str(GLB_PATH))
report["glb_bytes"] = os.path.getsize(GLB_PATH)

digest = hashlib.sha256()
with open(GLB_PATH, "rb") as handle:
    for block in iter(lambda: handle.read(1 << 20), b""):
        digest.update(block)
SOURCE_SHA256 = digest.hexdigest()
report["source_sha256"] = SOURCE_SHA256


def find_static_mesh(path):
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    for asset in registry.get_assets_by_path(path, recursive=True):
        if str(asset.asset_class_path.asset_name) == "StaticMesh":
            return unreal.load_asset(str(asset.package_name))
    return None


existing = find_static_mesh(DESTINATION)
if REQUEST.get("query_only"):
    # The caller is polling after a handler timeout. Starting another import
    # here would queue a fresh copy of the one already running, every poll.
    report["query_only"] = True
    mesh = existing
elif (existing is not None and not REQUEST.get("force_reimport")
      and str(unreal.EditorAssetLibrary.get_metadata_tag(existing, SOURCE_TAG))
      == SOURCE_SHA256):
    # Reuse only when the asset came from *this* file. Regenerating a mesh
    # keeps its filename, so matching on path alone silently keeps the previous
    # generation: a scene rendered with a mesh that had been replaced an hour
    # earlier, and nothing in any receipt said so.
    mesh = existing
    report["reused_existing"] = True
else:
    if existing is not None:
        report["replaced_stale_import"] = str(
            unreal.EditorAssetLibrary.get_metadata_tag(existing, SOURCE_TAG)) or "untagged"
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", GLB_PATH)
    task.set_editor_property("destination_path", DESTINATION)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    report["reused_existing"] = False
    report["imported_object_paths"] = [
        str(p) for p in (task.get_editor_property("imported_object_paths") or [])]
    mesh = find_static_mesh(DESTINATION)

if mesh is None:
    raise RuntimeError("import produced no StaticMesh under " + DESTINATION)

# Nanite reports and renders a coarse fallback proxy in place of the real
# geometry: a 500k-triangle reconstruction previously measured as 1,770.
settings = mesh.get_editor_property("nanite_settings")
if bool(settings.get_editor_property("enabled")):
    settings.set_editor_property("enabled", False)
    mesh.set_editor_property("nanite_settings", settings)
    report["nanite_disabled"] = True
unreal.EditorAssetLibrary.set_metadata_tag(mesh, SOURCE_TAG, SOURCE_SHA256)
unreal.EditorAssetLibrary.save_loaded_asset(mesh)

extent = mesh.get_bounds().box_extent
report.update({
    "classification": "PROVEN",
    "static_mesh": str(mesh.get_path_name()),
    "package_name": str(mesh.get_path_name()).split(".")[0],
    "triangles": int(mesh.get_num_triangles(0)),
    "vertices": int(mesh.get_num_vertices(0)),
    "extent_cm": [float(extent.x), float(extent.y), float(extent.z)],
    "nanite_enabled": bool(
        mesh.get_editor_property("nanite_settings").get_editor_property("enabled")),
})

result = json.dumps(report)
