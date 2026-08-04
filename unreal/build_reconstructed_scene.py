"""Assemble a viewable Unreal scene from a MoGe reconstruction.

This is the step that turns a GLB on disk into something an agent can render
and score. It imports the mesh, builds a material that actually shows the
reconstruction's vertex colours, places a camera reproducing the recovered
field of view, and saves the map.

The material matters more than it looks. A depth mesh carries its appearance in
vertex colours, and Unreal's default material ignores them entirely -- the mesh
imports "successfully" and renders black, which is the same class of failure
that made every earlier capture in this project look like a broken renderer.
So the material is built explicitly: VertexColor -> Emissive on an unlit
shading model, which needs no lighting rig to be inspectable.

Axis convention: `moge_reconstruct` exports glTF (Y up, Z toward the viewer),
and Unreal's importer maps glTF -Z onto +X. A camera looking down -Z in the
reconstruction therefore looks down +X here, which is yaw 0.

Configure with a `SCENE_REQUEST` global.

    python -m uemcp python @unreal/build_reconstructed_scene.py --json
"""

import json
import os

import unreal

REQUEST = globals().get("SCENE_REQUEST") or {}

GLB_PATH = REQUEST["glb"]
SCENE_ID = REQUEST.get("scene_id", "reconstruction")
PACKAGE_ROOT = REQUEST.get("package_root", f"/Game/AgentProof/{SCENE_ID}")
MAP_PATH = REQUEST.get("map_path", f"{PACKAGE_ROOT}/Maps/L_{SCENE_ID}")
FOV = float(REQUEST.get("fov_deg", 90.0))
ASPECT = float(REQUEST.get("aspect_ratio", 4.0 / 3.0))
# MoGe works in metres and Unreal in centimetres, but the glTF importer already
# applies that 100x conversion. Scaling the actor again put a 544 m scene at
# 25 km across, which pushes almost all of it outside the source frustum.
UNIFORM_SCALE = float(REQUEST.get("actor_scale", 1.0))

MESH_DIR = f"{PACKAGE_ROOT}/Meshes"
MATERIAL_PATH = f"{PACKAGE_ROOT}/Materials/M_{SCENE_ID}_VertexColor"

report = {"schema_version": "reconstructed_scene_receipt_v1", "scene_id": SCENE_ID}

if not os.path.isfile(GLB_PATH):
    raise RuntimeError(f"reconstruction not found: {GLB_PATH}")

# -- material ---------------------------------------------------------------
material = unreal.load_asset(MATERIAL_PATH)
if material is None:
    tools = unreal.AssetToolsHelpers.get_asset_tools()
    package, name = MATERIAL_PATH.rsplit("/", 1)
    material = tools.create_asset(name, package, unreal.Material, unreal.MaterialFactoryNew())
    if material is None:
        raise RuntimeError(f"could not create {MATERIAL_PATH}")

material.set_editor_property("shading_model", unreal.MaterialShadingModel.MSM_UNLIT)
material.set_editor_property("two_sided", True)

if not unreal.MaterialEditingLibrary.get_material_property_input_node(
        material, unreal.MaterialProperty.MP_EMISSIVE_COLOR):
    vertex_color = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionVertexColor, -400, 0)
    # Pin naming is per-expression and a wrong name fails *silently*, leaving an
    # unconnected emissive and a black mesh. VertexColor only accepts the
    # default output; TextureSample accepts "RGB". Try the default first and
    # verify, rather than trusting the return value.
    for pin in ("", "RGB", "RGBA"):
        unreal.MaterialEditingLibrary.connect_material_property(
            vertex_color, pin, unreal.MaterialProperty.MP_EMISSIVE_COLOR)
        unreal.MaterialEditingLibrary.recompile_material(material)
        if unreal.MaterialEditingLibrary.get_material_property_input_node(
                material, unreal.MaterialProperty.MP_EMISSIVE_COLOR):
            break
    else:
        raise RuntimeError("could not connect VertexColor to Emissive Color")

unreal.EditorAssetLibrary.save_loaded_asset(material)
report["material"] = MATERIAL_PATH
report["emissive_connected"] = bool(
    unreal.MaterialEditingLibrary.get_material_property_input_node(
        material, unreal.MaterialProperty.MP_EMISSIVE_COLOR))

# -- import -----------------------------------------------------------------
task = unreal.AssetImportTask()
task.set_editor_property("filename", GLB_PATH)
task.set_editor_property("destination_path", MESH_DIR)
task.set_editor_property("automated", True)
task.set_editor_property("replace_existing", True)
task.set_editor_property("save", True)

unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
imported = [str(p) for p in (task.get_editor_property("imported_object_paths") or [])]
report["imported_object_paths"] = imported

meshes = [p for p in imported if unreal.load_asset(p) and
          isinstance(unreal.load_asset(p), unreal.StaticMesh)]
if not meshes:
    raise RuntimeError(f"import produced no StaticMesh; got {imported}")
mesh = unreal.load_asset(meshes[0])
report["static_mesh"] = meshes[0]

# The glTF importer enables Nanite by default. Nanite then reports (and, in
# several capture paths, renders) a coarse fallback proxy -- 1770 triangles for
# a 500k-triangle reconstruction -- which quietly discards the detail the depth
# stage worked to produce. Reconstructions are authored geometry, not source
# art, so keep them as authored.
try:
    nanite = mesh.get_editor_property("nanite_settings")
    nanite.set_editor_property("enabled", False)
    mesh.set_editor_property("nanite_settings", nanite)
    unreal.EditorAssetLibrary.save_loaded_asset(mesh)
    report["nanite_disabled"] = True
except Exception as exc:
    report["nanite_error"] = str(exc)

try:
    report["triangles"] = int(mesh.get_num_triangles(0))
    report["vertices"] = int(mesh.get_num_vertices(0))
except Exception:
    pass

# -- map --------------------------------------------------------------------
subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
subsystem.new_level(MAP_PATH)

actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

shell = actor_subsystem.spawn_actor_from_object(
    mesh, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
shell.set_actor_label(f"{SCENE_ID}_ReconstructedMesh")
shell.set_actor_scale3d(unreal.Vector(UNIFORM_SCALE, UNIFORM_SCALE, UNIFORM_SCALE))

component = shell.get_component_by_class(unreal.StaticMeshComponent)
component.set_material(0, material)

camera = actor_subsystem.spawn_actor_from_class(
    unreal.CameraActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
camera.set_actor_label(f"{SCENE_ID}_Camera_Source")
camera_component = camera.get_component_by_class(unreal.CameraComponent)
camera_component.set_editor_property("field_of_view", FOV)
camera_component.set_editor_property("aspect_ratio", ASPECT)
camera_component.set_editor_property("constrain_aspect_ratio", True)

subsystem.save_current_level()

origin, extent = shell.get_actor_bounds(False)
report.update({
    "classification": "PROVEN",
    "map": MAP_PATH,
    "shell_label": str(shell.get_actor_label()),
    "camera_label": str(camera.get_actor_label()),
    "camera_fov_deg": FOV,
    "camera_aspect_ratio": ASPECT,
    "import_uniform_scale": UNIFORM_SCALE,
    "bounds_origin": [float(origin.x), float(origin.y), float(origin.z)],
    "bounds_extent": [float(extent.x), float(extent.y), float(extent.z)],
})

result = json.dumps(report)
