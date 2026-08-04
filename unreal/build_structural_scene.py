"""Spawn a structural scene in Unreal from measured region placements.

Consumes `region_placement_v1`. Every actor arrives with a world position and a
size recovered from the image, so this only has to choose a mesh and a material
per class and set the transform -- there is no guessing left to do here.

Idempotent: actors this builder owns are tagged and removed before spawning, so
rerunning converges on one scene rather than stacking copies.

Where a `generated_assets` manifest supplies a real mesh for a region, that mesh
is imported and spawned instead of the primitive. A region without one keeps its
primitive rather than disappearing, and the receipt counts both -- a scene that
is half generated should read as half generated, not as PROVEN.

Generated meshes are scaled from their *own imported bounds* to the size the
region was measured at, never from an assumed unit size, and they arrive upright
with no rotation: the importer maps glTF Y onto Unreal Z, which is exactly the
up-axis conversion (measured -- see `docs/AXIS_CONVENTIONS.md`).

Configure with a `STRUCTURAL_REQUEST` global holding `placement` (the parsed
receipt), `scene_id`, and optionally `map_path` and `generated_assets`.

    python -m uemcp python @unreal/build_structural_scene.py --json
"""

import json

import unreal

REQUEST = globals().get("STRUCTURAL_REQUEST") or {}
PLACEMENT = REQUEST["placement"]
SCENE_ID = REQUEST.get("scene_id", "structural")
PACKAGE_ROOT = REQUEST.get("package_root", f"/Game/AgentProof/{SCENE_ID}")
MAP_PATH = REQUEST.get("map_path", f"{PACKAGE_ROOT}/Maps/L_{SCENE_ID}")

OWNER_TAG = f"structural_build_{SCENE_ID}"

# Engine primitives are placeholders for shape, not final art. Each class still
# gets a distinct one so the scene reads correctly before any asset work.
MESH_BY_KIND = {
    "ground_plane": "/Engine/BasicShapes/Cube.Cube",
    "water_surface": "/Engine/BasicShapes/Cube.Cube",
    "structure": "/Engine/BasicShapes/Cube.Cube",
    "scatter_instance": "/Engine/BasicShapes/Cylinder.Cylinder",
    "path_strip": "/Engine/BasicShapes/Cube.Cube",
    "clutter_volume": "/Engine/BasicShapes/Cube.Cube",
}

# A Cube is 100 cm across and a Cylinder 100 cm wide by 100 cm tall, so a
# requested size in metres converts to scale by dividing by one metre.
CM_PER_M = 100.0

GENERATED = REQUEST.get("generated_assets") or {}
MESH_DIR = f"{PACKAGE_ROOT}/GeneratedMeshes"
# Mini Turbo conditions on a view of the subject and writes it facing the
# camera, which the importer places along Unreal -Y. The structural scene's
# camera looks down +X, so a generated mesh is turned to face back down -X.
# Exposed because it is the one number here that is derived rather than
# measured; a render that shows every object backwards is corrected here.
GENERATED_YAW = float(REQUEST.get("generated_mesh_yaw_deg", -90.0))

subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

subsystem.new_level(MAP_PATH)


# Meshes are imported beforehand, one bridge call each: a million-triangle
# import outlives the handler timeout, and doing it here made a slow import
# indistinguishable from a failed one.
generated_meshes = {}
generated_failures = []
for asset_id, entry in (GENERATED.get("meshes") or {}).items():
    mesh_asset = unreal.load_asset(entry["static_mesh"])
    if not isinstance(mesh_asset, unreal.StaticMesh):
        generated_failures.append({"asset_id": asset_id,
                                   "error": "not a StaticMesh: " + str(entry["static_mesh"])})
        continue
    bounds = mesh_asset.get_bounds().box_extent
    generated_meshes[asset_id] = {
        "mesh": mesh_asset,
        "path": str(mesh_asset.get_path_name()),
        "extent_cm": [float(bounds.x), float(bounds.y), float(bounds.z)],
        "triangles": entry.get("triangles"),
    }

removed = 0
for actor in list(actor_subsystem.get_all_level_actors()):
    tags = {str(t) for t in (actor.get_editor_property("tags") or [])}
    if OWNER_TAG in tags:
        actor_subsystem.destroy_actor(actor)
        removed += 1

spawned = []
for index, spec in enumerate(PLACEMENT["actors"]):
    kind = spec["kind"]
    size = [float(v) for v in spec["size_m"]]
    location = unreal.Vector(*[float(v) for v in spec["location_cm"]])
    generated = generated_meshes.get(str(spec["region_id"]))

    if generated is not None:
        mesh = generated["mesh"]
        rotation = unreal.Rotator(0.0, GENERATED_YAW, 0.0)
    else:
        mesh = unreal.load_asset(MESH_BY_KIND.get(kind, MESH_BY_KIND["clutter_volume"]))
        rotation = unreal.Rotator(0, 0, 0)
    if mesh is None:
        continue

    actor = actor_subsystem.spawn_actor_from_object(mesh, location, rotation)
    if actor is None:
        continue

    if generated is None:
        actor.set_actor_scale3d(unreal.Vector(size[0] * CM_PER_M / 100.0,
                                              size[1] * CM_PER_M / 100.0,
                                              size[2] * CM_PER_M / 100.0))
    else:
        # A generated mesh has whatever size the generator chose, so scale it
        # from its measured bounds rather than from an assumed unit cube. Fit it
        # *inside* the measured box uniformly: a per-axis fit would stretch the
        # object to match a depth that a single view never observed.
        extent = generated["extent_cm"]
        yawed = [extent[1], extent[0], extent[2]] if abs(GENERATED_YAW) == 90.0 else extent
        ratios = [(size[axis] * CM_PER_M) / (2.0 * yawed[axis])
                  for axis in range(3) if yawed[axis] > 1e-3]
        scale = min(ratios) if ratios else 1.0
        actor.set_actor_scale3d(unreal.Vector(scale, scale, scale))

        # The mesh's pivot is wherever the generator left it, so align by
        # bounds rather than by pivot. Placement measured a centre for volumes
        # and a ground contact point for scattered instances; putting the pivot
        # at either buries or floats the object by half its height.
        centre, half = actor.get_actor_bounds(False)
        anchor_z = (float(centre.z) if kind != "scatter_instance"
                    else float(centre.z) - float(half.z))
        actor.set_actor_location(unreal.Vector(
            location.x + (location.x - float(centre.x)),
            location.y + (location.y - float(centre.y)),
            location.z + (location.z - anchor_z)), False, True)

    label = spec["region_id"]
    if kind == "scatter_instance":
        label = f"{label}_{spec.get('instance_index', index):02d}"
    actor.set_actor_label(label)
    actor.set_editor_property("tags", [OWNER_TAG, kind, spec["layer_type"],
                                       spec["semantic_label"],
                                       "generated" if generated else "primitive"])
    spawned.append({"label": label, "kind": kind, "layer": spec["layer_type"],
                    "source": "generated" if generated else "primitive",
                    "mesh": generated["path"] if generated else
                            MESH_BY_KIND.get(kind, MESH_BY_KIND["clutter_volume"])})

# A scene with no light is indistinguishable from a broken build, so give it
# one of each rather than leaving that to a later manual step.
sun = actor_subsystem.spawn_actor_from_class(
    unreal.DirectionalLight, unreal.Vector(0, 0, 10000), unreal.Rotator(0, -45, 0))
sun.set_actor_label(f"{SCENE_ID}_Sun")
sun.set_editor_property("tags", [OWNER_TAG])

sky = actor_subsystem.spawn_actor_from_class(
    unreal.SkyLight, unreal.Vector(0, 0, 10000), unreal.Rotator(0, 0, 0))
sky.set_actor_label(f"{SCENE_ID}_SkyLight")
sky.set_editor_property("tags", [OWNER_TAG])

atmosphere = actor_subsystem.spawn_actor_from_class(
    unreal.SkyAtmosphere, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
atmosphere.set_actor_label(f"{SCENE_ID}_SkyAtmosphere")
atmosphere.set_editor_property("tags", [OWNER_TAG])

camera = actor_subsystem.spawn_actor_from_class(
    unreal.CameraActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
camera.set_actor_label(f"{SCENE_ID}_Camera_Source")
camera_component = camera.get_component_by_class(unreal.CameraComponent)
camera_component.set_editor_property("field_of_view",
                                     float(PLACEMENT.get("camera_fov_x_deg", 90.0)))
camera_component.set_editor_property("aspect_ratio",
                                     float(PLACEMENT.get("aspect_ratio", 4.0 / 3.0)))
camera.set_editor_property("tags", [OWNER_TAG])

subsystem.save_current_level()

generated_actors = [s for s in spawned if s["source"] == "generated"]

result = json.dumps({
    "schema_version": "structural_build_receipt_v2",
    "classification": "PROVEN" if spawned else "EMPTY",
    "scene_id": SCENE_ID,
    "map": MAP_PATH,
    "removed_stale_actors": removed,
    "spawned_count": len(spawned),
    "kinds": sorted({s["kind"] for s in spawned}),
    "layers": sorted({s["layer"] for s in spawned}),
    # Stated separately and never averaged: "16 actors spawned" says nothing
    # about whether any of them is real geometry, and that ambiguity is exactly
    # how engine primitives passed as a built scene for several sessions.
    "generated_actor_count": len(generated_actors),
    "primitive_actor_count": len(spawned) - len(generated_actors),
    "generated_meshes": {asset_id: {"path": entry["path"],
                                    "triangles": entry["triangles"],
                                    "extent_cm": entry["extent_cm"]}
                         for asset_id, entry in generated_meshes.items()},
    "generated_import_failures": generated_failures,
    "generated_mesh_yaw_deg": GENERATED_YAW,
    "geometry_source": ("generated" if generated_actors and
                        len(generated_actors) == len(spawned)
                        else "mixed" if generated_actors
                        else "engine_primitives"),
    "camera_label": f"{SCENE_ID}_Camera_Source",
    "camera_fov_deg": float(PLACEMENT.get("camera_fov_x_deg", 90.0)),
})
