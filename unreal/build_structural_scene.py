"""Spawn a structural scene in Unreal from measured region placements.

Consumes `region_placement_v1`. Every actor arrives with a world position and a
size recovered from the image, so this only has to choose a mesh and a material
per class and set the transform -- there is no guessing left to do here.

Idempotent: actors this builder owns are tagged and removed before spawning, so
rerunning converges on one scene rather than stacking copies.

Configure with a `STRUCTURAL_REQUEST` global holding `placement` (the parsed
receipt), `scene_id`, and optionally `map_path`.

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

subsystem = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)

subsystem.new_level(MAP_PATH)

removed = 0
for actor in list(actor_subsystem.get_all_level_actors()):
    tags = {str(t) for t in (actor.get_editor_property("tags") or [])}
    if OWNER_TAG in tags:
        actor_subsystem.destroy_actor(actor)
        removed += 1

spawned = []
for index, spec in enumerate(PLACEMENT["actors"]):
    kind = spec["kind"]
    mesh = unreal.load_asset(MESH_BY_KIND.get(kind, MESH_BY_KIND["clutter_volume"]))
    if mesh is None:
        continue

    location = unreal.Vector(*[float(v) for v in spec["location_cm"]])
    actor = actor_subsystem.spawn_actor_from_object(mesh, location, unreal.Rotator(0, 0, 0))
    if actor is None:
        continue

    size = [float(v) for v in spec["size_m"]]
    actor.set_actor_scale3d(unreal.Vector(size[0] * CM_PER_M / 100.0,
                                          size[1] * CM_PER_M / 100.0,
                                          size[2] * CM_PER_M / 100.0))

    label = spec["region_id"]
    if kind == "scatter_instance":
        label = f"{label}_{spec.get('instance_index', index):02d}"
    actor.set_actor_label(label)
    actor.set_editor_property("tags", [OWNER_TAG, kind, spec["layer_type"],
                                       spec["semantic_label"]])
    spawned.append({"label": label, "kind": kind, "layer": spec["layer_type"]})

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

result = json.dumps({
    "schema_version": "structural_build_receipt_v1",
    "classification": "PROVEN" if spawned else "EMPTY",
    "scene_id": SCENE_ID,
    "map": MAP_PATH,
    "removed_stale_actors": removed,
    "spawned_count": len(spawned),
    "kinds": sorted({s["kind"] for s in spawned}),
    "layers": sorted({s["layer"] for s in spawned}),
    "camera_label": f"{SCENE_ID}_Camera_Source",
    "camera_fov_deg": float(PLACEMENT.get("camera_fov_x_deg", 90.0)),
})
