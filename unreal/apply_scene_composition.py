"""Apply a pipeline-owned scene_content_manifest in Unreal.

This adapter contains no scene layout or actor-specific behavior.  It accepts
only the allowlisted asset/material classes represented by the manifest and
reports every mutation against the manifest hash.
"""

from __future__ import annotations

import json
from pathlib import Path

import unreal


_SCRIPT_FILE = globals().get("__file__")
DEFAULT_MANIFEST = (Path(_SCRIPT_FILE).resolve().parents[1] if _SCRIPT_FILE else Path.cwd()) / "evidence" / "latest-image-to-scene" / "scene_content_manifest.json"
MATERIALS = {
    "rock_or_ground": "/Game/JungleEnvironmentMegaPack/Materials/Instances/MI_JEM_Rock_Mossy.MI_JEM_Rock_Mossy",
    "local_architecture": "/Game/JungleEnvironmentMegaPack/Materials/Instances/MI_JEM_Cobblestone_Mossy.MI_JEM_Cobblestone_Mossy",
    "wood_or_local": "/Game/JungleEnvironmentMegaPack/Materials/Instances/MI_JEM_Wood_Planks_Wet.MI_JEM_Wood_Planks_Wet",
    "water": "/Game/JungleEnvironmentMegaPack/Materials/Instances/MI_JEM_Water_Shallow_Ripples.MI_JEM_Water_Shallow_Ripples",
    "biome_local": "/Game/JungleEnvironmentMegaPack/Materials/Instances/MI_JEM_Grass_Lush_Mixed.MI_JEM_Grass_Lush_Mixed",
}
MESHES = {
    "box": "/Engine/BasicShapes/Cube.Cube",
    "rock_cluster": "/Engine/BasicShapes/Cube.Cube",
    "cylinder": "/Engine/BasicShapes/Cylinder.Cylinder",
    "cone": "/Engine/BasicShapes/Cone.Cone",
    "foliage_cross": "/Engine/BasicShapes/Cube.Cube",
}
ALLOWED_EXTERNAL_MESH_PREFIXES = ("/Engine/BasicShapes/", "/Game/JungleEnvironmentMegaPack/Meshes/Foliage/")


def _read_manifest(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != "scene_content_manifest_v1":
        raise RuntimeError("scene_content_manifest_v1 is required")
    if value.get("classification") != "PROVEN" or value.get("manual_only_actor_count", 0):
        raise RuntimeError("manifest is not an executable proven composition")
    actors = value.get("actors")
    if not isinstance(actors, list):
        raise RuntimeError("manifest actors must be an array")
    return value


def _find(label: str, actors: unreal.EditorActorSubsystem):
    matches = [actor for actor in actors.get_all_level_actors() if str(actor.get_actor_label()) == label]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate manifest actor {label}")
    return matches[0] if matches else None


def _vector(values):
    return unreal.Vector(float(values[0]) * 100.0, float(values[1]) * 100.0, float(values[2]) * 100.0)


def _apply_static_actor(record: dict, actors: unreal.EditorActorSubsystem):
    actor_id = str(record["actor_id"])
    transform = record["world_transform"]
    geometry = record["geometry_parameters"]
    primitive = str(geometry.get("primitive", "box"))
    mesh_path = str(record.get("asset_path") or MESHES.get(primitive, ""))
    if mesh_path not in MESHES.values() and not mesh_path.startswith(ALLOWED_EXTERNAL_MESH_PREFIXES):
        raise RuntimeError(f"asset path is not allowlisted for {actor_id}")
    mesh = unreal.load_asset(mesh_path)
    if mesh is None:
        raise RuntimeError(f"missing mesh {mesh_path} for {actor_id}")
    actor = _find(actor_id, actors)
    created = actor is None
    if actor is None:
        actor = actors.spawn_actor_from_class(unreal.StaticMeshActor, _vector(transform["location_m"]), unreal.Rotator(0.0, 0.0, 0.0))
    if actor is None:
        raise RuntimeError(f"could not create {actor_id}")
    actor.set_actor_label(actor_id)
    actor.set_actor_location(_vector(transform["location_m"]), False, False)
    rotation = transform.get("rotation_deg", [0.0, 0.0, 0.0])
    actor.set_actor_rotation(unreal.Rotator(float(rotation[0]), float(rotation[1]), float(rotation[2])), False)
    scale = transform["scale_m"]
    actor.set_actor_scale3d(unreal.Vector(float(scale[0]), float(scale[1]), float(scale[2])))
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    if component is None:
        raise RuntimeError(f"missing mesh component for {actor_id}")
    component.set_static_mesh(mesh)
    material_class = str(record["material_class"])
    if material_class not in MATERIALS:
        raise RuntimeError(f"material class is not allowlisted for {actor_id}: {material_class}")
    material = unreal.load_asset(MATERIALS[material_class])
    if material is None:
        raise RuntimeError(f"missing material for {actor_id}: {material_class}")
    component.set_material(0, material)
    collision = str(record["collision_policy"]) not in {"none", "ignored"}
    component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS if collision else unreal.CollisionEnabled.NO_COLLISION)
    component.set_collision_profile_name("BlockAll" if collision else "NoCollision")
    component.set_editor_property("can_ever_affect_navigation", str(record["navigation_policy"]) in {"walkable", "walkable_openings"})
    actor.set_editor_property("tags", sorted({"scene_manifest_generated", "unpromoted", str(record["builder_id"]), str(record["semantic_region_id"])}))
    return {"actor_id": actor_id, "created": created, "class": str(actor.get_class().get_name()), "builder_id": record["builder_id"], "material_class": material_class, "location_m": [float(value) for value in transform["location_m"]], "rotation_deg": [float(value) for value in rotation], "scale_m": [float(value) for value in scale]}


def _apply_environment_actor(record: dict, actors: unreal.EditorActorSubsystem):
    actor_id = str(record["actor_id"])
    kind = str(record.get("geometry_parameters", {}).get("environment_kind", "atmosphere")).lower()
    actor_class = unreal.ExponentialHeightFog if "fog" in kind else unreal.SkyAtmosphere
    actor = _find(actor_id, actors)
    created = actor is None
    if actor is None:
        actor = actors.spawn_actor_from_class(actor_class, unreal.Vector(0.0, 0.0, 0.0), unreal.Rotator(0.0, 0.0, 0.0))
    if actor is None:
        raise RuntimeError(f"could not create environment actor {actor_id}")
    actor.set_actor_label(actor_id)
    actor.set_editor_property("tags", ["scene_manifest_generated", "unpromoted", "environment"])
    return {"actor_id": actor_id, "created": created, "class": str(actor.get_class().get_name()), "builder_id": record["builder_id"], "material_class": "environment", "assignment": "environment_class"}


def apply_manifest(manifest_path: str | Path = DEFAULT_MANIFEST, target_map: str | None = None) -> dict:
    path = Path(manifest_path)
    manifest = _read_manifest(path)
    level = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if target_map is not None and not bool(level.load_level(str(target_map))):
        raise RuntimeError(f"could not load target map {target_map}")
    actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    records = [_apply_environment_actor(record, actors) if record.get("semantic_class") in {"sky_or_ceiling", "lighting_source"} else _apply_static_actor(record, actors) for record in manifest["actors"]]
    if not bool(level.save_current_level()):
        raise RuntimeError("manifest composition save failed")
    receipt = {"schema_version": "scene_content_build_receipt_v1", "classification": "PROVEN", "manifest_hash": manifest["manifest_hash"], "actor_count": len(records), "records": records, "map": str(unreal.EditorLevelLibrary.get_editor_world().get_path_name()), "target_map_requested": target_map, "manual_only_actor_count": 0, "materials_pipeline_assigned": True, "transforms_pipeline_derived": True}
    output = path.parent / "scene_content_build_receipt.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for builder_id in ("terrain", "architecture", "water", "crossing", "vegetation", "environment"):
        layer_records = [item for item in records if item.get("builder_id") == builder_id]
        layer_receipt = {"schema_version": f"{builder_id}_build_receipt_v2", "classification": "PROVEN" if layer_records else "NOT_PROVEN", "builder_id": builder_id, "manifest_hash": manifest["manifest_hash"], "record_count": len(layer_records), "records": layer_records, "pipeline_owned": True, "manual_only_actor_count": 0}
        (path.parent / f"{builder_id}_build_receipt.json").write_text(json.dumps(layer_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    unreal.log("SCENE_COMPOSITION_MANIFEST_APPLIED=PROVEN")
    return receipt


if __name__ == "__main__":
    apply_manifest()
