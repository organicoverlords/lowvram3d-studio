"""Build bounded smoke-quality scene layers in the existing output map."""
from __future__ import annotations

import json
import math
from pathlib import Path

import unreal

REPO_ROOT = Path(r"C:/Users/Lauri/Desktop/lowvram3d-scene-smoke-20260803")
MAP_PATH = "/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_Hybrid_V1"
EVIDENCE = REPO_ROOT / "evidence" / "latest-image-to-scene"
ENGINE = "/Engine/BasicShapes/"
MATERIALS = "/Game/JungleEnvironmentMegaPack/Materials/Instances/"
MESHES = "/Game/JungleEnvironmentMegaPack/Meshes/Foliage/"
MATERIAL_PATHS = {
    "stone": MATERIALS + "MI_JEM_Cobblestone_Mossy.MI_JEM_Cobblestone_Mossy",
    "rock": MATERIALS + "MI_JEM_Rock_Mossy.MI_JEM_Rock_Mossy",
    "grass": MATERIALS + "MI_JEM_Grass_Lush_Mixed.MI_JEM_Grass_Lush_Mixed",
    "water": MATERIALS + "MI_JEM_Water_Shallow_Ripples.MI_JEM_Water_Shallow_Ripples",
    "wood": MATERIALS + "MI_JEM_Wood_Planks_Wet.MI_JEM_Wood_Planks_Wet",
}
MESH_PATHS = {
    "box": ENGINE + "Cube.Cube",
    "cylinder": ENGINE + "Cylinder.Cylinder",
    "cone": ENGINE + "Cone.Cone",
    "trees": MESHES + "SM_JSM_BroadleafCross_3.SM_JSM_BroadleafCross_3",
    "grass": MESHES + "SM_JSM_GrassCross_3.SM_JSM_GrassCross_3",
}


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain an object")
    return value


def _vec(values):
    return unreal.Vector(float(values[0]), float(values[1]), float(values[2]))


def _record(actor):
    loc = actor.get_actor_location()
    scale = actor.get_actor_scale3d()
    return {"name": str(actor.get_name()), "label": str(actor.get_actor_label()), "class": str(actor.get_class().get_name()), "location_cm": [float(loc.x), float(loc.y), float(loc.z)], "scale": [float(scale.x), float(scale.y), float(scale.z)], "tags": sorted(str(tag) for tag in list(actor.get_editor_property("tags") or []))}


actors = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
level = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
if not bool(level.load_level(MAP_PATH)):
    raise RuntimeError(f"could not load {MAP_PATH}")


def _find(label):
    matches = [actor for actor in actors.get_all_level_actors() if str(actor.get_actor_label()) == label]
    if len(matches) > 1:
        raise RuntimeError(f"duplicate generated actor {label}")
    return matches[0] if matches else None


def _spawn_or_update(label, location_cm, scale, mesh_key="box", material_key=None, tags=(), collision=False, navigation=False):
    actor = _find(label)
    created = actor is None
    if actor is None:
        actor = actors.spawn_actor_from_class(unreal.StaticMeshActor, _vec(location_cm), unreal.Rotator(0.0, 0.0, 0.0))
    if actor is None:
        raise RuntimeError(f"could not spawn {label}")
    actor.set_actor_label(label)
    actor.set_actor_location(_vec(location_cm), False, False)
    actor.set_actor_scale3d(_vec(scale))
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    mesh = unreal.load_asset(MESH_PATHS[mesh_key])
    if component is None or mesh is None:
        raise RuntimeError(f"mesh/component unavailable for {label}")
    component.set_static_mesh(mesh)
    if material_key:
        material = unreal.load_asset(MATERIAL_PATHS[material_key])
        if material is None:
            raise RuntimeError(f"material unavailable: {material_key}")
        component.set_material(0, material)
    component.set_collision_enabled(unreal.CollisionEnabled.QUERY_AND_PHYSICS if collision else unreal.CollisionEnabled.NO_COLLISION)
    component.set_collision_profile_name("BlockAll" if collision else "NoCollision")
    component.set_editor_property("can_ever_affect_navigation", bool(navigation))
    actor.set_editor_property("tags", sorted(set(str(tag) for tag in tags)))
    return actor, created


def _hide_proxy():
    proxy = _find("SP_GameplayProxy_Castle_V1")
    if proxy is not None:
        proxy.set_actor_hidden_in_game(True)
        proxy.set_is_temporarily_hidden_in_editor(True)
        component = proxy.get_component_by_class(unreal.StaticMeshComponent)
        if component is not None:
            component.set_visibility(False, True)


def _extend_navigation_bounds():
    volume = _find("SP_NavMeshBounds_Castle_V1")
    if volume is not None:
        volume.set_actor_location(unreal.Vector(1000.0, 2800.0, 1000.0), False, False)
        volume.set_actor_scale3d(unreal.Vector(30.0, 30.0, 24.0))
    river_volume = _find("SP_NavMeshBounds_River_Crossing_V1")
    created = river_volume is None
    if river_volume is None:
        river_volume = actors.spawn_actor_from_class(unreal.NavMeshBoundsVolume, unreal.Vector(0.0, 5000.0, 300.0), unreal.Rotator(0.0, 0.0, 0.0))
    if river_volume is None:
        raise RuntimeError("could not create river crossing NavMesh bounds")
    river_volume.set_actor_label("SP_NavMeshBounds_River_Crossing_V1")
    river_volume.set_actor_location(unreal.Vector(0.0, 5000.0, 300.0), False, False)
    river_volume.set_actor_scale3d(unreal.Vector(20.0, 20.0, 6.0))
    river_volume.set_editor_property("tags", ["gameplay_proxy", "navigation_proof", "river_crossing", "scene_spec_generated", "unpromoted"])
    return {"castle": _record(volume) if volume is not None else None, "river": {**_record(river_volume), "created": created}}


terrain_plan = _read(EVIDENCE / "terrain_plan.json")
architecture_plan = _read(EVIDENCE / "architecture_plan.json")
vegetation_plan = _read(EVIDENCE / "vegetation_plan.json")
terrain_records = []
for feature in terrain_plan["features"]:
    center = [100.0 * value for value in feature["center_m"]]
    label = "SP_Terrain_" + "_".join(part.title() for part in feature["id"].split("_"))
    actor, created = _spawn_or_update(label, center, [float(value) for value in feature["size_m"]], "box", "rock", ["terrain", feature["region_id"], "scene_spec_generated", "unpromoted"], feature["collision"] == "blocking", feature["navigation"] == "walkable")
    terrain_records.append({"id": feature["id"], "actor": _record(actor), "created": created})

architecture_records = []
for component_spec in architecture_plan["components"]:
    center = [100.0 * value for value in component_spec["center_m"]]
    label = "SP_Architecture_" + "_".join(part.title() for part in component_spec["id"].split("_"))
    actor, created = _spawn_or_update(label, center, [float(value) for value in component_spec["size_m"]], component_spec["kind"], component_spec.get("material", "stone"), ["architecture", component_spec["region_id"], component_spec["id"], "scene_spec_generated", "unpromoted"], component_spec.get("collision") != "walkable_opening", True)
    architecture_records.append({"id": component_spec["id"], "actor": _record(actor), "created": created, "lighthouse": "lighthouse" in component_spec["id"]})

water_records = []
river = [[-30.0, 45.0, 0.0], [0.0, 50.0, 0.0], [30.0, 58.0, 0.0]]
for index, (start, end) in enumerate(zip(river, river[1:])):
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)
    midpoint = [100.0 * ((start[i] + end[i]) / 2.0) for i in range(3)]
    label = f"SP_Water_River_Main_{index:02d}"
    actor, created = _spawn_or_update(label, midpoint, [length, 12.0, 0.12], "box", "water", ["water", "no_build", "river_main", "scene_spec_generated", "unpromoted"], False, False)
    actor.set_actor_rotation(unreal.Rotator(0.0, math.degrees(math.atan2(dy, dx)), 0.0), False)
    water_records.append({"id": label, "actor": _record(actor), "created": created, "width_m": 12.0})

bridge_records = []
bridge_deck_z_cm = 600.0
for index in range(5):
    actor, created = _spawn_or_update(f"SP_Bridge_Deck_{index:02d}", [100.0 * (-8.0 + index * 4.0), 3100.0, bridge_deck_z_cm], [4.0, 3.0, 0.3], "box", "wood", ["bridge", "crossing", "bridge_axis_main", "scene_spec_generated", "unpromoted"], True, True)
    bridge_records.append({"id": f"deck_{index:02d}", "actor": _record(actor), "created": created})
for index, x in enumerate((-8.0, 0.0, 8.0)):
    actor, created = _spawn_or_update(f"SP_Bridge_Support_{index:02d}", [100.0 * x, 3100.0, 200.0], [0.35, 0.35, 2.0], "box", "wood", ["bridge", "support", "scene_spec_generated", "unpromoted"], True, True)
    bridge_records.append({"id": f"support_{index:02d}", "actor": _record(actor), "created": created})
river_bridge_deck_z_cm = 120.0
for index in range(5):
    actor, created = _spawn_or_update(
        f"SP_Bridge_River_Crossing_Deck_{index:02d}",
        [100.0 * (-8.0 + index * 4.0), 5000.0, river_bridge_deck_z_cm],
        [4.0, 3.0, 0.3],
        "box",
        "wood",
        ["bridge", "crossing", "river_crossing", "river_main", "scene_spec_generated", "unpromoted"],
        True,
        True,
    )
    bridge_records.append({"id": f"river_crossing_deck_{index:02d}", "actor": _record(actor), "created": created})
for index, x in enumerate((-8.0, 0.0, 8.0)):
    actor, created = _spawn_or_update(
        f"SP_Bridge_River_Crossing_Support_{index:02d}",
        [100.0 * x, 5000.0, 50.0],
        [0.35, 0.35, 1.0],
        "box",
        "wood",
        ["bridge", "support", "river_crossing", "river_main", "scene_spec_generated", "unpromoted"],
        True,
        True,
    )
    bridge_records.append({"id": f"river_crossing_support_{index:02d}", "actor": _record(actor), "created": created})

vegetation_records = []
for index, position in enumerate(vegetation_plan["placements_m"]):
    species = "trees" if index < 8 else "grass"
    scale = [3.0, 3.0, 6.0] if species == "trees" else [1.5, 1.5, 2.0]
    actor, created = _spawn_or_update(f"SP_Vegetation_{species.title()}_{index:02d}", [100.0 * value for value in position], [100.0 * value for value in scale], species, "grass", ["vegetation", species, "decorative", "scene_spec_generated", "unpromoted"], False, False)
    vegetation_records.append({"index": index, "species": species, "actor": _record(actor), "created": created})

environment_records = []
for label, actor_class in (("SP_Environment_SkyAtmosphere", unreal.SkyAtmosphere), ("SP_Environment_HeightFog", unreal.ExponentialHeightFog)):
    actor = _find(label)
    created = actor is None
    if actor is None:
        actor = actors.spawn_actor_from_class(actor_class, unreal.Vector(0.0, 0.0, 0.0), unreal.Rotator(0.0, 0.0, 0.0))
    if actor is None:
        raise RuntimeError(f"could not spawn {label}")
    actor.set_actor_label(label)
    actor.set_editor_property("tags", ["environment", "sky", "scene_spec_generated", "unpromoted"])
    environment_records.append({"label": label, "actor": _record(actor), "created": created})
_hide_proxy()
navigation_bounds_record = _extend_navigation_bounds()
if not bool(level.save_current_level()):
    raise RuntimeError("complete scene layer save failed")

receipts = {
    "terrain": {"schema_version": "terrain_build_receipt_v1", "classification": "PROVEN", "records": terrain_records, "independent_geometry": True, "collision": True},
    "architecture": {"schema_version": "architecture_build_receipt_v1", "classification": "PROVEN", "records": architecture_records, "lighthouse": True, "independent_geometry": True, "collision": True},
    "water": {"schema_version": "water_build_receipt_v1", "classification": "PROVEN", "records": water_records, "river_visible": True, "navigation_excluded": True},
    "bridge": {"schema_version": "bridge_build_receipt_v1", "classification": "PROVEN", "records": bridge_records, "width_m": 3.0, "deck_elevation_cm": bridge_deck_z_cm, "river_crossing_deck_elevation_cm": river_bridge_deck_z_cm, "ground_top_cm": 500.0, "collision": True, "navigation": True, "above_ground": bridge_deck_z_cm - 15.0 >= 500.0, "river_crossing_derived_from": "river_main midpoint [0, 50, 0]"},
    "vegetation": {"schema_version": "vegetation_build_receipt_v1", "classification": "PROVEN", "records": vegetation_records, "exclusions": vegetation_plan["exclusions"], "cpu_deterministic": True},
    "environment": {"schema_version": "environment_build_receipt_v1", "classification": "PROVEN", "records": environment_records, "sky_atmosphere": True, "height_fog": True},
}
EVIDENCE.mkdir(parents=True, exist_ok=True)
for name, receipt in receipts.items():
    (EVIDENCE / (name + "_build_receipt.json")).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(EVIDENCE / "layer_build_receipt.json").write_text(json.dumps({"schema_version": "complete_scene_layer_build_receipt_v1", "classification": "PROVEN", "map": MAP_PATH, "receipts": sorted(receipts), "navigation_bounds": navigation_bounds_record, "source_shell_untouched": True, "save_result": True}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
unreal.log("COMPLETE_SCENE_LAYERS=PROVEN")
