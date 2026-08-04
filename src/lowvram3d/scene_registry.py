"""Data-driven capability registry for image-to-scene construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BuilderCapability:
    layer_type: str
    representations: tuple[str, ...]
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...]
    actor_types: tuple[str, ...]
    collision_policy: str
    navigation_policy: str
    material_policy: str
    resource_budget: Mapping[str, int]
    proof_gates: tuple[str, ...]
    fallback: str


CAPABILITIES: dict[str, BuilderCapability] = {
    "terrain": BuilderCapability("terrain", ("editable_geometry", "procedural_module"), ("regions",), ("landmarks",), ("StaticMeshActor",), "blocking_when_playable", "walkable_when_playable", "rock_or_ground", {"triangles": 100000}, ("geometry", "collision"), "visual_shell"),
    "ground_surface": BuilderCapability("ground_surface", ("editable_geometry", "gameplay_proxy", "procedural_module"), ("regions",), ("landmarks",), ("StaticMeshActor",), "blocking", "walkable", "ground", {"triangles": 50000}, ("geometry", "collision", "navigation"), "bounded_proxy"),
    "architecture": BuilderCapability("architecture", ("editable_geometry", "procedural_module"), ("regions",), ("landmarks",), ("StaticMeshActor",), "blocking_by_region", "walkable_openings", "stone_or_local", {"triangles": 200000}, ("geometry", "collision"), "conservative_blocker"),
    "interior_structure": BuilderCapability("interior_structure", ("editable_geometry",), ("regions",), ("landmarks",), ("StaticMeshActor",), "room_boundaries", "walkable_floor", "interior", {"triangles": 150000}, ("geometry", "navigation"), "visual_shell"),
    "water": BuilderCapability("water", ("gameplay_proxy", "procedural_module"), ("regions",), ("splines",), ("StaticMeshActor",), "none", "excluded", "water", {"triangles": 20000}, ("visibility", "exclusion"), "visual_shell"),
    "crossing": BuilderCapability("crossing", ("procedural_module", "editable_geometry"), ("regions",), ("splines",), ("StaticMeshActor",), "blocking", "walkable", "wood_or_local", {"triangles": 50000}, ("geometry", "collision", "navigation"), "unresolved"),
    "road_or_path": BuilderCapability("road_or_path", ("procedural_module", "editable_geometry"), ("regions",), ("splines",), ("StaticMeshActor",), "blocking", "walkable", "path", {"triangles": 50000}, ("geometry", "navigation"), "unresolved"),
    "vegetation": BuilderCapability("vegetation", ("procedural_population", "procedural_module"), ("regions",), ("landmarks",), ("StaticMeshActor",), "none", "ignored", "biome_local", {"instances": 1000}, ("population", "exclusion"), "visual_shell"),
    "hero_object": BuilderCapability("hero_object", ("editable_geometry", "procedural_module"), ("regions",), ("landmarks",), ("StaticMeshActor",), "local", "local", "object_local", {"triangles": 100000}, ("geometry", "material"), "visual_shell"),
    "background_geometry": BuilderCapability("background_geometry", ("visual_shell", "editable_geometry"), ("regions",), ("depth_bands",), ("StaticMeshActor",), "none", "ignored", "background", {"triangles": 100000}, ("coverage",), "visual_shell"),
    "sky_or_ceiling": BuilderCapability("sky_or_ceiling", ("sky", "visual_shell", "editable_geometry"), ("regions",), ("camera",), ("SkyAtmosphere", "ExponentialHeightFog"), "none", "ignored", "environment", {"triangles": 10000}, ("environment",), "flat_environment"),
    "lighting": BuilderCapability("lighting", ("lighting",), ("camera",), ("regions",), ("DirectionalLight", "SkyLight"), "none", "ignored", "lighting", {"draw_calls": 10}, ("lighting",), "default_lighting"),
    "gameplay_boundary": BuilderCapability("gameplay_boundary", ("gameplay_proxy", "procedural_module"), ("camera",), ("regions",), ("BlockingVolume",), "blocking", "blocked", "debug", {"actors": 32}, ("collision",), "bounded_box"),
}


def _layer_type(region: Mapping[str, Any]) -> str | None:
    explicit = region.get("layer_type") or region.get("layer")
    if isinstance(explicit, str) and explicit in CAPABILITIES:
        return explicit
    semantic = str(region.get("semantic_class", "")).lower()
    representation = str(region.get("representation", "")).lower()
    tags = {str(tag).lower() for tag in region.get("tags", [])}
    if representation in {"procedural_population", "procedural_module"} and ("decorative" in tags or "population" in semantic):
        return "vegetation"
    if representation == "sky" or "background" in tags and "sky" in str(region.get("id", "")).lower():
        return "sky_or_ceiling"
    if representation in {"editable_mesh", "gameplay_proxy"} and "hero_structure" in tags:
        return "architecture"
    for candidate, needles in (
        ("water", ("water",)),
        ("vegetation", ("vegetation", "grass", "forest", "tree")),
        ("crossing", ("bridge", "crossing")),
        ("road_or_path", ("road", "path", "street")),
        ("architecture", ("architecture", "building", "castle", "tower", "wall")),
        ("interior_structure", ("room", "corridor", "interior", "ceiling")),
        ("hero_object", ("hero", "object", "vehicle")),
        ("sky_or_ceiling", ("sky", "cloud", "ceiling")),
        ("terrain", ("terrain", "cliff", "island", "landscape", "ground")),
    ):
        if any(needle in semantic for needle in needles) or any(needle in tags for needle in needles):
            return candidate
    if representation in {"visual_shell", "source_visible_shell"}:
        return "visual_shell"
    if region.get("walkable") or region.get("interactive"):
        return "ground_surface"
    return "background_geometry"


def select_builders(spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for region in spec.get("regions", []):
        layer_type = _layer_type(region)
        if layer_type == "visual_shell":
            selected.setdefault("background_geometry", {"region_ids": [], "reason": "source_shell_or_distant_region"})["region_ids"].append(region.get("id"))
            continue
        if layer_type not in CAPABILITIES:
            continue
        selected.setdefault(layer_type, {"region_ids": [], "reason": "scene_spec_semantics"})["region_ids"].append(region.get("id"))
    return selected


def builder_manifest(spec: Mapping[str, Any]) -> dict[str, Any]:
    selected = select_builders(spec)
    return {
        "schema_version": "scene_builder_manifest_v1",
        "classification": "PROVEN",
        "scene_id": spec.get("scene_id"),
        "selected": {
            name: {"capability": CAPABILITIES[name].__dict__, **details}
            for name, details in sorted(selected.items())
            if name in CAPABILITIES
        },
        "unknown_semantics": [region.get("id") for region in spec.get("regions", []) if _layer_type(region) is None],
        "selection_source": "SceneSpec regions, representation, semantic_class, tags, and explicit layer_type",
        "filename_not_used_for_selection": True,
    }
