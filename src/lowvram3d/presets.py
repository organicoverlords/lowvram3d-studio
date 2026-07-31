from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class AssetType(str, Enum):
    AUTO = "auto"
    AVATAR = "avatar"
    CHARACTER = "character"
    CREATURE = "creature"
    VEHICLE = "vehicle"
    PROP = "prop"
    BUILDING = "building"
    ROOM = "room"
    SCENE = "scene"
    LEVEL = "level"


class QualityPreset(str, Enum):
    BACKGROUND = "background"
    GAMEPLAY = "gameplay"
    HERO = "hero"


QUALITY_TRIANGLES = {
    QualityPreset.BACKGROUND: (8_000, 15_000, 12_000),
    QualityPreset.GAMEPLAY: (25_000, 50_000, 45_000),
    QualityPreset.HERO: (50_000, 80_000, 70_000),
}

WORLD_OBJECT_TARGETS = {
    QualityPreset.BACKGROUND: 6_000,
    QualityPreset.GAMEPLAY: 15_000,
    QualityPreset.HERO: 30_000,
}

LOD_RATIOS = {
    QualityPreset.BACKGROUND: (0.45,),
    QualityPreset.GAMEPLAY: (0.55, 0.25),
    QualityPreset.HERO: (0.65, 0.35),
}


@dataclass(frozen=True, slots=True)
class AssetProfile:
    asset_type: AssetType
    preserve_hierarchy: bool
    preserve_continuous_body: bool
    split_loose_parts: bool
    separate_material_regions: bool
    merge_small_fragments: bool
    detect_round_parts: bool
    generate_rig: bool
    rigid_rig: bool
    atlas_mode: str
    lightmap_uv: bool
    collision_mode: str
    lod_count: int
    spatial_chunking: bool
    target_triangles: int
    target_min: int
    target_max: int
    texture_size: int
    uv_padding_px: int
    max_parts: int
    budget_mode: str
    per_object_target: int
    planar_angle_deg: float
    lod_ratios: tuple[float, ...]
    studio_retopo_mode: str
    studio_uv_mode: str
    retopo_options: dict
    uv_options: dict
    cell_divisions: int
    export_strategy: str
    texture_strategy: str

    def to_dict(self) -> dict:
        data = asdict(self)
        data["asset_type"] = self.asset_type.value
        return data


_BASE = {
    AssetType.AVATAR: dict(
        preserve_hierarchy=False, preserve_continuous_body=True, split_loose_parts=True,
        separate_material_regions=False, merge_small_fragments=True, detect_round_parts=False,
        generate_rig=True, rigid_rig=False, atlas_mode="shared", lightmap_uv=False,
        collision_mode="capsule_hulls", lod_count=2, spatial_chunking=False, max_parts=48,
        budget_mode="total", planar_angle_deg=0.0, studio_retopo_mode="off",
        studio_uv_mode="single_object", cell_divisions=1, export_strategy="animated_human_avatar",
        texture_strategy="identity_preserving_bake",
    ),
    AssetType.CHARACTER: dict(
        preserve_hierarchy=False, preserve_continuous_body=True, split_loose_parts=True,
        separate_material_regions=False, merge_small_fragments=True, detect_round_parts=False,
        generate_rig=True, rigid_rig=False, atlas_mode="shared", lightmap_uv=False,
        collision_mode="capsule_hulls", lod_count=2, spatial_chunking=False, max_parts=48,
        budget_mode="total", planar_angle_deg=0.0, studio_retopo_mode="off",
        studio_uv_mode="single_object", cell_divisions=1, export_strategy="skeletal_asset", texture_strategy="bake_transfer",
    ),
    AssetType.CREATURE: dict(
        preserve_hierarchy=False, preserve_continuous_body=True, split_loose_parts=True,
        separate_material_regions=False, merge_small_fragments=True, detect_round_parts=False,
        generate_rig=True, rigid_rig=False, atlas_mode="shared", lightmap_uv=False,
        collision_mode="capsule_hulls", lod_count=2, spatial_chunking=False, max_parts=64,
        budget_mode="total", planar_angle_deg=0.0, studio_retopo_mode="off",
        studio_uv_mode="single_object", cell_divisions=1, export_strategy="skeletal_asset", texture_strategy="bake_transfer",
    ),
    AssetType.VEHICLE: dict(
        preserve_hierarchy=True, preserve_continuous_body=False, split_loose_parts=True,
        separate_material_regions=False, merge_small_fragments=True, detect_round_parts=True,
        generate_rig=True, rigid_rig=True, atlas_mode="shared", lightmap_uv=False,
        collision_mode="per_major_part", lod_count=2, spatial_chunking=False, max_parts=96,
        budget_mode="total", planar_angle_deg=1.5, studio_retopo_mode="single_object",
        studio_uv_mode="single_object", cell_divisions=1, export_strategy="rigid_hierarchy", texture_strategy="bake_transfer",
    ),
    AssetType.PROP: dict(
        preserve_hierarchy=True, preserve_continuous_body=False, split_loose_parts=True,
        separate_material_regions=False, merge_small_fragments=True, detect_round_parts=True,
        generate_rig=False, rigid_rig=False, atlas_mode="shared", lightmap_uv=False,
        collision_mode="convex", lod_count=2, spatial_chunking=False, max_parts=48,
        budget_mode="total", planar_angle_deg=2.5, studio_retopo_mode="single_object",
        studio_uv_mode="single_object", cell_divisions=1, export_strategy="static_asset", texture_strategy="bake_transfer",
    ),
    AssetType.BUILDING: dict(
        preserve_hierarchy=True, preserve_continuous_body=False, split_loose_parts=True,
        separate_material_regions=True, merge_small_fragments=True, detect_round_parts=False,
        generate_rig=False, rigid_rig=False, atlas_mode="shared", lightmap_uv=True,
        collision_mode="architectural", lod_count=2, spatial_chunking=False, max_parts=192,
        budget_mode="total", planar_angle_deg=3.0, studio_retopo_mode="single_object",
        studio_uv_mode="single_object", cell_divisions=1, export_strategy="modular_building", texture_strategy="bake_transfer",
    ),
    AssetType.ROOM: dict(
        preserve_hierarchy=True, preserve_continuous_body=False, split_loose_parts=True,
        separate_material_regions=True, merge_small_fragments=False, detect_round_parts=False,
        generate_rig=False, rigid_rig=False, atlas_mode="shared", lightmap_uv=True,
        collision_mode="room_shell", lod_count=1, spatial_chunking=False, max_parts=256,
        budget_mode="total", planar_angle_deg=3.0, studio_retopo_mode="single_object",
        studio_uv_mode="single_object", cell_divisions=1, export_strategy="room_kit", texture_strategy="bake_transfer",
    ),
    AssetType.SCENE: dict(
        preserve_hierarchy=True, preserve_continuous_body=False, split_loose_parts=False,
        separate_material_regions=False, merge_small_fragments=False, detect_round_parts=False,
        generate_rig=False, rigid_rig=False, atlas_mode="per_object", lightmap_uv=True,
        collision_mode="per_object", lod_count=1, spatial_chunking=True, max_parts=512,
        budget_mode="per_object", planar_angle_deg=2.0, studio_retopo_mode="off",
        studio_uv_mode="off", cell_divisions=4, export_strategy="scene_bundle", texture_strategy="preserve_existing",
    ),
    AssetType.LEVEL: dict(
        preserve_hierarchy=True, preserve_continuous_body=False, split_loose_parts=False,
        separate_material_regions=False, merge_small_fragments=False, detect_round_parts=False,
        generate_rig=False, rigid_rig=False, atlas_mode="preserve_or_per_object", lightmap_uv=True,
        collision_mode="per_cell", lod_count=1, spatial_chunking=True, max_parts=1024,
        budget_mode="per_object", planar_angle_deg=2.0, studio_retopo_mode="off",
        studio_uv_mode="off", cell_divisions=8, export_strategy="world_partition_bundle", texture_strategy="preserve_existing",
    ),
}


def infer_asset_type(prompt: str, filename: str = "") -> AssetType:
    text = f"{filename} {prompt}".lower()
    groups = [
        (AssetType.LEVEL, ("level", "map", "world")),
        (AssetType.ROOM, ("room", "interior")),
        (AssetType.BUILDING, ("building", "house", "hotel", "barn", "lighthouse")),
        (AssetType.VEHICLE, ("vehicle", "car", "truck", "bus", "ship", "train", "tank", "motorcycle")),
        (AssetType.AVATAR, ("avatar", "digital twin", "dancing version", "photo of me", "photoreal human")),
        (AssetType.CHARACTER, ("character", "human", "humanoid", "person")),
        (AssetType.CREATURE, ("creature", "animal", "monster", "beast", "bird", "raccoon")),
        (AssetType.SCENE, ("scene", "diorama", "landscape", "environment")),
    ]
    for asset_type, words in groups:
        if any(word in text for word in words):
            return asset_type
    return AssetType.PROP


def _retopo_options(asset_type: AssetType, target: int, quality: QualityPreset) -> dict:
    shell_resolution = 160 if quality is QualityPreset.BACKGROUND else 192 if quality is QualityPreset.GAMEPLAY else 256
    hard_surface = asset_type in {AssetType.VEHICLE, AssetType.PROP, AssetType.BUILDING, AssetType.ROOM}
    open_surface = asset_type in {AssetType.BUILDING, AssetType.ROOM}
    return {
        "target_faces": target,
        "quads": False,
        "watertight": not open_surface,
        "shell_resolution": shell_resolution,
        "shell_close_iter": 1,
        "shell_smooth": 0.8 if hard_surface else 1.25,
        "shell_taubin": 6 if hard_surface else 10,
        "shell_samples_per_pitch": 2.0,
        "max_memory_gb": 3.0,
        "adaptive": True,
        "remesh_iters": 8,
        "feature_deg": 25.0 if hard_surface else 40.0,
        "calibrate_passes": 1,
        "preserve_features": hard_surface,
        "feature_angle": 22.0 if hard_surface else 35.0,
        "project": not hard_surface,
        "project_iters": 8,
        "project_clamp": 1.25,
        "relax_strength": 0.25 if hard_surface else 0.35,
        "device": "cpu",
        "seed": 0,
    }


def _uv_options(asset_type: AssetType, size: int, padding: int) -> dict:
    if asset_type in {AssetType.AVATAR, AssetType.CHARACTER, AssetType.CREATURE}:
        return {
            "max_cone_deg": 70.0, "sharp_weight": 0.15, "min_faces": 36,
            "min_area_frac": 0.002, "fold_cap_deg": 105.0, "refine": True,
            "refine_target_faces": 120, "refine_ad_thresh": 1.42,
            "method": "arap", "arap_iters": 8,
            "resolution": size, "padding_texels": padding,
            "weld": True, "weld_tol_frac": 0.05,
        }
    if asset_type in {AssetType.BUILDING, AssetType.ROOM}:
        return {
            "max_cone_deg": 32.0, "sharp_weight": 0.8, "min_faces": 12,
            "min_area_frac": 0.001, "fold_cap_deg": 60.0, "refine": True,
            "refine_target_faces": 64, "refine_ad_thresh": 1.20,
            "method": "planar", "arap_iters": 0,
            "resolution": size, "padding_texels": padding,
            "weld": True, "weld_tol_frac": 0.03,
        }
    return {
        "max_cone_deg": 48.0, "sharp_weight": 0.6, "min_faces": 16,
        "min_area_frac": 0.002, "fold_cap_deg": 75.0, "refine": True,
        "refine_target_faces": 80, "refine_ad_thresh": 1.28,
        "method": "auto", "arap_iters": 3,
        "resolution": size, "padding_texels": padding,
        "weld": True, "weld_tol_frac": 0.05,
    }


def get_profile(
    asset_type: str | AssetType,
    quality: str | QualityPreset = QualityPreset.GAMEPLAY,
    texture_size: int | None = None,
    lod_enabled: bool = True,
    prompt: str = "",
    filename: str = "",
) -> AssetProfile:
    resolved_type = AssetType(asset_type)
    if resolved_type is AssetType.AUTO:
        resolved_type = infer_asset_type(prompt, filename)
    resolved_quality = QualityPreset(quality)
    target_min, target_max, target = QUALITY_TRIANGLES[resolved_quality]
    size = texture_size or (4096 if resolved_quality is QualityPreset.HERO else 2048)
    size = min(4096, max(512, int(size)))
    padding = max(4, round(size / 256))
    base = dict(_BASE[resolved_type])
    if not lod_enabled:
        base["lod_count"] = 0
    lod_ratios = LOD_RATIOS[resolved_quality][: base["lod_count"]]
    per_object_target = WORLD_OBJECT_TARGETS[resolved_quality] if base["budget_mode"] == "per_object" else target
    return AssetProfile(
        asset_type=resolved_type,
        target_triangles=target,
        target_min=target_min,
        target_max=target_max,
        texture_size=size,
        uv_padding_px=padding,
        per_object_target=per_object_target,
        lod_ratios=lod_ratios,
        retopo_options=_retopo_options(resolved_type, target, resolved_quality),
        uv_options=_uv_options(resolved_type, size, padding),
        **base,
    )
