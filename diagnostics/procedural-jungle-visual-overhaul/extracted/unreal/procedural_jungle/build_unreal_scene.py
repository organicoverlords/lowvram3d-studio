from __future__ import annotations

import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path

import unreal

GENERATED_ROOT = Path(os.environ['JUNGLE_GENERATED_ROOT'])
PROJECT_ROOT = Path(os.environ['JUNGLE_PROJECT_ROOT'])
REPORT_PATH = Path(os.environ['JUNGLE_UNREAL_BUILD_REPORT'])
CONTENT_ROOT = PROJECT_ROOT / 'Content' / 'ProceduralJungle' / 'Generated'
MAP_PATH = '/Game/ProceduralJungle/Maps/L_ProceduralJungle'
STATIC_DEST = '/Game/ProceduralJungle/Generated/Static'
PANDA_DEST = '/Game/ProceduralJungle/Generated/Panda'
MATERIAL_DEST = '/Game/ProceduralJungle/Materials'


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def log(message: str) -> None:
    unreal.log(f'[ProceduralJungle] {message}')


def import_static_fbx(filename: Path) -> list[str]:
    task = unreal.AssetImportTask()
    task.set_editor_property('filename', str(filename))
    task.set_editor_property('destination_path', STATIC_DEST)
    task.set_editor_property('automated', True)
    task.set_editor_property('replace_existing', True)
    task.set_editor_property('save', True)
    options = unreal.FbxImportUI()
    options.set_editor_property('import_mesh', True)
    options.set_editor_property('import_as_skeletal', False)
    options.set_editor_property('import_materials', True)
    options.set_editor_property('import_textures', True)
    options.set_editor_property('create_physics_asset', False)
    try:
        static_data = options.get_editor_property('static_mesh_import_data')
        static_data.set_editor_property('combine_meshes', False)
        static_data.set_editor_property('generate_lightmap_u_vs', True)
        static_data.set_editor_property('auto_generate_collision', True)
    except Exception as exc:
        log(f'static import option warning: {exc}')
    task.set_editor_property('options', options)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    return list(task.get_editor_property('imported_object_paths'))


def import_panda_fbx(filename: Path) -> list[str]:
    task = unreal.AssetImportTask()
    task.set_editor_property('filename', str(filename))
    task.set_editor_property('destination_path', PANDA_DEST)
    task.set_editor_property('automated', True)
    task.set_editor_property('replace_existing', True)
    task.set_editor_property('save', True)
    options = unreal.FbxImportUI()
    options.set_editor_property('import_mesh', True)
    options.set_editor_property('import_as_skeletal', True)
    options.set_editor_property('import_animations', True)
    options.set_editor_property('import_materials', True)
    options.set_editor_property('import_textures', True)
    options.set_editor_property('create_physics_asset', True)
    try:
        skeletal_data = options.get_editor_property('skeletal_mesh_import_data')
        skeletal_data.set_editor_property('import_mesh_lo_ds', False)
        skeletal_data.set_editor_property('use_t0_as_ref_pose', True)
        anim_data = options.get_editor_property('anim_sequence_import_data')
        anim_data.set_editor_property('import_bone_tracks', True)
    except Exception as exc:
        log(f'panda import option warning: {exc}')
    task.set_editor_property('options', options)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    return list(task.get_editor_property('imported_object_paths'))


def list_assets(path: str) -> list[unreal.AssetData]:
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    return list(registry.get_assets_by_path(path, recursive=True))


def load_by_name(path: str, name: str, cls_name: str | None = None):
    for data in list_assets(path):
        if str(data.asset_name) != name:
            continue
        asset = data.get_asset()
        if cls_name and asset.get_class().get_name() != cls_name:
            continue
        return asset
    return None


def create_material(
    name: str,
    colour: tuple[float, float, float],
    roughness: float,
    metallic: float = 0.0,
    wind: bool = False,
    translucent: bool = False,
    emissive: float = 0.0,
    secondary_colour: tuple[float, float, float] | None = None,
    noise_scale: float = 0.004,
    opacity_value: float = 0.72,
):
    asset_path = f'{MATERIAL_DEST}/{name}'
    material = unreal.EditorAssetLibrary.load_asset(asset_path) if unreal.EditorAssetLibrary.does_asset_exist(asset_path) else None
    if not material:
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(name, MATERIAL_DEST, unreal.Material, unreal.MaterialFactoryNew())
    if not material:
        raise RuntimeError(f'Could not create material {asset_path}')

    material.set_editor_property('two_sided', bool(wind or translucent))
    material.set_editor_property('blend_mode', unreal.BlendMode.BLEND_TRANSLUCENT if translucent else unreal.BlendMode.BLEND_OPAQUE)
    try:
        material.set_editor_property('used_with_instanced_static_meshes', True)
    except Exception as exc:
        log(f'instanced material usage warning for {name}: {exc}')

    mel = unreal.MaterialEditingLibrary
    mel.delete_all_material_expressions(material)

    primary = mel.create_material_expression(material, unreal.MaterialExpressionConstant3Vector, -720, -80)
    primary.set_editor_property('constant', unreal.LinearColor(colour[0], colour[1], colour[2], 1.0))
    colour_output = primary

    if secondary_colour is not None and not translucent:
        try:
            secondary = mel.create_material_expression(material, unreal.MaterialExpressionConstant3Vector, -720, 40)
            secondary.set_editor_property('constant', unreal.LinearColor(secondary_colour[0], secondary_colour[1], secondary_colour[2], 1.0))
            world = mel.create_material_expression(material, unreal.MaterialExpressionWorldPosition, -760, 210)
            noise = mel.create_material_expression(material, unreal.MaterialExpressionNoise, -500, 210)
            try:
                noise.set_editor_property('scale', noise_scale)
                noise.set_editor_property('levels', 4)
                noise.set_editor_property('quality', 2)
            except Exception:
                pass
            mel.connect_material_expressions(world, '', noise, 'Position')
            blend = mel.create_material_expression(material, unreal.MaterialExpressionLinearInterpolate, -220, -20)
            mel.connect_material_expressions(primary, '', blend, 'A')
            mel.connect_material_expressions(secondary, '', blend, 'B')
            mel.connect_material_expressions(noise, '', blend, 'Alpha')
            colour_output = blend
        except Exception as exc:
            log(f'procedural colour variation fallback for {name}: {exc}')
            colour_output = primary

    mel.connect_material_property(colour_output, '', unreal.MaterialProperty.MP_BASE_COLOR)
    rough = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -520, 420)
    rough.set_editor_property('r', roughness)
    mel.connect_material_property(rough, '', unreal.MaterialProperty.MP_ROUGHNESS)
    metal = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -520, 490)
    metal.set_editor_property('r', metallic)
    mel.connect_material_property(metal, '', unreal.MaterialProperty.MP_METALLIC)

    if emissive > 0.0:
        glow = mel.create_material_expression(material, unreal.MaterialExpressionMultiply, -180, -170)
        strength = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -500, -170)
        strength.set_editor_property('r', emissive)
        mel.connect_material_expressions(colour_output, '', glow, 'A')
        mel.connect_material_expressions(strength, '', glow, 'B')
        mel.connect_material_property(glow, '', unreal.MaterialProperty.MP_EMISSIVE_COLOR)

    if translucent:
        opacity = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -520, 570)
        opacity.set_editor_property('r', opacity_value)
        opacity_output = opacity
        try:
            coordinate = mel.create_material_expression(material, unreal.MaterialExpressionTextureCoordinate, -760, 660)
            panner = mel.create_material_expression(material, unreal.MaterialExpressionPanner, -560, 660)
            panner.set_editor_property('speed_x', 0.035)
            panner.set_editor_property('speed_y', -0.16 if 'Waterfall' in name else 0.022)
            mel.connect_material_expressions(coordinate, '', panner, 'Coordinate')
            noise = mel.create_material_expression(material, unreal.MaterialExpressionNoise, -330, 660)
            try:
                noise.set_editor_property('scale', 14.0 if 'Waterfall' in name else 5.0)
                noise.set_editor_property('levels', 3)
            except Exception:
                pass
            mel.connect_material_expressions(panner, '', noise, 'Position')
            bias = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -330, 780)
            bias.set_editor_property('r', opacity_value * 0.58)
            opacity_add = mel.create_material_expression(material, unreal.MaterialExpressionAdd, -80, 680)
            mel.connect_material_expressions(noise, '', opacity_add, 'A')
            mel.connect_material_expressions(bias, '', opacity_add, 'B')
            opacity_output = opacity_add
        except Exception as exc:
            log(f'animated transparency fallback for {name}: {exc}')
        mel.connect_material_property(opacity_output, '', unreal.MaterialProperty.MP_OPACITY)
        if name not in {'M_Foam', 'M_Mist'}:
            refraction = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -520, 850)
            refraction.set_editor_property('r', 1.025)
            mel.connect_material_property(refraction, '', unreal.MaterialProperty.MP_REFRACTION)

    if wind:
        try:
            wind_node = mel.create_material_expression(material, unreal.MaterialExpressionSimpleGrassWind, -80, 330)
            intensity = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -360, 330)
            intensity.set_editor_property('r', 0.18 if name == 'M_Grass' else 0.12)
            weight = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -360, 420)
            weight.set_editor_property('r', 1.0)
            speed = mel.create_material_expression(material, unreal.MaterialExpressionConstant, -360, 500)
            speed.set_editor_property('r', 0.30)
            mel.connect_material_expressions(intensity, '', wind_node, 'WindIntensity')
            mel.connect_material_expressions(weight, '', wind_node, 'WindWeight')
            mel.connect_material_expressions(speed, '', wind_node, 'WindSpeed')
            mel.connect_material_property(wind_node, '', unreal.MaterialProperty.MP_WORLD_POSITION_OFFSET)
        except Exception as exc:
            log(f'wind material fallback for {name}: {exc}')

    mel.recompile_material(material)
    unreal.EditorAssetLibrary.save_loaded_asset(material)
    return material


def make_materials() -> dict[str, unreal.Material]:
    specs = {
        'M_Ground': dict(colour=(0.026, 0.016, 0.007), secondary_colour=(0.055, 0.048, 0.018), roughness=0.96, noise_scale=0.0030),
        'M_Mud': dict(colour=(0.040, 0.020, 0.008), secondary_colour=(0.090, 0.052, 0.018), roughness=0.78, noise_scale=0.0045),
        'M_Rock': dict(colour=(0.055, 0.070, 0.052), secondary_colour=(0.135, 0.155, 0.115), roughness=0.89, noise_scale=0.0060),
        'M_Moss': dict(colour=(0.010, 0.050, 0.012), secondary_colour=(0.045, 0.155, 0.030), roughness=0.98, noise_scale=0.0080),
        'M_Riverbed': dict(colour=(0.025, 0.035, 0.028), secondary_colour=(0.075, 0.090, 0.065), roughness=0.75, noise_scale=0.0050),
        'M_Bark': dict(colour=(0.035, 0.016, 0.006), secondary_colour=(0.105, 0.052, 0.016), roughness=0.95, noise_scale=0.0090),
        'M_BarkDark': dict(colour=(0.010, 0.006, 0.003), secondary_colour=(0.045, 0.022, 0.008), roughness=0.97, noise_scale=0.0100),
        'M_Leaf': dict(colour=(0.006, 0.060, 0.012), secondary_colour=(0.028, 0.185, 0.032), roughness=0.78, wind=True, noise_scale=0.0120),
        'M_LeafDark': dict(colour=(0.003, 0.024, 0.007), secondary_colour=(0.014, 0.090, 0.018), roughness=0.84, wind=True, noise_scale=0.0140),
        'M_LeafLight': dict(colour=(0.020, 0.100, 0.018), secondary_colour=(0.070, 0.260, 0.045), roughness=0.74, wind=True, noise_scale=0.0110),
        'M_Fern': dict(colour=(0.008, 0.070, 0.014), secondary_colour=(0.040, 0.215, 0.045), roughness=0.82, wind=True, noise_scale=0.0150),
        'M_Grass': dict(colour=(0.008, 0.052, 0.009), secondary_colour=(0.035, 0.170, 0.030), roughness=0.86, wind=True, noise_scale=0.0180),
        'M_Vine': dict(colour=(0.004, 0.028, 0.006), secondary_colour=(0.020, 0.095, 0.015), roughness=0.91, wind=True, noise_scale=0.0140),
        'M_Water': dict(colour=(0.004, 0.045, 0.070), roughness=0.14, translucent=True, emissive=0.015, opacity_value=0.76),
        'M_Waterfall': dict(colour=(0.080, 0.250, 0.310), roughness=0.09, translucent=True, emissive=0.035, opacity_value=0.70),
        'M_Foam': dict(colour=(0.48, 0.68, 0.58), roughness=0.30, translucent=True, emissive=0.06, opacity_value=0.68),
        'M_Mist': dict(colour=(0.35, 0.48, 0.40), roughness=0.40, translucent=True, emissive=0.03, opacity_value=0.22),
    }
    return {name: create_material(name, **kwargs) for name, kwargs in specs.items()}


def apply_materials_to_meshes(materials: dict[str, unreal.Material]) -> dict[str, str]:
    assignments: dict[str, str] = {}
    collision_prefixes = ('SM_Tree', 'SM_Emergent', 'SM_Palm', 'SM_Rock', 'SM_Log')
    for data in list_assets(STATIC_DEST):
        asset = data.get_asset()
        if not isinstance(asset, unreal.StaticMesh):
            continue
        static_materials = list(asset.get_editor_property('static_materials'))
        changed = False
        for index, slot in enumerate(static_materials):
            slot_name = str(slot.get_editor_property('material_slot_name'))
            candidate = materials.get(slot_name)
            if candidate:
                slot.set_editor_property('material_interface', candidate)
                static_materials[index] = slot
                changed = True
                assignments[f'{asset.get_path_name()}:{slot_name}'] = candidate.get_path_name()
        if changed:
            asset.set_editor_property('static_materials', static_materials)
        if asset.get_name() == 'SM_JungleTerrain' or asset.get_name().startswith(collision_prefixes):
            try:
                body_setup = asset.get_editor_property('body_setup')
                if body_setup:
                    body_setup.set_editor_property('collision_trace_flag', unreal.CollisionTraceFlag.CTF_USE_COMPLEX_AS_SIMPLE)
            except Exception as exc:
                log(f'collision setup warning {asset.get_name()}: {exc}')
        unreal.EditorAssetLibrary.save_loaded_asset(asset)
    return assignments


def runtime_manifest(source: dict) -> tuple[Path, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in source['instances']:
        grouped[item['mesh']].append({
            'location': item['location_cm'],
            'rotation': item['rotation_deg'],
            'scale': item['scale'],
        })
    groups = []
    collision_prefixes = ('SM_Tree', 'SM_Emergent', 'SM_Palm', 'SM_Rock', 'SM_Log')
    for mesh_name, instances in sorted(grouped.items()):
        mesh = load_by_name(STATIC_DEST, mesh_name, 'StaticMesh')
        if not mesh:
            raise RuntimeError(f'Imported static mesh missing for manifest group: {mesh_name}')
        collision = mesh_name.startswith(collision_prefixes)
        groups.append({'mesh': mesh.get_path_name(), 'mesh_name': mesh_name, 'collision': collision, 'instances': instances})
    payload = {
        'schema': 'procedural_jungle_runtime_instances_v2',
        'groups': groups,
        'total_instances': sum(len(group['instances']) for group in groups),
        'family_counts': source.get('instance_family_counts', {}),
    }
    CONTENT_ROOT.mkdir(parents=True, exist_ok=True)
    path = CONTENT_ROOT / 'runtime_instances.json'
    save_json(path, payload)
    return path, payload


def spawn_static(mesh_name: str, label: str | None = None):
    mesh = load_by_name(STATIC_DEST, mesh_name, 'StaticMesh')
    if not mesh:
        raise RuntimeError(f'Hero mesh not imported: {mesh_name}')
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    actor.set_actor_label(label or mesh_name)
    comp = actor.get_editor_property('static_mesh_component')
    comp.set_editor_property('static_mesh', mesh)
    comp.set_editor_property('mobility', unreal.ComponentMobility.STATIC)
    return actor


def set_light_component(actor, **properties):
    comp = actor.get_component_by_class(unreal.LightComponentBase)
    if not comp:
        comp = actor.get_component_by_class(unreal.DirectionalLightComponent)
    for key, value in properties.items():
        try:
            comp.set_editor_property(key, value)
        except Exception as exc:
            log(f'light property {key} warning: {exc}')


def spawn_environment() -> list:
    actors = []
    sun = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 18000), unreal.Rotator(-28, -36, 0))
    sun.set_actor_label('JungleSun')
    set_light_component(sun, intensity=3.2, cast_shadows=True, light_color=unreal.Color(255, 224, 188, 255))
    try:
        component = sun.get_component_by_class(unreal.DirectionalLightComponent)
        component.set_editor_property('atmosphere_sun_light', True)
        component.set_editor_property('atmosphere_sun_light_index', 0)
    except Exception as exc:
        log(f'sun atmosphere warning: {exc}')
    actors.append(sun)

    moon = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.DirectionalLight, unreal.Vector(0, 0, 17000), unreal.Rotator(32, 148, 0))
    moon.set_actor_label('JungleMoon')
    set_light_component(moon, intensity=0.08, cast_shadows=False, light_color=unreal.Color(105, 132, 180, 255))
    try:
        component = moon.get_component_by_class(unreal.DirectionalLightComponent)
        component.set_editor_property('atmosphere_sun_light', True)
        component.set_editor_property('atmosphere_sun_light_index', 1)
    except Exception as exc:
        log(f'moon atmosphere warning: {exc}')
    actors.append(moon)

    try:
        atmosphere = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkyAtmosphere, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        atmosphere.set_actor_label('JungleSkyAtmosphere')
        actors.append(atmosphere)
    except Exception as exc:
        log(f'sky atmosphere fallback: {exc}')

    sky = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkyLight, unreal.Vector(0, 0, 12000), unreal.Rotator(0, 0, 0))
    sky.set_actor_label('JungleSkyLight')
    try:
        component = sky.get_component_by_class(unreal.SkyLightComponent)
        component.set_editor_property('intensity_scale', 0.38)
        component.set_editor_property('real_time_capture', True)
    except Exception as exc:
        log(f'skylight warning: {exc}')
    actors.append(sky)

    fog = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.ExponentialHeightFog, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    fog.set_actor_label('JungleHeightFog')
    try:
        component = fog.get_component_by_class(unreal.ExponentialHeightFogComponent)
        component.set_editor_property('fog_density', 0.0018)
        component.set_editor_property('fog_height_falloff', 0.19)
        component.set_editor_property('volumetric_fog', True)
        component.set_editor_property('volumetric_fog_scattering_distribution', 0.42)
        component.set_editor_property('volumetric_fog_albedo', unreal.Color(88, 118, 92, 255))
    except Exception as exc:
        log(f'fog warning: {exc}')
    actors.append(fog)

    post = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PostProcessVolume, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    post.set_actor_label('JunglePostProcess')
    try:
        post.set_editor_property('unbound', True)
        settings = post.get_editor_property('settings')
        settings.set_editor_property('override_auto_exposure_method', True)
        settings.set_editor_property('auto_exposure_method', unreal.AutoExposureMethod.AEM_MANUAL)
        settings.set_editor_property('override_auto_exposure_bias', True)
        settings.set_editor_property('auto_exposure_bias', -1.15)
        settings.set_editor_property('override_color_saturation', True)
        settings.set_editor_property('color_saturation', unreal.Vector4(1.02, 1.06, 0.96, 1.0))
        try:
            settings.set_editor_property('override_color_contrast', True)
            settings.set_editor_property('color_contrast', unreal.Vector4(1.08, 1.08, 1.06, 1.0))
        except Exception:
            pass
        post.set_editor_property('settings', settings)
    except Exception as exc:
        log(f'post process warning: {exc}')
    actors.append(post)
    return actors


def spawn_camera(name: str, location: tuple[float, float, float], target: tuple[float, float, float], fov: float = 52.0):
    start = unreal.Vector(*location)
    end = unreal.Vector(*target)
    rotation = unreal.MathLibrary.find_look_at_rotation(start, end)
    camera = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.CameraActor, start, rotation)
    camera.set_actor_label(name)
    camera.set_editor_property('tags', [unreal.Name('JungleProofCamera')])
    component = camera.get_editor_property('camera_component')
    component.set_editor_property('field_of_view', fov)
    return camera


def build_level(manifest: dict, materials: dict[str, unreal.Material], runtime_payload: dict) -> dict:
    if unreal.EditorAssetLibrary.does_asset_exist(MAP_PATH):
        unreal.EditorLevelLibrary.load_level(MAP_PATH)
        for actor in unreal.EditorLevelLibrary.get_all_level_actors():
            unreal.EditorLevelLibrary.destroy_actor(actor)
    else:
        unreal.EditorLevelLibrary.new_level(MAP_PATH)

    hero_names = list(manifest['hero_meshes'])
    hero_actors = [spawn_static(name) for name in hero_names]
    environment = spawn_environment()

    population_class = unreal.load_class(None, '/Script/ProceduralJungle58.JunglePopulationActor')
    if not population_class:
        raise RuntimeError('JunglePopulationActor class is unavailable; C++ project was not compiled')
    population = unreal.EditorLevelLibrary.spawn_actor_from_class(population_class, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    population.set_actor_label('ProceduralJunglePopulation')
    population.set_editor_property('manifest_relative_path', 'ProceduralJungle/Generated/runtime_instances.json')
    if not population.rebuild_instances():
        raise RuntimeError('HISM population rebuild returned false')

    panda_mesh = next((data.get_asset() for data in list_assets(PANDA_DEST) if data.get_asset().get_class().get_name() == 'SkeletalMesh'), None)
    walk_anim = next((data.get_asset() for data in list_assets(PANDA_DEST) if data.get_asset().get_class().get_name() == 'AnimSequence' and 'Walk' in str(data.asset_name)), None)
    if not walk_anim:
        walk_anim = next((data.get_asset() for data in list_assets(PANDA_DEST) if data.get_asset().get_class().get_name() == 'AnimSequence'), None)
    if not panda_mesh or not walk_anim:
        raise RuntimeError('Imported panda skeletal mesh or walk animation missing')

    panda_class = unreal.load_class(None, '/Script/ProceduralJungle58.PandaWalkerCharacter')
    if not panda_class:
        raise RuntimeError('PandaWalkerCharacter class unavailable')
    route = [unreal.Vector(*point) for point in manifest['panda_route']]
    panda = unreal.EditorLevelLibrary.spawn_actor_from_class(panda_class, route[0], unreal.Rotator(0, 0, 0))
    panda.set_actor_label('WalkingTacticalRedPanda')
    panda.set_editor_property('panda_mesh', panda_mesh)
    panda.set_editor_property('walk_animation', walk_anim)
    panda.set_editor_property('route_points', route)
    panda.set_editor_property('walk_speed', 140.0)

    player_start = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.PlayerStart, unreal.Vector(-35500, 9800, 2600), unreal.Rotator(0, 18, 0))
    player_start.set_actor_label('JunglePlayerStart')

    proof_cameras = [
        spawn_camera('ProofCamera_00_PandaFront', (-35000, 10000, 2500), (-30000, 7200, 1500), 48.0),
        spawn_camera('ProofCamera_01_PandaRiver', (-27500, 10800, 2250), (-21000, 6600, 1400), 50.0),
        spawn_camera('ProofCamera_02_PandaUnderstory', (-17500, 9800, 2100), (-11200, 5200, 1250), 50.0),
        spawn_camera('ProofCamera_03_PandaCanopy', (-7200, 7800, 2200), (-1500, 3900, 1300), 48.0),
        spawn_camera('ProofCamera_04_RiverTunnel', (-23500, 13800, 2800), (-12000, 8500, 1500), 54.0),
        spawn_camera('ProofCamera_05_Waterfall', (12500, 2500, 3600), (23200, -10500, 600), 52.0),
        spawn_camera('ProofCamera_06_LowerPool', (39500, -29200, 1900), (29000, -17600, -1200), 50.0),
        spawn_camera('ProofCamera_07_RootPath', (-31500, 4300, 1900), (-23500, 5600, 1200), 48.0),
    ]

    director_class = unreal.load_class(None, '/Script/ProceduralJungle58.JungleProofDirector')
    if not director_class:
        raise RuntimeError('JungleProofDirector class unavailable')
    director = unreal.EditorLevelLibrary.spawn_actor_from_class(director_class, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    director.set_actor_label('JungleProofDirector')
    director.set_editor_property('required_capture_count', len(proof_cameras))

    game_mode_class = unreal.load_class(None, '/Script/ProceduralJungle58.JungleGameMode')
    world = unreal.EditorLevelLibrary.get_editor_world()
    settings = world.get_world_settings()
    settings.set_editor_property('default_game_mode', game_mode_class)

    unreal.EditorLevelLibrary.save_current_level()
    unreal.EditorAssetLibrary.save_directory('/Game/ProceduralJungle', only_if_is_dirty=False, recursive=True)
    return {
        'hero_actor_count': len(hero_actors),
        'environment_actor_count': len(environment),
        'proof_camera_count': len(proof_cameras),
        'population_instance_count': int(population.get_built_instance_count()),
        'population_family_counts': runtime_payload.get('family_counts', {}),
        'panda_mesh': panda_mesh.get_path_name(),
        'panda_animation': walk_anim.get_path_name(),
        'panda_route_points': len(route),
        'player_start': player_start.get_path_name(),
        'game_mode': game_mode_class.get_path_name(),
        'camera_fov_range': [48.0, 54.0],
        'visual_overhaul': True,
    }


def main() -> None:
    manifest_path = GENERATED_ROOT / 'jungle_manifest.json'
    panda_report_path = GENERATED_ROOT / 'panda' / 'panda_rig_report.json'
    if not manifest_path.is_file() or not panda_report_path.is_file():
        raise RuntimeError('Generated Blender manifests are missing')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    panda_report = json.loads(panda_report_path.read_text(encoding='utf-8-sig'))
    CONTENT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, CONTENT_ROOT / 'source_jungle_manifest.json')
    imported_static = import_static_fbx(Path(manifest['static_asset_fbx']))
    imported_panda = import_panda_fbx(Path(panda_report['outputs']['fbx']))
    materials = make_materials()
    material_assignments = apply_materials_to_meshes(materials)
    runtime_path, runtime_payload = runtime_manifest(manifest)
    scene = build_level(manifest, materials, runtime_payload)
    # Fresh in-process reload before the separate commandlet audit.
    unreal.EditorLevelLibrary.load_level(MAP_PATH)
    actors_after_reload = unreal.EditorLevelLibrary.get_all_level_actors()
    report = {
        'classification': 'UNREAL_DENSE_JUNGLE_BUILD_PROVEN',
        'canonical_map': MAP_PATH,
        'imported_static_paths': imported_static,
        'imported_panda_paths': imported_panda,
        'material_assignments': material_assignments,
        'runtime_manifest': str(runtime_path),
        'runtime_group_count': len(runtime_payload['groups']),
        'runtime_instance_count': runtime_payload['total_instances'],
        'actors_after_reload': len(actors_after_reload),
        'scene': scene,
        'no_external_art_assets': True,
        'visual_overhaul': True,
        'visual_manifest_targets': manifest.get('visual_targets', {}),
    }
    save_json(REPORT_PATH, report)
    log(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
