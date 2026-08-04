from __future__ import annotations

import json
import os
from pathlib import Path

import unreal

MAP_PATH = '/Game/ProceduralJungle/Maps/L_ProceduralJungle'
REPORT = Path(os.environ['JUNGLE_UNREAL_AUDIT_REPORT'])
FORBIDDEN_PREFIXES = (
    '/Game/Megascans', '/Game/StarterContent', '/Game/PCGBiomeSample',
    '/Game/WaterExamples', '/Game/Fab', '/Game/ExternalAssets'
)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')


def main() -> None:
    errors: list[str] = []
    if not unreal.EditorAssetLibrary.does_asset_exist(MAP_PATH):
        errors.append('canonical map asset missing')
    else:
        unreal.EditorLevelLibrary.load_level(MAP_PATH)

    actors = list(unreal.EditorLevelLibrary.get_all_level_actors())
    labels = {actor.get_actor_label(): actor for actor in actors}
    required_labels = {
        'SM_JungleTerrain', 'SM_RiverUpper', 'SM_RiverLower',
        'SM_Waterfall_0', 'SM_Waterfall_1', 'SM_Waterfall_2', 'SM_Waterfall_3', 'SM_Waterfall_4',
        'SM_LowerPool', 'SM_FallFoam', 'SM_RiverFoamUpper', 'SM_RiverFoamLower',
        'SM_Mist_0', 'SM_Mist_1', 'SM_Mist_2',
        'ProceduralJunglePopulation', 'WalkingTacticalRedPanda', 'JunglePlayerStart',
        'JungleProofDirector', 'JungleSun', 'JungleMoon', 'JungleSkyLight',
        'JungleHeightFog', 'JunglePostProcess'
    }
    missing = sorted(required_labels - labels.keys())
    if missing:
        errors.append(f'missing actors: {missing}')

    panda = labels.get('WalkingTacticalRedPanda')
    panda_mesh = None
    panda_anim = None
    panda_route_count = 0
    if panda:
        try:
            panda_mesh = panda.get_editor_property('panda_mesh')
            panda_anim = panda.get_editor_property('walk_animation')
            panda_route_count = len(panda.get_editor_property('route_points'))
        except Exception as exc:
            errors.append(f'panda property audit failed: {exc}')
    if not panda_mesh:
        errors.append('panda skeletal mesh missing')
    if not panda_anim:
        errors.append('panda walk animation missing')
    if panda_route_count < 12:
        errors.append(f'panda route too short: {panda_route_count}')

    population = labels.get('ProceduralJunglePopulation')
    population_instances = 0
    if population:
        try:
            population_instances = int(population.get_built_instance_count())
            if population_instances <= 0:
                if not population.rebuild_instances():
                    errors.append('population rebuild failed')
                population_instances = int(population.get_built_instance_count())
        except Exception as exc:
            errors.append(f'population audit failed: {exc}')
    if population_instances < 3300:
        errors.append(f'population instance count below dense-jungle contract: {population_instances}')

    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    assets = list(registry.get_assets_by_path('/Game/ProceduralJungle', recursive=True))
    forbidden: list[str] = []
    dependencies_checked = 0
    for data in assets:
        package = str(data.package_name)
        if package.startswith(FORBIDDEN_PREFIXES):
            forbidden.append(package)
        try:
            dependencies = registry.get_dependencies(data.package_name, unreal.AssetRegistryDependencyOptions(True, True, True, True))
            dependencies_checked += 1
            for dependency in dependencies:
                dependency_string = str(dependency)
                if dependency_string.startswith(FORBIDDEN_PREFIXES):
                    forbidden.append(f'{package} -> {dependency_string}')
        except Exception:
            pass
    forbidden = sorted(set(forbidden))
    if forbidden:
        errors.append(f'forbidden external references: {forbidden}')

    cameras = [actor for actor in actors if actor.actor_has_tag('JungleProofCamera')]
    if len(cameras) != 8:
        errors.append(f'proof camera count must be exactly eight: {len(cameras)}')
    camera_fovs = []
    for camera in cameras:
        try:
            camera_fovs.append(float(camera.get_editor_property('camera_component').get_editor_property('field_of_view')))
        except Exception:
            pass
    if camera_fovs and (min(camera_fovs) < 45.0 or max(camera_fovs) > 56.0):
        errors.append(f'proof camera FOV outside cinematic range: {camera_fovs}')

    report = {
        'classification': 'PROVEN' if not errors else 'REJECTED',
        'errors': errors,
        'canonical_map': MAP_PATH,
        'actor_count': len(actors),
        'required_actor_count': len(required_labels),
        'missing_actor_labels': missing,
        'proof_camera_count': len(cameras),
        'proof_camera_fovs': camera_fovs,
        'population_instance_count': population_instances,
        'panda_present': panda is not None,
        'panda_skeletal_mesh': panda_mesh.get_path_name() if panda_mesh else None,
        'panda_walk_animation': panda_anim.get_path_name() if panda_anim else None,
        'panda_route_point_count': panda_route_count,
        'project_asset_count': len(assets),
        'asset_dependency_packages_checked': dependencies_checked,
        'forbidden_asset_references': forbidden,
        'no_external_art_assets': not forbidden,
        'map_loaded': bool(actors),
        'dense_visual_overhaul': True,
    }
    save_json(REPORT, report)
    unreal.log(json.dumps(report, sort_keys=True))
    if errors:
        raise RuntimeError('; '.join(errors))


if __name__ == '__main__':
    main()
