from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    errors: list[str] = []
    jungle = load(root / 'jungle_manifest.json')
    rig = load(root / 'panda' / 'panda_rig_report.json')

    if jungle.get('schema') != 'procedural_jungle_manifest_v2':
        errors.append('unexpected jungle manifest schema')
    variants = list(jungle.get('variant_meshes', []))
    instances = list(jungle.get('instances', []))
    family_counts = dict(jungle.get('instance_family_counts', {}))
    if len(variants) < 40:
        errors.append('fewer than 40 generated mesh variants')
    if len(instances) < 3300:
        errors.append('fewer than 3300 jungle instances')
    if int(family_counts.get('canopy', 0)) + int(family_counts.get('emergent', 0)) < 380:
        errors.append('canopy coverage contract below 380 trees')
    understory = sum(int(family_counts.get(name, 0)) for name in ('sapling', 'shrub', 'fern', 'grass', 'vine'))
    if understory < 2500:
        errors.append('understory density contract below 2500 instances')
    for required_family in ('canopy', 'emergent', 'palm', 'sapling', 'shrub', 'fern', 'grass', 'rock', 'log', 'vine'):
        if int(family_counts.get(required_family, 0)) <= 0:
            errors.append(f'missing population family: {required_family}')

    route = jungle.get('panda_route', [])
    if len(route) < 12:
        errors.append('panda route has fewer than 12 points')
    for index, point in enumerate(route):
        if len(point) != 3 or not all(math.isfinite(float(value)) for value in point):
            errors.append(f'non-finite route point {index}')

    river = jungle.get('river_points_cm', [])
    if len(river) < 10:
        errors.append('river has fewer than 10 points')
    if river and not (river[0][2] > river[-1][2]):
        errors.append('river does not descend overall')
    waterfall = int(jungle.get('waterfall_segment', -1))
    if not (0 <= waterfall < len(river) - 1):
        errors.append('invalid waterfall segment')
    elif not (river[waterfall][2] - river[waterfall + 1][2] >= 3500):
        errors.append('waterfall drop is below 35 metres')

    hero_meshes = set(jungle.get('hero_meshes', []))
    for required_hero in ('SM_JungleTerrain', 'SM_RiverUpper', 'SM_RiverLower', 'SM_LowerPool', 'SM_FallFoam'):
        if required_hero not in hero_meshes:
            errors.append(f'missing hero mesh: {required_hero}')
    if sum(1 for name in hero_meshes if name.startswith('SM_Waterfall_')) < 5:
        errors.append('fewer than five waterfall sheets')
    if sum(1 for name in hero_meshes if name.startswith('SM_Mist_')) < 3:
        errors.append('fewer than three mist cards')

    required_bones = {
        'root', 'pelvis', 'spine_01', 'spine_02', 'neck', 'head',
        'thigh_L', 'thigh_R', 'calf_L', 'calf_R', 'foot_L', 'foot_R',
        'upperarm_L', 'upperarm_R', 'lowerarm_L', 'lowerarm_R',
        'tail_01', 'tail_02', 'tail_03', 'tail_04', 'tail_05'
    }
    bones = set(rig.get('bones', []))
    missing_bones = sorted(required_bones - bones)
    if missing_bones:
        errors.append(f'missing panda bones: {missing_bones}')
    animation = rig.get('animation', {})
    if animation.get('name') != 'Walk' or not animation.get('looping') or not animation.get('in_place'):
        errors.append('panda walk animation contract incomplete')

    for relative in ('jungle_static_assets.fbx', 'jungle_sources.blend', 'panda/tactical_red_panda_walk.fbx', 'panda/tactical_red_panda_walk.blend'):
        path = root / relative
        if not path.is_file() or path.stat().st_size < 1024:
            errors.append(f'missing or implausibly small output: {relative}')

    report = {
        'classification': 'DENSE_JUNGLE_GENERATED_PROVEN' if not errors else 'REJECTED',
        'errors': errors,
        'variant_count': len(variants),
        'instance_count': len(instances),
        'instance_family_counts': family_counts,
        'canopy_instance_count': int(family_counts.get('canopy', 0)) + int(family_counts.get('emergent', 0)),
        'understory_instance_count': understory,
        'hero_mesh_count': len(hero_meshes),
        'panda_route_points': len(route),
        'panda_bone_count': len(bones),
        'panda_animation': animation,
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(report, sort_keys=True))
    if errors:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
