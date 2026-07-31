# Upstream-inspired processing settings

These presets are explicit pipeline inputs, not hidden UI defaults. Every job writes the resolved profile into `proof/job_receipt.json`.

## 3D Gen Studio mesh tools

The pinned Studio mesh service supports CPU Auto Retopo and Auto UV. The pipeline only calls it when component analysis reports one part, because the Studio service concatenates multi-mesh GLBs before processing.

### Auto Retopo baseline for the 16 GB machine

- CPU backend
- memory budget: 3 GB
- adaptive isotropic remesh
- 160 / 192 / 256 voxel shell resolution for Background / Gameplay / Hero
- surface projection for organic or rounded single-shell assets
- hard-feature preservation for props, vehicles, buildings, and rooms
- watertight shell disabled for buildings and rooms so openings are retained

### Auto UV baseline

- 2K atlas by default, 4K for Hero
- 8 px padding at 2K, 16 px at 4K
- weld enabled with conservative tolerance
- ARAP-biased charts for characters and creatures
- planar charts and strong sharp-edge seams for buildings and rooms
- balanced automatic charts for props and single-shell vehicles

## Blender fallback and multipart path

Blender remains authoritative when hierarchy or multiple components must survive:

- organic: continuous body, angle-aware unwrap, selected-to-active bake
- hard surface: planar dissolve / collapse while respecting material, UV, seam, normal, and sharp boundaries
- rooms/buildings: preserve openings and add a second lightmap UV
- scenes/levels: per-object triangle budgets, existing materials, cell manifests, no shared overlapping atlas

## Game output classes

| Class | Budget mode | Gameplay target | UV/material strategy | Rig/export |
|---|---|---:|---|---|
| Character | total | 40–50k | shared 2K organic atlas | humanoid template, skeletal GLB/FBX |
| Creature | total | 40–50k | shared 2K organic atlas | creature template, skeletal GLB/FBX |
| Vehicle | total | 40–50k | shared 2K hard-surface atlas | rigid part hierarchy |
| Prop | total | 40–50k | shared 2K atlas | static or rigid hierarchy |
| Building | total | 40–50k | shared atlas + lightmap UV | modular static asset |
| Room | total | 40–50k | shared atlas + lightmap UV | room kit, openings retained |
| Scene | per object | 15k/object | preserve per-object materials | 4×4 spatial bundle |
| Level | per object | 15k/object | preserve materials/lightmaps | 8×8 world-partition-style bundle |

The target ranges are configurable and validation allows bounded tolerance rather than forcing a destructive exact count.
