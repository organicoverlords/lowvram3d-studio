# Proof matrix

| Capability | Status |
|---|---|
| Mini Turbo geometry on GTX 1660 SUPER | **PROVEN by user run** |
| TripoSR CPU marching-cubes compatibility on target PC | **PROVEN by installer output** |
| TRELLIS2 rejection | **PROVEN by repeated target failures** |
| Installer stage resume and degraded optional stages | **TARGET-OBSERVED + UNIT/INTEGRATION-TESTED** |
| Correct `uv pip install --python ...` order | **TARGET-PROVEN** |
| Stage-10 extensible JSON configuration merge | **IMPLEMENTED + REGRESSION-TESTED; target retry pending** |
| Nine asset-class profiles including human avatar | **UNIT-TESTED** |
| Pinned BiRefNet cache/runtime revision | **STATIC/UNIT-TESTED; target model download pending** |
| Largest-person mask isolation | **ALGORITHM-TESTED** |
| Pose-bounded mask recovery | **ALGORITHM-TESTED** |
| Edge-aware alpha and colour decontamination | **ALGORITHM-TESTED** |
| Transparent full-body normalization and landmark transform | **ALGORITHM-TESTED** |
| Source alpha preserved into texture-view preparation | **UNIT-TESTED** |
| Source appearance forwarded into PBR bake | **INTEGRATION-TESTED** |
| Avatar GLB armature/dance-action validation | **CONTRACT-TESTED; Blender target proof pending** |
| 3D Gen Studio provider registration | **UNIT-TESTED; target registration pending** |
| 3D Gen Studio SSE mesh-tool parser | **UNIT-TESTED** |
| Blender import, split, retopo, UV, bake, export on target PC | **IMPLEMENTED; real target artifact proof pending** |
| MV-Adapter SD2.1 peak VRAM on target PC | **NOT PROVEN** |
| Photorealistic likeness from one photograph | **NOT PROVEN** |
| Humanoid automatic-weight/dance deformation quality | **NOT PROVEN; manual review required** |
| Scene/level cell bundle imported into Unreal | **NOT PROVEN** |
| Strict GLB/PNG/JSON checkpoint validation | **UNIT-TESTED** |
| Control-service health/start/stop in build container | **SMOKE-TESTED** |
| Dedicated Studio port 8311 | **STATIC/UNIT-TESTED** |

A stage is **PROVEN** only when the real artifact exists, its receipt passes, previews are inspected, and the exported GLB survives a clean Blender re-import.
