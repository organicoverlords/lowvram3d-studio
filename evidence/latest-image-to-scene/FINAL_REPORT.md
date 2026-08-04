# Image-to-Scene Evidence Summary

## Castlegrounds fixture

- Classification: `IMAGE_TO_SCENE_SMOKE_PARTIAL`
- The source image was used only as the first integration fixture.
- The source map remained protected; independent gameplay layers, water exclusions, bridge traversal, navigation bounds, and fresh map reload are separately receipted.
- Visual proof: `BLOCKED_UNREAL_EDITOR_SLATE_ASSERT`.
- The live editor terminated during the last read-only PIE audit. No automatic retry or editor restart was performed.

### Gates

- `scene_spec_and_builder_plan`: `PROVEN`
- `complete_scene_layers`: `PROVEN`
- `navigation`: `PROVEN`
- `gameplay`: `PROVEN`
- `fresh_map_reload`: `PROVEN`
- `live_contract`: `PROVEN`
- `visual_proof`: `BLOCKED_UNREAL_EDITOR_SLATE_ASSERT`

## Generic pipeline

- Classification: `GENERIC_ONE_IMAGE_TO_SCENE_PIPELINE_PARTIAL`
- Selection is driven by SceneSpec layer semantics and a capability registry.
- Scene-local map/evidence paths and resource budgets are enforced.
- `treesandbarn` and `landscape` completed CPU bootstrap/resume dry runs with camera/depth still `REQUIRES_ANALYSIS`; they are not end-to-end Unreal scene proofs.
- `baatti.jpg` and `panda.jpg` are explicitly recorded as `input_kind=object` and are excluded from scene-generalization evidence.
- A second real end-to-end materially different scene has not been proven.

## Protected source

- Output map: `/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_Hybrid_V1`
- Source map SHA-256: `39547be52ab21f3f6b0d99c0f2a2f93103a5c0ebf9da56435e37feae04cc15f9`
- No GPU work was requested by the generic CPU lane.
