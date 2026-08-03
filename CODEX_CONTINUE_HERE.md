# Codex: continue the Castlegrounds scene pipeline here

Date: 2026-08-03  
Repository: `organicoverlords/lowvram3d-studio`  
Required branch: `agent/scene-pipeline-smoke-20260803`  
Execution owner: **Codex, not the GitHub local worker**

The previous panda texture task is deferred. Work on the Castlegrounds image-to-scene pipeline until this handoff is explicitly replaced.

## Operating rules

Before editing or running anything:

1. Verify repository origin, branch, local HEAD, remote HEAD, and clean worktree.
2. Work only on `agent/scene-pipeline-smoke-20260803`.
3. Do not work on `main`, merge, force-push, or rewrite preserved proof.
4. Do not trigger or depend on the GitHub self-hosted worker for this task.
5. Use the existing local repository and applications directly.
6. Do not kill or restart an already-running Unreal Editor unless the user explicitly authorizes it.
7. Do not rerun MoGe. The selected visual-shell mesh is already proven and must be reused.
8. Do not alter the source mesh, UVs, texture, triangle count, source hash, or imported Unreal asset unless a new failing proof demonstrates that the source asset itself is wrong.
9. Run one bounded GPU-capable process at a time. No neural workload is currently required.
10. Emit compact JSON receipts before promotion. Classify claims as `PROVEN`, `PARTIAL`, `BLOCKED`, or `REJECTED`.

Likely local repository from the preserved preflight:

```text
C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803
```

Unreal project:

```text
C:\Users\Lauri\Desktop\UnrealAITest58\UnrealAITest58.uproject
```

Blender:

```text
C:\Program Files\Blender Foundation\Blender 5.2\blender.exe
```

Unreal Engine 5.8:

```text
C:\Program Files\Epic Games\UE_5.8
```

## Current proven state

### Phase A — reusable SceneSpec contract: PROVEN

Implemented:

- `schemas/scene_spec_v1.schema.json`
- `src/lowvram3d/scene_spec.py`
- `tests/test_scene_spec.py`
- `configs/scene/castlegrounds_scene_spec_v1.json`
- `docs/SCENE_PIPELINE_V2_20260803.md`

Proof:

- `evidence/latest-scene-spec-local-worker/scene_spec_validation.json`
- classification `PROVEN`
- 0 validation errors
- 0 warnings
- unsafe gameplay proxy, GPU collision, GPU gameplay output, and GPU concurrency cases correctly fail closed

### Phase B — legacy evidence migration: PROVEN

Implemented:

- `src/lowvram3d/scene_spec_legacy.py`
- `tests/test_scene_spec_legacy.py`
- `configs/scene/castlegrounds_source_mesh_v2.json`

Proof:

- `evidence/latest-scene-spec-local-worker/legacy_migration_receipt.json`
- `evidence/latest-scene-spec-local-worker/migrated_scene_spec.json`
- `evidence/latest-scene-spec-local-worker/migrated_scene_spec_validation.json`

Preserved exactly:

- source image SHA-256 and dimensions
- legacy camera payload
- landmarks
- selected mesh identity
- mesh transform contract

No neural, Blender, Unreal, or GPU work was started by the migration.

### Phase C — deterministic Blender preparation: PROVEN

Implemented:

- `src/lowvram3d/scene_preparation.py`
- `tests/test_scene_preparation.py`
- `blender/prepare_scene_from_spec.py`

Proof:

- `evidence/latest-scene-preparation-local-worker/blender_scene_preparation_receipt.json`

Proven asset contract:

```text
GLB: C:\AI\ScenePipelineSmoke\20260803\castlegrounds\castlegrounds_source_mesh_v2.glb
SHA-256: 2ee23fd3816f44931a25628392fa5962b23a55dcc17e207540dd3d937e940011
Triangles: 374,959
Vertices after fresh Blender import: 190,432
Mesh edits: 0
Generated mesh operations: 0
```

Required collections exist:

```text
SCENE_VISUAL_SHELL
SCENE_EDITABLE
SCENE_GAMEPLAY_PROXY
SCENE_PROCEDURAL_MODULES
SCENE_REFERENCE_ONLY
```

Prepared blend:

```text
C:\AI\ScenePipelineSmoke\20260803\castlegrounds\scene_camera_v1\castlegrounds_source_locked_v1_prepared.blend
```

### Authoritative CameraContract: PROVEN

Important correction: `scene_interpretation.json` contained a stale 48-degree estimate. It is superseded by the proven MoGe calibration and exact Blender camera receipt.

Implemented:

- `src/lowvram3d/scene_camera.py`
- `tests/test_scene_camera.py`
- `blender/apply_scene_camera_contract.py`

Proof:

- `evidence/latest-scene-camera-local-worker/camera_contract.json`
- `evidence/latest-scene-camera-local-worker/camera_contract_receipt.json`
- `evidence/latest-scene-camera-local-worker/blender_camera_application_receipt.json`
- `evidence/latest-scene-camera-local-worker/authoritative_scene_spec.json`

Authoritative camera:

```text
Horizontal FOV: 66.50838470458984 degrees
Vertical FOV: 52.37591552734375 degrees
Principal point: [0.5, 0.5]
Basis: normalized, orthogonal, right-handed
Legacy 48-degree estimate: superseded
Mesh edits while applying camera: 0
```

Exact Blender basis:

```text
origin  = [-1.4999749120901782e-16, 2.7755575615628914e-17, -2.220446049250313e-16]
forward = [ 8.201049757676981e-17, -1.0,  1.6653345369377348e-16]
right   = [ 1.0,  8.201049757676981e-17, -8.201049757676981e-17]
up      = [-8.201049757676981e-17, -1.6653345369377348e-16, -1.0]
```

### Existing Unreal scene: previously proven, current re-audit blocked

Preserved proof:

- `proof/scene/20260803-image-to-scene-smoke/scene_build_receipt.json`
- `proof/scene/20260803-image-to-scene-smoke/map_reload_receipt.json`

Existing map:

```text
/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_ImageToScene_Smoke
```

Existing imported source mesh:

```text
/Game/AgentProof/ImageToSceneSmoke_20260803/Geometry/CastlegroundsSourceMeshV2/castlegrounds_source_mesh_v2/StaticMeshes/castlegrounds_source_mesh_v2.castlegrounds_source_mesh_v2
```

Existing source camera actor from the proven receipt:

```text
CameraActor_5
```

Implemented read-only audit:

- `unreal/audit_scene_contract.py`
- `.github/workflows/scene-unreal-audit-local-worker.yml`

Latest audit result:

```text
REJECTED / NOT RUN
Reason: Unreal Editor PID 10044 was already running.
The worker refused to interfere before launching UnrealEditor-Cmd.
No Unreal commandlet, mutation, render, GPU workload, import, or save occurred.
```

This is an execution-route blocker, not a scene-content failure.

## Immediate Codex task

Continue through the already-running local Unreal Editor instead of the GitHub worker.

### Step 1 — inspect the live Unreal process safely

1. Confirm whether PID `10044` is still the intended `UnrealAITest58` editor.
2. Do not terminate it.
3. Prefer the already-enabled Unreal MCP / Python Editor Script route to inspect the live editor.
4. If MCP is unavailable, create a user-invoked editor Python script that runs inside the existing editor. Do not launch a second editor process.
5. Reuse the read-only logic in `unreal/audit_scene_contract.py` rather than rewriting the contract from scratch.

Required receipt:

```text
evidence/latest-scene-unreal-live-audit/unreal_scene_contract_audit.json
```

The audit must prove or reject:

- the exact map is loaded or loadable in the existing editor;
- the exact imported static mesh asset exists;
- exactly one actor references that source mesh;
- source camera exists;
- source camera horizontal FOV matches 66.50838470458984 within 0.0001 degrees;
- source camera forward direction matches the preserved Unreal camera convention;
- no actor, asset, import, save, or render mutation was performed during audit.

Do not classify the audit as proven merely because the old receipt exists. Inspect the live project.

### Step 2 — finish authoritative hybrid composition

Current implementation:

- `src/lowvram3d/scene_hybrid.py`

Add focused tests in:

```text
tests/test_scene_hybrid.py
```

Generate:

```text
evidence/latest-scene-hybrid/authoritative_hybrid_scene_spec.json
evidence/latest-scene-hybrid/hybrid_composition_receipt.json
```

Composition rules:

- start from `configs/scene/castlegrounds_scene_spec_v1.json`;
- replace all source and offset camera values from `camera_contract.json`;
- offsets must be parallel translations along the proven camera right vector;
- never hard-code 48 degrees again;
- preserve the proven Unreal map and imported mesh paths;
- mark castle proxy, bridge tile, grass cluster, PCG placement, collision, and navigation as `not_promoted` / unproven until independent evidence exists;
- SceneSpec validation must pass with 0 errors.

### Step 3 — only after the live read-only audit is PROVEN

Build the smallest independently useful gameplay layer in the existing map:

1. one simple walkable gameplay proxy;
2. one blocking castle proxy;
3. no PCG yet;
4. no GPU population yet;
5. no source-shell modification;
6. no source-camera change;
7. save to a new map or deterministic bounded layer so the proven source-only map remains recoverable.

Suggested new map:

```text
/Game/AgentProof/ImageToSceneSmoke_20260803/Maps/L_Castlegrounds_Hybrid_V1
```

Required proof:

```text
GAMEPLAY_PROXY_CREATED=PROVEN
SOURCE_MESH_REFERENCE_PRESERVED=PROVEN
SOURCE_CAMERA_CONTRACT_PRESERVED=PROVEN
COLLISION_PROBE=PROVEN
NAVIGATION_PROBE=PROVEN
MAP_SAVE_RELOAD=PROVEN
NO_COLLATERAL_CHANGE=PROVEN
```

Do not proceed to bridge grammar, river spline, grass, GPU PCG, or visual rendering until this bounded gameplay layer is proven.

## Tests to run before Unreal mutation

```text
python -m compileall -q src tests unreal blender
python -m pytest -q \
  tests/test_scene_spec.py \
  tests/test_scene_spec_legacy.py \
  tests/test_scene_preparation.py \
  tests/test_scene_camera.py \
  tests/test_scene_hybrid.py
```

Existing focused suite before the new hybrid tests: 20 tests should pass.

## Promotion gates

Current classification remains:

```text
IMAGE_TO_SCENE_PARTIAL
```

Already proven:

```text
SCENE_SPEC_VALID
LEGACY_EVIDENCE_MIGRATION
SOURCE_MESH_IDENTITY
SOURCE_MESH_TRIANGLE_CONTRACT
BLENDER_PREPARATION
AUTHORITATIVE_CAMERA_CONTRACT
AUTHORITATIVE_CAMERA_APPLIED_IN_BLENDER
UNREAL_INTERCHANGE_IMPORT (preserved prior proof)
UNREAL_SAVE_RELOAD (preserved prior proof)
```

Not yet proven under the current authoritative pipeline:

```text
UNREAL_LIVE_READ_ONLY_AUDIT
GAMEPLAY_PROXY
COLLISION
NAVIGATION
PCG_REFERENCE_GRAPH
UNREAL_SOURCE_RENDER
UNREAL_PARALLAX
GPU_BUDGET
```

Do not promote the overall scene until every required gate has direct evidence.

## Commit policy

Commit and push only validated changes on `agent/scene-pipeline-smoke-20260803`.

Before finishing:

- verify focused tests;
- verify local HEAD equals remote HEAD;
- leave the worktree clean;
- report exact commit SHA;
- report which claims are `PROVEN`, `REJECTED`, `BLOCKED`, or still `NOT_PROVEN`;
- stop at `USER_REVIEW_REQUIRED` if a live Unreal action needs the user to click or approve something.
