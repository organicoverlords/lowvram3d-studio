# Direct local-worker task: playable procedural jungle with walking panda

No Codex, Claude, or autonomous coding agent may be invoked by this lane. GitHub Actions on the trusted Windows self-hosted worker executes repository scripts directly. MagicMusic is reserved only for recovery if GitHub cannot reach the worker.

## Deliverable

Create a dedicated UE 5.8 project at:

`C:\Users\Lauri\Desktop\ProceduralJungle58\ProceduralJungle58.uproject`

Canonical map:

`/Game/ProceduralJungle/Maps/L_ProceduralJungle`

Large proof root:

`C:\AI\ProceduralJungle\20260804`

The result must be a realistic, playable jungle generated without downloaded or marketplace art assets. It must include procedural terrain, dense varied vegetation, rocks, roots, vines, a winding river, waterfall, lower pool, moving water, mist/spray/foam, wind, lighting, fog, collision, and a walking player.

## Panda integration

Use the user's existing tactical red panda mesh from the canonical local benchmark tree. Select the newest accepted textured candidate only when its own acceptance receipt is proven; otherwise use the newest structurally valid repaired panda GLB and record the downgrade.

The panda must be an animated in-world character, not a static prop:

- generate a game skeleton in Blender;
- preserve existing mesh/material assignments;
- include root, pelvis, spine, chest, neck, head, left/right arms, left/right legs, feet, and a multi-bone tail chain;
- generate an in-place looping walk cycle with opposing arm/leg swing, knee flexion, body bob, head stabilization, and secondary tail sway;
- export skeletal mesh and animation to FBX;
- import both into Unreal;
- move the panda along a deterministic path through the jungle near the river and waterfall;
- match movement speed to animation stride closely enough to avoid obvious foot sliding;
- keep the capsule grounded and prove it does not fall through terrain;
- capture at least four distinct in-game panda-walking frames, including one riverbank and one waterfall view.

A static-mesh transform animation, sliding mesh, or rotating billboard is rejected.

## Execution policy

Before edits or generation, the worker must record and verify repository root, branch, HEAD, remote, clean status, engine path/version, Blender path/version, GPU/VRAM, RAM, target-project identity marker, and relevant running processes. It must refuse to overwrite an unrelated project or interfere with an open Unreal Editor.

The build is deterministic and stage-resumable. Repository scripts own all decisions and write machine-readable receipts. A successful process exit is not proof.

## Required terminal classification

- `JUNGLE_PANDA_PLAYABLE_PROVEN`
- `HARD_BLOCKER`

`JUNGLE_PANDA_PLAYABLE_PROVEN` requires:

1. dedicated UE 5.8 project exists;
2. canonical map saves and reloads in a fresh process;
3. no forbidden external art references;
4. jungle, river, waterfall, lower pool, lighting, fog, wind, and effects are visibly present;
5. player movement, camera, gravity, jump, and collision initialize in game mode;
6. panda is a skeletal mesh with a looping walk animation and moves along the intended route;
7. panda movement is grounded and does not visibly slide through the world;
8. six or more non-black 1920x1080 scene captures, including four panda-walk frames;
9. 1080p evidence route averages at least 30 FPS and records the minimum;
10. project reload audit, provenance audit, tests, and supported one-command rebuild all pass.

Required compact repository evidence:

- `proof/scene/20260804-procedural-jungle/FINAL_REPORT.md`
- `evidence/latest-procedural-jungle/acceptance.json`
- `evidence/latest-procedural-jungle/workflow_receipt.json`

Do not commit the generated Unreal project, `.uasset` files, large meshes, or full-resolution screenshots to this repository.