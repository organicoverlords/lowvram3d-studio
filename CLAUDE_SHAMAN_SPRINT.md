# Claude sprint: finish the shaman geometry model

## Objective

Produce the first clearly recognizable, upright, high-detail shaman geometry GLB from the verified source PNG on the user's Windows 10 / GTX 1660 SUPER machine. Do not texture, rig, pose, decimate, or optimize until the geometry is visibly acceptable.

## Exact source

- Path: `C:\Users\Lauri\Downloads\ChatGPT Image 29.7.2026 klo 20.00.45.png`
- Dimensions: `1122x1402`
- SHA-256: `4d23adc758c5b700dd29939e37c043ce61919792b566bdcf13f58b1409d6cf6f`

## Never use as generation input

The uploaded `shaman.fbx` is comparison-only. Do not feed it into reconstruction, cleanup, retopo, UV, texture, optimization, pose, rigging, geometry transfer, or texture transfer. It may only be used after a candidate exists for normalized silhouette and proportion comparison.

## Current branch and run

- Repository: `organicoverlords/lowvram3d-studio`
- Branch: `infra/windows-self-hosted-runner-20260731`
- PR: `#3`, draft, do not merge
- Queued workflow run: `30666945507`
- Queued job: `91276078679`, name `geometry-first`
- Current head: `2a44f13c4acee78dcf7f040e5f9d4dcad9346c6c`

## Proven current blocker

The Windows runner is online:

- scheduled task: `LowVRAM3D-GitHub-Runner`
- `Runner.Listener.exe` is running
- workflow remains queued with no steps

This means GitHub has not matched the queued job to the runner. Diagnose runner registration / labels / runner group eligibility first. Do not keep retriggering blindly.

Current workflow expects a Windows self-hosted runner. Verify actual registered labels through GitHub or local `.runner` / `.credentials` state, then make `runs-on` match exactly. Prefer the minimum reliable selector:

```yaml
runs-on: [self-hosted, Windows, X64]
```

Only add custom labels after proving they exist on the online runner. Do not merge the PR.

## Fastest acceptable execution route

Once runner assignment works, use the geometry-first lane already present:

- `.github/workflows/shaman-local-worker.yml`
- `scripts/run-shaman-geometry-iteration.ps1`
- `workers/run_geometry_iteration.py`
- `blender/geometry_iteration_validate.py`

The intended local output is:

`C:\AI\LowVRAM3D-benchmarks\outputs\antlered_bird_shaman_anchor\geometry-latest\`

Expected deliverables:

- `shaman_geometry_master.glb`
- `shaman_geometry_working.glb`
- `geometry_validation.json`

## Geometry acceptance gate

Reject the candidate unless all are true:

1. Upright, not horizontal or compressed.
2. Birdlike head and beak are readable.
3. Branching antlers are present and coherent.
4. Torso, shoulders, robes, arms, and legs read as one character.
5. Staff remains separate from the body.
6. No merged staff/body mass.
7. No duplicate limbs or major floating debris.
8. No major silhouette feature is missing.
9. GLB imports in a fresh Blender process with finite bounds and nonzero triangles.

Do not call a structural GLB pass a visual success.

## Generator priority

1. Attempt Mini Turbo only if the ComfyUI client returns or discovers a real mesh file.
2. The client was hardened to accept direct filesystem GLB outputs even when ComfyUI history has no file entry.
3. If Mini Turbo still does not produce a usable GLB, use the installed TripoSR fallback with the explicit geometry iteration override rather than the old 192-resolution proxy.
4. Preserve the raw generated master byte-for-byte before any cleanup.
5. Use conservative cleanup only after a recognizable candidate exists.

## Existing measured lessons

- Prior shaman generation reached a valid 55,684-face manifold mesh.
- The earlier full run failed because component audit treated two tiny ambiguous components as fatal.
- Current policy preserves the original mesh when ambiguity is the only problem and topology is safe.
- A previous 50,103-face textured recovery was sideways/compressed/distorted and must not be used as a baseline.
- Black/identical MV-Adapter views are invalid even if the worker receipt says success.
- Do not texture wrong geometry.

## Sprint execution order

1. Fix runner/job matching.
2. Run one geometry-only candidate.
3. Validate in a fresh Blender process.
4. Inspect the candidate against the geometry acceptance gate.
5. If rejected, change only the generation stage/settings and rerun.
6. Stop after two failed candidates and report the exact generator, settings, face count, bounds, and failure reason.
7. If accepted, copy the master and validation JSON to `geometry-latest` and publish the workflow receipt/artifact.
8. Do not start texture, UV, A-pose, rigging, or final export in this sprint.

## Reporting contract

Return one of:

- `SHAMAN_GEOMETRY_ACCEPTED`
- `SHAMAN_GEOMETRY_REJECTED`
- `BLOCKED_RUNNER_MATCH`
- `BLOCKED_GENERATOR`

Include:

- branch and HEAD
- workflow run ID and job ID
- generator used
- source hash verification
- output GLB path and SHA-256
- vertices / triangles / components / bounds
- fresh Blender validation result
- concise visual-gate verdict
- exact next action

Do not claim success without the GLB and validation evidence.
