# LowVRAM 3D Studio 0.6.1

A resumable, proof-first local 3D production pipeline for Windows 10, GTX 1660 SUPER 6 GB, and 16 GB RAM. **3D Gen Studio is the control layer.**

Read `docs/CURRENT_STATE.md` before changing the active high-resolution route. It records the TurboBird reference success, the verified shaman source, the latest failed-run evidence, and the current preserve-first cleanup policy.

## One-click paths

```text
Source image or high-poly GLB
→ Mini Turbo geometry when starting from an image
→ import and topology validation
→ conservative component audit
→ preserve the original mesh when audit evidence is ambiguous
→ guarded 3D Gen Studio Auto Retopo/Auto UV or object-aware Blender fallback
→ measured LOD selection and source-aware PBR transfer
→ optional class-specific rig and animations
→ collision, sockets, GLB/FBX and previews
→ clean Blender re-import validation
```

Supported classes:

- Photoreal human avatar + dance
- Character
- Creature
- Vehicle
- Prop
- Building
- Room / interior
- Scene / environment
- Level / world chunk

The package registers Auto Class, class-specific one-click generation providers, and class-specific post-processing providers in 3D Gen Studio.

## Core production principle

TurboBird is the user's first real successful result and the current qualitative reference. New gates must detect demonstrated corruption without rejecting a useful TurboBird-class mesh merely because a classifier is uncertain.

Component cleanup therefore distinguishes **damage** from **ambiguity**:

- subprocess failure, invalid artifacts, changed main geometry, or increased boundary/non-manifold edges fail closed;
- unresolved component classification alone is recorded as a warning and `manual_review_required=true`;
- when topology remains safe, the generated master is preserved byte-for-byte and downstream LOD, texture, rig/export and independent validation continue.

See `docs/CURRENT_STATE.md` for the exact proof boundary.

## Human avatar mode

Choose:

```text
LowVRAM One-Click — Photoreal Human Avatar + Dance
```

Use a single **full-body photograph** with one person, visible hands/knees/ankles/feet, limited occlusion, and space around the body. A cropped face selfie cannot supply faithful full-body geometry or clothing.

Avatar processing adds:

```text
pinned BiRefNet foreground matte
→ MediaPipe 33-landmark pose and person-mask analysis
→ edge-aware hair/cloth alpha refinement
→ foreground colour decontamination
→ largest-person selection
→ transparent square full-body normalization
→ identity-weighted source projection
→ humanoid rig + idle + dance_loop
→ validation that the exported GLB still contains its armature and dance action
```

See `docs/HUMAN_AVATAR_PIPELINE_0.6.0.md` for the exact limitations and proof boundary.

## Install or resume

Run:

```text
INSTALL-ONE-CLICK.cmd
```

After any interruption or package update, run:

```text
CONTINUE-INSTALL.cmd
```

It does not restart the installation. Verified stages are checkpointed under the install root, and only a failed, invalid, or changed stage runs. `INSTALL-STATUS.cmd` shows the stage table. Optional backends that fail are recorded as **degraded** and skipped on normal resumes; `RETRY-OPTIONAL.cmd` retries only those optional stages.

Failed asset processing is resumable too. Rerunning the same failed 3D Gen Studio card continues its saved job. `Resume Last 3D Job` continues the latest failed full/post-process job without re-uploading the source.

For an existing 0.6.0 installation, 0.6.1 keeps all completed stages and the packages already downloaded into the MV-Adapter environment. Stage 08 removes conflicting OpenCV wheel variants, installs only the pinned contrib wheel from cache, and writes a per-component readiness report. Completed TripoSR stage 09 remains reusable, and stage 10 then retries the corrected configuration merge.

The installer accepts an existing Blender installation and uses:

```text
uv pip install --python <venv-python> ...
```

It never installs model dependencies into the existing ComfyUI environment. 3D Gen Studio uses `http://127.0.0.1:8311`, avoiding the existing port-3001 router branch.

## Low-memory rules

- One GPU-heavy worker at a time.
- Default total GPU ceiling: 5,600 MB.
- BiRefNet uses 1024×1024 FP16 inference when CUDA is available; high-resolution 2048 matting models are not the 6 GB default.
- Retopo, UV analysis, splitting, scene chunking, mesh extraction, and most baking run on CPU.
- 2K textures by default; maps process sequentially.
- Validated installer and asset stages are retained before retry.
- Hugging Face receipts are checked against the real offline cache.
- Source/settings fingerprints prevent incompatible checkpoint reuse.
- Mini Turbo, BiRefNet, MV-Adapter, and proxy generation share one heavy-GPU lock.
- TRELLIS2 is not a backend.

## Asset-class safeguards

See `docs/UPSTREAM_SETTINGS.md` for the complete profile settings.

- Avatar/character/creature bodies remain continuous rather than being split into rigid limbs.
- Avatar source alpha is preserved through view creation instead of being removed twice.
- Ambiguous tiny character/creature components do not block a topology-safe generated master.
- Vehicles preserve detached wheels and likely movable rigid parts.
- Buildings and rooms preserve openings and receive a lightmap UV.
- Scenes and levels use per-object budgets and spatial manifests instead of one global triangle target.
- Studio Auto Retopo/UV is guarded because its service flattens a multi-mesh scene into one mesh.
- Rigging or pose normalization must never overwrite a valid unrigged asset when they fail.

## Output

```text
output/<asset>/
  source/original.glb
  meshes/high.glb
  meshes/game_ready.glb
  meshes/game_ready.fbx
  meshes/lod1.glb
  meshes/lod2.glb
  textures/*.png
  previews/front.png
  previews/side.png
  previews/rear.png
  previews/perspective.png
  previews/uv_layout.png
  reports/pipeline-report.json
  reports/validation-report.json
  reports/rig.json
  reports/parts.json
  reports/scene_manifest.json
  cells/*.glb                  # scene/level only
```

Avatar jobs also retain:

```text
preprocess/subject.png
preprocess/subject_mask.png
preprocess/mask_preview.png
preprocess/avatar_report.json
```

## Current proof status

- TurboBird: **USER-VALIDATED QUALITATIVE REFERENCE** and the project's first real successful result. Its exact canonical hashes and quantitative receipt still need consolidation.
- Mini Turbo geometry on the target GTX 1660 SUPER: **PROVEN by local successful generation**, including the shaman run reaching a valid 55,684-face manifold mesh.
- Verified shaman source PNG identity and Windows self-hosted runner execution: **PROVEN**.
- Full shaman route through LOD, texture, rigging, final export and visual validation: **NOT PROVEN yet**.
- Preserve-original-on-audit-ambiguity policy: **IMPLEMENTED AND TARGET-TESTED**, pending proof in the replacement full run.
- TripoSR CPU marching-cubes compatibility on the target PC: **PROVEN by the resumed installer**.
- Resume contracts, foreground-mask mathematics, source-alpha preservation, source appearance forwarding, animation validation, class profiles, Studio registration, installer checkpoints, and command ordering: **UNIT/INTEGRATION-TESTED**.
- Control-service `/health` startup and clean shutdown in the build container: **SMOKE-TESTED**.
- A-pose policy and normalizer exist, but main-route integration and Blender/Unreal deformation quality: **NOT PROVEN yet**.
- Automatic organic weights require visual inspection before shipping.
- P3-SAM remains experimental and disabled by default.
