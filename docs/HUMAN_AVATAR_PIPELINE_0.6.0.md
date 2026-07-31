# Local human avatar and dance pipeline — 0.6.0

## Purpose

This mode turns one **full-body photograph of one person** into a locally processed, textured, rigged GLB with a loopable dance action. It is designed for the GTX 1660 SUPER 6 GB target and therefore avoids the heavier research stacks that normally assume 8–24 GB VRAM.

The registered 3D Gen Studio provider is:

```text
LowVRAM One-Click — Photoreal Human Avatar + Dance
```

## Recommended source photograph

Use a high-resolution full-body image with:

- one person only;
- head, hair, hands, knees, ankles, and feet visible;
- arms separated from the torso where practical;
- limited motion blur and no large foreground occluders;
- even lighting without hard coloured backlight;
- clothing and accessories that should appear in the final model;
- enough space around the body so no part touches the image edge.

A face-only selfie cannot supply body shape, clothing, feet, or hidden-surface evidence. The pipeline will continue in degraded mode when possible, but it cannot honestly infer a faithful full-body digital double from a cropped face photograph.

## Local execution path

```text
photograph
→ pinned BiRefNet foreground inference
→ MediaPipe 33-landmark pose and person-mask analysis
→ edge-aware alpha refinement
→ foreground colour decontamination
→ largest-person selection
→ centred 1024×1024 transparent full-body reference
→ Mini Turbo high-poly geometry
→ character-preserving retopo and UV
→ authoritative source-facing appearance projection
→ MV-Adapter hidden-view completion or deterministic fallback
→ PBR bake
→ pose-guided humanoid template rig
→ automatic skin weights
→ idle + loopable dance action
→ GLB/FBX export
→ clean re-import validation of mesh, textures, armature, and dance action
```

The user photograph and generated assets remain in the local job directory. Model weights are fetched from their upstream repositories during installation; asset processing does not require sending the photograph to a hosted avatar API.

## Background-removal safeguards

The avatar preprocessing worker uses the pinned model snapshot:

```text
model: ZhengPeng7/BiRefNet
revision: e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4
input: 1024×1024
```

The final matte is not accepted directly without post-processing. The worker:

1. uses MediaPipe segmentation only as a bounded body-completeness prior;
2. keeps the largest connected foreground subject;
3. closes small holes without merging distant people or objects;
4. uses a grayscale guided filter around uncertain boundaries;
5. decontaminates semi-transparent edge colours to reduce green/white backdrop fringes;
6. preserves the resulting alpha instead of removing the background a second time;
7. records source framing, normalized framing, component counts, landmarks, and warnings.

## Identity and hidden surfaces

The cleaned source image is weighted three times more strongly than generated front-view colour during projection. This makes the visible face, hair, clothing, and accessories the authoritative appearance source where the generated geometry is visible from the original camera.

A single photograph contains no true evidence for the back of the body or hidden surfaces. Lane A uses MV-Adapter to produce coherent additional views. Lane B mirrors and palette-matches the source deterministically. Neither is represented as ground-truth identity for unseen details.

## Animation

The generated GLB includes:

- a humanoid armature fitted from the source pose proportions;
- automatic mesh weights;
- an `idle` action;
- a loopable `dance_loop` action when the dance preset is selected.

Final validation rejects an avatar export when the cleanly re-imported GLB lacks an armature, animation actions, or a dance action. Automatic weighting can still produce elbow, shoulder, hand, clothing, or skirt deformation defects, so visual review remains mandatory.

## Resume behavior

Avatar preprocessing is a normal checkpointed asset stage. A later UV, bake, rig, or export failure reuses the validated mask, normalized photograph, raw high-poly mesh, and earlier completed processing stages. The installer similarly reuses all completed environments and retries only fingerprints changed by 0.6.0.

For an existing 0.5.1 installation:

- stages 01–07 remain reusable;
- stage 08 reruns only to add/verify MediaPipe and the compatible Hugging Face runtime;
- completed TripoSR stage 09 remains reusable;
- stage 10 retries with safe JSON property insertion;
- stage 12 downloads or verifies the pinned BiRefNet snapshot.

## Proof boundary

### PROVEN

- The target machine has previously produced Mini Turbo geometry on the GTX 1660 SUPER.
- The resumed installer has already completed TripoSR with the CPU marching-cubes compatibility layer.

### TESTED in the build environment

- largest-subject isolation;
- pose-mask hole recovery bounded to the subject envelope;
- edge-aware alpha refinement;
- foreground colour decontamination;
- square-canvas normalization and landmark transformation;
- pinned model revision in runtime and cache prefetch;
- source alpha preservation through view preparation;
- source appearance passed into the PBR bake;
- dance-action validation contract;
- stage-10 JSON property insertion contract;
- installer and asset resume contracts.

### NOT PROVEN yet

- final likeness on the physical GTX 1660 SUPER machine;
- peak BiRefNet and MV-Adapter VRAM on that machine;
- automatic skin-weight quality on the first real avatar;
- dance deformation quality in 3D Gen Studio, Blender, or Unreal;
- faithful geometry for body regions not visible in the source photograph.
