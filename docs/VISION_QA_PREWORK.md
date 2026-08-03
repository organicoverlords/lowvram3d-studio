# Vision QA and bounded pipeline supervision

Status: **implementation prework only**. No new model has been downloaded, loaded, or allowed to control a pipeline stage.

## Purpose

Add visual understanding to the existing measured 3D pipeline without replacing deterministic proof. The vision layer classifies visible defects, labels important regions, requests a second opinion when uncertain, and selects one bounded recovery action. The state machine remains the controller.

The current Castlegrounds source of truth remains `IMAGE_TO_SCENE_PARTIAL`. Its exact source-camera gate is rejected on mesh coverage while raw reprojection and coordinate transforms are proven. This lane must not rerun MoGe or reinterpret that failure as a camera problem.

## Architecture

```text
stage output
  -> deterministic mesh/UV/texture/import checks
  -> Blender evidence pack with exact camera and diagnostic passes
  -> specialist models: labels, masks, independent depth
  -> primary small VLM
  -> optional independent VLM second opinion
  -> fail-closed policy and bounded retry budget
  -> whitelisted script action or quarantine
```

The model does not receive shell access. It returns `vision_qa_decision_v1`; `policy.py` rejects unknown actions, exhausted retry budgets, mismatched packet identities, missing evidence references, or attempts to pass rejected hard gates.

## Evidence pack

Each artifact is content-addressed and referred to by an ID. The prompt intentionally omits local file paths. Required geometry evidence is:

- locked source image;
- exact source-camera unlit render;
- silhouette pass;
- wireframe pass;
- depth pass;
- normal pass;
- metrics receipt.

Texture adds beauty, albedo, atlas and semantic masks. UV adds atlas, seam and exact-overlap receipts. Rigging adds neutral and extreme-pose evidence plus projected keypoint metrics. Export adds a fresh import receipt, engine render and log.

## Controller rules

1. `REJECTED`, `BLOCKED`, and `NOT_PROVEN` hard gates cannot be converted into `PASS` by a VLM.
2. A user visual rejection is authoritative even when numeric gates pass.
3. A retry can select only one action explicitly allowed for that stage.
4. Retry counts are stored in the packet and cannot be reset by the model.
5. Threshold changes are not model actions.
6. Confidence from `0.65` to `0.90` requires a second independent model.
7. Conflicting model verdicts require user review.
8. Models run in separate processes and only one heavy GPU process may exist at once.
9. Model compatibility requires a target-PC receipt; parameter count and file size are not proof.
10. A successful process benchmark does not prove visual quality.

## Initial action whitelist

### Geometry

- `repair_mesh_coverage_from_saved_arrays`
- `build_bounded_edge_threshold_candidate`
- `build_bounded_winding_candidate`
- `rerender_exact_source_camera`

The Castlegrounds lane may use saved `points.npy`, `depth.npy`, `normal.npy`, `mask.npy`, and intrinsics. It must not rerun MoGe during a coverage repair.

### UV

- `rerun_uv_with_locked_geometry`
- `repack_uv_without_geometry_change`

### Texture

- `rerun_projection_for_region`
- `increase_region_projection_priority`
- `rerender_texture_evidence`

### Rig and export

- region-bounded reweight or evidence rerender;
- fresh reimport or engine-evidence rerender.

## Runtime separation

Use isolated environments because the current pipeline has already encountered incompatible Torch, Transformers, CUDA and custom-extension combinations.

- `vision-controller`: dependency-free contracts, policy and local HTTP client.
- `vision-vlm-primary`: one quantized Qwen3.5-2B server candidate.
- `vision-vlm-secondary`: one quantized MiniCPM-V 4.6 server candidate.
- `vision-labels`: Florence-2 and, only after compatibility is proven, EdgeTAM.
- `vision-da3`: Depth Anything 3 Small.
- existing MoGe environment remains unchanged.

No environment is installed into Blender, ComfyUI, Unreal, or the existing MoGe venv.

## Target-PC benchmark order

1. CPU preflight and contract tests.
2. Load-only model test.
3. One-image inference at 448 or 512 long edge.
4. JSON/output validation.
5. Peak VRAM and wall-time receipt.
6. Process-exit and VRAM-release validation.
7. Fixed visual benchmark with known pass/fail pairs.
8. Only then enable advisory classification.
9. Enable bounded retries only after advisory accuracy is proven.

Start with Florence-2 and DA3 Small, then MiniCPM-V, then Qwen3.5-2B. The supervisor is not the first model to install because specialist outputs are easier to validate objectively.

## Acceptance boundary

This branch can prove only CPU contracts, policy behavior, prompt construction, evidence hashing and process-benchmark instrumentation. It cannot prove:

- any model fits in 6 GB VRAM;
- any model sees the demonstrated face/atlas defects correctly;
- Depth Anything 3 is independent enough for a particular source;
- a VLM should be trusted to trigger automatic retries;
- Unreal visual capture.
