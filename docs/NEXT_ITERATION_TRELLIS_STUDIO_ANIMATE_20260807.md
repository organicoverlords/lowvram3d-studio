# Next Iteration Design: TRELLIS → 3D Gen Studio → Segment → Rig → Animate

**Date:** 2026-08-07  
**Branch:** `agent/studio-rig-pipeline-20260807` (PR #18 draft)  
**Foundation:** `agent/scene-pipeline-smoke-20260803` @ `9533cf8eef4309cdcacc213b9f1bbfbb920e044e`  
**Status:** DESIGN PROPOSAL — no GPU runtime claimed; vendor-first test lane  
**Pinned vendors:**  
- `PozzettiAndrea/ComfyUI-UniRig@69ee59dc459d2da7cb0291930c1f944886c31d7c`  
- `visualbruno/3DGenStudio@4c2c3f8da9cbd4ef04ebf59f79738be7c9d774ad` (`mcp/tools/workflows.js`)

---

## 1. Objective

Move beyond pure TRELLIS image→mesh to a **pinned, reproducible, evidence-driven** studio iteration that sequences:

```
source image
 → TRELLIS.2 geometry (high-quality structured latents, not Mini Turbo on this path)
 → native Stage 6 finalization + Hunyuan3D-Paint + Blender raster QA (preserved textured LOD0)
 → 3D Gen Studio import & ComfyUI orchestration (reuse, not reimplementation)
 → conditional segmentation (recovery-only)
 → stock rig (MIA for humanoid, UniRig for creature) via ComfyUI-UniRig @ FP16
 → stock animation retarget (UniRigApplyAnimation → Studio → Unreal IK Retargeter)
 → engine export + skeletal LOD + 5-pose + performance validation
```

**Non-goal retained from `docs/RIGGING_PIPELINE_V3.md`:** do not replace the working TRELLIS/Hunyuan foundation, do not patch MIA/UniRig/skinning/attention before stock is measured and rejected on the target GTX 1660 SUPER 6 GB. Unreferenced adapter notions stay out until the proof boundary is crossed.

---

## 2. Research Synthesis (evidence for the design)

### 2.1 3D Gen Studio — what exists to reuse

Studio is an open-source AI 3D mesh production layer: React + Node/Express, visual graph editor + mesh editor + kanban, local storage and ComfyUI API-mode execution. Tagline from upstream: orchestrate text→image, image edit, mesh gen, UV unwrap, texturing in one workspace powered by ComfyUI and external APIs.

**Exactly what this iteration reuses (pinned @ `4c2c3f8da9cbd4ef04ebf59f79738be7c9d774ad`, `mcp/tools/workflows.js`):**

| Studio MCP tool | Role in this iteration | Already proven in Studio |
|---|---|---|
| `inspect_workflow` | Enumerate literals/terminal outputs of a ComfyUI **API-format** graph before import | Studio expects `nodeId → {class_type, inputs}` not UI `nodes/links` |
| `import_workflow` | Save graph to Studio workflow library with mesh/image/string/number/boolean params + result nodes | Declares parameters, not just a file drop |
| `run_workflow` | Upload local mesh via `fileInputs` or Studio asset id, submit to configured ComfyUI, SSE progress, persist + attach results to graph/kanban, handle prompt ids + asset ancestry | This is the client we do NOT reimplement in `src/lowvram3d/comfyui_client.py` unless rejected with evidence |
| `get_run_status` | Poll a workflow that outlives the initial timeout | Required for rig jobs >20s |

**Design consequence:** No second ComfyUI submission/progress/asset-attachment client is built until `run_workflow` on the target machine is exercised and rejected. The lock file, receipts, and provenance live in Studio's asset/version system, not a parallel lowvram3d queue.

### 2.2 Segmentation — BiRefNet, SAM family, RMBG

**BiRefNet (Bilateral Reference Network):** SOTA salient object segmentation, native ComfyUI support (PR #12747), packaged as `viperyl/ComfyUI-BiRefNet`. The repo's own `workers/avatar_preprocess.py` already hit the key low-VRAM finding: **FP16 on sm_75 (Turing GTX 1660 SUPER) produces all-NaN mattes → illegal memory access → poisoned CUDA context, no in-process recovery.** The fix shipped is `cuda/fp32@768` (identical mask to fp32@1024 at ~2.3 GB vs ~3.4 GB). Background-removal matte at 1024 FP16 is not the 6 GB default.

**SAM family in ComfyUI:** `1038lab/ComfyUI-RMBG` bundles RSD 2026-era engines: RMBG-2.0, INSPYRENET, BEN/BEN2, BiRefNet, Lucida, SDMatte, **SAM, SAM2, GroundingDINO** + real-time background replacement. SAM 3.1 adds text-prompted segmentation (InVideo 2026 survey: SAM 3.1 vs BiRefNet vs BEN v2 vs depth/clean-plate). Studio itself can call SAM3 for 2D source masks.

**3D segmentation:** `VAST-AI-Research/UniRig` / `artem-milos-sergeev/puppeteer` style semantic mesh segmenters exist for accessory isolation, but on the shaman asset `98.97%` of faces sit in one component (`docs/CURRENT_STATE.md` — `pipeline-v2-validation` audit) so loose-part splitting cannot isolate staff/antlers/cords. `docs/SHAMAN_RIG_PROGRESS.md` shows horizontal-slab `PARTS` flagged `POSITIONAL_BANDS_ONLY_ANATOMY_NOT_PROVEN` on a robed figure precisely to avoid invented confidence.

**Design stance:** Segmentation stays **recovery-only**, not mandatory pre-rig. BiRefNet stays for 2D matting (FP32/768), SAM3 via Studio/ComfyUI is available for accessory isolation only if deformation QA proves weight bleed or a rigid staff/lantern must move independently. No new semantic-mesh segmenter is authored before that gate.

### 2.3 Animation pipelines

ComfyUI-UniRig ships two pinned graphs:

- **`mia_humanoid.json`** — `UniRigLoadMesh → MIALoadModel (fp32→fp16 on change) → MIAAutoRig → UniRigPreviewRiggedMesh`. First smoke target.
- **`apply_animation.json`** — `UniRigLoadRiggedMesh + UniRigApplyAnimation (mixamo/animation_type) → Preview3D`. Requires a valid rigged FBX first.

Studio's existing rigging/animation lane integrates **TokenRig** (Studio-native) with rig transfer + naming templates + animation workflow. Upstream SkinTokens docs place TokenRig well above 6 GB direct-CUDA. Decision from `docs/STUDIO_COMFY_RIG_INTEGRATION.md`: keep TokenRig in the comparison set (bounded load test ok) but **do not make it the first 6 GB runtime target** — re-attempting OOM tuning without an upstream low-VRAM mode is not productive.

For game deliverable, **Unreal IK Retargeter / Auto Retarget** outranks custom Python retarget math (`integrations/unreal/`). Validate idle + walk + one high-motion clip (Breakdance.mixamo.fbx in the pinned workflow is a useful high-motion probe).

### 2.4 ComfyUI custom-node integration pattern

ComfyUI-UniRig is `GPL-3.0`, manager-installable (`PozzettiAndrea/ComfyUI-UniRig`, 415 stars/46 forks), with three install routes: Manager search, Manager Git URL, or manual `git clone + pip install -r requirements.txt + python install.py`. It bundles its own Blender + UniRig/MIA code and recommends MIA for humanoids. `nodes/__init__.py` registers: `UniRigLoadMesh`, `MIALoadModel`, `MIAAutoRig`, `UniRigPreviewRiggedMesh`, `UniRigLoadRiggedMesh`, `UniRigApplyAnimation`. `comfy-env-root.toml` uses `comfy-env`+pixi for isolated install — **do not let it mutate the managed 3D Gen Studio ComfyUI** without a preflight.

### 2.5 Low-VRAM optimizations for 6 GB

| Lever | Setting for this iteration | Source |
|---|---|---|
| Precision | MIA/UniRig **FP16** first, SDPA where exposed (UniRig exposes `attn_backend`=auto/sdpa). Do not change model code. | Vendor-first rule + APatero 2025 low-VRAM guide principle: "cut VRAM peaks into weights/latents/VAE/attention, not one flag" |
| Resolution | Texture bake **uses Hunyuan Paint `--render-size`/`--texture-size` as fixed** (late-binding bug fixed in smoke `9533cf8`). Re-run size A/B only with fixed renderer and receipt `baked_atlas_size` | `docs/RIGGING_PIPELINE_V3.md` |
| Batch & queue | **One GPU-heavy worker at a time**, `gpu.lock` serialized (`%LOCALAPPDATA%/LowVRAM3DStudio/locks/gpu.lock`), TRELLIS / Hunyuan Paint / MIA never concurrent | `README.md` low-memory rules |
| Attention | SDPA preferred on 6 GB; `enable_attention_slicing("max")` is proven to clobber MV attention (`DecoupledMVRowColSelfAttnProcessor2_0` → `SlicedAttnProcessor`) so it stays disabled until excluded from UNet attn_procs | `LOWVRAM3D_HANDOFF.md` #15 |
| Offload & cleanup | `enable_model_cpu_offload()` + `/free` unload between jobs; `vram_optimizer` custom node (strawberryPunch) pattern — clear unused VRAM between queue executions | ComfyUI docs |
| Ceiling | `5600 MB` configured; ceiling applies to **stage growth = total − baseline − 256 MB reserve**, not whole-GPU, because WDDM `nvidia-smi --query-compute-apps` returns `[N/A]` on this host | `src/lowvram3d/runner.py::gpu_budget` |
| Tiling | No tiled VAE needed for rigging (not a large-latent image node), but retained for any texture-refinement sub-step | Apatero guide |
| Quant | GGUF Q5 / fp8 noted for FLUX/video but **not applied** to MIA/UniRig on first pass — measure stock FP16 before quantizing | Vendor-first |

---

## 3. Pinned Iteration — reproducible sequence

### 3.0 Phase 0 — Environment proof (no CUDA, no download)

**Script:** `tools/rig_vendor_preflight.ps1 -Output evidence/rigging/vendor_preflight.json`

- Locates ComfyUI roots (`GENSTUDIO_DATA_ROOT/comfyui`, `%APPDATA%/comfyui`, `%LOCALAPPDATA%/comfyui`, `C:\AI\ComfyUI`, `C:\ComfyUI`), probes `main.py`
- Resolves python (`comfy-venv/Scripts/python.exe` → `venv/...` → `python_embeded/...` → system python)
- Checks `custom_nodes/ComfyUI-UniRig/.git` HEAD vs `69ee59dc…`, workflow presence, light imports (`trimesh`, `comfy_env`) **without importing torch/CUDA**, optional `/object_info` sweep `8188–8205` for `MIAAutoRig`/`MIALoadModel`
- Emits `evidence/rigging/vendor_preflight.json` with `status ∈ {BLOCKED_NO_COMFY, BLOCKED_NO_PYTHON, INSTALL_REQUIRED, PIN_MISMATCH, PACK_INCOMPLETE, READY_OFFLINE, READY_RUNNING}`

**Workflow:** `.github/workflows/rig-vendor-preflight-local-worker.yml` runs this on `[self-hosted, Windows, X64]`, path-scoped to `agent/studio-rig-pipeline-20260807`, 15 min timeout.

**Gate to advance:** `READY_OFFLINE` or `READY_RUNNING`. Any other status stops the lane and requests an isolated pinned install before patching code.

### 3.1 Phase 1 — TRELLIS → Hunyuan Paint (unchanged foundation)

```
source image (verified SHA-256, e.g. shaman `4d23adc758c5b...` from docs/CURRENT_STATE.md)
 → TRELLIS.2 (structured latents) → weld/smooth/validate → native Stage 6 finalize
 → Hunyuan3D-Paint (fixed render-size/texture-size) → Blender raster QA
 → preserve `game_ready_unrigged.glb` + `game_ready_uv.glb` + atlas 1024 (default production res)
```

- Hook: `src/lowvram3d/raster_route.py` + `workers/raster_project.py` + `blender/raster_export.py` remain the fast path; Cycles `project_texture.py` stays as disabled diagnostic only.
- Receipt truth: `uv_island_occupancy`, `observed_semantic_coverage`, `synthesized_surface_coverage` never conflated; `real_world_size` metadata carried forward (smoke fix).

### 3.2 Phase 2 — Studio import

- On the coordinator (no GPU): `Studio get_run_status` / `run_workflow` not yet; just **asset upload**: `fileInputs` → Studio project asset with provenance (source SHA, TRELLIS seed, Paint baked size, QA receipt).
- No format invention: the GLB is already Studio-consumable (`FILE_3D_GLB`).

### 3.3 Phase 3 — Stock vendor smoke (isolation test)

**Do this before touching our mesh.**

1. Run pinned `integrations/vendor/ComfyUI-UniRig/mia_humanoid.json` **unchanged** except the precision widget `fp32 → fp16`, against vendor `assets/realistic_male_character.glb`.
2. Serialize through `gpu.lock`.
3. Record: wall time, `peak_dedicated_vram_mb`, `peak_shared_spill_mb`, `system_ram_peak_mb`, `armature_present`, `skin_weights_present`, `materials_preserved` (original texture/material count vs rigged), output `mixamo.fbx` SHA-256.
4. Immediately fail-closed on OOM/hardware failure before diagnosing our topology.

### 3.4 Phase 4 — Stock MIA on our production humanoid

Only after Phase 3 passes:

```
Studio asset (preserved textured LOD0, .blend carrier for welded topology)
 → run_workflow with the same stock MIA graph (API-format export, see §4)
 → stock MIA FP16 → rigged FBX attached back to Studio with full provenance
```

- **Carrier rule from `docs/SHAMAN_RIG_PROGRESS.md`:** ship topology as **`.blend`**; GLB re-splits every welded vertex per corner (537k → ~3.2M verts) and destroys heat-diffusion connectivity.
- **Weld policy:** `blender/common.py::weld_vertices(remove_doubles)` at `1e-4` producing ~537k verts / 1.07M faces; lost-face gate is measured collapsible count, not arbitrary ratio.

### 3.5 Phase 5 — Conditional segmentation (recovery only)

Trigger only if deformation QA (§6) sets `weight_bleed_detected=true` or `rigid_accessory_requires_isolation=true` (`src/lowvram3d/rigging_policy.py::needs_segmentation_recovery`).

- 2D path (preferred for 6 GB): Studio/ComfyUI **SAM3** or **BiRefNet FP32@768** on rendered views for mask; not a full mesh segmentation rebuild.
- 3D path (only if 2D insufficient): existing 3D segmenter for staff isolation as `fused_staff_control` (9,845 verts pattern from milestone 1). No boolean hole cut.

### 3.6 Phase 6 — Animation

Only after rig passes §6 promotion gates:

1. `integrations/vendor/ComfyUI-UniRig/apply_animation.json` stock: `UniRigLoadRiggedMesh(fbx_file=mixamo.fbx) → UniRigApplyAnimation(animation_type=mixamo, animation_file=Breakdance.fbx)` → `Preview3D` verify.
2. Then Studio animation path (same graph via `run_workflow`).
3. Then Unreal IK Retargeter validation for at least **idle, walk, high-motion (Breakdance)** — proves engine import before LOD.

### 3.7 Phase 7 — Export & skeletal LOD

Order enforced by `src/lowvram3d/rigging_policy.py::PIPELINE_ORDER` / `pipeline_stage_order`:

```
preserve_textured_lod0 → rig_and_skin → static_rig_qa → deformation_qa
 → animation_retarget → engine_export → skeletal_lod_generation
```

Do not revert to `LOD0/LOD1/LOD2 → rig LOD0` without proven weight transfer across independent LODs.

---

## 4. Workflow Format Conversions (the non-obvious boundary)

| Artifact | Format | Where it lives | Conversion |
|---|---|---|---|
| `mia_humanoid.json` / `apply_animation.json` | **ComfyUI UI format** (`nodes`/`links`, widget `widgets_values`) | `integrations/vendor/ComfyUI-UniRig/` (pinned blob SHA beside `VENDOR_LOCK.json`) | None — run directly in ComfyUI for Phase 3 |
| API-format graph | **ComfyUI API format** (`nodeId → {class_type, inputs}`) | `integrations/vendor/ComfyUI-UniRig/api/*.json` (new provenance artifact, added at conversion) | In ComfyUI: `File → Export (API)` on the passing UI graph (or `comfy --export-api`). Do NOT hand-write. Verify node types match: `UniRigLoadMesh`, `MIALoadModel`, `MIAAutoRig` survive export. |
| Studio import | API-format only | Studio `inspect_workflow` input | Feed the exported API JSON to `inspect_workflow` → `import_workflow` with params `{file_path: mesh, precision: string(enumerated fp32/fp16), attn_backend: combo}` and `result_nodes=[MIAAutoRig.fbx_output_path]` |

**Provenance:** keep all three side-by-side: `VENDOR_LOCK.json` (pin) + `*.json` (UI) + `api/*.json` (API export) + export receipt (export timestamp, ComfyUI version, frontendVersion). A drift between UI and API graphs is a defect, not a silent edit.

**Asset handoff:** TRELLIS GLB → Studio asset `fileInputs` is a direct file upload, not a format transcode. All metadata transcodes are JSON receipts (`source SHA`, `TRELLIS seed`, `Paint baked_atlas_size`, `real_world_size`, `peak VRAM`).

---

## 5. Commit Pins & Reproducibility

### 5.1 Exact pins (do not float)

```json
// integrations/vendor/ComfyUI-UniRig/VENDOR_LOCK.json
{ "repository": "PozzettiAndrea/ComfyUI-UniRig", "commit": "69ee59dc459d2da7cb0291930c1f944886c31d7c",
  "workflows": {
    "mia_humanoid.json": { "blob_sha": "851db1ce915652f4d6ff28cbdd1a9699f55487c3" },
    "apply_animation.json": { "blob_sha": "e7b5258fbc3507e964d8220cea2eae8ffc24452c" }
  }
}
// integrations/vendor/3DGenStudio/VENDOR_LOCK.json
{ "repository": "visualbruno/3DGenStudio", "commit": "4c2c3f8da9cbd4ef04ebf59f79738be7c9d774ad",
  "source_path": "mcp/tools/workflows.js",
  "reuse": ["inspect_workflow","import_workflow","run_workflow","get_run_status"] }
```

 Smoke base is intentionally pinned at `9533cf8` until the vendor-first lane has runtime proof; syncing before Phase 3 passes risks re-baselining without evidence.

### 5.2 Failure ladder (so pins mean something)

- Stock MIA fails on **vendor asset** → diagnose install/hardware/backend, do not touch our mesh.
- Stock MIA passes vendor but fails **our humanoid** → diagnose pose/topology/material handoff; then test **stock UniRig** (`UniRigAutoRig` SDPA/FP16) before any patch.
- Stock UniRig fails on same asset → only then justify patched Puppeteer+SDPA (FlashAttention-2 → SDPA is a code change, not a config tweak).
- Stock animation fails after valid rig → test Studio/Unreal retarget before custom animation math.

### 5.3 Reproducibility contract per run

Each run writes a machine-readable receipt next to the asset:

```json
{
  "schema": "lowvram3d_studio_rig_run_v1",
  "source": {"sha256": "…", "path": "C:\\Users\\Lauri\\Downloads\\…png"},
  "gen": {"backend": "trellis2", "seed": 7, "octree_res": 512},
  "paint": {"renderer": "hunyuan_fixed", "baked_atlas_size": 1024, "real_world_size_m": {"x":1.65,"y":0.59,"z":1.98}},
  "preflight": "evidence/rigging/vendor_preflight.json",
  "comfy": {"root": "…\\comfyui", "python": "…\\python.exe", "server": "http://127.0.0.1:8188"},
  "vendor_pin": {"comfy_unirig": "69ee59dc…", "studio": "4c2c3f8…"},
  "workflows": {"ui_sha": "851db1ce…", "api_sha": "…"},
  "perf": {"wall_s": 42.3, "peak_dedicated_vram_mb": 5120, "peak_shared_mb": 340, "system_ram_mb": 8100},
  "rig": {"iqr": {"armature_present": true, "skin_weights_present": true, "materials_preserved": true, "bones": 67}},
  "deformation": {"rest_pose": true, "elbow_bend": true, "knee_bend": true, "hip_crouch": true, "shoulder_raise": true},
  "animation": {"mixamo_Breakdance": true, "studio_retarget": true, "unreal_ik_retarget": true},
  "fresh_import": {"blender": true, "triangles": 219967},
  "promotion": "PROVEN | REJECTED | NOT_PROVEN",
  "gpu_lock": "%LOCALAPPDATA%/LowVRAM3DStudio/locks/gpu.lock"
}
```

---

## 6. Evidence-Driven Validation (VRAM/time, 5-pose)

### 6.1 Machine gates (fail-closed, from `src/lowvram3d/rigging_policy.py`)

`evaluate_rig_promotion(report, plan)` fails if:

- `armature_present != true`
- `skin_weights_present != true` (checked via `bpy.data.armatures` + vertex groups, not FBX byte size)
- `materials_preserved != true` when `preserve_textured_lod0=true` (material count + texture SHA before/after)
- `peak_vram_mb > vram_ceiling_mb` (5600) when measured
- any of `DEFORMATION_POSES = (rest_pose, elbow_bend, knee_bend, hip_crouch, shoulder_raise)` not `passed=true`

Visual rejection overrides metrics. A written FBX/GLB is **not** success until rendered evidence passes.

`needs_segmentation_recovery(report)` → `weight_bleed_detected || rigid_accessory_requires_isolation`. Alone it never auto-segments.

### 6.2 5-pose deformation evidence (Blender proof)

Script: `blender/render_rig_deformation_set.py` (to be added; follows `blender/rig_animate.py` + `blender/render_glb_diagnostic_set.py` convention, run with `--python-use-system-env` and `PYTHONPATH=src;blender`).

- Load rigged FBX/GLB in a **fresh Blender process** (not the export session).
- Apply five deformations:
  1. **rest_pose** — neutral, check bind pose not collapsed
  2. **elbow_bend** 90° — left & right forearm, detect cape bleed (`ARMS_DOWN_AT_SIDES_ELEVATES_ARM_TORSO_WEIGHT_BLEED_RISK`)
  3. **knee_bend** 90° — robe/skirt must not drag rigidly nor invert
  4. **hip_crouch** — spine + hips, staff control stays rigid if `PARTS=staff`
  5. **shoulder_raise** 160° — overhead reach, reveals antler/cord topology stress (the `+32 non-manifold` class of failure)
- For each pose: render 4 views (front/side/rear/three-quarter, STUDIO ortho `ortho_scale=2.6`, `BASECOLOR_EMISSION` — never lit), record vertex-group heatmap, write `deformation_poses[pose].png` + `deformation_poses[pose].json` (`passed`, `weight_bleed_px`, `unweighted_verts`).

**Pass =** no major body region moves with wrong bone, no detached screen-space islands in any of six views, no unweighted verts (`max influences=4` envelope), staff group bound to staff bone (not torso), zero new boundary/non-manifold edges vs rig base.

### 6.3 Performance evidence

- Measure `wall_time_s` from `run_workflow` prompt submission to `get_run_status==succeeded` (or ComfyUI `/history` for direct runs).
- Sample `peak_dedicated_vram_mb` + `peak_shared_mb` via `nvidia-smi --query-gpu=memory.used,memory.total` deltas or `runner.py::gpu_budget` baseline-delta method (per-process `query-compute-apps` is `[N/A]` on WDDM, so delta is authoritative). Also record `system_ram_peak_mb`.
- Report per stage, not aggregated: TRELLIS, Paint, MIA, Animation are separate rows; OOM in one must not be averaged away.

### 6.4 Full promotion checklist (all required)

```
GEOMETRY_REUSED=PROVEN            (textured LOD0 SHA unchanged)
UV_PRESERVED=PROVEN
VENDOR_PIN_EXACT=PROVEN            (69ee59dc / 4c2c3f8 SHAs match)
MIA_STOCK_ON_VENDOR_ASSET=PROVEN   (vendor rig passes machine gates)
MIA_STOCK_ON_HUMANOID=PROVEN        (our humanoid, FP16, gpu.lock serialized)
MATERIALS_PRESERVED=PROVEN
ARMATURE_AND_WEIGHTS=PROVEN
FIVE_POSE_DEFORMATION=PROVEN        (all 5 poses, contact sheet)
ANIMATION_RETARGET=PROVEN           (Breakdance + idle/walk via Studio path)
ENGINE_IMPORT=PROVEN                (Unreal import after deformation proof)
SKELETAL_LOD=PROVEN                 (only after rig, not before)
FRESH_IMPORT_VALIDATED=PROVEN       (fresh Blender process, triangles/components match)
PEAK_VRAM_RECORDED=PROVEN           (± wall time)
```

Any `REJECTED` on a stock path is diagnosed in that order before a patch is authored; the working unrigged asset is **never overwritten** by a failed rig (`LOWVRAM3D_HANDOFF.md` rule 6 retained).

---

## 7. Configuration & Documentation Changes

### 7.1 No pipeline.py replacement needed on first pass

The iteration is orchestrated via existing entrypoints:

- `scripts/prefetch_models.py` (weights)
- `workers/rig_backend_preflight.py` + `tools/rig_vendor_preflight.ps1` (policy)
- Studio MCP for the heavy step (no new `comfyui_client.py` path until proven inadequate)
- `src/lowvram3d/rigging_policy.py` already exposes `build_rigging_plan(asset_type, rig_kind="auto")` — humanoid→`mia`/`unirig` fallback, creature→`unirig`, `segmentation_before_rig=False`, `generate_lods_after_rig=True`.

### 7.2 Docs to update

| File | Action |
|---|---|
| `docs/STUDIO_COMFY_RIG_INTEGRATION.md` | Keep as integration contract; add this doc's §4 format table as appendix |
| `docs/VENDOR_RIG_TEST_PLAN.md` | Keep Phase 0/1/2 + failure ladder; reference this doc for 5-pose script |
| `docs/RIGGING_PIPELINE_V3.md` | Retain ordering + TokenRig note; mark "next: 5-pose receipt" after smoke sync |
| `docs/CURRENT_STATE.md` | On first promotion, append `STUDIO_RIG_V1_PROVEN` row with artifact SHA + runtime table |
| `docs/HARDWARE_RESEARCH.md` | Append low-VRAM table from §2.5 (BiRefNet fp32@768, gpu.lock, WDDM N/A) |
| `integrations/vendor/**` | No float; add `api/*.json` provenance artifacts only after successful export |

### 7.3 CI

`.github/workflows/rig-vendor-preflight-local-worker.yml` remains the only workflow touching this lane. PR #18 stays **draft, unmerged** until `READY_RUNNING` + 5-pose receipt exist. Do not add a full-GPU CI job that collides with the shared lock or reintroduces the `test_core` collection collision owned by the parallel session.

---

## 8. Commit Pin Guidance

- Sync PR #18 to `agent/scene-pipeline-smoke-20260803` only after Phase 3 passes on the vendor asset; syncing earlier without new evidence merely moves the baseline.
- Pin commits are verification inputs, not version wishes: if `rig_vendor_preflight.ps1` reports `PIN_MISMATCH`, create an **isolated pinned stock test env** (second ComfyUI root) rather than force-updating the Studio-managed ComfyUI in place.
- Exported API graphs are content-addressed (SHA of the exported file) and versioned alongside the UI graph — the blob SHA in `VENDOR_LOCK.json` is the test oracle.

---

## 9. Risks & Mitigations Retained from Prior Sessions

- **Triangle-soup re-split:** Mitigated by `.blend` carrier + explicit welded-base receipt (537k verts datum).
- **BiRefNet NaN on Turing FP16:** Mitigated by FP32@768 default; not revisited until fp32 fails.
- **MV-Adapter NaN after dtype fix:** Not on this path (TRELLIS/Hunyuan route), but reminder: `cond_encoder.to(dtype)` alone did not fix it — module-level NaN probe was never completed.
- **Attention slicing clobber:** Stay disabled for MIA/UniRig until proven excluded from `attn_processors`.
- **Late-binding Paint sizes:** Use fixed renderer receipt, not requested value.

---

## 10. Next Action (ordered)

1. On target PC, run `tools/rig_vendor_preflight.ps1` → `evidence/rigging/vendor_preflight.json`; publish artifact.
2. Run vendor `mia_humanoid.json` FP16 on `realistic_male_character.glb` through ComfyUI directly; collect `peak VRAM + 5-pose` receipt.
3. Export the passing graph to API format → `integrations/vendor/ComfyUI-UniRig/api/mia_humanoid.api.json` → `inspect_workflow` → `import_workflow` → prove Studio `run_workflow` on vendor asset.
4. Repeat (3) on one preserved production humanoid from `TRELLIS.2 → Hunyuan Paint` (textured LOD0 `.blend`).
5. Run 5-pose Blender proof + Mixamo Breakdance retarget + Unreal import; write promotion receipt.
6. Only on measured stock failure, test stock UniRig; only after both stock routes fail, propose Puppeteer+SDPA patch with its own VENDOR_LOCK.

---

*This document is the pinned iteration design. Runtime proof is not claimed here. Every downstream claim must cite a `vendor_preflight.json`, a stock workflow SHA, a `peak VRAM/wall time` row, a 5-pose contact sheet, and a fresh-import validation.*
