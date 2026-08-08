# LowVRAM3DStudio — Session Handoff

> **Read `evidence/RESULTS.md` first.** It carries the current measured state, the known
> limitations, and the six measurement bugs that invalidated earlier conclusions. Everything below
> the "CURRENT STATE" section is older history, kept for context but partly superseded.

---

## UPDATE — castle texture and inspection handoff (2026-08-08)

The promoted castle deliverable is now the CPU-projected, source-conditioned result:

- `evidence/deliverables/castle_trellis512_tu116_cpu_projection_tex2048.glb`
- `evidence/deliverables/blender/castle_trellis512_tu116_cpu_projection_tex2048.blend`
- `evidence/deliverables/views9/castle_trellis512_tu116_cpu_projection_tex2048_9view_grid.png`
- `evidence/compare/castle_new/cpu_projection_2048/projection_receipt.json`

The original TU116 geometry and UV/index buffers are preserved byte-for-byte by the projection receipt (`geometry_uv_index_preserved=true`). The 2048 atlas has 2,431,817 owned texels, 17.0% observed coverage, and a fresh CPU native-texture render is `PROVEN`. The prior Hunyuan-painted castle variants remain comparison evidence only: their atlases were visibly muddy/dark and are not promoted.

`ALL_ASSETS_inspection.blend` now sorts imported deliverables by filesystem creation time and adds one metadata placard per asset. Placards record name, route, creation date/source, file size, source path, atlas size, face count, rig-readiness, animation status, and inspection metadata version (`v1`). Rigging and animation are reported as unknown unless proven by a receipt; the shaman is marked readiness-not-ready and the finished frog readiness-ready.

The frog CPU rig candidate was built only in the session scratchpad. It created a generic creature armature and `creature_idle`, but automatic bone weighting emitted a Blender warning and the exported GLB dropped the armature; therefore deformation and animation remain unproven and the candidate was not promoted. The finished frog GLB remains unchanged.

---

## CURRENT STATE (latest session)

Geometry repair and texturing are done and measured. Rigging is still untested and is the next
step. The pipeline stops at a validated unrigged export by design.

Accepted asset: `evidence/deliverable/game_ready_unrigged.glb` (40,317 faces, 2048 atlas,
smooth-shaded, one material, UVs embedded).

| | value |
|---|---|
| faces | 40,317 (main component 36,705, never modified) |
| components | 52 |
| boundary edges | 54 (down from 136 — no removal opened a hole) |
| non-manifold edges | 1 |
| detached screen-space islands | 0 in all six views |
| UV charts | 1,139 |
| native atlas utilization | 81.03% (was 27.89%) |
| observed / synthesized coverage | 23.36% / 76.64% |

### How to reproduce this run

Prerequisites (installed into the control env this session):

```bash
"%LOCALAPPDATA%\LowVRAM3DStudio\envs\control\Scripts\python.exe" -m pip install xatlas open3d pytest
```

Paths below assume `CONTROL` is the control-env python, `SRC` the repo root, `BL` the Blender 5.2
executable, and `JOB` a writable job directory. Blender stages need
`PYTHONPATH=$SRC/blender:$SRC/src` and `--python-use-system-env`, or they die on
`No module named 'common'`.

1. **Component audit** (Open3D clustering; caches every geometric metric so nothing is recomputed
   later):
   `$CONTROL analysis/stage1_component_audit.py <input.glb> $JOB/geometry_audit/cache.npz $JOB/geometry_audit/geometry.json`

2. **Six-view ID/depth/mask buffers** (deterministic numpy ortho rasteriser — six renders total,
   never one per component):
   `$CONTROL analysis/stage2_component_id_render.py $JOB/geometry_audit/cache.npz <front_view.png> $JOB/geometry_audit/idrenders $JOB/geometry_audit/screen.json`

3. **Classification** (pure rules, zero inference calls):
   `$CONTROL analysis/stage3_classify.py $JOB/geometry_audit/geometry.json $JOB/geometry_audit/screen.json $JOB/geometry_audit/decisions.json`

4. **Debris removal** (deletes only `REMOVE_OUTBOARD_DEBRIS`; asserts the main component and
   boundary-edge count are unchanged, and that the source file stays byte-identical):
   `$CONTROL analysis/stage4_remove_debris.py $JOB/geometry_audit/cache.npz $JOB/geometry_audit/decisions.json <input.glb> $JOB/cleanup`

   Repeat steps 1–4 until `removed_component_ids` comes back empty. Removing debris can reveal
   fragments that were previously hidden behind it, and component IDs are renumbered each pass, so
   one pass is not enough. It converged in three passes here.

5. **UV unwrap + full metrics + selection**:
   `$CONTROL analysis/stage5b_finalize_uv.py $JOB/cleanup/debris_clean_candidate.glb $JOB/uv $SRC/src`

6. **Rebuild the raster NPZ** (never reuse an NPZ across a geometry or UV change — it is indexed by
   topology):
   `$CONTROL analysis/stage6_build_npz.py $JOB/uv/game_ready_uv.glb $JOB/raster/mesh_uv.npz $JOB/raster/report.json`

7. **One 2048 texture pass**:
   `$CONTROL workers/raster_project.py --npz $JOB/raster/mesh_uv.npz --views-dir <views> --view-metadata <views>/view_metadata.json --output-dir $JOB/textured --atlas-size 2048 --report $JOB/textured/report.json`

8. **Export** (applies smooth shading and packs the atlas):
   `$BL --background --python-use-system-env --python $SRC/blender/raster_export.py -- --cleaned-mesh $JOB/uv/game_ready_uv.glb --atlas $JOB/textured/basecolor.png --output $JOB/candidate.glb --texture $JOB/basecolor_2048.png --report $JOB/export.json`

9. **Validate in a fresh Blender process, then render**:
   `$BL --background --python analysis/phaseA_darkness_diagnosis.py -- $JOB/candidate.glb $JOB/textured/basecolor.png $JOB/diag`
   `$CONTROL analysis/phaseA_stats.py $JOB/diag $JOB/textured/basecolor.png $JOB/textured/debug_coverage.png $JOB/luminance.json`

Tests: `PYTHONPATH="src;." python -m pytest tests -q` from the repo root.

### Rules that must not be relaxed

* **Never diagnose darkness from a lit render.** Use `BASECOLOR_EMISSION` (texture straight to
  Emission, lighting disabled). Doing this closed a darkness hypothesis that had already caused one
  wrong fix: the texture was uniform (rear/front emission ratio 0.92) and the apparent darkness was
  the QA render's single-sun lighting plus a genuinely dark ghillie suit.
* **Never use raster pixel collision or area/coverage ratio as a UV overlap gate.** Use
  `src/lowvram3d/uv_overlap.py`. See `evidence/RESULTS.md` for why both are wrong.
* **Weld by position before computing 3D connectivity** on any mesh that has been UV-unwrapped.
* **Never treat mirrored fallback views as real observations.** `view_metadata.json` governs this.
* **Keep observed, synthesized and inpainted coverage separate** in every receipt.
* **Promote outputs only after fresh-process validation.** A stale canonical GLB was published once
  because stages were verified in scratch directories without re-running the export chain.

### Next step: rigging

Rigging has never been executed successfully in any session. Start at `blender/rig_animate.py`,
dry-run it against `evidence/deliverable/game_ready_unrigged.glb`, and if it damages the mesh,
preserve the unrigged asset and report rigging as a separately failed stage. Do not let a rig
failure overwrite a working texture result.

Also open: the rear carries no real material detail because 77% of the surface has no observation.
That needs generated rear views (the MV-Adapter lane, still NaN-blocked on this GPU), not another
repair pass.

---

## UPDATE (continuation session, same day)

Steps 1-7 of the "Immediate next steps" list below are now DONE:
1. `phase2_raster_v2.py` output-naming edit verified: 512 output is byte-identical (sha256) to
   `baseline_v2_512/`. Syntax-checked, ran clean.
2. Bounded 1024 pass run: 2.3s raster + ~4s Blender preview render, both well under 60s budget.
3. 512 vs 1024 compared: 1024 has meaningfully more real coverage (17.4% vs 11.7% observed
   semantic, 45.8% vs 77.7% synthesized-fill) at negligible extra cost. **Chose 1024 as production
   resolution.** Did not run 2048 (per instruction).
4. First textured unrigged GLB exported:
   `jobs\a20a421f-.../output\a20a421f-.../meshes\game_ready_unrigged.glb` (39,722 faces, Principled
   BSDF not the QA-only emission shader, atlas packed/embedded, single `UVMap` layer).
5. Reimport-validated in a **fresh Blender process**: parses, mesh/material/UV counts correct,
   face count matches cleaned mesh (39,722), front/side/back render correctly (front shows real
   projected detail, side/back show synthesized fill with no duplicated face — confirmed visually).
6. Correct receipt written (`meshes\texture_receipt_1024.json`) with properly separated fields —
   `observed_semantic_coverage_percent` (17.41) is never conflated with
   `synthesized_surface_coverage_percent` (82.59).
7. **Raster route integrated into the real pipeline**, replacing the Cycles bake
   (`project_texture.py`, previously measured 600s+ and killed by timeout) as the DEFAULT texture
   path:
   - `workers/make_fallback_views.py` now emits `view_metadata.json` (source_type/confidence per
     view) — it didn't before; the raster projector depends on this to bar mirrored views from
     semantic projection.
   - New `blender/raster_cleanup_extract.py` (component cleanup + geometry/visibility extraction),
     `workers/raster_project.py` (pure numpy/opencv UV-atlas projector), `blender/raster_export.py`
     (Principled BSDF material assign + GLB export) — direct ports of the proven scratch scripts,
     adapted to the project's `common.py` helper conventions and CLI/receipt patterns.
   - `PipelineEngine._texture_projection` in `pipeline.py` now branches on new config flag
     `use_raster_texture_route` (default `True`); old Cycles path kept intact as the fallback when
     the flag is `False`. Atlas size explicitly capped at 1024 in the pipeline wiring — 2048 was
     never validated for this route, so an untested `texture_size=2048` config value is not
     trusted blindly.
   - **All three new stages were run end-to-end for real** (not just imagined): cleanup+extract via
     Blender, raster via the control env, export via Blender — numbers matched the standalone
     proven run exactly (17.4% real coverage, 39,722 faces, atlas packed not path-referenced).
   - Full 76-test suite passes. One test (`test_postprocess_is_split_into_focused_module`,
     line-count guard on `pipeline.py`) had its threshold bumped 550→650 per explicit user
     instruction ("ignore the 550 threshold") rather than extracting a new module — if a future
     session wants that module split, it's still worth doing for its own sake, just wasn't done now.
   - All edits synced from source (`Downloads\Siistimättömät\...`) into the installed app
     (`AppData\Local\LowVRAM3DStudio\app`) — verified byte-identical via diff.

**Not done: step 8 (rig → final `game_ready.glb` → reimport → render proof → `export_validate` →
finalize receipt).** Rigging remains completely untested (per the original handoff) — this is
higher-risk, unvalidated territory and needs its own careful pass, not tacked onto an already-large
session. Start here next: read `blender/rig_animate.py`, dry-run it against
`game_ready_unrigged.glb`, and if it damages the mesh, preserve the verified unrigged GLB and report
rigging as a separately-failed stage per the user's original instruction — don't let a rig failure
erase the working texture result.

---

Working job: `C:\Users\Lauri\AppData\Local\LowVRAM3DStudio\jobs\a20a421f-ea17-4009-8691-5b9da86c8b42`
Source install: `C:\Users\Lauri\Downloads\Siistimättömät\LowVRAM3DStudio-one-click-v0.6.1\LowVRAM3DStudio` (edit here, then copy into the installed app)
Installed app: `C:\Users\Lauri\AppData\Local\LowVRAM3DStudio\app`
Scratch scripts: `C:\Users\Lauri\AppData\Local\Temp\claude\C--Users-Lauri-Desktop\bfb8eeb3-461f-4c11-ac55-bb425cbedb7d\scratchpad`

## Goal

Turn `C:\Users\Lauri\Downloads\babaer.jpg` (red panda sniper in ghillie suit) into a textured,
game-ready 3D asset via the LowVRAM3DStudio pipeline, fixing defects encountered along the way,
persisting fixes to source, and eventually wiring the fast custom texture route into the pipeline
proper.

## Status right now (mid-edit, DO NOT run yet)

`scratchpad/phase2_raster_v2.py` is being edited to make output filenames resolution-scoped
(so a 1024 pass can't overwrite the frozen 512 baseline). The edit is **incomplete/unverified** —
last action (syntax check) was interrupted by the user before running. Before doing anything else:

1. Read `scratchpad/phase2_raster_v2.py` fully and check the `out_name()` helper and its 3 call
   sites (basecolor, source_view/debug, confidence, coverage) are self-consistent.
2. Run a syntax check: `python -c "import ast; ast.parse(open(r'...').read())"`.
3. Only then run the bounded 1024 pass (command below).

## What has been proven so far (do not redo)

### Install (source `v0.6.1`) — COMPLETE, 14/15 stages, 1 optional degraded
Fixed and persisted to source (`Downloads\Siistimättömät\...\LowVRAM3DStudio`), then copied into
installed app (`AppData\Local\LowVRAM3DStudio\app`):

1. `requirements-control.txt` — added `opencv-python-headless==4.11.0.86`, `httpx==0.27.2`
   (control env was missing both; stage-02 probe now imports `cv2, httpx` too).
2. `INSTALL-ONE-CLICK.ps1` — unit-test discovery only put `src` on PYTHONPATH; `service`/`workers`/
   `scripts` at app root were unimportable. Fixed: `PYTHONPATH = src + PathSeparator + AppRoot`.
3. `INSTALL-ONE-CLICK.ps1` — TripoSR `rembg` needs `onnxruntime`, never installed. Added.
4. `scripts/prefetch_models.py` — HF cache symlinks fail on Windows without Dev Mode/elevation
   (`WinError 1314`). Added `HF_HUB_DISABLE_SYMLINKS=1`.
5. `INSTALL-ONE-CLICK.ps1` — stage-10 config bug from a *previous* session was already fixed in
   v0.6.1 (`Set-ObjectProperty` helper + regression test) — verified, did not need re-fixing.

Regression tests added to `tests/test_layout.py` / `tests/test_core.py`. Suite is 76 tests, all
passing (`python -m unittest discover -s tests` with `PYTHONPATH=src;.` from repo root).

### Blender pipeline bugs (source `blender/*.py`, `src/lowvram3d/*.py`) — FIXED
6. **`INSTALL-ONE-CLICK.ps1`/`pipeline.py` `_blender_stage`** — Blender's embedded Python ignores
   inherited `PYTHONPATH` without `--python-use-system-env`. Every `blender/*.py` script died on
   `No module named 'common'`. This means **no Blender stage had ever executed successfully**
   before this session. Fixed in `pipeline.py:_blender_stage`.
7. **`blender/ingest_validate.py`, `blender/package_validate.py`** — `obj.data.vertices[::step]`
   throws `TypeError: slice steps not supported` on `bpy_prop_collection`. Fixed to index by
   range instead.
8. **THE BIG ONE — no vertex welding anywhere in the pipeline.** glTF export splits one vertex
   per face-corner wherever normals/UVs aren't shared. A flat-shaded mesh (what TripoSR/marching-
   cubes produces) re-imports as **fully disconnected triangles**: every edge non-manifold, one
   loose component per face. This silently wrecked every downstream stage (split, retopo, UV,
   bake, rig) on every asset ever processed. Root cause was actually **flat shading**, not just
   missing weld: 93,284/98,402 faces flat-shaded → 295,156 verts on reimport.
   Fixed in `blender/common.py`: added `shade_smooth()` and `weld_vertices()` (bmesh
   `remove_doubles`), wired into `blender/ingest_validate.py` before validation/export.
   Verified: 295,156→49,289 verts, 295,156→0 non-manifold edges, 98,481→67 loose components.
9. **`workers/proxy_generate.py`** — TripoSR's `--bake-texture` flag routes export through
   `xatlas.export()`, which **always writes Wavefront OBJ** even when the target filename ends
   in `.glb`. Produced a file that is not a valid glTF (`Bad glTF: json error`). Fixed: dropped
   `--bake-texture` (Blender bakes PBR from the source image anyway).
10. **`pipeline.py:generate()`** — gated geometry-lane success on `raw_mesh.is_file()` instead of
    `artifact_is_valid()`, so a corrupt/wrong-format file from a failed lane silently passed
    through to Blender, surfacing 3 stages later as a confusing "Bad glTF" instead of a clean
    lane failure. Fixed to check `artifact_is_valid()`.
11. **`blender/common.py: configure_render`** — hard-coded `BLENDER_EEVEE_NEXT`, which only
    exists in Blender 4.2. Blender 5.2 (installed here) exposes `BLENDER_EEVEE`. Every render
    stage (texture projection, control-view renders) failed with
    `enum "BLENDER_EEVEE_NEXT" not found`. Fixed: added `preferred_render_engine()` that reads
    the actual enum the running build exposes.

### GPU / VRAM bugs (source `src/lowvram3d/runner.py`, `blender/common.py`) — FIXED
12. **Cycles baking was pinned to CPU** in 3 places (`bake_transfer.py`, `bake_maps.py`,
    `project_texture.py`), *and* nobody set a sample count, so it inherited Blender's scene
    default of **4096 samples**. A 512px bake on a 12k-tri mesh ran 15+ min with no end in
    sight. Added `enable_cycles_gpu()` in `common.py`: tries OptiX→CUDA→HIP→oneAPI, sets
    32 samples (overridable via `LOWVRAM3D_BAKE_SAMPLES`), adaptive sampling, denoising off.
    Wired into all 3 call sites.
13. **VRAM ceiling charged stages for other processes' memory.** `runner.py` killed any stage
    when *whole-GPU* usage crossed the configured ceiling (5600 MB), even though ~1-2GB was
    routinely held by the user's desktop/other apps (Unreal Editor, Brave, LM Studio). Killed
    TripoSR at 5620MB (20MB over) while the desktop held ~1.1GB of that.
    Fixed: added `gpu_budget(ceiling_mb, baseline_mb)` in `runner.py` — budgets stage growth
    against `total_gpu_memory_mb() - baseline - GPU_RESERVE_MB(256)`, plus an absolute hard cap
    just below physical card size to still catch genuine runaways. `total_gpu_memory_used_mb()`
    is sampled as `baseline_vram` when each stage starts. 3 regression tests added
    (`GpuBudgetTests` in `tests/test_core.py`).
    **Per-process GPU attribution is impossible on this machine**: `nvidia-smi
    --query-compute-apps=pid,used_memory` returns `[N/A]` under Windows WDDM. The baseline-delta
    approach is the best available substitute.

### MV-Adapter (Lane A texturing) — dtype bug fixed, but STILL PRODUCES NaN (unresolved)
14. **`workers/mv_adapter_from_controls.py`** — `_init_custom_adapter` (upstream MV-Adapter repo,
    `thirdparty/MV-Adapter`) builds the `T2IAdapter` condition encoder at default fp32 and never
    casts it to the pipeline's fp16, and it's not a registered pipeline component so
    `enable_model_cpu_offload()` never places it. First conv died:
    `Input type (c10::Half) and bias type (float) should be the same`.
    Fixed: `pipe.cond_encoder.to(device="cuda", dtype=dtype)` after `enable_model_cpu_offload()`.
15. **After the dtype fix, all 6 generated views are pure black (NaN in latents).** Controlled
    A/B test proved this is **NOT** `enable_attention_slicing("max")` clobbering the MV attention
    processors (that IS a real separate bug — it replaces `DecoupledMVRowColSelfAttnProcessor2_0`
    with `SlicedAttnProcessor`, silently dropping `mv_scale`/`num_views` — but is not the cause of
    the NaN). With slicing removed AND cond_encoder cast fixed, latents are still 100% NaN.
    **Root cause not yet isolated to a specific UNet module/timestep** — a module-level
    forward-hook instrumentation script was written (`scratchpad/diag_first_nan.py`) but never
    run to completion (interrupted to pivot to Mini Turbo). Likely candidate: SD 2.1 base UNet
    itself is numerically unstable in fp16 on sm_75 (Turing), same failure class as BiRefNet
    (see #16). This has NOT been proven.
    **Status: `MV_ADAPTER_FINITE_OUTPUT_VERIFIED = NOT PROVEN`, `FULL_LANE_A_EXPORT_VERIFIED = NOT
    PROVEN`.** Do not claim Lane A works.

### BiRefNet masking (background removal) — FIXED, now GPU
16. **`workers/avatar_preprocess.py`** — BiRefNet in fp16 on sm_75 (GTX 1660 SUPER) produces an
    all-NaN matte, sometimes escalating to `CUDA error: an illegal memory access` (which poisons
    the CUDA context — no in-process recovery possible). Root-caused via isolated per-process
    probe (varying precision/resolution/SDP backend independently): fp16 NaNs at every
    resolution/backend tested; fp32 is correct. fp32 @1024px peaks ~3.4GB (blew the old VRAM
    ceiling); fp32 @768px peaks ~2.3GB with **identical mask coverage to 4 decimal places**
    (0.4618 vs 0.4610) because the matte is resampled to full image size regardless.
    Fixed: `_biref_mask` now runs `cuda/fp32@768` by default (`LOWVRAM3D_BIREFNET_SIZE` env
    override). This is on GPU, not CPU — do not "fix" it back to CPU.

### Geometry lane comparison — Mini Turbo dramatically better than TripoSR
- TripoSR (Lane C, the shipped fallback): 49,340 verts / 98,504 faces from `babaer.jpg` — an
  unrecognisable blob. This is Lane C's ceiling on this hardware; not a bug, just a weak model.
- **Hunyuan3D-2 mini-turbo** (NOT wired into the shipped pipeline — run standalone via
  `scratchpad/hunyuan_geom.py`): 237,731 verts / 979,620 faces — clearly recognisable creature
  (ears, muzzle, vest, tail, rifle protrusion). Generated in **98 seconds** using the FlashVDM
  turbo VAE (without it, volume decoding alone takes ~16 min — do not skip FlashVDM).
  Weights: `C:\AI\HY3D2\HuggingFaceHub\hunyuan3d-2mini-direct\` (DiT 3.8GB safetensors, VAE 407MB
  safetensors — downloaded via direct streaming HTTP with resume, see
  `scratchpad/dl_hunyuan.py`, because `huggingface_hub.snapshot_download` silently stalls with
  no timeout on large files — this bit us twice).
  Runtime: `C:\AI\HY3D2\python_standalone\python.exe` (torch 2.8+cu126, CUDA works), hy3dgen code
  at `C:\AI\HY3D2\Hunyuan3D-2`. Loaded via `Hunyuan3DDiTFlowMatchingPipeline.from_single_file()`
  (bypasses HF resolution entirely — needed since `tencent/Hunyuan3D-2mini` HF repo access is
  flaky/slow).
  **This mesh (`scratchpad/hunyuan_mesh.glb`) is what postprocess is currently running on.**

### Postprocess run on Hunyuan mesh — reached export chain, weld fix validated at scale
Ran `engine.postprocess()` directly (bypassing `full()`/geometry lanes) via
`scratchpad/run_postprocess.py` on `hunyuan_mesh.glb` (980k faces). Stage timings:
```
ingest_validate    5.4s   analyse   4.4s   split   6.5s   retopologize_blender  9.7s
uv_blender  4.4s   prepare_projection_views  1.2s   project_fallback_views  FAILED (633.8s, killed)
```
**`split` on a 980k-face mesh took 6.5 seconds** (same stage ground for 6+ min on a 98k mesh
before the weld fix) — this is the clearest possible validation that fix #8 actually works at
scale, not just in isolated tests.
`project_fallback_views` (the shipped Cycles-based texture-projection stage,
`blender/project_texture.py`) took 633.8s and was killed by the 10-minute tool timeout — this
triggered the pivot to a custom fast texturing route (see below). **This shipped stage is
confirmed pathologically slow and is the next thing to replace in the real pipeline** (item #7
in the user's latest instructions — not yet done).

## Custom fast texture route (in progress — this is the "handoff" reason)

Per explicit user direction, built a from-scratch UV-atlas texture projector to replace the slow
Cycles path, entirely in NumPy/OpenCV (no Cycles). Blender is used only to extract geometry/UVs/
visibility, never to render/bake.

Scripts (all in scratchpad, NOT yet copied into the real pipeline):
- `phase1_camera_proof.py` — camera-alignment sanity check (unlit emission projection through
  the pipeline's own ORTHO camera convention: `ortho_scale=2.6`, front at `(0,-3,0)` looking at
  origin). **PASSED**: face/scarf/rifle/tail all land correctly on the mesh. 4s runtime.
- `phase2_extract.py` — original extractor (Blender): verts/tris/UVs/normals/per-view visibility
  via raycasting, saved to `.npz`. Superseded by `phase2b_clean_extract.py` (below), which adds
  mesh cleanup first.
- `phase2b_clean_extract.py` — **current extractor**. Does bounded floating-component removal
  first (bmesh connected-components flood fill; removes components that are both small
  relative to face count/body diagonal AND outside the main silhouette bbox — protects thin
  ears/rifle/tail), then extracts geometry+visibility from the cleaned mesh. On the Hunyuan
  mesh: removed 1922/2634 components (5259/44981 faces, 11.7%), 6s total.
  Outputs: cleaned `.glb`, `.npz` (verts/tris/uvs/normals/centroids/view visibility), JSON
  cleanup report.
- `phase2_raster.py` — **v1 raster** (SUPERSEDED — averaged all 4 views including mirrored
  fallback views as if they were real observations → duplicated the front face onto the back).
  Preserved as `baseline_v1... ` no wait, actually preserved under
  `textured/projection/benchmark_v1_512/` in the job dir per user instruction "preserve as
  benchmark, do not overwrite".
- `phase2_raster_v2.py` — **current/correct raster**, mid-edit right now (see "Status" above).
  Key fixes vs v1:
  - Reads `view_metadata.json` (`textured/projection/view_metadata.json` in job dir) which
    labels each of the 4 prepared views (front/right/back/left) with `source_type`
    (`real`/`mirrored`/`synthetic`/`generated`) and `confidence`. Only views with
    `source_type` in `{"real","generated"}` and confidence > 0 are used for semantic
    projection — for this single-image job, that's **only `front`**. The mirrored
    right/back/left from `make_fallback_views.py` are explicitly barred from semantic
    projection (this is what fixed the duplicated-face-on-back bug).
  - Per-triangle: reject if not visible from camera (precomputed), reject if facing score
    below threshold (`facing_min=0.15`), rasterize into UV-atlas space via barycentric
    coordinates, sample the source image with confidence = `source_confidence *
    facing^3 * alpha * edge_distance_falloff`.
  - **Winner-takes-most** per atlas pixel (best confidence wins, never averaged) — this
    sharpened the face instead of smearing it.
  - Uncovered UV area filled via mask-aware push-pull diffusion (iterative blur-weighted
    growth) then heavily Gaussian-blurred (σ=9) so no recognisable feature (eye, muzzle,
    scope) survives into the synthesized fill — only low-frequency material colour.
  - Writes diagnostics: `debug_source_view.png` (green=real projection, grey=UV island,
    lighter grey=synthesized fill), `debug_confidence.png`, `debug_coverage.png`.
  - **Verified results on the cleaned Hunyuan mesh at 512**: front=only semantic view used;
    11.7% of UV-island area got real projected colour, 77.7% synthesized fill, ~1.5s raster
    runtime. Rendered 4 views (front/three-quarter/side/back) via
    `phase2_preview_v2.py` — **quantitatively confirmed no face on the back**
    (`light_frac=0.001` on back vs `0.030` on front reference; v1's back had
    `light_frac=0.015` with visible eye/nose structure). All user acceptance criteria
    (§7 of the second instruction block) passed — this was accepted as
    `PRODUCTION_ROUTE_WORKABLE`.
  - **THE EDIT IN PROGRESS**: adding resolution-scoped output naming
    (`out_name()` helper + `NAME_SUFFIX` from optional 6th CLI arg) so that running at
    ATLAS=1024 doesn't overwrite the frozen 512 baseline files. At 512 (default, no
    suffix), output names are unchanged (`preview_v2_basecolor.png` etc. — safe, matches
    what's already frozen). At any other resolution, outputs become
    `basecolor_{N}.png` / `source_map_{N}.png` / `confidence_{N}.png` / `coverage_{N}.png`,
    and progress file becomes `raster-progress_{N}.json`. **This edit was in the middle of
    being verified (syntax check) when interrupted for this handoff.**
- `phase2_preview.py` / `phase2_preview_v2.py` — Blender: assign atlas as unlit UVMap material,
  render front/three-quarter/side/back via the pipeline's ORTHO camera convention.

### Frozen baselines (job dir: `.../a20a421f-.../textured/projection/`)
- `benchmark_v1_512/` — the flawed v1 result (mirrored-view averaging, face-on-back bug).
  Preserved per user instruction, DO NOT overwrite or delete.
- `baseline_v2_512/` — the corrected, accepted result. Contains full provenance manifest
  `BASELINE_MANIFEST.json` (source image sha256, cleaned mesh sha256, camera transform,
  projection parameters, script sha256s, coverage/runtime numbers, cleanup stats). This is the
  proven reference — **DO NOT overwrite**. Also contains copies of the 3 scripts as they were
  at freeze time, the cleaned mesh, the `.npz`, and the source image.
- `view_metadata.json` (live, in the job's `textured/projection/` dir, not frozen) — drives which
  views are semantic vs fill-only. Will be needed by any future run against this same job.

## Immediate next steps (in the order the user specified, second instruction block)

1. **Finish and verify the `phase2_raster_v2.py` output-naming edit** (syntax check, then confirm
   a 512 run still writes to the original `preview_v2_*` names unchanged — diff against
   `baseline_v2_512/` to be sure nothing drifted).
2. **Run one bounded 1024 pass**: `phase2b_clean_extract.py` (only if not reusing the existing
   `mesh_clean.npz`/cleaned GLB — the mesh/UVs don't change with atlas resolution, so this step
   can likely be skipped and the existing `.npz` reused directly) →
   `phase2_raster_v2.py ... 1024` → `phase2_preview_v2.py` (needs a resolution-aware version too,
   or pass output names explicitly) for front_1024/threequarter_1024/side_1024/back_1024.
   Target: under 60s total. Required output files listed in the user's second instruction §2.
3. **Compare 512 vs 1024** automatically (sharpness, seams, unfilled texels, semantic
   duplication, runtime, file size) and decide which to carry forward. Do not run 2048.
4. **Export the first textured, unrigged GLB**:
   `output/<job>/meshes/game_ready_unrigged.glb`, atlas embedded, correct UV layer, no absolute
   paths in the material. This has never been done in this session — no run has reached
   `export_validate` yet with a real texture on it.
5. **Reimport validation in a fresh Blender process** (not the in-memory scene that built it):
   parses, mesh/material/UV counts correct, texture resolves and is embedded, face count matches
   cleaned mesh, no removed floating components reappear, front/side/back render correctly.
6. **Correct receipt metrics** — the user was explicit: never call synthesized-fill pixels
   "observed coverage". Record `uv_island_occupancy_percent`,
   `observed_semantic_coverage_percent`, `synthesized_surface_coverage_percent`,
   `final_filled_uv_percent`, `removed_component_count`, `removed_face_count`,
   `removed_faces_percent` as separate fields.
7. **Integrate into the real pipeline**: replace `project_fallback_views`'s implementation in
   `src/lowvram3d/appearance.py`/`postprocess.py` (whichever calls
   `blender/project_texture.py`) with the proven raster route (cleanup → extract → raster →
   preview chain), keeping the old Cycles path only as an explicitly-disabled diagnostic
   fallback. Must write progress/partial artifacts. No manual approval step.
8. **Continue downstream**: rig (if configured) → final `game_ready.glb` → reimport → render
   proof → `export_validate` → finalize receipt. If rigging damages anything, preserve the
   verified unrigged GLB and report rigging as a separately failed stage — do not let a rig
   failure erase a working texture result.

## Things NOT to do / traps already hit once

- Don't reinstate `enable_attention_slicing("max")` in MV-Adapter without also excluding it from
  clobbering `pipe.unet.attn_processors` — it silently disables multi-view attention.
- Don't set BiRefNet back to CPU or back to fp16 — both are wrong, fp32/GPU/768px is correct and
  measured.
- Don't re-add `--bake-texture` to the TripoSR invocation in `proxy_generate.py` — it corrupts
  the GLB export.
- Don't use `huggingface_hub.snapshot_download` for large files without a wrapper that shows
  throughput and has a real timeout — it can silently stall for 10+ minutes doing nothing.
- Don't run any Blender subprocess without `--python-use-system-env` if it imports from
  `blender/common.py` or similar sibling modules.
- Watch background-task timeouts: this tool's max is 10 minutes; several runs were killed by
  this ceiling rather than by real failure (project_fallback_views Cycles bake, first TripoSR
  attempt). Don't mistake a timeout kill for a crash — check the log for actual error text vs.
  just "exit 143 / no completion record".
- Two full pipeline-driver processes + duplicate Blender instances were once left running
  simultaneously by accident, competing for the same GPU. Always confirm `ps -W | grep -i
  blender` / `nvidia-smi` is quiet before starting a new run.
- The install source lives in `Downloads\Siistimättömät\...\v0.6.1\`; edits must be copied into
  `AppData\Local\LowVRAM3DStudio\app\` to take effect on the next run (robocopy mirrors this on a
  real installer run, but ad-hoc testing needs a manual `cp`).

## Key file locations quick-reference

- Source (edit here): `C:\Users\Lauri\Downloads\Siistimättömät\LowVRAM3DStudio-one-click-v0.6.1\LowVRAM3DStudio\`
- Installed/running app: `C:\Users\Lauri\AppData\Local\LowVRAM3DStudio\app\`
- Control env python: `C:\Users\Lauri\AppData\Local\LowVRAM3DStudio\envs\control\Scripts\python.exe`
- MV-Adapter env python: `C:\Users\Lauri\AppData\Local\LowVRAM3DStudio\envs\mv-adapter\Scripts\python.exe`
- TripoSR env python: `C:\Users\Lauri\AppData\Local\LowVRAM3DStudio\envs\triposr\Scripts\python.exe`
- Blender: `C:\Program Files\Blender Foundation\Blender 5.2\blender.exe`
- Hunyuan3D-2 code: `C:\AI\HY3D2\Hunyuan3D-2\` — run with `C:\AI\HY3D2\python_standalone\python.exe`
- Hunyuan weights (direct download): `C:\AI\HY3D2\HuggingFaceHub\hunyuan3d-2mini-direct\`
- Current working job: `C:\Users\Lauri\AppData\Local\LowVRAM3DStudio\jobs\a20a421f-ea17-4009-8691-5b9da86c8b42\`
- Scratch scripts: `C:\Users\Lauri\AppData\Local\Temp\claude\C--Users-Lauri-Desktop\bfb8eeb3-461f-4c11-ac55-bb425cbedb7d\scratchpad\`
- Source benchmark image: `C:\Users\Lauri\Downloads\babaer.jpg`

## Memory files written this session

- `C:\Users\Lauri\.claude\projects\C--Users-Lauri-Desktop\memory\lowvram3d-stack-state.md` —
  ComfyUI is unusable on this machine (no python_embeded anywhere), SD2.x pulled from HF (use
  local copy at `C:\AI\HY3D2\HuggingFaceHub\models--stabilityai--stable-diffusion-2-1-base`),
  TripoSR is the shipped working geometry lane (Mini Turbo is much better but not wired in).
