# Handoff — 2026-08-08

Repo: `C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803`
Branch: `agent/scene-pipeline-smoke-20260803`
Head: `4939fad`, pushed to `origin`.

Interpreters:

- control venv — `C:/Users/Lauri/AppData/Local/LowVRAM3DStudio/envs/control/Scripts/python.exe`
- HY3D2 standalone — `C:\AI\HY3D2\python_standalone\python.exe` with
  `PYTHONPATH=C:/AI/HY3D2/Hunyuan3D-2`
- Blender 5.2 for all rendering.

The system Python has a phantom torch and will mislead you. Run the test suite
in the control venv.

---

## Read this first: two conclusions were wrong, and how

### 1. The renderer invented a texture defect

`workers/render_textured_views.py` sampled textures with an **unfiltered
nearest-neighbour** lookup. A face covering one or two texels takes a single
texel's colour across its whole area; neighbouring faces landing in different
texels meet at a hard edge. Every low-density texture therefore rendered as a
mosaic of hard-edged plates. **The textures were fine.** It samples bilinearly
now.

Before anyone checked the renderer, that one artifact produced:

- two fennec paints declared unusable — by me **and by both vision models**;
- a "multiview hallucination" root cause from Luna at high effort, ruling out
  texel density, baked lighting and gamma by name;
- a 4× rebake to chase it. 2048 vs 1024, measured in Blender: **mean absolute
  difference 0.45/255**, 1.54% of pixels differing by more than 8. Median
  texels/face 1.40 → 5.59 bought nothing visible, for 5% more wall-clock
  (510.6s → 535.7s);
- a whole runbook section on UV fragmentation, written to explain plates that
  were never in the asset.

The standing verification rule is two vision models plus my own eyes. All three
agreed — **because all three were shown the same bad picture.** Model agreement
cannot detect a defect in the image itself; it only detects disagreement about
what the image shows.

**The rule that follows:** a defect seen in exactly one renderer is a renderer
bug until a second renderer shows it too. Re-render through
`workers/render_asset_views.py --native` (Blender/EEVEE, seconds) before writing
down any conclusion about a texture, surface or shading defect.

### 2. RUNBOOK §14 and §19 are partially retracted

The **measurements stand**: 5,609 charts, median 14 faces per chart, 36.7%
atlas utilisation, 85% of the atlas unowned, 1.36 mean / 1.00 median texels per
face, and UV fragmentation universal across all seven meshes (3,232–9,806
charts, 33–45% coverage). It is not a regression.

The **causal story built on top of them does not stand.** "Visibly damaged by
texel starvation" was explaining the renderer's artifact. Note also that an
early "3.69 texels/face" figure assumed 100% atlas utilisation; weighting by UV
triangle area gives 1.36, roughly 3× lower.

---

## The GPU fault: measured, reduced, not eliminated

It is **driver-level**, not thermal (57–70 °C, 37–52 W) and not seed-dependent.
Over six hours: 8× `nvlddmkm` Id 13 (graphics exception) and 20× Id 153 (TDR /
engine reset), correlating to the second with every CUDA failure across **both**
ggml and PyTorch. The seed hypothesis was disproved directly — seed 12345 on the
same image passed 5 times and failed 7. **There is no safe-seed list.**

Two failure modes that look alike and are not:

| symptom | cause |
|---|---|
| CUDA illegal access / misaligned address, **with** an nvlddmkm 13 or 153 at the same second | GPU fault |
| Python traceback ending in `_ArrayMemoryError`, **no** driver event | host RAM exhaustion |

The fennec SIGSEGV at 22:23:01 had no driver event and was host-side heap
corruption (`run_after_guard` + `run_final`) — a separate bug, still unfixed.

### System state, verified 2026-08-08

| | wanted | actual |
|---|---|---|
| Driver | 610.88 | **610.88** (32.0.16.1088, 2026-07-22) ✓ |
| HAGS | off | **`HwSchMode: 1`** ✓ |
| Clocks | reference | **1530 MHz** vs 2100 max ✓ (Debug Mode) |
| `TdrDelay` | 10–12 s | **96** ✗ |

`TdrDelay` is the one outstanding item. It does not affect whether the fault
happens, only what happens afterwards: at 96 seconds a poisoned context hangs
the machine for a minute and a half instead of the driver resetting the engine
and the process dying cleanly. `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers`,
DWORD `TdrDelay` = `10`, reboot. **System changes are the user's to make.**

### Result

`sh tools/driver_validation.sh 10 "debugmode+hagsoff+610.88"` —
**9 pass / 1 fail out of 10.** Baseline for the identical configuration
(panda_matte, seed 12345, res 512) was 7 fail in 12 = 58%. Under that rate,
≤1 failure in 10 is **p ≈ 0.003**. Per-run data in
`evidence/driver_validation/runs.jsonl`.

The single failure carried an `nvlddmkm=1`. **The fault mode still exists — it
is rarer, not gone.** Long jobs still need to be resumable, and a failed paint
still means a retry rather than a diagnosis.

**Do not use `tools/gpu_fault_probe.py` to validate anything.** It ran 104,900
launches clean *at the broken driver state*, so it cannot distinguish a fixed
machine from a broken one. It is kept for what it rules out. Validation goes
through `driver_validation.sh`, which repeats the workload that actually faults
and checks the System log inside each run's own window.

---

## What else changed

- **`render_asset_views.py --native`** wires `COLOR_0` through a
  `ShaderNodeVertexColor` into Base Color, so Mini Turbo geometry shows its own
  appearance instead of being forced to clay. Two views added —
  `three_quarter_rear` and `three_quarter_far` — because that is where a paint
  has the least conditioning and is most likely to be wrong.
- **`hunyuan_paint_texture.py` moves the multiview RNG to the CPU.** `hy3dgen`
  builds `torch.Generator(device=cuda)` at `multiview_utils.py:69`, and that is
  the exact call the bluetree paint died in. Patched around the vendor call so
  the checkout stays pristine; defaults on, `--gpu-rng` reverts, recorded in the
  receipt. It removes one class of kernel launch from the fault window. **It is
  not a fix for the driver fault and must not be reported as one.**
- **Receipt bug**: it read `.visual` off a Scene, which has none, and logged a
  null atlas size for bakes that were exactly the size requested. It resolves to
  the geometry first now — but **still logged null on the 2048 run**, so the fix
  is incomplete. Worth ten minutes.
- **`tools/paint_view_provenance.py`** replays the vendor projection with flat
  per-camera debug colours, and replays `fast_bake_texture` to record which
  views it drops at the 99%-already-painted threshold.
- **15 deliverables** re-rendered as 9-view sheets at 2000 px per view in
  `evidence/deliverables/views9/`, reflowed to 3×3 grids. Roughly 1,000–1,500 px
  of subject per view — above the threshold where a feature judgement is
  possible at all.

---

## Facts worth not rediscovering

- **Mini Turbo produces no colour.** `Hunyuan3DDiTFlowMatchingPipeline` from
  `hy3dgen.shapegen` is shape-only; every Mini Turbo GLB carries `POSITION` and
  nothing else. TRELLIS carries `NORMAL, POSITION, TEXCOORD_0` plus two images.
  **Paint is the only route to appearance for a Mini Turbo asset.**
- **`trimesh` fabricates `visual.vertex_colors` on access.** It returned
  2,126,340 "vertex colours" that were a single uniform grey `(102,102,102)`.
  Never treat its presence as evidence of colour.
- **Two renderers, different jobs.** `render_asset_views.py` is the fast one —
  Blender/EEVEE, seconds, every mesh and view in one launch.
  `render_textured_views.py` is a software rasteriser with an O(faces) Python
  loop, roughly 2 minutes.
- **Both existing trellis-cpp builds JIT Pascal/Ampere PTX onto Turing**
  (`61-virtual;80-virtual`, never native sm_75).

---

## Open work

1. **The trees.** `bluetree` and `greentree` still have no paint — one died to
   the GPU fault, one to a host OOM I caused by letting a compile compete for
   RAM. Mini Turbo geometry exists for both (`*_mt_uv.glb`); TRELLIS geometry
   was never produced for either. **This is the largest gap, and at 9/10 it is
   now a reasonable bet.**
2. **`build-sm75`** at `C:\AI\trellis-cpp\build-sm75` is configured and verified
   emitting `arch=compute_75,code=[compute_75,sm_75]` with `FORCE_MMQ`, roughly
   two-thirds compiled, killed before linking. Resume with
   `cmake --build build-sm75 --config Release` — **not while a paint is running.**
   Watch free RAM; that is what killed greentree.
3. **`baked_texture_size` still logs null** despite the Scene→geometry fix.
5. **Task #14**, a lightweight model viewer, still pending.
6. **Unreal Showroom import** — `tools/unreal_showroom_import.py` needs the user
   to run it in-editor.

---

## The parallel session, and why the worktree is dirty

A second agent (`command-code`, session `8f3d93cb-c7a1-4e90-991f-6ce38ef06a0d`)
worked this same repo in parallel through the night. **It hit its weekly usage
limit mid-commit at 02:57 and is dead for ~5 days 16 hours.** Its work is
finished and validated but **never committed**, and it is sitting in the working
tree right now:

```
 M LOWVRAM3D_HANDOFF.md                    <- its castle/frog handoff section
 M evidence/deliverables/MANIFEST.json
 M evidence/deliverables/blender/MANIFEST.json
 M tools/collect_deliverables.py
 M tools/make_blend_files.py
 M tools/make_inspection_scene.py
 M workers/render_asset_views.py           <- its changes stacked on top of mine
?? blender/rig_five_pose_proof.py
?? evidence/compare/castle_new/  evidence/compare/frog_new/
?? evidence/compare/shaman/shaman_finished_rig_readiness.json
?? evidence/deliverables/views9/castle_trellis512_tu116_*
```

`4939fad` did not clobber any of it — that commit predates these edits, and the
only shared file it touched was `render_asset_views.py` carrying my own
`--native` work. **Verify that before trusting it, and do not `git add -A` in
this repo.** `conftest.py`, `pytest.ini` and
`tests/test_mv_adapter_direct_camera_runtime.py` belong to yet another session
and must stay out of every commit.

What that agent actually produced, from its own receipts:

- **castle — done, and the GPU paint was rejected.** The promoted deliverable is
  `castle_trellis512_tu116_cpu_projection_tex2048.glb`, textured by a
  **numpy vectorised rasteriser** rather than the diffusion paint:
  `blender_used: false`, `scene_ray_cast_calls: 0`, `bpy_pixel_accesses: 0`.
  Conditioning was a BiRefNet hard mask at 0.298 alpha coverage over two
  foreground components; observed coverage 0.169968. The earlier
  `mattepaint_tex1024` castle is superseded and its 9-view sheets are stale.
- **frog — done, and rigged only as far as honesty allows.**
  `frog_trellis512_tex2048.glb`, 159,834 vertices / 145,084 triangles, profile
  `humanoid_complex_accessories`, `ready: true` with no failure codes — arms,
  legs, shins, feet and depth all measured clear. But the rig candidate
  **exported a GLB with the armature dropped**, so the five-pose action preview
  failed closed and the receipt correctly records **`deformation_proven:
  false`**. The candidate is real; the deformation is not proven. Do not promote
  it as rigged.

So **castle and frog are no longer open work** — a claim to the contrary in an
earlier draft of this document was wrong, written before I read that session.
The open item there is the **armature-dropping GLB export** in the rig path.

---

## Standing constraints

- **Do not edit** `C:\Users\Lauri\Desktop\lowvram3d-two-character-production-20260804` —
  another agent owns it.
- **Do not commit** `conftest.py`, `pytest.ini`, or
  `tests/test_mv_adapter_direct_camera_runtime.py` — a parallel session owns them.
  They are deliberately left dirty in the working tree.
- **Never touch** opencode / codex / browser caches during disk cleanup.
- **Do not kill the user's applications** — Traycer, Brave, ChatGPT, opencode,
  ComfyUI, 3DGenStudio, Blender, the control service.
- **Do not delete model weights** unilaterally, and **do not empty the Recycle Bin.**
- **Do not make system, driver, registry or control-panel changes.** Those are
  the user's to perform; hand over a checklist instead.
- **Luna 5.6 only through `command-code`, never codex. DeepSeek has no vision.**
  Visual verification uses Luna and Spark **and** my own eyes — with the caveat
  from §1 above, which that rule did not previously cover.
- **Show the render before any analysis**, and show native textures alongside
  clay.
- **No model-specific fixes.** Tuning per asset *class* is allowed; per
  individual asset is not.
- **Do not run a global remesh** to fix the fringe. **Turntable is retired.**
- **Never judge fine geometry from a contact sheet** — crop at full resolution
  first; under ~600 px the honest answer is "cannot tell".
