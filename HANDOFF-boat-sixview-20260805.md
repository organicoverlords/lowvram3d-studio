# Handoff — boat six-view texture route, 2026-08-05

Working tree: `C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803`

Standing instruction from the user: **"keep improving the boat any way you can"**, plus a
new one issued mid-session: **"the background removal has to be improved also"**.

---

## 1. Where the work actually stands

The six-view chain now runs end to end on the boat. This is the first time it has
completed on a scene-pipeline asset with usable output.

| Stage | State | Artefact |
|---|---|---|
| UV unwrap | PROVEN | `evidence/compare/boat/boat_uv.glb` — charts 3232, util 0.8017, overlap 1.613e-06, oob 0 |
| CPU controls @384 | PROVEN, basis `y_up_z_front` | `evidence/compare/boat/controls_384/` |
| Camera contract finalised | PROVEN | same dir, sha `cfb6b97c…` |
| Six-view inference | **EXECUTED**, QA_REJECTED | `evidence/compare/boat_sixview_384/` |
| Composite bake | **NOT STARTED** | — |
| Render / review | **NOT STARTED** | — |

### The inference run

- 251.8 s wall, peak 2359 MB VRAM (comfortable on the 6 GB card), 6/6 non-black.
- Exit code 2 because QA said `passed: false`.
- **Only the colour gate failed.** Structural, rear-correlation and semantic gates
  all passed. The gate is `foreground_saturation >= 0.08` on every view
  (`mvadapter_sd21_six_view_inference.py:496`); views 0/2/3/5 came in at
  0.016 / 0.060 / 0.037 / 0.035.
- That threshold was calibrated on the panda. A weathered near-monochrome riverboat
  legitimately sits under it. **Do not treat QA_REJECTED here as "the run failed"** —
  read the per-gate booleans, not the summary.

Command that worked (note the two non-obvious parts):

```bash
PYTHONPATH=C:\AI\mvadapter-upstream-inspection "C:\AI\3d-studio-pipeline\workers\mv_adapter\.venv\Scripts\python.exe" workers/run_sixview_no_cudnn.py --config configs/boat_mvadapter_ig2mv_sd21_384x20.json --output-dir evidence/compare/boat_sixview_384
```

---

## 2. Traps that cost time this session — read before re-running

**cuDNN must be off or the views come back pure black.** The pipeline's own guard
does *not* install itself: its self-test reports `fp16_cudnn_finite_fraction: 1.0`
and `unet_cudnn_disabled: false` even on this GPU, because the probe builds a fresh
random Conv2d and the defect is weight-dependent. Always launch through
`workers/run_sixview_no_cudnn.py`, which sets `cudnn.enabled = False` process-wide.

**The run needs ≥2048 MB of *system* RAM, not VRAM.** Preflight fails with
`MVADAPTER_RAM_PAGEFILE_INSUFFICIENT`. Sequential offload holds weights in host RAM.
It blocked at 1537 MB available. Fixed non-destructively by calling `EmptyWorkingSet`
on every process over 50 MB (1537 → 2786 MB) — this pages to standby without killing
anything. **Do not kill the user's Brave / Traycer / opencode / ChatGPT processes.**

**Output dir must not exist** or you get `MVADAPTER_HEARTBEAT_PATH_ALREADY_EXISTS`.

**Do not edit** `C:\Users\Lauri\Desktop\lowvram3d-two-character-production-20260804`
(another agent has a job running there). It is read-only reference.
**Do not commit** `conftest.py`, `pytest.ini`, or
`tests/test_mv_adapter_direct_camera_runtime.py` — a parallel session owns those.

---

## 3. Which view carries the photograph: **index 0**

This changed during the session and the earlier answer was wrong.

- The old answer, index 1, came from silhouette IoU against the **control masks**,
  and won by a margin of 0.013. That is a coin toss, and I said so at the time.
- The generated views settle it. Indices 0 and 2 are the **broadsides** (paddlewheel
  visible, coverage 0.396 / 0.398); 1 and 3 are the narrow bow/stern ends
  (0.314 / 0.288). The source photograph is a broadside, so it cannot be index 1.
- Between the two broadsides: the photo has the paddlewheel on the **left**, matching
  view 0. View 2 is the mirrored side.
- `workers/identify_photographed_view.py` (new) scores IoU against the generated
  images and picks 0 with margin 0.053.

**Independently corroborated.** GPT-5.6 Luna, shown only the source image and the
contact sheet and told not to trust any label, answered **view 0 at 90% confidence**
from the same features (paddlewheel left in image 1 and view 0, right in view 2).
Three lines now agree: silhouette IoU, my own visual check, and an independent model.

**Naming trap worth recording.** Luna also reported "the labels are wrong — 0 and 2
are the two long sides, not the two ends." It is right about the geometry and wrong
about it being a defect. The contract's front/rear/left/right are *mesh-canonical
directions*, not the boat's bow and stern; the mesh's front axis simply happens to
point at a broadside. Nothing needs fixing in the contract. It matters only because
one thing downstream does assume "front = the photographed face" — the bake's
`--photograph-view` — which is why that index had to be established empirically
rather than read off a label.

**Caveat, stated plainly:** that worker's second axis — appearance correlation —
produced near-zero values for all six views (max 0.082, "winner" index 5, which is
the bottom of the hull). It has no signal on this asset, because the photograph is a
3/4 view being bbox-stretched against straight-on renders. The worker therefore
reports `INCONCLUSIVE`. **The index-0 conclusion rests on silhouette IoU plus the
direct visual paddlewheel check, not on that correlation number.** Either fix the
appearance axis for 3/4 sources or drop it.

---

## 4. The colour problem — measured, unfixed

Foreground statistics inside the matte:

| | mean RGB | luminance |
|---|---|---|
| photograph | `[47.5, 44.1, 38.1]` | **44.4** |
| view 0 front | `[124.5, 123.7, 121.4]` | 123.7 |
| view 1 right | `[99.5, 87.9, 72.3]` | 89.2 |
| view 2 rear | `[127.3, 120.5, 112.0]` | 121.3 |
| view 3 left | `[129.2, 124.9, 120.2]` | 125.5 |
| view 4 top | `[150.1, 142.1, 129.2]` | 142.8 |
| view 5 bottom | `[58.2, 53.7, 49.3]` | 54.3 |

The generated views are ~2.8× too bright and have lost the warm cast — view 0 is
`R≈G≈B`, i.e. neutral grey, where the photograph is warm brown (`R>G>B`).

**Baking as-is will put a dark warm photograph next to pale grey synthesis and show a
hard seam.** Before baking, colour-transfer the generated views' foreground statistics
onto the photograph's — *that direction*. An earlier session made this mistake in
reverse (matched the photo down to the synthesis, gain 0.703) and it was wrong; the
photograph is the ground truth.

### Independent visual review of the six views (Luna, high effort)

Identity ranking, best first: **0, 1, 2, 3, 4, 5**. Two findings that change the bake
plan and that I had not caught:

- **Views 3 and 4 contain invented structure** — a curved green/glass canopy and
  bulky end construction on view 3, exposed deck machinery, boxes and circular
  fittings on view 4. Neither is supported by the source. These are exactly the
  views that would supply the side and top texels, so they should be down-weighted
  relative to views 0/1/2, not trusted equally.
- **Clipped highlights and baked lighting**, distinct from the brightness offset:
  silver-grey surfaces, near-white blown highlights, flat midtones, near-black
  bottom. For an albedo map that is contamination, not exposure, and a mean/std
  transfer will not remove it — it will only redistribute it.

Other defects it ranked: cross-view structural inconsistency (railings, decks,
towers and stair blocks move between views), melted fine geometry (window mullions,
paddlewheel spokes, ornaments fused), and distorted signage lettering.

⚠ One earlier number in this session was wrong and is corrected here: a first pass
reported photo luminance as 11.2 (a 10× gap). That measurement used
`Image.open(matte).convert("L")`, which **drops the alpha channel** and selected the
background instead of the boat. The correct figure is 44.4. Always key off
`np.asarray(im)[..., 3]`, never `convert("L")` — `matte.png` is RGBA and its RGB
channels render as the subject on white, so the mistake is invisible by eye.

---

## 5. Background removal — the user's new request

Producer: `workers/pipeline_matte.py`, border flood-fill key.
Current output for the boat: `evidence/compare/boat/matte.png` + `matte.json`
(tolerance 42, background `[248,248,248]`, subject fraction 0.5398).

Measured defects, in order of how much evidence supports them:

1. **Alpha is strictly binary.** `pipeline_matte.py:137` is
   `alpha = np.where(subject, 255, 0)`. Measured: `alpha partial (1..254) fraction
   = 0.0` — not one antialiased pixel in the whole image. Every edge is a hard step,
   and any partially-covered thin structure (rigging, masts, railings, the flag) is
   rounded to all-or-nothing.
   *Fix:* solve the compositing equation in a narrow band around the boundary.
   With background `B` known and foreground `F` taken from the nearest interior
   subject pixel (distance-transform indices),
   `α = clamp(((C−B)·(F−B)) / |F−B|², 0, 1)`.

2. **A 67-pixel detached speck is kept as subject**, and it is *below the hull*.
   Measured impact: bounding box goes from `(11, 8, 604, 460)` for the main body to
   `(11, 8, 604, 481)` with the speck — **21 px of vertical stretch on a 473 px
   subject, 4.4%.** Registration in the bake fits the photo's bbox to the silhouette
   bbox, so this directly mis-scales the photograph. *Fix:* drop detached components
   under a size floor. This is the cheapest real win available.

3. **Honest scope limit on defect 1 for *this* image.** Pixels in the ambiguous band
   (colour distance 20–60 from background) number only 548, or 0.17% of the frame,
   and *zero* of them are currently misclassified as background. So on this
   particular crisp render soft alpha buys less than I first assumed — it is a
   correctness fix for the general case (photographs, soft edges, motion blur) more
   than a visible win on the riverboat. Do not oversell it in the receipt.

### Status: DONE, with two defects found by review and fixed

Both defects were implemented, then caught by a Luna high-effort review, then
**reproduced with a failing case before being fixed** — neither was taken on trust.

| | boat | shaman |
|---|---|---|
| partial-alpha pixels | 0 → 1.49% | 0 → 1.90% |
| islands dropped | 35 (319 px) | 31 (175 px) |
| bbox correction | — | left +41, top +63, right −22 px |
| undecidable band | 25.1% | 0.04% |

Visible win: **the flagpole**. A 1–2 px mast the binary key dropped entirely, leaving
the flag floating detached, is recovered as a connected line
(`evidence/compare/boat/matte_alpha_compare.png`, left binary, right feathered).

Two defects found and fixed, both with regression tests in
`tests/test_pipeline_matte_feather.py` (11 pass):

1. **The island drop was a no-op for nearby specks.** A dropped speck within the
   feather radius was re-solved by the band and returned at **alpha 255 — fully
   opaque**, not a faint ghost. `ndimage.label` is 4-connected, so even a diagonal
   neighbour is a separate component sitting well inside the band. Fixed by
   excluding dropped pixels last, after the solver.
2. **Thin structures took their colour from unrelated geometry.** Anything under
   three pixels erodes to nothing, and the nearest-interior lookup is global rather
   than per-component. Measured: an isolated 1 px red mast borrowed its foreground
   from a distant dark body and keyed to **alpha 173** — a solid structure rendered
   two-thirds opaque. Fixed by letting any subject pixel no interior can reach act
   as its own colour source.

Remaining known limit: 25.1% of the boat's feather band is undecidable (foreground
too close to the plate colour to project). Those pixels keep their binary value and
are counted in the receipt as `feather_degenerate_pixels` rather than being given a
fabricated fraction.

---

## 5b. Colour correction — BUILT AND PROVEN

`workers/match_view_colour.py`. Single per-channel gain, fitted in **linear light**
on the photographed view, applied identically to all six.

| | before | after | target |
|---|---|---|---|
| view 0 mean RGB | `[124.5, 123.7, 121.4]` | `[61.7, 57.4, 51.2]` | `[60.6, 56.4, 50.4]` |
| channel spread | 3 (neutral) | **10.5 (warm, R>G>B)** | 10.2 |
| top/bottom ratio | 6.597 | 6.600 | must not move |

**Fit in linear light, not sRGB.** The gap reads as 2.8x in sRGB and is **5.58x**
in linear. Fitting through the transfer curve would under-correct shadows and
over-correct highlights — a gain applied through sRGB is not a gain.

Visual: `evidence/compare/boat_colour_match_compare.png` (original / corrected /
photograph). The amber window glow is lost, which for an albedo map is arguably
correct — it is baked lighting.

Two guards, both of which earned their place:

- **Inter-view ratio.** A global gain cannot change a ratio between views, so if
  the top/bottom ratio moves, the transform silently became per-view and genuine
  lighting variation has been flattened. It fired at 23% drift on the first run
  and was right that something was wrong and **wrong about what**: the fault was
  my own measurement re-keying the subject mask on the darkened output with an
  absolute threshold, which then rejected most of the subject. Masks are now keyed
  once on the original. Drift is 0.0005.
- **Shadow crush.** The threshold is 0.10, not 2%, and deliberately loose: per-view
  normalisation drives this ratio toward 1.0 (drift ~0.85 here), while 8-bit
  quantisation of a 5-8x darkening produces 0.0005–0.02. Nothing lands in between.
  A separate `shadow_crush_warning` reports when the darkest view has been crushed
  into too few levels — correct gain, lost detail.

Tests: `tests/test_match_view_colour.py`, 5 cases including the ratio invariant and
a refusal when the matte is not RGBA.

## 5c. CFG is NOT available in this pipeline — do not retry blind

`guidance_scale` and `negative_prompt` are **literals** at
`mvadapter_sd21_six_view_inference.py:844-848`; there is no config route. Patching
them from `run_sixview_no_cudnn.py --guidance-scale` works (the upstream base class
implements CFG properly and the LowVRAM subclass does not override `__call__`), but
the run then dies in 48.9 s:

```
RuntimeError: The size of tensor a (12) must match the size of tensor b (6)
```

12 = 6 views x 2 for the conditional and unconditional batches. The LowVRAM
pipeline builds its reference cache and control tensor for **6**, and its tiling
guard at `lowvram_mvadapter_i2mv_sd21.py:281` only tiles when the cache batch is
exactly **1**, so the doubled batch is never accommodated. Two structural
assumptions of CFG-off, both in the production repo.

It *is* reachable: `install_rowcol_reference_cache_compatibility` returns early if
`_lowvram_cache_compatibility` is already set on the processor class, so a launcher
could install its own tiling wrapper first and pre-set that flag. That is a real
change to inference behaviour rather than an ablation, so it was not done. If CFG
is wanted, that is the hook — and the control tensor path needs the same treatment.

Evidence: `evidence/compare/boat_sixview_384_cfg3/`, config
`configs/boat_mvadapter_ig2mv_sd21_384x20_cfg3.json`.

## 5d. GEOMETRY — the actual problem, and the fix that worked

The user's verdict partway through: *"the model is shit, that is the problem
currently."* Correct, and everything in sections 4–5c was texture work on a mesh
that was never going to hold up.

The old mesh came from **mini turbo, one photograph, 5 steps** (`steps: 5`, the
turbo default, a speed setting). Three sides of that hull were invention.

The reference sheet the user supplied changed what was possible: real front, side
and back elevations, plus a deck plan. That unlocked the **mv checkpoint**, which
takes up to 4 named views and had been unusable for want of inputs.

| | old | new |
|---|---|---|
| checkpoint | mini turbo | **mv turbo** |
| conditioning | 1 photograph | **2 real elevations** (bow + long side) |
| steps | 5 | **30** |
| octree | 384 | **320** (coarser!) |
| time | ~7 min | 49 min (2935 s) |
| result | `boat_lod.glb` | `evidence/compare/boat/boat_mv2_refsheet.glb` |

**The headline: conditioning beats resolution.** The new mesh is clearly better at
a *coarser* octree. Luna, reviewing independently: "more resolution cannot recover
the correct unseen structure from weak or ambiguous conditioning." That
retroactively explains two earlier dead ends — feature-preserving smoothing
failing at every setting, and the 1022 px conditioning upscale being harmful. The
lumps were never noise to filter or pixels to add; they were the model guessing.

Improvements, agreed by inspection and by Luna: coherent boat hull and footprint
(the top-down view is the clearest evidence), stacked decks readable, bow/stern
structure present, far less single-view facade extrusion.

**Remaining defects:** melted/intersecting railings and trim, decks reading as
fused terraces, thick soft edges, blocky bow/stern transitions, residual
asymmetry, and the mesh is **not watertight**. Luna's verdict: fine as a blockout,
proxy or distant background asset; not ready for close gameplay, collision or hero
use without retopology.

**Not attributed:** steps (5→30) and view count (1→2) were changed together,
because each attempt costs ~50 minutes. A 2x2 ablation is still owed.

### VRAM ceiling, measured

- **4 views do not fit** on the 6 GB card. OOM mid-diffusion at step 17 of 30
  after 35 minutes. The cuBLAS error names it: `m 3072 n 5480 k 102`, where
  n=5480 is the 4-view conditioning token count.
- **2 views fit**, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and
  the ladder `320:2000,256:1500`. Peak allocated 4966 MB, ~250 MB free.
- **Peak allocated is flat across steps** (4-view 5091 MB, 2-view 4966 MB, no
  growth). An OOM at step N is therefore a transient spike against a fragmented
  heap, not accumulation. There is no step after which a run becomes safe, and
  surviving the step where a previous run died proves nothing — a correction to
  a claim made earlier in this session.

## 5e. The online quality bar, and two claims withdrawn

The user supplied 17 online-service GLBs at ~1.5M triangles each
(`C:\AI\LowVRAM3D-benchmarks\production\panda_online_model_targets_20260805`,
note: a concurrent job renames files there — resolve by wildcard). One is the
same riverboat subject, mislabelled `ornate_tower`.

**The online model is uniformly crisp and fully 3D on every side**: ornate arch
facade with statuary, complete broadside with resolved window rows and railings,
paddlewheel, staircase, tiered decks all round. Produced from **one picture**.

### Withdrawn claim 1: "the online model has a flat featureless back"

False. It was an artefact of a bug in `preview_generated_mesh.py` (see below).
Everything built on it is void, including a Luna review recommending a "hybrid
asymmetric pipeline" that locks the observed facade and closes the back with a
smooth shell. That recommendation was answering a question about a corrupted
render. Do not act on it without redoing the analysis.

### Withdrawn claim 2: "the online exports are Z-up"

False. They are Y-up, standard glTF. The upright-looking "top" view was another
symptom of the same bug.

### The actual bug

`preview_generated_mesh.py` concatenated `scene.geometry.values()` **without
applying glTF node transforms**. Hunyuan3D emits a single node so it never
surfaced; multi-node exports rendered with parts collapsed to the origin at
authored scale. Fixed by flattening with `scene.to_geometry()`. A `--up-axis`
flag was added while mis-diagnosing this; it defaults to `auto` and is harmless,
but it is not what fixed anything.

### What the comparison actually shows

The gap is **model capability**, not a pipeline trick and not view count. The
user's point stands: perfect models from one picture are achievable, so the
single image is not the limiting factor. Local is Hunyuan3D-2**mini** turbo at
0.6B parameters, consistency-distilled, on a 6 GB card.

### Settled: diffusion steps do not matter here

5 steps vs 30 steps, identical otherwise, produce visually equivalent geometry
(1.01M vs 0.93M triangles). Confirmed by Luna: "no meaningful difference... not
worth 4.5x the time." Both checkpoints use
`ConsistencyFlowMatchEulerDiscreteScheduler`, which is distilled to converge in
~5 steps. **Never raise steps on a turbo checkpoint.** This brings the mv
two-view route to ~11 minutes, not 49.

## 6. Next steps, in order

1. ~~Improve background removal~~ — **done**, §5.
2. ~~Colour-transfer the six views~~ — **done**, §5b.
3. Finish the `reference_conditioning_scale` 1.0 → 1.5 ablation
   (`configs/boat_mvadapter_ig2mv_sd21_384x20_ref15.json`). Attempt 1 died at
   103.4 s with `CUDA error: an illegal memory access was encountered` — not a
   clean rejection, and this GPU has a documented fp16 defect, so attempt 2 is a
   retry. **If it dies at the same point again the parameter is the cause, not the
   hardware.** This is the only remaining lever on colour-at-source, since CFG is
   unavailable (§5c), and the only lever at all on the invented structure in views
   3 and 4 — colour transfer provably cannot touch geometry.
4. **Re-texture the NEW mesh.** All the texture work in §5–5c applies but must be
   redone against `boat_mv2_refsheet.glb`, because the geometry changed: fresh
   `unwrap_mesh_uv`, fresh controls at 1024, then `build_reference_views` with the
   sheet panels and `bake_multiview_atlas` with `--albedo-target 0.18`
   (**not** the 0.45 default — this subject is genuinely dark and the default
   gamma lift washes it out).
5. **Owed ablation:** 2x2 over steps (5 vs 30) and views (1 vs 2), to attribute
   the geometry gain. ~50 min per cell.
6. Consider a 3-view run (front + left + back). 2 fit and 4 do not, so 3 is
   untested and is the obvious next question.
7. Re-run the barn at 512 conditioning to see how much of its mush was the double
   resample rather than the wide subject.

## 7. Older open items, untouched

- Enforce "never upsample conditioning; resize to 512" in
  `src/lowvram3d/asset_generation.py`. The 1022 upscale was harmful — `ImageProcessorV2`
  already crops/squares/borders and INTER_AREA-downsamples to 512, so upscaling first
  causes a double resample. Every earlier good mini turbo run used `[512, 512]`.
- Feature-preserving smoothing (`workers/refine_mesh_surface.py`) was **measured and
  abandoned** — worse at every setting tried (thresholds 20/35/55, and pre-decimation
  at 35 with max displacement 13.9% of extent). It assumes noise superimposed on
  structure; Hunyuan3D's lumps *are* the geometry. Do not retry without a new idea.
- Nothing has been built in Unreal since welding, the facing gate, or the placement
  objective. No current asset has been placed.
- `scripts/measure_offaxis_stability.py` has still never been executed.
- The castle and diorama source images are still not on disk.

## 8. Capacity test — the search for a local fix is closed

**Question:** the mini turbo checkpoint is 0.6B. Online services make crisp,
fully-3D geometry of this same riverboat from one picture. Is the gap simply
model capacity?

**Test.** Fetched the full Hunyuan3D-2 shape model — 1.1B, `depth 16` /
`depth_single_blocks 32`, same class as the mv checkpoint that already runs at a
4.97 GB peak — via `workers/fetch_hunyuan3d2_shape.py`, which pulls only the DiT
turbo subfolder and its VAE (5.35 GB) rather than the 22.95 GB repo. Ran it on
the *same* photo, *same* seed 12345, *same* octree ladder `320:2000,256:1500`,
*same* 5 steps as the mini run, so capacity is the only variable:

| row | model | params | conditioning | tris | time |
|---|---|---|---|---|---|
| 1 | online service | — | 1 picture | 1.50M | — |
| 2 | Hunyuan3D-2 full turbo | 1.1B | 1 picture | 0.72M | 542 s |
| 3 | Hunyuan3D-2 **mini** turbo | 0.6B | 1 picture | 1.63M | ~7 min |
| 4 | Hunyuan3D-2**mv** turbo | 1.1B | 2 elevations | 1.01M | ~11 min |

Contact sheet: `evidence/compare/boat/boat_capacity_test.png`. Row 2 mesh:
`evidence/compare/boat/boat_full11b.glb`.

**Result: the capacity hypothesis is wrong.** It fits on 6 GB, and it does not
close the gap. Luna's independent read (`--format json`, high effort, image
only, no context about which row was which): row 2 is better than row 3 in
*structural coherence* — steadier hull and cabin proportions, fewer hallucinated
protrusions — but not in fine detail, and the extra 0.5B "does not solve the
fundamental reconstruction bottleneck." Ranked on crispness and on overall
usability: **row 1 > row 2 > row 4 > row 3**.

**Triangle count is not the explanation either.** Row 3 carries *more* triangles
than the online row 1 and still looks melted. Luna: the online mesh spends
triangles on railings, window recesses and sharp feature boundaries; row 3
spends them on "bumps, ripples, fused ornaments, and topology noise." Density is
not information.

**Not a normal-map or post-process trick.** Worth stating because it was the
last cheap explanation available: these are flat-shaded, textureless renders,
and row 1's thin features hold their silhouette and occlusion across all four
angles. Normal maps cannot create a railing gap in a silhouette. Luna reads row
1 as better reconstruction *plus* substantial cleanup — feature-preserving
remesh, hole fill, component removal — not cleanup standing in for generation.

**Every locally reachable knob has now been measured and eliminated:** diffusion
steps (5 ≡ 30 on a distilled checkpoint, §6), octree resolution (320 beat 384),
conditioning views (helped structure, not detail), feature-preserving smoothing
(worse at every setting, §7), model capacity, triangle count. What is left is
structural to the online pipeline — higher-resolution latent, a better geometry
decoder, feature-aware surface extraction instead of uniform marching cubes,
probably retopology — and none of it is a setting on this card.

**Verdict, asked for bluntly and given bluntly:** "No realistic tweak on a 6 GB
GPU will close this gap." The working split is the online service for hero
assets, local generation for blockouts, proxies and background pieces. Within
local, prefer row 2 (full 1.1B) for single-image work and row 4 (mv, 2 views)
when silhouette consistency across views matters.

Do not re-open this by trying a larger checkpoint, a finer octree, or more
triangles. Those are the three things that have already been falsified. A real
re-open needs a *different pipeline* — SPAR3D, TRELLIS, TripoSG — not a bigger
dial on this one.

## 9. Surface extraction tested — also not the bottleneck

**Question:** every mesh this project has made was extracted with Lewiner
marching cubes (`mc_algo='mc'`). MC places one vertex per grid edge crossing and
so cannot represent a crease -- a sharp edge becomes a staircase. Was the
"melted" look an extraction artefact rather than a generation failure?

**A bug first, because it nearly produced a false negative.** `--mc-algo` was
added to `workers/mini_turbo_generate.py`, and the first run reported success
with `dmc` while actually running `mc`. `pipeline.enable_flashvdm()` swaps in
the turbo VAE (`replace_vae=True`) and then sets the extractor from its own
`mc_algo='mc'` default, so an extractor assigned *before* that call is discarded
along with the VAE object it was attached to. It is silent. It was caught only
because both meshes came out with byte-identical face connectivity; the
differing vertex hash was fp16 nondeterminism, not a real difference. The
extractor assignment now happens after `enable_flashvdm`, and the receipt
records `surface_extractor` -- the class actually in place -- rather than
echoing the requested flag back.

**Controlled comparison.** Same photo, seed 12345, octree 320, 5 steps, mini
turbo; only the extractor differs.

| extractor | tris | bodies | watertight | euler | front coverage |
|---|---|---|---|---|---|
| `mc`  | 1,089,998 | 13 | **yes** | -714 | 48.25% |
| `dmc` | 1,088,546 | 13 | no | -630 | 48.17% |

Previews: `evidence/compare/boat/preview_boat_mini_mc320.png` and
`preview_boat_mini_dmc.png`.

**Result: no meaningful difference.** The two render as near-identical. DMC is
marginally tidier on the balcony rails and slightly less staircased, and it
recovers no window, no railing gap, no architectural line. It also *loses*
watertightness and emits at half scale (`DiffDMC(normalize=True)` +
`center_vertices` puts the mesh in a unit box instead of applying the `bounds`
transform), so adopting it would need a rescale.

**Why this matters more than the result itself.** A better extractor cannot
invent a feature the field does not contain. Getting the same mesh from a
primal and a dual method says the decoded occupancy field is genuinely smooth
where the windows should be. That moves the blame upstream of everything
measured so far -- past steps, octree, views, smoothing, capacity, triangle
count and now extraction -- and onto the **latent representation**.

**Operational note:** `dmc` needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
and the `256:1500` fallback rung. Loading diso's CUDA module on top of a run
that already peaks at 5.7 GB caused an illegal memory access mid-diffusion
(step 2) on the first attempt. With the allocator flag it peaks at 5473 MB.

**Do not retry a third extractor.** The next question is not how the surface is
extracted but what is being extracted from.
