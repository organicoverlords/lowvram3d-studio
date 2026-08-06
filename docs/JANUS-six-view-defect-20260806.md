# The duplicated face in the six-view texture route

> ## RETRACTED, 2026-08-06 — there is no duplicated face
>
> Everything below the line was written before the review sheet existed. The
> sheet disproved it within the hour. **Keep reading only for the QA history;
> the diagnosis is wrong.**
>
> On the panda, the source photograph was projected onto **the back of the
> head**. Index 0's geometry is featureless ghillie fur, and the photographic
> face sits on it as a decal. The mesh's actual sculpted face — eyes, muzzle,
> whiskers, the rifle across the chest — is at index 3, and it received
> camouflage invention. `evidence/compare/panda2/photo_vs_geometry.png` shows
> both hemispheres, photo atlas beside geometry, same camera.
>
> So what I called "a second face on the true rear" was MV-Adapter correctly
> painting the model's **real** face, and what I called "the sharp front" was a
> photograph pasted on the back of its head. **MV-Adapter behaved correctly
> throughout.** The defect is a 180° error in the front-axis solve, upstream of
> generation entirely.
>
> The paint-based front audit then confirmed the error rather than catching it,
> exactly as designed: it reports *where paint landed*, which is a fact about
> the projection, not about the shape. When the projection is 180° off, the
> audit agrees with it. That is a fifth check that could not see what it was
> named for — and this one I built.
>
> This also explains both user reports precisely. "The front is all pixelated
> and glitchy": the real front got invented camouflage. "There are faces at back
> and the front has 2 faces": the photo decal on the back plus the real
> geometric face.
>
> **Corrections to specific claims below:**
> - *"The labels are rotated by one"* — panda only. Whale and shaman have their
>   true opposite at index 2, as labelled. Measured, not assumed.
> - *"The generated-to-control assignment is identity (0.91–0.99)"* — not
>   supported. It cannot be reconciled with the sheet, and I have not re-derived
>   it.
> - *"`reference_conditioning_scale` is the knob"* — it is a real, untuned knob,
>   but it is not the fix for this. The ablation was **not** run.
> - *The whale is fine.* Its photo landed on the profile its geometry actually
>   has; the mirror flank is invented, which is expected.
>
> **What is still true and worth keeping:** the QA repair. Grayscale correlation
> cannot see a face (0.162 on a rear that has one), `semantic_gate_passed` was
> hardcoded `True`, and promotion now requires a human verdict that fails closed.
> That work stands regardless of the diagnosis.
>
> Next: re-solve the panda's front axis geometrically rather than by paint, and
> add a gate that compares where the photograph landed against where the mesh's
> detail actually is.

---

**Status 2026-08-06.** Four assets have been through the six-view route. The
geometry is good, the fronts are sharp, the coverage numbers are the best this
pipeline has produced — and every one of them has a second face on the true
rear. This document states what the defect is, what was measured, what was ruled
out, and what has not been tried. It is written to be handed to a reviewer
without the conversation that produced it.

## Where things stand

| asset | resolution | directly observed | seen by ≥2 views | front | rear |
|---|---:|---:|---:|---|---|
| sky whale courier | 512 | **69.59%** | 36.27% | sharp, photographic | duplicated face |
| red panda | 512 | 66.43% | 29.82% | sharp, photographic | duplicated face |
| boat | 384 | 64.94% | 19.50% | sharp | not re-audited |
| bird-skull shaman | 512 | 53.85% | 26.18% | sharp, photographic | duplicated face |

"Directly observed" means the texel was painted by at least one real view rather
than filled. The remainder is surface-space kNN fill. The boat's 64.94% is the
`boat_mv_fixed` run; its three earlier runs at 5.50%, 5.50% and 16.03% were the
stale-`CANONICAL_TRANSFORM` bug, not a texture problem.

## The defect

MV-Adapter paints the reference image's frontal appearance onto the view that
faces **away** from the reference camera. This is the classic Janus / multi-face
failure of multiview diffusion, and it is not an indexing bug on our side:

```
true opposite pairs, measured from the contract's own camera directions:
  index 0 (front )  <->  index 3 (left  )   dot -1.000
  index 1 (right )  <->  index 2 (rear  )   dot -1.000
  index 4 (top   )  <->  index 5 (bottom)   dot -1.000
```

The generated-to-control assignment was measured as **identity** (correlation
0.91–0.99 per tile), so the images are not rotated. The **labels** are: the
builder emits `front, right, rear, left`, so the tile labelled `rear` sits 90°
from the true opposite of `front`. The boat's config already recorded
`builder_labels_are_rotated_by_one` and nothing downstream ever read it.

## Why no gate caught it

The QA's Janus detector compared index 0 against index 2 — i.e. the front
against a **side** view, which are 90° apart and legitimately uncorrelated. It
reported an innocent 0.16 while a full second face sat at index 3. The detector
now derives the pair from camera directions rather than from label order, so it
cannot be rotated out of correctness again. **The underlying artifact is
unfixed.**

This is the fourth silent-pass failure in this route. The others:

1. `build_mvadapter_cpu_controls.py` hardcodes `world_up = [0,0,1]` while glTF is
   Y-up — the shaman was built lying on its side in all four horizontals, with
   `passed: true`.
2. `fit_to_mask` had no image-space Y negation, so every camera search scored an
   upside-down silhouette and never rose above 0.585.
3. `preview_textured_mesh.collect()` ignored scene node transforms, so the tool
   used to verify orientation fixes could not see them. A correct 151.67°
   rotation was reported to the user as "did not take".

The pattern is consistent: **a check that cannot see the thing it checks returns
pass.** Every gate added since is required to derive its reference frame from
measured geometry, not from a label, a constant, or a convention.

## The mechanism, located in the source

`mvadapter/models/attention_processor.py:362`:

```python
if use_ref:
    hidden_states = hidden_states + hidden_states_ref * ref_scale
```

`hidden_states_ref` is shaped `(B*num_views, seq, dim)` — the view axis lives
**inside** the batch dimension — and `ref_scale` is a scalar. So the reference
photograph is added into the residual stream of every view at equal strength,
including the view pointing directly away from it. That is the duplicated face,
in one line.

## The knob nobody turned

`reference_conditioning_scale` and `control_conditioning_scale` are both exposed
pipeline arguments (`pipeline_mvadapter_i2mv_sd.py:262,258`) and are already
plumbed through `workers/mvadapter_sd21_six_view_inference.py:886`. Every run in
this project has used the default:

```json
"reference_conditioning_scale": 1.0,
"control_conditioning_scale": 1.0
```

A second-opinion research pass (DeepSeek V4, 2026-08-06) asserted that
MV-Adapter has *no* exposed conditioning-scale parameter. That is wrong —
verified against `C:\AI\mvadapter-upstream-inspection\mvadapter`. Its diagnosis
was right, though: the face arrives through the decoupled **image
cross-attention**, not through the geometry guider, which is exactly the channel
`reference_conditioning_scale` scales.

Because the view axis is inside the batch dimension, a `(B*V, 1, 1)` tensor
broadcasts correctly at line 362. **Per-view reference strength** is a few lines
and costs no VRAM: full 1.0 at index 0, ~0.15 at the measured true opposite,
~0.6 on the sides.

## What was tried and failed

**CFG 3.0 with an anti-Janus negative prompt.** Aborted at step 0 with
`RUNTIME_REJECTED_SEQUENCE_CONSUMED`. The receipt shows the launcher's
`--guidance-scale` and `--negative-prompt` never reached the worker, because the
worker hardcodes them at the call site:

```python
guidance_scale=1.0,          # mvadapter_sd21_six_view_inference.py:885
negative_prompt=None,        # :881
```

System RAM was also down to 1.6 GB at the time. Separately, this pipeline's CFG
implementation stacks the batch (`torch.cat([latents] * 2)`,
`pipeline_mvadapter_i2mv_sd.py:568`), giving a 12-view UNet call. That is an
implementation choice, not a property of CFG — two sequential 6-view calls per
step with `eps = (1+w)·eps_cond − w·eps_uncond` would fit 6 GB.

**A rolled control bundle** (photographed view moved to index 1, to test whether
the anchor is positional or a model prior) was built at
`evidence/compare/panda2/controls_512_rolled/` but is **not runnable**: the
permutation `[3,0,1,2,4,5]` was derived assuming 0↔2 opposites, so the contract's
`front_rear_direction_dot` and `left_right_direction_dot` came out 0.0 against a
gate that requires ≤ −0.999. It needs rebuilding against the true pairs (0↔3,
1↔2) before it can answer anything.

**A surface-space kNN fill** replacing the atlas-space dilation. It removes
cross-chart bleeding by construction — UV neighbours are not surface neighbours,
and this atlas has 7,371 charts — but it scores *worse* on flatness (30.1% vs
20.8%) and is visually near-indistinguishable. It was kept for the correctness
property. It did not fix the pixelation complaint and should not be credited
with having done so.

## What the alternatives cost (research, 2026-08-06)

Nothing else fits 6 GB. Figures from upstream repos and issue threads:

| candidate | VRAM | verdict |
|---|---|---|
| Hunyuan3D-Paint-v2-1 (PBR) | 21 GB texture / 29 GB total; ~15 GB `--low_vram_mode` | no. A 16 GB mobile 5080 cannot run the paint stage |
| Hunyuan3D-Paint-v2-0 (RGB) | +10 GB over the 6 GB shape stage | no |
| TEXTure / Text2Tex / Paint3D | ~8 / 12 / 16 GB | no |
| Meta 3D TextureGen | ~20 GB+, non-commercial weights | no |
| MVPaint / TexPainter / TexGaussian | >6 GB, research code, text-conditioned | no |
| Era3D | ~2–3 GB, 512px, row-wise attention | **only real alternative**; reduces Janus, does not guarantee it away |
| MV-Adapter SD2.1 ig2mv (current) | fits 6 GB, measured | keep |

Multi-image geometry is also out. VGGT falls back to fp32 on non-bf16 GPUs
(~45 GB for 10 frames) and TU116 has neither bf16 nor FlashAttention; Fast3R
needs 6.33 GB *on an A100 in fp16 with FA2*; DUSt3R/MASt3R at 8 views ≈ 24 GB;
COLMAP and Meshroom fail on stylised low-texture art. Hunyuan3D-2mv is genuinely
multiview-conditioned but ~14–16 GB nominal and ~8 GB offloaded, and **there is
no multiview mini** — the 0.6B is single-view only. TRELLIS multi-image is a
tuning-free approximation its own authors call suboptimal, with a 16 GB floor.

The decisive point on multi-image: extra views generated by a
single-image-conditioned model carry the **same front bias**, so feeding them to
a geometry model launders the Janus error into geometry rather than fixing it.
Independently confirmed — feeding Hunyuan3D-2mv the real front plus a
*generated* back view scored **worse than the single-view path** on ULIP and
Uni3D. Multi-image only wins with a genuinely independent rear, i.e. a drawn
turnaround sheet. And the model that wants one does not fit.

## Proposed order of work

1. **Sweep `reference_conditioning_scale`** 1.0 → 0.6 → 0.35 on the panda. One
   GPU sequence each, no code change, no extra VRAM. If the rear face fades this
   is over.
2. **Per-view `ref_scale` tensor** if the global sweep costs too much side-view
   fidelity. ~5 lines at the call site.
3. **Bilateral mirror of the front UV into the rear**, feathered, plus side-view
   back-projections. Correct regardless of 1 and 2 — the rear hemisphere is
   observation-free and all four subjects are bilaterally symmetric.
4. **Two-pass reference swap**: run MV-Adapter a second time with the *rear
   geometry render* as the reference image and keep only the tile aligned to
   that camera. The model cannot paint a face it was never shown. Use it to seed
   an inpaint rather than trusting it outright.
5. **Sequential CFG** (two UNet calls per step, first ~60% of steps only) if
   1–4 leave a visible face.

Open and unexamined: the panda's tail corruption, reported by the user and not
yet looked at; the boat and whale have not been re-run through the corrected
Janus detector.

## Questions for a reviewer

1. Is per-view `ref_scale` sound, or does breaking the uniform reference weight
   across views destabilise the multi-view attention that keeps the six tiles
   consistent with each other?
2. Is the bilateral-mirror rear an acceptable production answer, or does it
   read as obviously fake on asymmetric detail (the shaman's hanging props, the
   whale's cargo rigging)?
3. Is Era3D worth the integration cost as an alternate multiview source, given
   it is a reconstruction model being used as a texture source?
4. Is there a defensible way to use a *drawn* rear view here, given the user can
   author one, without paying the 8 GB Hunyuan3D-2mv floor?
