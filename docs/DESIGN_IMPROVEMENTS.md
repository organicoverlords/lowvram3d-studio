# Where this pipeline is actually behind, and what to borrow

Research pass against published single-image-to-scene work, filtered by what
runs on a 6 GB Turing card. Ranked by measured defect, not by novelty.

## The useful finding

[3D-RE-GEN](https://github.com/cgtuebingen/3D-RE-GEN) (TU Tübingen,
[arXiv:2512.17459](https://arxiv.org/abs/2512.17459)) is current state of the
art for single-image compositional scene reconstruction, and **it generates
geometry with Hunyuan3D-2 — the same generator this pipeline already uses.**

Its stages are:

| 3D-RE-GEN | here | gap |
|---|---|---|
| Grounded-SAM **instance** segmentation | SegFormer **semantic** segmentation | real |
| generative inpainting of occluded objects | none | real |
| Hunyuan3D-2 per object | Hunyuan3D-2 mini turbo per object | **none** |
| multi-view texturing | single-view object-space projection | real |
| 4-DoF differentiable placement on the ground plane | heuristics + a post-hoc overlap audit | real |
| generated background | terrain primitives | real |

So the generator is not the problem, and swapping it would not fix anything we
have observed. Every remaining defect is in a stage we hand-rolled. That is good
news: the fixes are modular and individually testable.

---

## 1. Instance segmentation — retracted as the top item

> **Superseded by measurement.** This section was written before anyone looked
> at the photograph. SAM 2.1 tiny was downloaded and run, and its masks plotted
> over the region, which is when the premise collapsed: the "tree line" is **one
> wind-swept tree** arching over the barn, plus one small tree at the right.
>
> There was nothing to instance-segment. Connected components already declines
> to split that tree, and SAM declines too, at every threshold tried — both are
> right. The defect was never missing instance segmentation; it was that
> clustering *invented* twelve instances from a pixel budget, and every symptom
> below followed from that one number.
>
> Fixed by emitting one instance per connected component. SAM is downloaded and
> works with no install, and remains the right tool for a scene that genuinely
> contains several touching objects — but it is not what was wrong here, and
> shipping it would have papered over the real bug.
>
> Kept, unedited below, because the reasoning was sound and the conclusion was
> still wrong. The pipeline had three independent symptoms and I inferred a
> cause from them instead of opening the image.

**Defect it was thought to fix.** SegFormer returns one mask per *class*. Every
tree in the photograph is a single "tree" region, which is why the conditioning
crop was a wall of foliage running off its own frame on two sides, why the
generator returned a 1.96 x 1.13 x 0.17 m slab, and why placement had to invent
instances by k-means over a semantic blob. It is also why a tree-trunk shadow
could be promoted to a building: with no instance boundary, "is this one thing?"
is never asked.

### Option A — SAM 2.1 tiny automatic masks, intersected with the SegFormer class map (recommended)

Run SAM's automatic mask generator for instance boundaries, then label each
instance by majority vote against the SegFormer map we already compute. Neither
model needs to know about the other.

- 39 M parameters, ~0.5 GB. Never contends with generation.
- Apache-2.0, no gated download, no contact form.
- Reuses the existing segmentation stage rather than replacing it.
- Over-segments (a tree becomes crown + branches). Needs a merge pass —
  cheap, since regions we merge share a class and are spatially adjacent.

### Option B — SAM 3 promptable concept segmentation

Prompt `"tree"`, get one mask and ID per tree directly. Strictly better output
and no merge pass.

- 0.9 B parameters, ~1.8 GB at fp16. Fits, but only with the editor closed.
- **Meta SAM License, gated behind a contact form.** Not Apache. This is a
  project-level licensing decision, not a technical one.
- Turing has no bf16, and the examples are written for it. Worse, this card has
  a *recorded* fp16 cuDNN convolution defect (~25% NaN) — see
  `HARDWARE_RESEARCH.md`. SAM 3 is mostly attention rather than convolution, so
  it will probably be fine, but "probably" needs a NaN check before trusting it.

**Recommendation: A.** Same defect fixed, an order of magnitude less VRAM, no
licence question, and it composes with what is already here rather than
replacing it.

---

## 2. Amodal completion — inpaint the occluded part before generating

**Defect it fixes.** The barn is partly behind trees. We hand the generator the
visible fragment and it reconstructs a fragment. The `trunk_support` primitives
are a hand-rolled special case of exactly this problem: a tree crown is visible,
its trunk is not, so we *invent a cylinder*. 3D-RE-GEN treats occlusion as an
image-editing task — inpaint the hidden pixels first, then reconstruct once,
with no per-class hack.

**Why it is cheap here.** SD 2.1 is already on this machine (kept for the
MV-Adapter work, per `lowvram3d-mvadapter-fp32-and-projection-cost`). SD 2.1
inpainting needs no new download and runs comfortably in the headroom.

**Caveat worth stating plainly.** This *invents* geometry. It must be recorded
per asset — which pixels were observed and which were inpainted — or the
pipeline starts asserting things about the world that were never photographed.
The same reason the border-contact check refuses a subject cut off by the
photograph's own edge.

---

## 3. Multi-view texturing — the right fix, with a bad track record here

**Defect it fixes.** `project_crop_texture.py` projected one view along object
space −Z onto *every* face, so surfaces perpendicular to the camera got a few
pixels stretched down their whole length and the back got the front painted on
it. Those faces are now flat-filled instead, which stops the pipeline asserting
appearance it never observed — but it does not create the missing appearance.

**How much is missing, measured twice, independently:**

| asset | metric | observed |
|---|---|---|
| barn (this session) | faces facing the conditioning camera | **39.8 %** |
| ship (`20260803-ship-production-texture.json`) | directly observed texels of owned area | **19.0 %** |

Different assets and different units, same conclusion: a single view sees a
minority of a closed surface, and everything else is synthesis.

**Why this is not just wiring.** An earlier version of this section said MV-
Adapter TG2MV was installed and ready. The models are indeed local and the
config is complete, but the record says otherwise about readiness:

- the last recorded run is `EXECUTED_384_QA_REJECTED`;
- the ship receipt's own root cause reads *"5 of the 6 conditioning views are
  mirrored fills at confidence 0"* — MV-Adapter was not producing usable novel
  views, and a fallback filled them;
- that projection run took **190 minutes**.

So the honest position: multi-view is the correct fix for the largest remaining
defect, it is the reason this lane exists, and it has never yet worked on this
machine. It should be approached as a debugging project with its own budget,
not as a step in a scene run. The FP32 conditioning encoder and the fp16 cuDNN
convolution defect are both known and recorded, which is where to start.

Hunyuan3D-Paint is the alternative and is a second large model on the same card.

---

## 4. Turn the overlap audit into an objective

**Defect it fixes.** Placement is heuristics, and correctness is checked
*afterwards* by `audit_actor_overlaps`. When the audit fails, nothing acts on it
— it reports. 3D-RE-GEN instead optimises 4 DoF per object (position + yaw)
against the ground plane.

The same thing is available here at almost no cost, because every term is
already computed:

- **ground contact** — `ground_plane_unreal` is already fitted by least squares;
- **non-penetration** — the audit's AABB overlap is already the penalty;
- **reprojection agreement** — the object's silhouette rendered back into the
  source camera should match its segmentation mask, which we now have on disk.

That is a small torch optimisation over a handful of parameters, on CPU, in
seconds. It converts three existing measurements from *reports* into a *loss*.

`measure_offaxis_stability.py` is the honest evaluation of the result, and
should not be used as its objective — source-view similarity has already
certified a world-space-wrong mesh twice in this project.

---

## Not recommended now

- **Generated background** (3D-RE-GEN's fifth stage). Real capability, but it is
  a second generative model and the terrain primitives are not currently what
  makes the output look wrong.
- **Swapping the geometry generator.** SOTA uses the one we have.
- **PartCrafter / P3-SAM.** Already triaged as not a 6 GB baseline.

## Order

~~1 first~~ — retracted, see above. 1 was ranked top from three symptoms that
shared a cause nobody had checked. Remaining order: **2** (amodal completion,
which subsumes the `trunk_support` hack), then **3** (multi-view texturing,
already installed), with **4** done.

## The lesson worth keeping

Three separate diagnostics — crop border contact, projected scale disagreement,
and pairwise overlap — each independently pointed at the vegetation region, and
each produced a plausible, self-consistent story about instance segmentation.
All three were downstream of one wrong number: a cluster count taken from the
pixel budget. Opening the image took two minutes and settled what a day of
inference had got wrong.

Agreement between measurements is not evidence when they share an input.

## Sources

- [3D-RE-GEN](https://github.com/cgtuebingen/3D-RE-GEN) · [arXiv:2512.17459](https://arxiv.org/abs/2512.17459)
- [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- [facebookresearch/sam3](https://github.com/facebookresearch/sam3) · [facebook/sam3 on Hugging Face](https://huggingface.co/facebook/sam3)
- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) · [Hunyuan3D-Paint texture generation](https://deepwiki.com/Tencent/Hunyuan3D-2/4.2-texture-generation-(hunyuan3d-paint))
- [Diorama: zero-shot single-view indoor scene modeling](https://arxiv.org/pdf/2411.19492)
