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

## 1. Instance segmentation — fixes the worst defect

**Defect it fixes.** SegFormer returns one mask per *class*. Every tree in the
photograph is a single "tree" region, which is why the conditioning crop was a
wall of foliage running off its own frame on two sides, why the generator
returned a 1.96 x 1.13 x 0.17 m slab, and why placement had to invent instances
by k-means over a semantic blob. It is also why a tree-trunk shadow could be
promoted to a building: with no instance boundary, "is this one thing?" is never
asked.

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

## 3. Multi-view texturing — fixes the stretched sides

**Defect it fixes.** `project_crop_texture.py` projects one view along object
space −Z, so every surface not facing the camera smears. The receipt already
admits this: `"observed_from": "single view; sides stretch along the projection
axis"`.

MV-Adapter TG2MV is already installed and already has its FP32 conditioning
encoder fix. This is wiring, not new capability. Hunyuan3D-Paint is the
alternative but is a second large model to host on the same card.

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

1 first: it is the largest measured defect, the cheapest option is 0.5 GB and
Apache-2.0, and both 2 and 4 get better inputs once regions are instances rather
than class blobs.

## Sources

- [3D-RE-GEN](https://github.com/cgtuebingen/3D-RE-GEN) · [arXiv:2512.17459](https://arxiv.org/abs/2512.17459)
- [facebookresearch/sam2](https://github.com/facebookresearch/sam2)
- [facebookresearch/sam3](https://github.com/facebookresearch/sam3) · [facebook/sam3 on Hugging Face](https://huggingface.co/facebook/sam3)
- [Hunyuan3D-2](https://github.com/Tencent-Hunyuan/Hunyuan3D-2) · [Hunyuan3D-Paint texture generation](https://deepwiki.com/Tencent/Hunyuan3D-2/4.2-texture-generation-(hunyuan3d-paint))
- [Diorama: zero-shot single-view indoor scene modeling](https://arxiv.org/pdf/2411.19492)
