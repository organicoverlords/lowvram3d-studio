---
name: visual-verify
description: Verify any visual claim about an image — renders, textures, atlases, mattes, masks, screenshots, plots — by sizing the evidence correctly and getting independent readings from Luna and Spark before stating a verdict. Use whenever about to say a render looks better, a defect is fixed, a matte is clean, or any other judgement made by looking at a picture.
---

# Visual verification

A visual verdict from one pair of eyes is not evidence. This skill is the
procedure that has to run before saying what an image shows.

It applies to **every** image used to decide something: clay renders, textured
renders, texture atlases, mattes and masks, segmentation previews, conditioning
images, multiview sheets, before/after comparisons, viewport screenshots, plots.

The order is fixed: **inspect the image yourself → show the user → ask the
models → report all three.** Skipping the first step poisons everything after
it.

## 0. Open the image and confirm it is usable — before anything else

Never show a render or dispatch a vision job without opening the file and
checking, at minimum:

- **Is the subject actually in frame?** Not "did the script exit 0" — look.
- **Is the feature in question inside the crop?** A face crop with no face is
  worse than no crop.
- **Is it legible at the size it will be judged at?**
- **Do the panel labels match the panels?**

If any answer is no, fix the render. Do not send it and caveat it.

This step exists because a cropping bug produced a sheet containing only reed
tips and empty backdrop — no head, no beak anywhere in the frame — and it was
sent to both models anyway. **Luna described a beak yawing sideways with a
downward roll. Spark reported "beak fully visible in all 9 views, crop adequate,
no occlusion" and gave a confident straight-midline verdict.** Both fabricated a
detailed reading of something that was not in the image, and neither said "I
cannot see a beak". Vision models will not reliably tell you the frame is
empty — they will answer the question you asked as if the evidence existed.

A model's confident answer is therefore not evidence that the image was valid.
The only thing that establishes that is opening it.

### Cropping traps that produce empty frames

- **Contact sheets have label bands.** Sampling the background colour at
  `a[5,5]` hits a white caption strip, so "not background" matches the entire
  backdrop, the bounding box becomes the whole image, and every relative crop
  lands in empty space. Sample the backdrop from inside the render area.
- **The topmost pixel is not the top of the head.** On this heron, reeds project
  above the skull, so a head band anchored to the bbox top captures reeds only.
  Anchor to the feature, not to the extremes.
- **Prefer alpha when the renderer emits it.** Where there is no alpha, verify
  the derived mask by printing its bounding box and per-row widths before using
  it — a mask covering 100% of every row is a broken mask, not a large subject.

## 1. Size the evidence before looking at it

A claim about a small feature — separated toes, a staff shaft, a reed stalk, a
hood brim, ornament count, texture blotches — cannot be made from a contact
sheet. Multi-view sheets put each subject at 250–400 px, below the size at which
the feature resolves at all.

Check the pixel height of the subject in the image. Under ~600 px:

1. Re-render one or two views at 2000 px or more.
2. Crop to the disputed region.
3. Upscale the crop.
4. Judge that.

If the feature is still not legible in the crop, the answer is **"cannot tell"**,
not a verdict. `workers/reflow_view_sheet.py` exists because seven views in one
row is ~285 px each.

## 2. Get two independent readings

Both models, through `command-code` — never codex. DeepSeek is never used here;
it has no vision.

```bash
command-code -p "Look at <absolute image path>. <specific question>" -m gpt-5.6-luna
command-code -p "Look at <absolute image path>. <specific question>" -m meta/muse-spark-1.2-contributor
```

Write the question so it can come back negative:

- **Bad:** "Does this look better than the previous version?"
- **Good:** "The left figure is octree 384, the right is 448. Are the toes
  separated into distinct digits in either, or fused? Answer per figure. If the
  resolution is too low to tell, say so."

Name the feature in dispute, state what each side of the comparison is, and give
an explicit escape hatch for "cannot tell". A leading question gets a leading
answer from both models and produces false confirmation.

## 3. Report all three readings

State Luna's, Spark's, and mine — including disagreement.

- If the models split, say they split. Do not pick the one that agrees.
- Do not paraphrase a hedge into agreement.
- If both models say the difference is marginal, the difference is marginal,
  regardless of what the measurement said.

## Why this exists

Every rule above is a correction the user made, not a precaution:

| claim | what happened |
|---|---|
| shaman staff fixed | asserted twice from thumbnails, wrong both times |
| octree 448 clearly better | asserted from a sheet at ~250 px/figure; user disagreed on sight |
| Mini Turbo paint is good | atlas underneath was flat-filled at 4.5 texels/face |
| TRELLIS geometry is bad | judged before the vendor finalizer had run |

The pattern is never a vision problem. It is one perspective stating a verdict
with nothing independent able to contradict it.
