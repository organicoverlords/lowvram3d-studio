# The front axis, and why every criterion so far answered a different question

The red panda's source photograph was projected onto the back of its head. This
is the fix, and the two adversarial reviews that shaped it.

## Why the silhouette fit failed, stated correctly

I had this half right and DeepSeek sharpened it. I said "front and back
silhouettes of a mirror-symmetric subject are near-identical". The real
degeneracy is stronger and has nothing to do with left-right symmetry:

> For a convex body, the orthographic silhouette from direction **u** and from
> **−u** is **the same set**, by construction.

A hooded character in a ghillie suit is very nearly convex. The muzzle protrudes
too little to break it. So the objective has two near-equal optima and
initialisation decides which one the search lands in. **IoU 0.915 is not
confidence — it is exactly what the wrong answer looks like** on a degenerate
objective.

The sky whale's axis came out correct because its silhouette is genuinely
asymmetric — a distinct tail fluke. That is not evidence the fit works; it is
evidence the fit works when the objective is non-degenerate, which is the
subjects that never needed it.

## The four criteria, and what each actually measures

| criterion | question it answers | verdict |
|---|---|---|
| tail colour | which side is the tail on **in the source image** | fallback tie-breaker, wrong on the panda |
| silhouette IoU | which yaw's **outline** matches | degenerate under fore-aft flip |
| painted-texel coverage | which camera **the projection painted from** | **circular** — confirms a backwards projection at 0.8364 vs 0.0016 |
| photo/geometry detail correlation | — | **does not work**, 0.081 wrong vs 0.124 right |
| DINOv2 geometry-vs-source | which yaw's **shape** matches the source | the fix, see below |

The last row is the only one that consults evidence the silhouette does not
contain. A face protrudes; a hood does not. Comparing the source image against a
shaded render of the **bare geometry** uses that, and it touches no atlas, no
paint and no projection, so it cannot inherit the error it exists to catch.

## What was built

`workers/resolve_front_axis_dino.py` sweeps yaw, clay-renders the geometry at
each, and scores DINOv2 cosine similarity against the source. On the panda the
score curve peaks at the face and troughs at the back:

```
yaw 345  0.4224   <- face and rifle visible
yaw   0  0.4178   <- face and rifle visible
yaw 180  0.2288   <- back of hood, tail
```

**Report the antipode margin, not the runner-up margin.** The winner's runner-up
is its own neighbour: 345 beats 0 by 0.0045, which reads as hopeless ambiguity
and is nothing of the kind — both show the face. Against its antipode the same
winner leads by **0.194**. My first summary of this run quoted the 0.0045 and
very nearly retired a criterion that works.

`fast_texture_projection.fit_camera` now calls `_break_fore_aft_tie` after the
silhouette search: it renders the fitted yaw and yaw+180 **in the projector's own
rotation convention**, embeds both plus the source, and keeps the better. Nothing
is translated between frames — cross-frame translation is itself a bug class this
file has had. If the two candidates are within `FORE_AFT_MIN_SEPARATION` (0.05)
or the model is unavailable, the fitted yaw is kept unchanged and the receipt
says so. It is a tie-breaker, not an override.

## What the reviewers said, including where they were right against me

**Luna** rejected DINO as a sole decision rule: *"global CLIP or DINO similarity
between a source image and a normal render is weak by itself. The modalities
differ too much, especially for stylized art."* That is a fair warning and it is
why this ships as a **tie-breaker gated on a separation threshold**, applied only
to the fore-aft pair, rather than as the camera solver. Luna also pointed out
that face detectors (MediaPipe, RetinaFace) will not fire on a hooded red panda
or on a normal map — correct, and it rules out the obvious alternative.

**DeepSeek** raised the hypothesis I had not: **H1, the geometry is flipped and
the projection is correct.** Under H1 a 180° rotation does not fix anything, it
exchanges the two defects. The discriminator it proposed settles it: the sculpted
geometry at the true front carries **the rifle across the chest and the ammo
belt, in the same places as the source**. A generator does not sculpt the rifle
onto the character's back. The geometry is right; the camera was aimed at the
back. H2 confirmed, H1 rejected.

DeepSeek also killed the metric I was about to report:

> Directly observed texel coverage is a *completeness* statistic and is
> near-invariant to correctness. A 180°-wrong projection saturated an entire
> hemisphere and still scored 66.4%.

## What to report instead of coverage

1. **Photo↔geometry face agreement** — fraction of texels on the *sculpted* face
   (defined from geometry curvature, never from texture) that the photograph
   paints. Non-circular: the face set comes from geometry, the paint from the
   projection, and the metric tests their agreement.
2. **Front fidelity** — DINO/CLIP similarity between the finished mesh rendered
   from the true front and the source.
3. **Back-view face absence** — the metric that would have caught this.
4. **Cross-view disagreement** in overlapping atlas regions.
5. **The human verdict**, which is already wired and fails closed.

## Chirality: settled, and the mirror was a confound

A mirrored copy of the mesh fitted at IoU 0.9151 and produced by far the best
looking texture, which suggested the reconstruction was left-right reversed. It
is not. The mirror only looked better because its **camera fit** was better, and
fit quality and handedness were being read off the same picture.

`workers/measure_chirality.py` separates them. Chirality is carried by the
geometry -- a rifle sculpted across the chest, a tail sculpted to one side -- so
it clay-renders the bare mesh at the resolved front and asks whether the source
or the **mirrored source** matches better in DINOv2 space. No texture is
involved, so the projection under suspicion contributes nothing to the verdict.

```
                          vs source   vs mirrored source   verdict
real mesh                    0.5136        0.3664          MATCHES_SOURCE
deliberately mirrored copy   0.3898        0.4614          MIRRORED
```

The second row is the control, and it is the only reason the first row means
anything. Three signals agree: this measurement, a human read of the clay render
against the source (the tail is on the viewer's right in both), and two vision
models independently reporting that the *mirrored* mesh's contact sheet is
reversed relative to the source. **Do not mirror.**

### The version that passed and was wrong anyway

The first working version compared pooled DINOv2 **CLS** vectors and reported
MATCHES_SOURCE for the real mesh at separation 0.0144. That looked like a
result. The control demolished it: the deliberately mirrored mesh also returned
MATCHES_SOURCE, at a *larger* separation of 0.0155. The criterion was not
detecting mirroring at all, and both numbers were noise.

The cause is that **DINOv2 trains with random horizontal flip as an
augmentation**, so its pooled features are deliberately invariant to handedness.
Asking a flip-invariant embedding which way round something is cannot work, and
no threshold would have rescued it.

The patch tokens are position-indexed, so flipping the image permutes the grid.
Comparing token (i,j) of the render against token (i,j) of the source restores
the sensitivity, and the separations rise by an order of magnitude -- 0.147 and
0.072 against a 0.0144 that meant nothing. Same model, same renders; only the
pooling changed.

This is the second criterion in this file to survive its subject and die to its
control. That is now the standard: **a chirality or axis criterion ships only
with a deliberately-wrong control that it correctly rejects.**

### The colour test that was written and thrown away

The first version of this script found the subject's one saturated lateral
feature -- the rust tail -- and reported which side it fell on. It worked on the
panda. It was deleted anyway, because it needs a hue window per subject and this
pipeline also has to handle a whale, a boat, a shaman and a castle. A criterion
that must be re-tuned for every asset is not a criterion. Tuning per asset
*class* is legitimate; tuning per asset is how a pipeline becomes a demo.

Its one useful contribution was negative: with a wide window it called 15 percent
of the subject "tail" and returned AMBIGUOUS rather than a verdict, which is the
behaviour every criterion in this file should have and most did not.

### A frame error, caught by its own symptom

The first run of the finished script returned UNRESOLVED at separation 0.0045. It
had been handed yaw 178.333 -- the **projector's** convention -- while
`clay_render` speaks the **resolver's**, where this subject's front is yaw 345.
The tell was the absolute score: 0.2249, against the 0.4224 the resolver already
recorded at its own front. Mixing those two conventions is the specific mistake
that produced the previous three wrong answers, and it very nearly produced a
fourth inside the script written to stop it.

## Still open

- Both reviewers warn a second invented face can still appear after the fix: the
  back hemisphere remains unobserved, and multiview diffusion has a strong face
  prior that inserts symmetric face-like detail onto flat fur. Luna suggests
  re-running with several seeds — consistent means geometry or controls,
  intermittent means hallucination.
- The shaman and boat have not been checked at all.
- Re-run the whale after any axis-logic change, as a regression on the case that
  already worked.
