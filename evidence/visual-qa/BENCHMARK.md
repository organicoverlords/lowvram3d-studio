# Tiny visual QA - measured benchmark, 2026-08-02

> **Status: `TINY_VISUAL_MODEL_DISCRIMINATION_NOT_PROVEN`.**
> The tiny VLM cannot currently judge generic repairs. It runs in `advisory_only` mode and has no
> authority to promote anything. The working gate is the deterministic one, below.

Local run on the workstation. Nothing here is projected or estimated.

## Headline: the staff regression is NOT met

Both tiny SmolVLM variants answer **C (insufficient evidence)** on the rejected oversized staff
hole. The required result was **reject** with `VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH` or
`VISUAL_GENERIC_REPAIR`. Per the instruction "do not connect this globally until the staff
regression is proven", the layer is therefore **not wired into any pipeline stage**.

This is a model-capacity limit, not a prompt bug: SmolVLM-256M answers C even for the
`control_unrelated_crop` fixture, where the candidate image is a piece of robe rather than the
staff at all. A judge that cannot tell the staff from a robe cannot be trusted to grade a repair.

## Environment

| item | value |
|---|---|
| environment | `%LOCALAPPDATA%\LowVRAM3DStudio\envs\visualqa` (isolated venv) |
| torch | 2.13.0+cpu |
| transformers | 5.14.1 (uses `AutoModelForImageTextToText`) |
| device | **cpu** - deliberate: the 6 GB GPU stays free for Blender |
| crops | 384 px, `do_image_splitting=False` |

At 512 px with sub-image splitting, three crops exceeded the 20 s budget during prefill alone and
every run returned `VISUAL_TIMEOUT`. The contract permits up to 512 px; 384 px with splitting off
is what actually fits the timeout on CPU.

## Models

| model | disk | cold load | warm inference | peak RSS | peak VRAM |
|---|---|---|---|---|---|
| SmolVLM-256M-Instruct | 516,584,538 B (492.7 MiB) | 8.14-8.45 s | 3.47 / 3.47 / 3.61 s | 1829.1 MiB | n/a (CPU) |
| SmolVLM-500M-Instruct | 1,018,581,548 B (971.4 MiB) | 8.55 s | 4.17 / 4.27 / 4.30 s | not captured | n/a (CPU) |

Output stability across 3 identical 256M runs: **perfectly stable** - same letter (C), same
confidence (0.8035) every time. Greedy decoding plus logit-derived confidence makes the judge
deterministic, which is the one property that did work as designed.

## Fixture results

### SmolVLM-256M

| fixture | decision | confidence | reason |
|---|---|---|---|
| `staff_hole_rejected` (the real negative) | uncertain | 0.8035 | `VISUAL_INSUFFICIENT_EVIDENCE` |
| `control_no_change` | uncertain | 0.7919 | `VISUAL_INSUFFICIENT_EVIDENCE` |
| `control_front_view` | uncertain | 0.7640 | `VISUAL_INSUFFICIENT_EVIDENCE` |
| `control_unrelated_crop` | uncertain | 0.8054 | `VISUAL_INSUFFICIENT_EVIDENCE` |
| `control_missing_image` | **unavailable** | 0.0 | `VISUAL_MODEL_UNAVAILABLE` |

Letter probabilities on the staff negative: `A=0.0156, B=0.1748, C=0.7787`. The model does rank
A lowest - it does not believe the candidate is faithful - but takes C as an escape hatch.

### SmolVLM-500M

| fixture | answer | decision | confidence |
|---|---|---|---|
| `staff_hole_rejected` | C | uncertain | 0.3623 |
| `control_no_change` | A | uncertain (low confidence) | 0.3968 |
| `control_unrelated_crop` | C | uncertain | 0.3674 |

500M is directionally better - it is the only configuration that answers A on the unchanged
control - but confidence never approaches the 0.80 promotion threshold, and the negative is still
not rejected.

## What did work

- **Safety properties hold in every case.** No fixture ever produced a promotable verdict. A
  judge that says "uncertain" cannot promote anything, so a weak model degrades to "preserve the
  baseline" rather than to a wrong accept.
- The missing-image control fails closed (`unavailable`).
- `auto` mode never blocked the pipeline; `run-tiny-visual-qa.ps1` exits 0 on an inconclusive
  optional check.
- No implicit downloads: production runs are pinned offline to a local model directory.
- Deterministic output across repeats.

## The deterministic hybrid gate - this one works

`workers/deterministic_visual_gate.py` + `src/lowvram3d/visual_delta_policy.py`. No model, no
VRAM, no timeout risk; numpy and Pillow only, so it runs in the ordinary pipeline environment.

The key measurement is scale-invariant, which is what makes concept art comparable to a clay
render: the **enclosed-opening fraction** - the equivalent diameter of the largest background
region that cannot reach the image border, divided by the subject's equivalent diameter. The
source art measures **0.3260**; the rejected repair measures **0.4403**.

| fixture | passed | feature_scale_ratio | reason codes |
|---|---|---|---|
| `staff_hole_rejected` | **false** | 1.35053 | `VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH`, `VISUAL_REPAIR_EXCEEDS_SOURCE_BOUNDARY` |
| `control_good_repair` | **true** | 0.98057 | `VISUAL_DELTA_OK` |
| `control_collateral_change` | false | 0.99724 | `VISUAL_COLLATERAL_CHANGE` |
| `control_no_change` | false | 0.62452 | `VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH` |
| `control_unrelated_crop` | false | 0.16442 | `VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH` |
| `control_missing_image` | false | n/a | `VISUAL_INPUT_MISSING` |

Success criteria, all met:

- oversized staff repair **rejected**, with a required reason code;
- identical control **not** falsely rejected for collateral damage (it fails on scale only,
  correctly: the unrepaired baseline genuinely does not match the source feature);
- unrelated crop **rejected**;
- a correctly scaled repair **passes**, so the gate discriminates rather than rejecting everything;
- the collateral fixture fails on `VISUAL_COLLATERAL_CHANGE` alone while its scale ratio is fine
  (0.997), proving the two axes are independent;
- every failure preserves the baseline.

### Two metrics are reported but deliberately not gated

`edge_similarity` and `alignment_confidence` measure ~0 (-0.024, -0.003) even for a correct
repair, because the source crop is textured concept art and the candidate is an untextured clay
render at different framing, scale and rotation. Gating on them rejected all six fixtures. They
are computed and stored, and `DeltaThresholds(gate_edge_similarity=True, gate_alignment=True)`
turns them on - but only a real registration step would make them trustworthy.

### Known limitation

A 2D enclosed-opening measurement cannot distinguish a real through-hole from a dark recess: the
unrepaired baseline still measures an opening fraction of 0.204 because its recess renders dark.
That is acceptable only because openness is already proven in 3D by the ray-cast and backlit
gates in `blender/repair_staff_ring_hole.py`. This gate judges **scale and shape**, not openness.
The two must stay paired.

## Recommended next step, and a cheaper one

1. **Escalate the model.** A ~2B VLM (SmolVLM2-2.2B, Qwen2-VL-2B) is the realistic floor for this
   comparison. On CPU that will breach the 20 s budget, so it needs the CUDA wheel and a VRAM
   budget negotiated against Blender.
2. **Do not rely on a VLM for this particular defect at all.** The rejected repair is detectable
   deterministically, and more reliably: the concept art's staff hole measures ~37% of the outer
   disc diameter, the rejected cut measured 43%, and the measured recess perimeter is 40.6%. An
   aperture-ratio plus circularity check on the existing silhouette mask catches "oversized generic
   donut" with no model, no VRAM and no timeout. The VLM should be reserved for judgements that
   cannot be reduced to a measurement.
