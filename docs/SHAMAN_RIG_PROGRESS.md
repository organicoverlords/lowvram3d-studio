# Shaman Rig / Animation Progress

Branch: `magicmusic/parts-pose-materials-20260802`

Canonical geometry source (immutable, never written by this work):

```
C:\AI\LowVRAM3D-benchmarks\pipeline-v2-validation\shaman_v2_validation\state\CLEAN\proven\shaman_v2_validation_stance_clean.glb
SHA256 2f712b49a88a39cb10fb08e6bfb08becef2025b153ce87103e86fde97dfb8c80
```

Generated artifacts run root:

```
C:\AI\LowVRAM3D-benchmarks\shaman-rig-animation\run-20260802-123123\
```

Rejected evidence preserved, never promoted:

```
...\state\TEXTURE\proven\shaman_v2_validation_textured_lod0.glb   (rejected visual baseline)
previous staff-hole cutter result                                  (rejected; approach not reused)
```

Status vocabulary: `PROVEN`, `REJECTED`, `NOT PROVEN`, `BLOCKED`.

## System status

| SYSTEM | STATUS | PROOF | NEXT ACTION | BLOCKER |
| --- | --- | --- | --- | --- |
| SOURCE_IDENTITY | PROVEN | `proof/shaman-rig/latest/source-audit.json`, SHA256 match | none | none |
| SOURCE_AUDIT | PROVEN | `source-audit.json`, `shaman_source_contact_sheet.png` | none | none |
| SOURCE_PREVIEW | PROVEN | 5 workbench renders + contact sheet | none | none |
| RIG_BASE_TOPOLOGY | NOT PROVEN | weld analysis in audit (537041 verts, 1 dominant component) | build and gate the welded rig base | none |
| PARTS (semantic regions) | NOT PROVEN | — | deterministic region worker | none |
| STAFF_POLICY | NOT PROVEN | staff candidate detected at x ≈ -0.42 | fused_staff_control implementation | none |
| POSE_PREP (A-pose) | NOT PROVEN | — | Blender-side A-pose proof with gates | none |
| RIG_READINESS | NOT PROVEN | audit verdict `humanoid_enough=true` | re-derive on rig base | none |
| RIG (skeleton + controls) | NOT PROVEN | — | build modular game skeleton | none |
| FINGER_TIER | NOT PROVEN | audit recommends TIER 1, capped at TIER 1 | re-derive from segmented hands | none |
| SKIN / SKIN_QA | NOT PROVEN | — | automatic weights + deformation QA | none |
| MOTION / MOTION_QA | NOT PROVEN | — | 4 clips + loop validation | none |
| EFFECTS | NOT PROVEN | — | socket manifest + preview collection | none |
| EXPORT / EXPORT_QA | NOT PROVEN | — | GLB + FBX + fresh import | none |
| TEXTURE | REJECTED | `shaman_v2_validation_textured_lod0.glb` is a rejected visual baseline | keep out of production manifest | texture CUDA lane |
| TEXTURE_CUDA_LANE | BLOCKED | `cond-fp32-20260802-121425` | instrumented CUDA checkpoint diagnostic | CUDA illegal memory access mid-denoise |

## Milestone 0 findings

Counts (canonical source, as imported by Blender 5.2):

| Metric | Value |
| --- | --- |
| Objects | 1 |
| Mesh objects | 1 (`geometry_0`) |
| Vertices (as stored) | 3,258,048 |
| Triangles | 1,086,965 |
| Materials | 0 |
| Armatures / bones | 0 |
| Actions | 0 |
| Vertex groups | 0 |
| Shape keys | 0 |
| Dimensions (X,Y,Z) | 1.6502 x 0.5952 x 1.9809 |
| Up axis | Z |

### The source is corner-split

`vertices / triangles = 3.00`. The GLB stores a separate vertex per triangle
corner, so almost no edges are shared. This is the single most important audit
finding: **heat-diffusion skinning cannot propagate across a corner-split mesh**,
because there is effectively no connectivity to diffuse along.

Welding at 1e-4 recovers real topology:

| Metric | Value |
| --- | --- |
| Welded vertices | 537,041 |
| Welded edges | 1,611,294 |
| Welded faces | 1,074,083 |
| Connected components | 60 |
| Largest component | 531,389 verts (98.95%) |
| Components > 1000 verts | 1 |
| Boundary edges | 348 |
| Non-manifold edges | 13 |

Consequence for later milestones: rigging and skinning run on a **welded rig
base**, not on the raw corner-split buffer. That transformation is reported
explicitly and gated; it is not silently folded into "canonical topology
unchanged". Topology invariance is enforced from the rig base onward.

### Geometry reality (from the rendered contact sheet)

The rendered proof corrected two heuristics that the first audit pass got wrong:

- The model's **1.65 full width is not arm span**. It is a horizontal antler bar
  at head height carrying hanging ornaments. Raw per-slice width is therefore
  meaningless for landmarks; the audit now isolates a body core by density
  (`torso_core.width = 1.2657` vs `head_band_raw_width = 1.6086` against a
  `head_core.width = 0.7205`).
- The **arms hang down at the sides**, they are not outstretched. The first pass
  read the accessory span as arm span and wrongly blocked automatic rigging.

Current geometry summary: bird/corvid shaman, beaked head, antler bar with
pendant ornaments, layered ragged cloth cape and skirt, visible separated
fingers, staff held on the -X side, feet planted.

### Landmarks and limbs

| Item | Value | Confidence |
| --- | --- | --- |
| Model height | 1.9809 | high |
| Shoulder candidate | height fraction 0.703 | 0.6 |
| Hip candidate | height fraction 0.469 | 0.6 |
| Neck candidate | height fraction 0.984 (contaminated by antlers) | low |
| Left lateral limb lobe | x ∈ [-0.579, -0.386], height fraction 0.280–0.520 | detected |
| Right lateral limb lobe | x ∈ [0.424, 0.786], height fraction 0.281–0.520 | detected |
| Left hand distal region | height fraction 0.331, 22,471 verts | detected |
| Right hand distal region | height fraction 0.336, 14,909 verts | detected |
| Staff candidate | narrow band x ≈ -0.46..-0.39, z coverage 0.93 | 0.5 |

### Finger capability

Audit recommends **TIER 1 (grouped curl)** and hard-caps itself at TIER 1.
Distal density peaks (4 left, 10 right) are noisy — the right-hand count is
inflated by cloth fringe, not fingers. TIER 2 is not claimable from an audit
heuristic; it requires per-finger vertex segmentation proven at the RIG stage.

### Automatic rigging verdict

`humanoid_enough_for_automatic_rigging = true`, no blocking reasons.

Advisory: `ARMS_DOWN_AT_SIDES_ELEVATES_ARM_TORSO_WEIGHT_BLEED_RISK` — arms
resting against the cape mean automatic weights are likely to bleed between arm
and torso, which SKIN_QA must catch rather than assume away.

## Milestone log

| Milestone | Commit | Result |
| --- | --- | --- |
| 0 — audit and truth packet | (this commit) | PROVEN |
