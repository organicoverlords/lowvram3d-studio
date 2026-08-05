# Panda control-geometry bar — separated classifications (2026-08-03)

`PANDA_CONTROL_GEOMETRY_CLEAN` was a claim about one rendered control bundle. It was being
read as a claim about the production mesh. The two are now tracked separately. The earlier
audit under `20260803-panda-control-bar-audit` is retained unchanged apart from its
classification wording.

| Classification | State | Evidence |
| --- | --- | --- |
| `PANDA_CONTROL_BAR_LOCALIZATION_PROVEN` | holds | `20260803-panda-control-bar-audit/panda_control_geometry_bar_audit.json` |
| `PANDA_REPAIRED_CONTROL_BUNDLE_BAR_ABSENT` | holds | same audit, `repaired_control_bundle_comparison` |
| `PANDA_PRODUCTION_MESH_REPAIR_NOT_PROVEN` | **superseded** | replaced by `PANDA_PRODUCTION_MESH_BAR_REPAIR_PROVEN` below |
| `PANDA_OLD_MV_OUTPUTS_UNPROMOTED` | holds | the 384x20 sequence and `spike_repair_v3` remain unpromoted |

## Results of this task

| Phase | Classification |
| --- | --- |
| 2–4 production mesh repair | `PANDA_PRODUCTION_MESH_BAR_REPAIR_PROVEN` |
| 5 regenerated controls | produced at 384, raw-index bundle `sd21_cpu_controls_384_v8_bar_repaired_raw` |
| 6 camera semantics | `PANDA_CAMERA_SEMANTICS_USER_REVIEW_REQUIRED` |
| 7 bounded MV-Adapter validation | **not run** — phase 6 is a stated precondition |

## Why the repair removes 64 faces and not 14

The bar is a 14-triangle fan. Its apex (vertex 91337, the mesh's minimum-x point) is used by
no other face, and every one of its rim edges is shared with a 50-triangle shell at the
opposite end of the mesh. That shell has no other connection to the body: the fan was its
only anchor. Removing only the 14 fan faces therefore raises the connected-component count
from 1 to 2, which the required structural gates forbid.

Two candidates were built and gated:

* **Candidate A** — remove 14, re-triangulate the exposed 14-gon with 12 faces.
  Passes every gate except `components_not_increased` (1 → 2). Retained as
  `rejected_candidate_a_closure_only_report.json`.
* **Candidate B (accepted)** — remove the 14 fan faces plus the 50-face orphaned shell.
  No faces are synthesised, no vertex, normal or UV changes, and every structural gate
  passes. The shell's own footprint in the proven control renders is at most 17 pixels and
  is occluded in every view, so nothing visible is lost with it.

The earlier `spike_repair_v3` candidate removed 79 faces across three chained passes without
identifying this fragment; the accepted repair removes 64 in one pass with a stated reason
for each.
