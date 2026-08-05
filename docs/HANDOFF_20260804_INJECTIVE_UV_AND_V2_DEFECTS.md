# Handoff — injective UV integration and twelve Pipeline V2 defects

Date: 2026-08-04
Branch: `integration/unified-pipeline-v2-20260802`
Commits: `d10e116` (rewrap + reprojection), `ed2776c` (canonical integration), `7727153` (defect fixes)

## Delivered

**1. Injective UV rewrap and six-view reprojection**, proven on `tactical_red_panda_scout`.
See `UV_REWRAP_TEXTURE_REPAIR_20260804.md` for the full measurement. Summary: the previous atlas
was non-injective *by area* — 1.062e8 texels of UV triangle area against a 4.19e6 texel atlas at
2048 — so no fusion policy could have fixed the front face appearing on the back of the head.

**2. Both workers integrated as canonical V2 stage routes** — `uv.route = "injective"` and
`texture.route = "mvadapter_sixview"`. Algorithms unchanged. See the stage-route table in
`PIPELINE_V2.md`.

**3. Canonical UV master designated** —
`configs/uv/tactical_red_panda_scout_canonical_uv_master.json`, carrying every proof hash. The UV
stage adopts it after checking sha256, geometry fingerprint and injectivity; it does not re-unwrap.

**4. Supported-command reproduction passed**, bit-identical:

| artifact | sha256 |
|---|---|
| `tactical_red_panda_scout_rewrapped.glb` | `950343dd7ff76877ce6adb83d6d4a80a8d123e7aafb397b566a762d454a9a5f1` |
| `..._basecolor.png` | `8b834f3338c5569f639162d0b697cd2cfabdda7bdd9483f7463a916f8eafcf9f` |
| `..._textured.glb` | `8f24b7d1e3245cd96b5cdc40a350df1a33cd2103502cc0d9e4903f865426bc17` |

Stage results: `LOD passed, UV passed, BAKE not_applicable, TEXTURE passed, TEXTURE_QA
not_applicable, EXPORT_QA passed` (13/13 fresh-import checks).

**5. `xatlas==0.0.11` pinned** in `requirements-control.txt`. Exact, because chart decomposition
and pack order are what make a rewrap reproducible and neither is covered by semver.

**6. 19.08 GB reclaimed** — duplicate `.bin`/`.ckpt` encodings removed only after each
`.safetensors` survivor was fully materialised tensor-by-tensor off disk. No model lost. Disk
25 GB → 44 GB free. Manifest of every deleted path was written at delete time.

## Performance, which is the thing to understand before judging this

| path | UV | TEXTURE |
|---|---|---|
| canonical, adopting a UV master | ~30 s verify | **80 s** (observe 58.7, fuse 6.1, donor 13.2, write 2.1) |
| cold start, unwrapping from scratch | **25–40 min per attempt** at 800k faces | not reached |

Production throughput comes from the canonical-master path. The unwrap is a **once per asset,
ever** cost, which is exactly why designating a canonical UV master matters. Hardware here is a
GTX 1660 SUPER (6 GB) with ~3 GB RAM free, driving 1.59M-face meshes at 4096 atlases.

## Twelve defects

Found by running the canonical stage range from `INGEST` on a real source image for the first
time. Ten were pre-existing; two were mine, introduced earlier on this branch.

| # | Defect | Evidence |
|---|---|---|
| 1 | `GENERATE` used the pipeline's interpreter; Mini Turbo needs its own + `hy3dgen` | `No module named 'hy3dgen'` |
| 2 | `Pipeline.run` replaced `PYTHONPATH` on `env_extra`, taking repo workers off the path | code |
| 3 | stance repair ran on any humanoid asset, not only riggable ones | `FEET_TOO_CLOSE_FOR_RIGGING` on a declared-unrigged asset |
| 4 | `GENERATE` pinned to the ladder's bottom rung; gates reported a hardcoded `256` regardless | 705k → **1.59M** faces after fix |
| 5 | no way to declare a legitimately connected pose | repair could only open a gap by deleting 35k faces of garment |
| 6 | LOD topology gate emitted no failure code | `failure_codes=[]` → straight to `needs_human` |
| 7 | **post-LOD debris repair non-convergent by construction** | support 0.375→0.125, 0.5→0.0, 1.0→0.0 on *identical* components |
| 8 | *(mine)* injective UV route fabricated codes from an absent report | `UV_GEOMETRY_CHANGED`+`UV_OVERLAP` from an empty file |
| 9 | UV stage ignored repair `overrides["route"]` | attempts 0 and 1 identical codes with override applied between |
| 10 | *(mine)* `uv_rewrap_injective` duplicates `uv_xatlas_repair`, worse | padding 4→8 halved collisions 6→3, cost 40% utilisation |
| 11 | `uv_xatlas_repair` default budget is 3 attempts × 4 rounds × 1200 s | unbounded in spirit, inside a pipeline whose principle is hard budgets |
| 12 | **`uv_xatlas_isolated` could never report success** | AST-confirmed unreachable code; `UnboundLocalError` reported as `failed` |

### The two that matter most

**#7 — a repair graded in a frame its own deletions moved.** `component_support` scores a
component by projecting it into a frame derived from percentiles of the mesh's own positions. So
deleting debris shifts the frame and re-scores every survivor. The cleaner removed 19 components,
and three it had just scored *supported* re-scored as *unsupported* by the gate that verifies it.
No retry could ever converge, and `--debris-blocking` made it terminal. Fixed by
`--support-reference-mesh`, so the gate grades in the pre-cleanup frame. Verified against the stuck
artifact: 3 shards → 0, `preserved_small` 96 → 99.

**#12 — a route that could never succeed.** In `uv_xatlas_isolated.run_one` the timeout `return`
sat in the outer `try` body rather than the `except` handler. It therefore ran unconditionally, the
two statements reading the child's real status were unreachable, and on the success path
`child_report` was unbound — an `UnboundLocalError` the outer handler reported as `failed`. The
documented `xatlas` UV route, and the `UV_OVERLAP` recipe that selects it, have never worked. **#9
hid it**, because the recipe could never fire to expose it.

Three of V2's repair paths were non-functional in ways no green build would ever have shown.

## Not delivered: the from-source asset

`INGEST → LOD` all passed. Blocked at `UV`.

- `GENERATE` succeeded at the **top** ladder rung (octree 384, 3000 chunks, 5 steps), first attempt.
- `CLEAN` kept all 1,591,614 faces; stance repair skipped by declaration.
- `LOD` produced 4 rungs (800k/400k/150k/50k) + micro-clean. LOD0 topology *improved*
  (boundary 63→60, non-manifold 109→85); LOD1–3 regressions recorded as advisory.
- `UV` failed: `UV_REWRAP_NOT_INJECTIVE:6` at padding 4, `:3` at padding 8. xatlas guarantees chart
  *bitmap* separation at its internal resolution, which can disagree with exact texel-centre
  geometry in a handful of spots. Padding is the wrong lever — it halves collisions and costs 40%
  of the atlas.

Output root: `...\tactical_red_panda_scout\pipeline_v2_from_source_20260804_full_ladder\`
Manifest: `configs/pipeline/tactical_red_panda_scout_from_source_v2.json`

### To resume

The correct fix is **not** another declared tolerance. `uv_xatlas_repair.py` already solves both
failure modes — it shrinks only the colliding charts toward their own centroid until the collision
clears (shrinking can never create a new overlap), and drops near-zero-area faces before re-unwrapping.
The `injective` route should **delegate its unwrap to that worker** and keep `atlas_raster.injectivity()`
purely as the acceptance gate, which is what it is genuinely better at — it is a stronger test than
the positive-area detector and it is what caught the panda's 80.8% catastrophe.

Note the one real difference: `uv_xatlas_repair` drops faces, changing topology. That is correct for
a fresh asset and **wrong for a canonical UV master**, where triangle-for-triangle preservation is
the contract. Keep the strict path for masters.

If wall-clock matters more than fidelity for this experiment: LOD0 at 400k instead of 800k roughly
halves both UV and BAKE, at the cost of the fringe tearing that is already declared advisory. `BAKE`
at 4096 / 48 samples from 1.59M→800k was still ahead and would likely have been the longest stage.

## Environment facts worth keeping

- Mini Turbo needs `C:\AI\HY3D2\python_standalone\python.exe` with `C:\AI\HY3D2\Hunyuan3D-2` on
  `PYTHONPATH`. The mv_adapter venv has no `hy3dgen`.
- There are **two parallel HuggingFace trees** and each is incomplete alone. The 3.82 GB Mini Turbo
  DiT weights live under `HuggingFaceHub\hub\models--tencent--Hunyuan3D-2mini\snapshots\f90a0f7d…\`;
  the direct model root had only a `config.yaml` for the DiT. They are now hardlinked together. If
  `MINI_TURBO_FAILED error=Mini Turbo weights not found` reappears, look in the `hub\` tree before
  re-downloading.

## Open

- The original source mesh `bar_local_closure_v1\tactical_red_panda_scout_bar_repaired.glb`
  (sha `78c5513…`) was removed from disk by something outside this pipeline during the rewrap run.
  The rewrap cannot be bit-reproduced until it is restored; the canonical UV master is the artifact
  of record in its place, and its geometry is verified against the preserved textured baseline.
- `uv_rewrap_injective.py` duplicates `uv_xatlas_repair.py` (#10). Consider collapsing them per the
  resume note above.
- Stage-level budget caps for `uv_xatlas_repair` (#11) are passed by hand, not yet wired into the
  stage. They belong where V2 states its retry policy, not in a worker's argparse defaults.
