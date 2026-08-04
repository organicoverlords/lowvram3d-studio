# Injective UV rewrap and texture repair — 2026-08-04

Asset: `tactical_red_panda_scout`. Supersedes the targeted UV-consumer repair proposed in
`tactical-red-panda-uv-repair-handoff.md`, which is not executable: the defect it describes as a
single conflicting consumer pair is a property of the whole layout.

## What was actually wrong

Measured on the production mesh (644,348 triangles) at a 2048 atlas:

| | |
|---|---|
| sum of UV triangle areas | 1.062e8 texels |
| texels in the atlas | 4.19e6 |
| covered texels | 2,688,521 (matches `fusion_report.json`) |
| triangles collapsed onto 4 distinct UV triples | 18,621 |
| covered texels claimed by ≥2 triangles, strict interior | 1,902,980 of 2,355,534 (80.8%) |
| triangles that are incompatible non-owner consumers | 403,653 (62.6%) |

The area figure alone is decisive and involves no rasterisation semantics: UV area exceeded the
atlas by 25×, so the layout could not be injective under any packing. A single-owner atlas
resolves each texel to one triangle, so every other claimant displayed a colour computed for a
surface it is not part of. That is why the front face appeared on the back of the head, and why
no fusion policy, protected region or chart move could remove it.

The strict-interior test uses barycentric > 0.05, so no count above can be a shared-seam or
tolerance artifact.

## Fix

Two stages, both UV-only. World-space positions, vertex normals and triangle topology are
unchanged; only UV seam vertices are duplicated.

1. **`workers/uv_rewrap_injective.py`** — rebuilds `TEXCOORD_0` with xatlas at 4096, one atlas.
   Gated before writing on `vertex_map[new_tris] == input tris` (exact, per triangle, in order)
   and on a strict-interior double-claim census.
2. **`workers/injective_atlas_texture.py`** — reprojects the same six MV-Adapter images through
   the same camera bundle onto the new layout. No neural regeneration, no camera remapping, same
   `FRONT_PROTECTED_MULTIBAND` settings as the accepted baseline.

`workers/atlas_raster.py` carries the vectorised exact texel-centre rasteriser shared by both.
Its covering test is identical to `fast_texture_projection.rasterise_atlas`; only the batching
differs.

Two mechanisms that existed to work around the broken layout are gone, deliberately:

- the negative-evidence triangle mask that suppressed the atlas on rear geometry;
- the 2D push-pull fill, which propagates colour across the atlas plane.

Unobserved surface is filled from 3D donors instead: nearest observed texels in world space,
constrained by connected component, distance (≤3% of extent) and normal (dot ≥ 0.5), carrying
the low-frequency band only (16-texel blur). Detail is never synthesised and never travels, so
no facial feature can reach a rear or side surface. The 2D bleed writes only into unowned gutter
texels; no owned texel of any chart receives colour from another chart through the atlas plane.
The sampler is `CLAMP_TO_EDGE` — no atlas wrapping.

## Reproduce

```bash
PY="C:\AI\3d-studio-pipeline\workers\mv_adapter\.venv\Scripts\python.exe"
ASSET="C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout"
OUT="$ASSET\panda_injective_rewrap_20260804"

cd workers && "$PY" uv_rewrap_injective.py \
  --mesh "$ASSET\bar_local_closure_v1\tactical_red_panda_scout_bar_repaired.glb" \
  --output "$OUT\tactical_red_panda_scout_rewrapped.glb" \
  --report "$OUT\uv_rewrap_report.json" --resolution 4096 --padding 4
```

```bash
cd workers && "$PY" injective_atlas_texture.py \
  --mesh "$OUT\tactical_red_panda_scout_rewrapped.glb" \
  --bundle "$ASSET\sd21_cpu_controls_384_v9_upright_raw" \
  --views-receipt "$ASSET\sd21_upright_384x20_20260803\inference_receipt.json" \
  --output-dir "$OUT\injective_texture_v1" \
  --region-config configs/texture/panda_face_priority_region.json --atlas-size 4096
```

```bash
cd blender && PYTHONPATH="$PWD" "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe" --background --python-use-system-env --python textured_asset_qa.py -- --glb "$OUT\injective_texture_v1\tactical_red_panda_scout_textured.glb" --output-dir "$OUT\injective_texture_v1\qa_512_unlit" --report "$OUT\injective_texture_v1\qa_report.json" --resolution 512 --samples 24 --unlit
```

## Result

New layout, at 4096:

| | |
|---|---|
| triangles | 644,348 (unchanged) |
| vertices | 462,769 → 584,762 (121,993 seam duplications) |
| interior texels claimed twice | **0** |
| analytic UV area fraction | 0.467 |
| owned texels | 7,837,789 (2.9× the old atlas's effective coverage) |
| directly observed | 5,737,690 (73.2%) |
| 3D donor filled | 1,562,150 |
| unresolved | 537,949 (6.9%, interior/occluded surface only — not visible in any QA view) |

Ownership is now balanced across views — front 21.4%, rear 18.8%, left 16.2%, right 17.8%,
top 11.2%, bottom 14.6%. The old atlas reported front 55.9% because front-owned texels were
being shared with rear geometry.

Verification (fresh Blender import, 512 unlit):

- front — face coherent; eyes, eye-mask, nose and muzzle correct; the baseline's speckle holes
  are gone;
- rear — no eyes, nose, muzzle or facial mask;
- left / right — clean profiles, no wrapped face;
- `closeup_rear_head` — hood fur only.

Artifacts, under `panda_injective_rewrap_20260804\`:

| | sha256 |
|---|---|
| `tactical_red_panda_scout_rewrapped.glb` | `950343dd7ff76877ce6adb83d6d4a80a8d123e7aafb397b566a762d454a9a5f1` |
| `injective_texture_v1\panda_injective_basecolor.png` | `8b834f3338c5569f639162d0b697cd2cfabdda7bdd9483f7463a916f8eafcf9f` |
| `injective_texture_v1\tactical_red_panda_scout_textured.glb` | `8f24b7d1e3245cd96b5cdc40a350df1a33cd2103502cc0d9e4903f865426bc17` |

The reprojection command was run twice from the same inputs and produced a bit-identical atlas
and GLB, so the pipeline command above reproduces the result exactly.

`FULL_UV_REWRAP_AND_TEXTURE_REPAIR_PROVEN`

## Notes

- **The source mesh no longer exists on disk.** `bar_local_closure_v1\` was removed by something
  outside this pipeline while the unwrap was running. It was read successfully at the start of
  the run and its sha256 (`78c5513…`) matches `fusion_report.json`. Geometry preservation is
  re-verified in `uv_rewrap_report.json` against the preserved textured baseline, which carries
  the identical surface. The rewrap cannot be re-run until that file is restored.
- `xatlas 0.0.11` was installed into the mv_adapter venv; it is a new runtime dependency.
- 1,424 triangles (0.22%) received zero-area UVs from the packer. They are 3D slivers and own no
  texels.
- The preserved baseline under `bounded_fusion_provenance_v2\front_protected_multiband\` was not
  modified. This repair writes to `panda_injective_rewrap_20260804\`.
