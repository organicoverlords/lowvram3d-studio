# Panda textured asset — 2026-08-03

Classification: **`PANDA_TEXTURED_ASSET_VISUAL_REPAIR_REQUIRED`**

Not `PROVEN`: the front view of the finished GLB does not show a recognisable panda face,
which is texture gate 1. Not a hard blocker either — the defect is isolated to the atlas
fusion stage, and every stage before it now passes.

## The coordinate defect, found and fixed

`build_mvadapter_cpu_controls` mapped mesh **Z** to the rig's up axis. This asset is
standard glTF: up is mesh **+Y**, forward is mesh **+Z**.

| evidence | value |
| --- | --- |
| longest mesh extent | Y, 1.952 (X 1.668, Z 1.522) |
| first principal axis | (0.298, −0.955, 0.013) |
| orange tail centroid | mesh (0.557, −0.572, −0.118), 22% along Y |
| axis probe, up=+Y | camera on +Z renders the face, hood, scarf, rifle |

The rig therefore orbited the wrong axis: every view rendered the character on its side, and
the camera actually facing the head was handed the elevation ±89.99 embedding, so it
produced a top-down image instead of a face. That single error explains the sideways
framing, the missing face and the earlier inability to resolve the facing axis.

Fixed by adding `canonical_basis=y_up_z_front` (mesh +Y → rig up, mesh +Z → azimuth 0).
The legacy basis remains the default so no earlier receipt or test changes.

## Anchored camera permutation

Anchored to the render the user named the visual front — camera on +Z, up +Y — which the
rebuilt rig produces as raw1.

```
raw_to_semantic = {0: left, 1: front, 2: right, 3: rear, 4: top, 5: bottom}
```

Opposition dots: front·rear −1.0, left·right −1.0, top·bottom −0.99999994. raw2 sees the
tail lobe, matching the tail centroid at mesh +X. Raw order and control arrays untouched;
the previous mapping is retained as superseded.

## Foot artifacts

`TEXTURE_ONLY_ARTIFACT` — 0 detached-geometry pixels, 8018 dark-atlas pixels on sound
surface. No mesh repair warranted; the marks lived in the legacy atlas that the fusion
replaces, and they are absent from the new one.

## 384x20 on the corrected rig

`PROVEN`. 20/20 steps, finite gate passed, structural gate passed, colour and semantic gates
passed. Per-view silhouette IoU: left 0.888, front 0.980, right 0.976, rear 0.982, top
0.966, bottom 0.962. Front/rear correlation 0.259 direct, 0.562 mirrored.

**The front view shows the face; the rear view does not.** That is the duplicated-face
requirement satisfied at the source.

## Multiview fusion

Confidence-gated per texel: foreground-mask, depth, occlusion, bounds and back-facing gates,
then a confidence product of viewing angle, depth agreement, distance from the silhouette
boundary and semantic reliability, blended only among colour-compatible observations.

| coverage | value |
| --- | --- |
| directly observed | 72.30% |
| multiview observed (2+ views) | 40.07% |
| blended from multiple views | 30.81% |
| synthesized (push-pull fill) | 27.70% |
| unresolved | 0.00% |

Synthesized area is reported separately and is **not** folded into the observed figure.
Geometry, UVs and index buffers are byte-identical before and after texture binding.

## Gate results on the finished GLB

Passing: upright orientation in all eight views; no protruding bar or tip blob; rifle
preserved; tail colour and shape coherent; backpack and webbing recognisable; left/right
consistent; top and bottom correspond to real upper and lower surfaces; no background colour
on geometry; no checkerboard; no black outputs; no chart-edge darkening; fresh import
resolves the 2048x2048 base colour.

Failing: **no recognisable panda face in the front view**. The face reads as a flat cream
muzzle without eye or nose markings, despite the generated front view containing a clean
face. Colour distribution is also darker and muddier than the orange/black/cream reference.

## Why it fails and what to do next

The face is observed — the front view contributes 446,919 valid texels — so this is a
fusion-weighting problem, not a coverage problem. Bilinear sampling and cubed confidence
(v4) removed the texel-scale mosaic that v3 had, but did not restore facial contrast: the
head is also seen at grazing angles by left, right and top, and those washed-out
observations survive the 60-unit colour-compatibility test and average the small dark eye
and nose markings away.

Next: make high-confidence regions winner-take-all instead of blended — when the leading
view's confidence exceeds the runner-up by a clear margin, take it alone, and reserve
blending for genuinely comparable observations. Re-run fusion only; no new GPU pass is
needed, because the six source views are already proven.

## Artifacts

* textured GLB: `…\panda_multiview_texture_v4\tactical_red_panda_scout_textured.glb`
* atlas: `…\panda_multiview_texture_v4\panda_multiview_basecolor.png` (2048x2048)
* six source views: `…\sd21_upright_384x20_20260803\view_{0..5}_{semantic}.png`
* eight-view QA: `…\panda_multiview_texture_v4\qa\`
* repaired mesh: `…\bar_local_closure_v1\tactical_red_panda_scout_bar_repaired.glb`
  (sha256 `78c55133165e931bc8d6765610a679d1d18badcdc178820a69e31b7b32bcbfb8`)
