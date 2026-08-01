# Shaman next-stage manifest (prepared, not executed)

Prepared at the end of the Stage 6 texture sprint. Nothing here has been run. Inputs are the
Stage 6 outputs; the canonical `game/shaman_lod0.glb` and `high/shaman_high_master.glb` remain
untouched and must stay that way.

## 0. Inputs this stage inherits

| Artifact | Path |
| --- | --- |
| Textured LOD0 | `final-pipeline/textured/shaman_textured_lod0.glb` |
| UV mesh (no material) | `final-pipeline/game/shaman_lod0_uv.glb` |
| Component identity | `final-pipeline/textures/shaman_material_id_4k.png` (62 welded-position components) |
| Projection tiers | `final-pipeline/reports/texture_projection_metrics.json` |

The material-ID map is the key asset for every step below: it already partitions the mesh into the
62 welded-position components, so part separation is a lookup rather than a fresh segmentation.

## 1. Separate the staff as a standalone object

- Identify the staff component from the material-ID partition: it is the single component with the
  largest vertical extent-to-width ratio that does not intersect the head bounding box.
- Confirm against the source silhouette before splitting - the staff is held *next to* the body and
  must not be fused to the hand in the mesh. Stage 1 recorded 62 components; if the staff shares a
  component with the body, this becomes a cut operation, not a split, and needs its own review.
- Export as a sibling object in the same GLB, not a separate file, so the texture atlas is shared.
- Risk: cutting introduces a new boundary that the current atlas does not have a seam for. Cut
  along an existing UV chart boundary where possible.

## 2. Identify rigid hanging ornaments

Candidates, from the source and confirmed present in geometry: bowl charm, leaf pendants, hollow
pod, lantern, far-right pendant, staff ring and staff charm.

- Select components whose bounding box sits below the antler bar and above the robe hem, with a
  small volume relative to the body.
- Each rigid ornament gets exactly one bone. They do not deform.

## 3. Identify cords and fringe needing secondary bones

- Cords: high aspect ratio (length / max cross-section > 8), suspended from the antler bar.
- Robe fringe and fabric strips: thin, hanging from the hem, high in count.
- These need 2-3 bone chains each, not single bones, or they will read as rigid sticks in motion.
- Expect this to be the largest bone-count contributor. Cap it deliberately rather than letting one
  bone per strip explode the skeleton.

## 4. Side-depth and rig-readiness assessment

- Measured depth extent is 0.5919 against a height of 1.9664 - the subject is markedly flat
  front-to-back. Verify the arms are genuinely separated from the torso before skinning; if they
  are fused, weight painting cannot recover the separation.
- Confirm the legs are distinct components. Stage 1's component list should be re-read for this.
- Report any component that is thinner than two voxels of the generation resolution, since those
  will collapse under deformation.

## 5. A-pose conversion

- The source pose is already close to a relaxed A-pose with arms down. The conversion is small:
  rotate the upper arms outward to a consistent angle and straighten the wrists.
- Do this on a copy. Preserve `shaman_textured_lod0.glb` as the bind-pose reference.
- The staff must move with the hand if it was separated in step 1; decide parenting before posing.

## 6. Body rig and secondary-motion bones

- Body: pelvis, spine chain, neck, head, two arm chains, two leg chains.
- Head carries the antler bar; the antler bar carries the cords; each cord carries its ornament.
- Secondary motion is the point of this asset - the hanging elements are its silhouette. Budget
  bones there rather than in the torso.

## Constraints carried forward

- Do not modify `game/shaman_lod0.glb` or `high/shaman_high_master.glb`.
- Do not re-unwrap or alter UVs; the exact overlap gate result is tied to this exact UV set.
- Do not use `shaman.fbx` or its textures.
- Do not regenerate geometry.
- PR #3 stays draft and unmerged; its base conflict is out of scope.
