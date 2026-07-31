# Architecture

## Control plane

3D Gen Studio owns projects, graph/Kanban state, source assets, generated versions, previews, downloads, and MCP automation. The local worker returns binary GLBs; Studio stores them as assets or versions.

```text
3D Gen Studio :8311
  ├─ class-specific one-click providers
  ├─ class-specific post-process providers
  └─ MCP / asset versioning
        ↓
LowVRAM FastAPI :8400
  ├─ existing ComfyUI Mini Turbo :8188
  ├─ 3D Gen Studio CPU mesh tools :8200
  ├─ Blender 5.x headless workers
  ├─ isolated BiRefNet + MediaPipe avatar preprocessing
  ├─ isolated MV-Adapter SD2.1
  └─ isolated TripoSR emergency fallback
```

Interactive browser-only projection tools are not required by automation. Blender performs deterministic projection and baking.

## Geometry and post-processing

The one-click image endpoint preserves Mini Turbo’s raw high-poly GLB and sends it through the same post-process engine used for imported GLB/OBJ/FBX assets. It does not use the older shallow generate → texture → rig route.

Stages:

1. ingest and clean import validation;
2. analysis of objects, materials, loose islands, symmetry, repeated topology, likely round parts;
3. class-aware geometric split;
4. guarded Studio retopo or Blender object-aware optimization;
5. guarded Studio UV or Blender class-aware unwrap;
6. selected-to-active basecolor, normal, AO, roughness, and metallic transfer;
7. optional humanoid, creature, or rigid mechanical template rig;
8. collision, sockets, LODs, previews, GLB/FBX export, clean re-import validation.

## Human avatar path

The avatar provider treats the photograph as identity evidence, not merely as a generic text-to-3D reference. A pinned BiRefNet worker produces a soft alpha; MediaPipe supplies 33 body landmarks and a bounded segmentation prior. Edge refinement, fringe-colour decontamination, largest-subject selection, and square normalization run before Mini Turbo. The transformed landmarks are retained for Blender rig fitting.

The source-facing projection receives higher weight than generated front colour. Additional views remain synthesized because one photograph does not contain hidden-surface truth. The final packaging validator reopens the GLB and requires a humanoid armature plus a dance action for `animated_human_avatar` exports.


## Guarded Studio mesh tools

The Studio service receives one Trimesh and concatenates a multi-mesh GLB. Therefore:

- use Studio Auto Retopo/UV only for a verified single-part result;
- use Blender for multipart hierarchy, character equipment, vehicles with moving parts, rooms, scenes, and levels;
- preserve explicit failed Studio receipts before the Blender fallback.

## Scenes and levels

Scenes and levels do not share a single destructive atlas or global 45k budget. They preserve object/material hierarchy, simplify per object, add lightmap UVs, and export cell manifests. Level cells are intended as import metadata for engine-side World Partition or equivalent streaming systems; the pipeline does not pretend to create an entire playable game level from one image.
