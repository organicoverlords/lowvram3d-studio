# Two pipelines, two different bets

This project kept oscillating between two incompatible goals under one name.
Splitting them makes each one honest, and makes it obvious which is failing.

| | **A — Photometric** | **B — Structural** |
|---|---|---|
| Bet | Reproduce *this photograph* in 3D | Build a *real scene* the photo describes |
| Geometry | Recovered from the image (MoGe) | Authored: primitives, kitbash, library assets |
| Appearance | The source image itself | Materials from the project's library |
| Best view | The source viewpoint, near-perfectly | Any viewpoint, equally |
| Fails when | The camera moves off-axis | You compare it to the source image |
| Navigable | No — no collision, no backsides | Yes |
| Honest claim | "a photo you can lean around" | "a scene inspired by a photo" |

Neither is better. They answer different questions, and the mistake this project
made repeatedly was grading one with the other's test — declaring a photometric
result `PROVEN` because it matched the source image, which it does *by
construction*, and then being surprised it looked like a flat picture.

---

## A — Photometric

```
image ─► MoGe-2 ─► point map + mask + FOV
                     ├─► mesh   (depth-edge culled, UV textured)  ─► Unreal StaticMesh
                     └─► splats (INRIA 3DGS PLY)                  ─► viewer / UE plugin
```

Two back ends over one reconstruction, because the artefacts differ:

**Mesh** carries connectivity, which is what breaks. Triangles spanning a depth
discontinuity stretch foreground into background; culling them (`edge_rtol`)
removes the smears and leaves holes. Cheap, renders anywhere, imports as an
ordinary static mesh, can take collision.

**Splats** have no connectivity, so the smear/tear artefact cannot occur.
Unobserved regions are simply absent. Higher fidelity off-axis, but needs a
plugin in Unreal and cannot take collision.

Known ceiling: a single view cannot observe what it cannot see. Backsides and
occluded regions are missing in both back ends. Closing that gap requires
adding information — depth-aware inpainting, or diffusion-based hallucination —
not tuning.

## B — Structural

```
image ─► semantic analysis ─► regions, instances, support relationships
                                     │
                            asset strategy (reuse ▸ procedural ▸ generated)
                                     │
                            layer builders ─► terrain, architecture, vegetation, water
                                     │
                            gameplay: collision, navmesh, player start
```

Every actor is real geometry with real materials, so it holds up from any angle
and can be walked through. The cost is that it will never match the source
image pixel-wise, and grading it that way is a category error.

Its current failure is upstream: semantic analysis is a stub, so every region
collapses to one `visual_shell` and the builders emit scaled cubes. That is why
the generated castlegrounds scene is a field of white boxes.

---

## Shared evaluation contract

Both pipelines emit `pipeline_result_v1` so they can be compared without
pretending they are the same thing. Each is graded on its **own** test:

| Metric | A | B |
|---|---|---|
| `source_view_similarity` | primary | not applicable |
| `offaxis_stability` | primary | primary |
| `navigable_fraction` | not applicable | primary |
| `actor_semantic_variety` | not applicable | primary |
| `unobserved_fraction` | primary | not applicable |

`offaxis_stability` is the one metric both must satisfy, and the only one that
would have caught the flat-shell result early.

---

## Current state

| | A/mesh | A/splats | B |
|---|---|---|---|
| Reconstruction | working | working | stub semantics |
| Unreal import | working | needs plugin | working |
| Appearance | UV texture | SH DC colour | placeholder materials |
| Verified | render + score | z-buffer validator | actor manifest only |

Next per pipeline:

- **A/mesh** — settle the glTF import axis mapping by measurement; sky backdrop.
- **A/splats** — install a UE splat plugin; anisotropic extents from local normals.
- **B** — replace stub semantics with real segmentation; this is the only change
  that stops the builders emitting cubes.
