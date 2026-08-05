# Codex continuation — Shaman V3 and multi-asset pipeline validation

## Repository identity

- Repository: `organicoverlords/lowvram3d-studio`
- Expected worktree: `C:\Users\Lauri\Desktop\lowvram3d-magicmusic-asset-systems`
- Branch: `magicmusic/parts-pose-materials-20260802`
- Last verified Claude head before this handoff: `3b7381f8e1168507533414a8b5bb6ec1793ce015`
- Never work on `main`.
- Never merge or force-push.
- Before edits, verify origin, branch, local HEAD, remote HEAD, and clean worktree.

## User goal

Continue Claude's work, but stop optimizing only for the shaman. Keep the shaman as a mandatory regression case and begin validating the same pipeline on three substantially different assets:

1. **Lucky Drown casino boat** — large architectural/vehicle asset and the user's chosen boat benchmark.
2. **Steampunk snapping turtle** — quadruped creature with shell/saddle and mechanical/organic detail.
3. **Frog salvage diver** — nonhuman upright character with bulky equipment, backpack, hoses, lantern, and clothing.

The user already generated online 3D versions and wants the local pipeline compared against those reference models. Do not substitute the ragged rowboat. The benchmark is the existing **Lucky Drown casino boat**.

## Source concept filenames to discover locally

Search the user's normal asset roots for these exact filenames and sensible stem variants:

- `Tumman Lucky Drown -jokilaivan konseptilevy.png`
- `Suomuinen höyrykilpikonna sumuisessa suossa.png`
- `Mutainen sammakkosukeltaja ruostevarusteissa.png`

Search at minimum:

- `C:\Users\Lauri\Downloads`
- `C:\Users\Lauri\Desktop`
- `C:\AI`
- existing LowVRAM3D benchmark/input roots

Do not guess a path. Record every candidate path, byte count, SHA-256, dimensions, alpha/background status, and modification time. Select only an exact visual/source match. If duplicates differ, preserve all and fail closed until deterministic evidence selects one.

## Discover the user's online-generated comparison models

For each asset, search non-destructively for existing `.glb`, `.gltf`, `.fbx`, `.obj`, `.ply`, `.blend`, `.stl`, and packaged ZIP outputs under the same roots plus known Hunyuan/online-download folders.

Match candidates using:

- source filename stem and nearby timestamps;
- containing folder names;
- embedded object/material names;
- file dimensions and modification times;
- generated preview renders compared with the source concept;
- existing logs/manifests, when present.

Never call a model the online reference merely because it is the newest file. Produce a contact sheet for every plausible candidate and record the selection evidence.

Each asset gets immutable inputs:

- `source_concept`
- `online_reference_model`
- `local_pipeline_candidate`

If the online reference cannot be found, mark only that case `BLOCKED_MISSING_ONLINE_REFERENCE`; continue source audit and local pipeline preflight for the other cases.

## Mandatory benchmark registry

The benchmark matrix is:

| ID | Asset | Expected automatic base classification | Key traits |
|---|---|---|---|
| `antlered_bird_shaman_anchor` | existing shaman | humanoid / generic character shell | robed, fused staff, antlered, hanging accessories, uncertain fingers |
| `lucky_drown_casino_boat` | Lucky Drown casino riverboat | vehicle/building hybrid | multi-deck, paddlewheel, doors, railings, windows, lights, large architectural structure |
| `steampunk_snapping_turtle` | snapping turtle | quadruped creature | shell/saddle, tail, four limbs, mechanical and organic regions |
| `frog_salvage_diver` | frog diver | humanoid creature | bulky suit, backpack/tank, hoses, lantern, webbed feet, attached equipment |

The expected classifications above are review expectations, not manual profile inputs. Normal execution must infer the classification automatically and report confidence, evidence, alternatives, and safe fallback. The user must not choose a profile.

## Work scheduling on the 6 GB GPU

Do not run concurrent GPU-heavy jobs. Use this sequence:

1. CPU/file discovery and immutable-input manifest for all four assets.
2. Fast Blender structural audit and standardized preview renders for all found reference models.
3. Implement/finish a data-driven `PROFILE_DISCOVERY` preflight that produces base profile plus composable traits without modifying geometry.
4. Run a cheap pipeline smoke test on the casino boat first.
5. Run turtle smoke test.
6. Run frog-diver smoke test.
7. Keep the shaman regression suite running after every generic pipeline change.
8. Resume expensive generation/texture jobs one at a time only after preflight gates pass.

A failure on one asset must not stop unrelated cases. Preserve a per-case status and an overall matrix.

## Casino boat requirements

This is not a humanoid and must never enter humanoid rigging.

Minimum local pipeline proof:

- source ingest/background handling;
- geometry generation or import;
- front/side/rear/top orientation audit;
- hull and superstructure component analysis;
- railings, smokestacks, paddlewheel, doors, windows, flag, and thin-feature preservation;
- LOD and topology report;
- UV/material/texture audit;
- export/fresh-import validation;
- standardized 360-degree contact sheet.

Optional articulation when topology supports it:

- paddlewheel rotation;
- entrance door hinges;
- flag/cloth bone or simple secondary motion;
- light/emissive sockets.

Do not fabricate interior rooms from a single exterior image and call them proven. The concept sheet may contain multiple views and a cutaway; use all panels only when their correspondence is explicit and no panel labels/text contaminate generation inputs.

## Turtle requirements

Minimum proof:

- automatic quadruped/creature classification;
- shell, head, jaw, four limbs, feet, tail, saddle/seat, and mechanical details identified as traits/components;
- no humanoid A-pose;
- source-pose or neutral quadruped rig readiness only when anatomy is observable;
- jaw, neck, legs, tail, and shell/saddle controls when defensible;
- shell/saddle remains rigid relative to soft body unless topology proves another behavior;
- walk test deferred until isolated limb and shell/saddle deformation gates pass.

## Frog-diver requirements

Minimum proof:

- automatic humanoid-creature classification;
- body versus backpack/tank, lantern, hoses, straps, clothing, and webbed feet;
- attached rigid equipment gets rigid controls/sockets rather than smeared body weights;
- no fabricated finger rig when hands are not separable;
- source-pose articulation tests before locomotion;
- hoses and hanging equipment classified for rigid, bone-chain, or cloth behavior.

## Comparison against online references

Normalize each local candidate and online reference into a non-destructive review scene. Do not overwrite either.

Produce comparable metrics and renders:

- source-image resemblance from canonical camera views;
- silhouette agreement;
- dimensions and orientation;
- mesh/object/component counts;
- vertices, triangles, connected components, boundaries, non-manifold counts;
- thin-feature survival;
- material and texture counts/resolutions;
- UV coverage/overlap where available;
- visible front/back/side texture quality;
- topology density distribution;
- runtime, peak VRAM, peak RAM, and failure stage;
- file size and fresh-import survival;
- model-specific articulation readiness.

Do not reduce the comparison to a single score. Produce a metric table plus contact sheets and explicit strengths/weaknesses. The online model is a comparison reference, not automatically the production winner.

Required output root:

`C:\AI\LowVRAM3D-benchmarks\multi-asset-validation\run-<timestamp>\`

Required files:

- `inputs/discovery_manifest.json`
- `benchmark_matrix.json`
- `profile_discovery/<asset-id>.json`
- `reference_audit/<asset-id>/...`
- `local_candidate/<asset-id>/...`
- `comparison/<asset-id>/metrics.json`
- `comparison/<asset-id>/contact_sheet.png`
- `comparison/overall_matrix.json`
- `CONTINUE_HERE.md`

## Shaman current state and immediate V3 task

The shaman remains required after every generic change.

Proven:

- proxy anatomy and source-pose rig;
- robe-first skinning;
- staff exclusivity;
- textured debug bind;
- semantic exclusion repair removed old torso/rear-cape/side-cape arm bleed;
- isolated protected-region movement counts are zero.

Still rejected:

- V2 arm articulation and wave: the previous body-wide cape fan is gone, but the free-side hanging sleeve behaves as a rigid triangular panel;
- breathing is too subtle;
- production texture quality;
- walk;
- export topology parity.

Implement exactly one V3 sleeve repair attempt:

1. split anatomical arm/hand core from sleeve anchor and hanging sleeve drape;
2. add cross-semantic triangle, edge stretch, area ratio, aspect ratio, flipped-normal, inversion, degeneration, and shear diagnostics;
3. broad sleeve drape must not be directly driven by the hand bone;
4. add a small damped sleeve-drape bone chain;
5. rerender isolated clavicle/upper-arm/elbow/wrist/sleeve tests;
6. rerender Milestone B and wave with original curves initially;
7. if the same rigid panel remains, stop with `BLOCKED_FUSED_SLEEVE_TOPOLOGY_REQUIRES_GARMENT_SEPARATION` rather than repeated radius tuning.

Do not let multi-asset work erase or postpone this shaman regression indefinitely. It is acceptable to complete cheap multi-asset discovery and audits while Blender renders run, but do not launch simultaneous heavy GPU jobs.

## Automatic profile discovery

Normal production execution must not require a profile dropdown or user decision.

Return:

- `base_profile`
- composable `traits`
- ranked alternatives and confidence
- deterministic evidence
- optional advisory visual evidence
- contradictions
- selected safe strategy
- safe fallback
- `user_input_required`

Initial base profiles:

- humanoid
- quadruped
- avian
- aquatic
- serpentine
- multi_limb
- generic_character_shell
- static_prop
- articulated_prop
- building
- vehicle
- unknown

The casino boat should prove that building/vehicle assets bypass character rigging. The turtle should prove quadruped routing. The frog diver and shaman should exercise two different character/garment/equipment cases.

## Proof and source control

After each bounded milestone:

- run focused tests;
- run affected regression tests;
- update proof documents;
- commit;
- push;
- verify local SHA equals remote SHA;
- verify clean worktree.

Classify every claim as `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED`.

Do not claim completion from code/tests alone. Execute Blender workers and create visible artifacts.

Stop only for:

- `USER_REVIEW_REQUIRED`
- `WAIT_FOR_USER_DECISION`
- `HARD_BLOCKER`
- `PROJECT_DONE`
- `TASK_DONE_NO_NEXT_ACTION`

Otherwise continue automatically through the bounded queue.
