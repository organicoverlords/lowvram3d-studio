# Benchmark addendum — walking industrial building

## Correction

The previous interpretation as a vending machine was wrong and is rejected.

The user-provided image shows a large, weathered, multi-storey industrial building or inhabited structure elevated on long mechanical walking supports in a sandy wasteland. It has external balconies and walkways, windows and doors, pipes and utility hardware, roof-mounted lights/sensors/equipment, signs and attached clutter. The image is a single perspective view, so the exact number and full arrangement of legs/supports is not proven from the source alone.

Canonical working name:

`walking_industrial_building`

## Visual identity used for discovery

The local source image may be unnamed. Match by the combined visual evidence:

- tall, weathered, green multi-storey industrial building or inhabited structure;
- external balconies, railings, stairs/walkways, windows, doors, and signs;
- long articulated mechanical supports or walking legs beneath the structure;
- large industrial feet contacting sandy ground;
- pipes, tanks, cables, lights, cameras/sensors, and rooftop utility equipment;
- post-industrial or post-apocalyptic desert setting;
- orange traffic cones and ground clutter are scene context, not part of the target asset unless physically attached;
- no humanoid body and no ordinary wheeled chassis.

Do not assume:

- that it is a vending machine;
- an exact leg count from this one view;
- that every visible background object belongs to the structure;
- that unseen rear architecture or interiors are proven;
- that the supports are static rather than articulated, or articulated rather than static, until topology and geometry provide evidence.

## Automatic profile expectations

Expected ranked base profiles:

1. `building`
2. `articulated_prop`
3. `vehicle`
4. `unknown` as fail-safe

Expected composable traits:

- `walking_or_stilted_structure`
- `large_scale_structure`
- `multi_storey`
- `hard_surface`
- `rigid_body_segments`
- `mechanical_supports`
- `external_balconies_and_walkways`
- `railings_and_thin_structures`
- `doors_and_windows`
- `roof_utility_equipment`
- `pipes_cables_and_signage`
- `asymmetric_surface_detail`

Forbidden routes:

- humanoid rig;
- humanoid A-pose;
- quadruped creature rig by assumption;
- organic heat-diffusion skinning across the whole structure;
- cloth classification for rigid architectural plates;
- inventing a leg count or hinge layout not supported by geometry.

The expected profile is a validation expectation, not a manually supplied runtime answer. Normal execution must infer the route and report confidence, alternatives, and contradictions.

## Why it belongs in the benchmark matrix

This asset tests a large architectural/mechanical hybrid that differs from the other cases:

- Shaman: fused robed humanoid character.
- Frog diver: equipped nonhuman humanoid.
- Turtle: organic/mechanical quadruped.
- Casino boat: large rigid vehicle/building hybrid.
- Walking industrial building: inhabited multi-storey architecture carried by mechanical supports.

It tests whether the pipeline preserves architectural detail, recognizes thin railings and balconies, separates rigid structural regions, avoids character skinning, and only adds articulation when the model topology supports it.

## Minimum pipeline proof

### Ingest and source cleanup

- isolate the complete elevated building and its mechanical supports from the desert background;
- exclude traffic cones, sand, distant buildings, dust, ground shadows, and loose debris unless physically attached;
- preserve balconies, railings, stairs, windows, doors, signs, pipes, roof equipment, supports, and feet;
- record occluded rear structure and support ambiguity rather than hallucinating it as proven.

### Geometry and component audit

Identify and report candidates for:

- main structural body;
- individual floors or major architectural masses;
- balconies, walkways, railings, ladders/stairs;
- doors and windows;
- roof lights, cameras/sensors, antennas, tanks, and pipes;
- every visible support or leg root;
- visible support segments and possible pivots;
- every visible foot/contact pad;
- hanging cables, signs, and attached props;
- scene objects that must be rejected as background contamination.

Required metrics:

- object and connected-component counts;
- vertices and triangles;
- thin-feature survival;
- support/leg count confidence with `observed`, `occluded`, and `unknown` categories;
- body-to-support attachment confidence;
- boundary and non-manifold counts;
- rigid-component candidates;
- possible hinge axes and uncertainty;
- architectural opening preservation;
- background contamination.

### Mechanical preparation

Treat the building body as rigid architecture unless geometry proves movable sections.

Prefer rigid-part parenting and hinge controls over organic skinning.

When topology supports it, create:

- one root/body control;
- one chain per proven mechanical support;
- foot/contact-pad controls;
- optional pan/tilt controls for roof-mounted lights or sensors;
- optional door, hatch, or panel hinges;
- sockets for lights, effects, interaction points, and cables.

Do not claim articulation for fused or visually ambiguous supports. Use a precise `NOT_PROVEN_FUSED_MECHANICAL_JOINT` or `NOT_PROVEN_SUPPORT_LAYOUT_OCCLUDED` classification.

### Motion proof

Only after support count and joints are proven, run isolated tests:

- each proven support independently;
- each proven hinge independently;
- each foot/contact pad independently;
- body lift/lower while planted contacts remain stable where feasible;
- roof light/sensor pan/tilt;
- doors/hatches if actual separable topology exists.

Required gates:

- architectural walls and floors retain rigid shape;
- balconies, railings, signs, and roof equipment remain attached;
- neighboring supports do not move unexpectedly;
- no plate bending from organic weights;
- planted-contact drift is measured;
- hinges do not stretch rigid geometry beyond tolerance;
- transforms remain finite;
- export and fresh import preserve objects, materials, controls, and actions where supported.

Do not attempt a walking cycle unless the support layout and articulation are actually proven.

### Texture and material proof

Preserve distinct material roles when observable:

- weathered green painted metal;
- rust and exposed metal;
- glass windows and lamps;
- dark mechanical joints/supports;
- signs and decals;
- pipes, cables, railings, and utility equipment;
- interior darkness or emissive windows where present.

Do not bake sand, sky, cones, distant structures, or ground shadows into the asset texture and call it valid coverage.

## Comparison against online reference

When the user's online-generated model is discovered, compare:

- overall multi-storey silhouette and proportions;
- number and arrangement of supports, with uncertainty for occluded source regions;
- balconies, railings, windows, doors, signs, pipes, and roof equipment;
- thin-feature survival;
- rigid architectural topology;
- support segmentation and articulation readiness;
- background contamination;
- material and texture separation;
- front, side, rear, and top plausibility;
- export and fresh-import survival;
- runtime, VRAM, RAM, and file size.

Do not reduce this to one aggregate score.

## Required classification fields

- `SOURCE_MATCH`
- `PROFILE_DISCOVERY`
- `STRUCTURE_CLASSIFICATION`
- `SUPPORT_COUNT_CONFIDENCE`
- `MECHANICAL_COMPONENTS`
- `HINGE_READINESS`
- `ARCHITECTURAL_DETAIL_PRESERVATION`
- `RIGID_PART_PRESERVATION`
- `BACKGROUND_CONTAMINATION`
- `TEXTURE_QUALITY`
- `ARTICULATION_PROOF`
- `EXPORT_QA`
- `ONLINE_REFERENCE_COMPARISON`

Use only `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED`, with precise reason codes.
