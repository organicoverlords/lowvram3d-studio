# Benchmark addendum — walking industrial settlement

## Required benchmark asset

Canonical working ID:

`walking_industrial_settlement`

The user-provided concept shows a building-sized mobile settlement or industrial tenement elevated on long mechanical walking legs. The inhabited upper structure has several levels, balconies and catwalks, railings, windows, doors, signs, lamps, pipes, tanks, cables, rooftop equipment, and accumulated utility clutter. It stands in a sandy wasteland. Traffic cones, loose ground objects, distant structures, sand, dust, sky, and ground shadows are scene context rather than part of the target asset unless physically attached.

This is **mobile architecture**: a building/settlement combined with a walking mechanical platform. It is not a vending machine, ordinary kiosk, humanoid robot, animal, or conventional wheeled vehicle.

## Source-reading policy

The concept is a single perspective view. The audit must distinguish:

- directly visible structure;
- partially occluded structure;
- inferred but unproven rear structure;
- environment/background elements;
- support or leg geometry that is visible versus hidden.

Do not invent unseen rear architecture, interior rooms, support count, hinge axes, or mechanical details and call them proven.

The visible arrangement strongly suggests a multi-legged walker, but the exact support count must be established from the source and generated geometry rather than hard-coded into the routing policy.

## Visual identity for bounded discovery

Search unnamed images and models using the combined evidence:

- large weathered green industrial building or settlement;
- several inhabited levels;
- exterior balconies, catwalks, railings, windows, doors, awnings, and signs;
- long mechanical walking legs/supports beneath the whole building;
- broad industrial foot pads contacting sandy ground;
- external pipes, tanks, cables, lamps, cameras/sensors, and rooftop utility equipment;
- rust, chipped paint, grime, patched metal, and asymmetrical additions;
- sandy post-industrial wasteland scene;
- orange traffic cones near the structure in the source image.

Reject false matches such as:

- standalone vending machines or kiosks;
- static buildings on ordinary columns;
- humanoid mechs;
- animal-shaped walkers;
- cranes;
- four-wheeled or tracked vehicles;
- the Lucky Drown casino boat.

## Automatic profile expectations

Expected ranked base profiles:

1. `building`
2. `articulated_prop`
3. `vehicle`
4. `unknown`

Expected composable traits:

- `mobile_architecture`
- `walking_or_stilted_structure`
- `multi_legged_mechanical_supports`
- `large_scale_structure`
- `multi_storey`
- `inhabited_structure`
- `hard_surface`
- `rigid_body_segments`
- `external_balconies_and_catwalks`
- `railings_and_thin_structures`
- `doors_and_windows`
- `roof_utility_equipment`
- `pipes_tanks_cables_and_signage`
- `weathered_patched_construction`
- `asymmetric_surface_detail`

Forbidden routes:

- humanoid rig;
- humanoid A-pose;
- quadruped creature rig by assumption;
- organic heat-diffusion skinning over the whole structure;
- cloth classification for rigid walls, balconies, or support plates;
- fixed support-count assumptions before audit;
- invented interior geometry presented as proven.

Expected profiles are review expectations only. Production routing must infer and explain the selected profile, alternatives, confidence, contradictions, and fallback.

## Why this benchmark matters

The matrix now spans:

- shaman — fused robed humanoid;
- frog diver — equipped nonhuman humanoid;
- turtle — organic/mechanical quadruped;
- casino boat — large rigid vehicle/building hybrid;
- walking industrial settlement — inhabited mobile architecture on mechanical supports.

This case tests whether the pipeline can preserve architectural complexity and thin structures while preparing rigid mechanical articulation without misrouting the asset into character skinning.

## Minimum pipeline proof

### Ingest and masking

- isolate the complete settlement and every attached mechanical support;
- exclude sand, sky, traffic cones, distant buildings, dust, ground shadows, and loose debris;
- preserve foot pads, railings, catwalks, signs, cables, pipes, tanks, lamps, rooftop equipment, windows, doors, and overhangs;
- report occluded regions instead of hallucinating them.

### Geometry and architectural audit

Identify and report candidates for:

- main structural body;
- individual floors or major masses;
- balconies, catwalks, railings, ladders, and stairs;
- windows, doors, hatches, and awnings;
- signs, pipes, tanks, cables, lamps, cameras/sensors, and rooftop equipment;
- each observed support/leg root;
- each observed support segment and possible pivot;
- each visible foot/contact pad;
- rigid attached props;
- rejected background objects.

Required measurements:

- objects, meshes, vertices, triangles, and connected components;
- boundary and non-manifold counts;
- thin-feature survival;
- architectural-opening preservation;
- observed/occluded/unknown support count;
- support-to-body attachment confidence;
- candidate rigid components;
- candidate hinge axes with uncertainty;
- background contamination;
- source-view coverage and unobserved regions.

### Mechanical preparation

Treat the inhabited upper structure as rigid architecture unless separable moving parts are proven.

Prefer rigid-part parenting and explicit hinges over organic weighting.

When supported by topology, create:

- one root/body control;
- one chain per proven mechanical support;
- foot/contact-pad controls;
- optional pan/tilt controls for roof lights or sensors;
- optional door, hatch, awning, or panel hinges;
- sockets for lights, effects, interaction points, and cables.

Use precise classifications for ambiguity, including:

- `NOT_PROVEN_SUPPORT_COUNT_OCCLUDED`
- `NOT_PROVEN_FUSED_MECHANICAL_JOINT`
- `NOT_PROVEN_UNSEEN_REAR_STRUCTURE`

### Motion proof

Do not begin a walk cycle first.

After support segmentation is proven, run:

- each support root independently;
- each proven hinge independently;
- each foot/contact pad independently;
- controlled body lift/lower with planted-contact measurements;
- roof light or sensor pan/tilt;
- door/hatch tests only for separable topology.

Required gates:

- walls, floors, balconies, and roof retain rigid shape;
- railings, signs, lamps, pipes, tanks, and equipment remain attached;
- neighboring supports remain stationary during isolated tests;
- no rigid plate bends from soft-body weights;
- no unexpected object detachment;
- planted-contact drift is measured;
- hinges do not stretch rigid geometry beyond tolerance;
- transforms remain finite;
- export/fresh-import preserves applicable objects, materials, controls, and actions.

Only after these pass may the pipeline attempt:

- weight shift;
- one controlled step;
- a short multi-leg gait loop.

### Texture and material proof

Preserve distinct observable material roles:

- weathered green painted metal;
- rust and exposed metal;
- glass windows and lamps;
- dark joints and support mechanisms;
- signs, decals, and painted markings;
- railings, pipes, tanks, cables, and utility equipment;
- dark or emissive interiors/windows where visible.

Reject sand, sky, cones, distant buildings, and ground shadows baked into asset textures.

## Online-reference comparison

When the user's online-generated model is found, compare:

- overall multi-storey silhouette and proportions;
- support count and arrangement, with source uncertainty explicitly recorded;
- preservation of balconies, catwalks, railings, windows, doors, signs, pipes, tanks, lamps, and rooftop equipment;
- thin-feature survival;
- architectural rigidity and topology;
- support segmentation and articulation readiness;
- environmental contamination;
- material and texture separation;
- front, rear, side, and top plausibility;
- export and fresh-import survival;
- runtime, peak VRAM, peak RAM, and file size.

Do not collapse the result into one score.

## Required classification fields

- `SOURCE_MATCH`
- `PROFILE_DISCOVERY`
- `MOBILE_ARCHITECTURE_CLASSIFICATION`
- `SUPPORT_COUNT_CONFIDENCE`
- `ARCHITECTURAL_COMPONENTS`
- `MECHANICAL_COMPONENTS`
- `HINGE_READINESS`
- `ARCHITECTURAL_DETAIL_PRESERVATION`
- `RIGID_PART_PRESERVATION`
- `BACKGROUND_CONTAMINATION`
- `TEXTURE_QUALITY`
- `ARTICULATION_PROOF`
- `EXPORT_QA`
- `ONLINE_REFERENCE_COMPARISON`

Use only `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED`, each with a precise reason code.
