# Benchmark addendum — giant tripod vending-machine walker

## Required benchmark asset

Add the user-provided concept shown in chat as a fifth mandatory benchmark.

Canonical working name:

`giant_tripod_vending_machine_walker`

Human-readable description:

A building-scale, weathered green vending-machine or kiosk body mounted on three long industrial walking legs. The body has a vending/display facade, rust and grime, attached signs and equipment, hanging cables/hoses, and a camera/floodlight/sensor cluster on top. The scene is a sandy or post-industrial wasteland with traffic cones, but the environment is not part of the target asset.

## Visual identity used for discovery

The local source image may be unnamed. Search and pair by these combined visual traits rather than filename:

- tall rectangular green vending-machine/kiosk body;
- three long articulated mechanical legs in a tripod arrangement;
- wide industrial feet touching sandy ground;
- prominent central front vending/display area;
- rusted pipes, cables, hoses, signs, panels, and utility attachments;
- camera, lamp, sensor, or floodlight cluster above the body;
- weathered post-apocalyptic industrial styling;
- orange traffic cones near the feet in the source scene;
- no organic body, no humanoid torso, and no wheels.

Do not confuse it with:

- an ordinary vending machine;
- a four-legged mech;
- a humanoid robot;
- a crane;
- a building with static supports;
- the Lucky Drown casino boat.

## Automatic profile expectations

Expected ranked base profiles:

1. `articulated_prop`
2. `vehicle`
3. `building`
4. `unknown` as fail-safe

Expected composable traits:

- `three_legged_mechanical`
- `hard_surface`
- `rigid_body_segments`
- `mechanical_hinges`
- `large_scale_structure`
- `vending_machine_facade`
- `camera_or_sensor_head`
- `hanging_cables_or_hoses`
- `thin_structures`
- `attached_signage_and_props`
- `asymmetric_surface_detail`

Forbidden routes:

- humanoid rig;
- humanoid A-pose;
- quadruped creature rig;
- organic heat-diffusion skinning across the whole model;
- cloth classification for rigid metal plates;
- treating the three legs as one deforming soft mesh when separable rigid segments exist.

The expected profile is a validation expectation, not a manually supplied runtime answer. Normal execution must infer and explain the route automatically.

## Why it belongs in the benchmark matrix

This case exercises a pipeline path not covered by the other assets:

- Shaman: fused robed humanoid character.
- Frog diver: equipped nonhuman humanoid.
- Turtle: organic/mechanical quadruped.
- Casino boat: large mostly rigid vehicle/building hybrid.
- Tripod vending walker: articulated hard-surface, three-legged mechanical structure.

It tests whether the pipeline can detect nonstandard limb count, preserve rigid parts, infer hinge candidates, avoid soft-body weighting, and create usable mechanical controls.

## Minimum pipeline proof

### Ingest and source cleanup

- isolate the walker from the sandy environment;
- exclude traffic cones, distant structures, dust, ground shadow, and background debris unless physically attached;
- preserve the full silhouette of all three legs, feet, top sensor cluster, cables, signs, and protrusions;
- record any occluded leg or rear-body ambiguity rather than hallucinating it as proven.

### Geometry and component audit

Identify and report candidates for:

- main vending/kiosk body;
- top camera/light/sensor assembly;
- each of three leg roots;
- upper and lower segments of each leg where visible;
- hinge or pivot regions;
- each foot;
- doors, panels, vending slots, signs, pipes, cables, and hoses;
- rigid attached props versus unsupported background objects.

Required metrics:

- object and connected-component counts;
- vertices and triangles;
- thin-feature survival;
- symmetry/asymmetry evidence;
- leg count confidence;
- body-to-leg attachment confidence;
- non-manifold and boundary counts;
- component rigidity candidates;
- likely hinge axes and uncertainty.

### Mechanical preparation

Prefer rigid-part parenting and hinge bones/controls over organic skinning.

When topology supports it, create:

- one root/body control;
- one chain per leg with explicit segment controls;
- foot controls;
- top sensor/camera pan and tilt controls;
- optional door/panel hinge controls;
- sockets for lights, effects, cables, and interaction points.

Do not claim a joint proven when topology is fused and the pivot cannot be established. Use `NOT_PROVEN_FUSED_MECHANICAL_JOINT` or a similarly precise classification.

### Motion proof

Before any walk cycle, run isolated tests:

- each leg root independently;
- each visible knee/hinge independently;
- each foot independently;
- body lift/lower while all feet remain planted where possible;
- sensor-head pan/tilt;
- door/panel hinge if detected.

Required gates:

- rigid panels retain shape;
- neighboring legs do not move;
- body does not shear;
- no plate bending from organic weights;
- no detached cables or signs unless classified for secondary motion;
- planted-foot drift measured;
- hinge rotation does not stretch rigid geometry beyond tolerance;
- transforms remain finite;
- export/fresh-import preserves controls and actions where the target format supports them.

Only after isolated gates pass may it attempt:

- weight-shift tripod stance;
- one controlled step;
- short three-leg gait loop.

### Texture and material proof

Preserve distinct material roles when observable:

- painted green metal;
- rust and exposed metal;
- glass/display/vending window;
- emissive lamps or screens;
- rubber or dark mechanical joints;
- signs and decals;
- cables and hoses.

Do not bake the sandy background into the model texture and call it coverage.

## Comparison against online reference

When the user's online-generated model is discovered, compare:

- correct three-leg count;
- silhouette and proportions;
- front vending-machine readability;
- sensor/light cluster preservation;
- leg segmentation and hinge readiness;
- foot shape and contact area;
- thin cable, sign, pipe, and railing survival;
- rigid-part topology quality;
- background contamination;
- texture/material separation;
- export and articulation readiness;
- runtime, VRAM, RAM, and file size.

Do not reduce this to a single score.

## Required classification fields

- `SOURCE_MATCH`
- `PROFILE_DISCOVERY`
- `LEG_COUNT`
- `MECHANICAL_COMPONENTS`
- `HINGE_READINESS`
- `RIGID_PART_PRESERVATION`
- `BACKGROUND_CONTAMINATION`
- `TEXTURE_QUALITY`
- `ARTICULATION_PROOF`
- `EXPORT_QA`
- `ONLINE_REFERENCE_COMPARISON`

Use only `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED`, with a precise reason code.
