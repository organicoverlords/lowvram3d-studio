# Pipeline progress and tracing architecture v1

## Goal

Every pipeline run must tell both a human and an agent:

- what is running now;
- how far the current operation has progressed;
- how much of the whole run is complete;
- what has already completed;
- what remains;
- what the last proven checkpoint is;
- whether the process is alive or stalled;
- where a failure occurred;
- what can safely resume without repeating completed work.

This is a small shared telemetry layer integrated into the existing runners and stage receipts. It is **not** a new pipeline engine, scheduler, UI framework, or replacement for provenance receipts.

## Non-negotiable truth rules

1. Never fabricate progress percentages.
2. Use exact unit progress only when the total is known: denoise steps, chunks, views, files, frames, triangles, components, or stages.
3. When the total is unknown, report `progress_mode=INDETERMINATE`, elapsed time, current operation, heartbeat age, and the last completed checkpoint.
4. ETA must include confidence. Hide the ETA when evidence is insufficient.
5. A process writing heartbeats is alive, not necessarily making progress. Track both heartbeat time and last meaningful progress time.
6. `100%` means the planned unit completed, not that the artifact passed QA.
7. Stage completion and stage classification are separate fields.
8. A generated GLB is not pipeline completion until export, fresh import, and required QA stages finish.

## Canonical files per run

Each run root contains:

```text
telemetry/
  plan.json
  status.json
  events.jsonl
  metrics.jsonl
  heartbeat.json
  last_error.json
  summary.md
```

### `plan.json`

Immutable after execution starts except for explicit versioned plan amendments. It records:

- run identity;
- repository, branch, HEAD, input hashes, configuration hash;
- ordered stage and substage tree;
- stage weights for overall progress;
- expected units when known;
- optional versus required stages;
- resume checkpoints;
- final acceptance requirements.

### `status.json`

Atomically replaced current snapshot for fast polling. Required fields:

- `schema_version`
- `run_id`
- `asset_id`
- `state`
- `classification`
- `overall_progress`
- `current_stage`
- `current_substage`
- `current_operation`
- `progress_mode`
- `units_completed`
- `units_total`
- `stage_percent`
- `elapsed_seconds`
- `eta_seconds`
- `eta_confidence`
- `started_at`
- `updated_at`
- `last_heartbeat_at`
- `last_progress_at`
- `last_completed_checkpoint`
- `current_artifact`
- `completed_stages`
- `remaining_stages`
- `blocked_reason`
- `user_action_required`
- `resume_command_or_token`
- `resource_snapshot`

Writes use temporary file plus atomic rename so agents never read partial JSON.

### `events.jsonl`

Append-only machine-readable history. Every line contains:

- monotonic `sequence`;
- UTC timestamp;
- run/stage/substage identity;
- event type;
- message;
- unit progress;
- resource snapshot when relevant;
- artifact/checkpoint path when relevant;
- exception class and operation when relevant.

Required event types:

```text
RUN_PLANNED
RUN_STARTED
STAGE_STARTED
SUBSTAGE_STARTED
OPERATION_STARTED
PROGRESS
HEARTBEAT
CHECKPOINT_WRITTEN
ARTIFACT_WRITTEN
QA_STARTED
QA_RESULT
RETRY_STARTED
STAGE_COMPLETED
STAGE_REJECTED
STAGE_BLOCKED
STAGE_FAILED
RUN_COMPLETED
RUN_REJECTED
RUN_BLOCKED
RUN_FAILED
RUN_CANCELED
```

### `metrics.jsonl`

Time-series resource and diagnostic measurements:

- CPU and process RAM;
- GPU allocated/reserved/peak memory;
- GPU utilization and temperature when safely available;
- current denoise step;
- current chunk range;
- mesh counts and active object where relevant;
- operation duration;
- queue/backoff or retry delay.

### `heartbeat.json`

Written at least every 10 seconds by the supervising process and no more than every 2 seconds. It includes:

- process ID;
- child process ID when relevant;
- current stage and operation;
- last progress sequence;
- process start time;
- heartbeat time;
- child stdout/stderr activity time.

A monitor classifies:

- `ACTIVE`: heartbeat age <= 30 s;
- `SLOW_OR_SILENT`: heartbeat current, no meaningful progress for configured stage threshold;
- `STALE`: no heartbeat for > 30 s while process should exist;
- `PROCESS_EXITED_WITHOUT_TERMINAL_EVENT`: process ended without a terminal status.

### `last_error.json`

Written before process exit whenever possible. It records:

- exact stage, substage, operation;
- last successful operation;
- last checkpoint;
- exception type and message;
- traceback/log tail;
- command and environment deltas;
- tensor shapes/dtypes/devices when CUDA-related;
- CUDA memory immediately before failure;
- first failed step/chunk when known;
- safe resume point;
- whether an unchanged retry is forbidden.

### `summary.md`

Generated from telemetry for humans. It always begins with:

```text
RUN: <id>
STATE: <state/classification>
OVERALL: <percent or indeterminate>
NOW: <stage > substage > operation>
CURRENT UNIT: <completed>/<total or unknown>
ELAPSED: <duration>
ETA: <duration + confidence, or unavailable>
LAST CHECKPOINT: <name/path>
DONE: <completed stages>
LEFT: <remaining stages>
BLOCKER: <reason or none>
NEXT SAFE ACTION: <resume/inspect/stop>
```

## State model

Execution states:

```text
PLANNED
RUNNING
PAUSED
COMPLETED
FAILED
CANCELED
```

Proof classifications remain independent:

```text
PROVEN
REJECTED
NOT_PROVEN
BLOCKED
```

Examples:

- `state=COMPLETED`, `classification=REJECTED`: command finished but QA rejected output.
- `state=FAILED`, `classification=BLOCKED`: process crashed and the lane cannot proceed safely.
- `state=RUNNING`, `classification=NOT_PROVEN`: expected during execution.

## Progress calculation

### Stage progress

Use one of:

- `EXACT_UNITS`: completed/total known;
- `WEIGHTED_OPERATIONS`: sub-operations have fixed plan weights;
- `INDETERMINATE`: total unknown.

### Overall progress

Overall percent is calculated from the immutable stage plan, not from elapsed time:

```text
overall = completed_required_stage_weights
        + current_stage_weight * current_stage_fraction
```

Optional stages excluded by policy are marked `SKIPPED_NOT_SELECTED` and removed from the denominator through a recorded plan amendment.

Never silently change stage weights after observing how long a stage takes.

### ETA

ETA sources in priority order:

1. rate from completed units in this exact stage/run;
2. median duration from matching prior successful runs with the same model/config envelope;
3. unavailable.

Confidence values:

```text
HIGH     exact units, stable rate, enough samples
MEDIUM   exact units but variable rate, or close historical match
LOW      weak historical estimate
NONE     do not display ETA
```

## CUDA and long-model tracing

For iterative CUDA workloads, telemetry must expose the real nested progress. Example:

```text
GENERATE_GEOMETRY
  model_load 1/1
  encode_image 1/1
  denoise_step 3/5
    chunk 1840/3000
  decode_latent 0/1
  octree_extract level=384
  write_glb 0/1
```

At each denoise step and configurable chunk interval record:

- step index and total;
- chunk start/end and total;
- tensor name, shape, dtype, device for stage boundaries;
- CUDA allocated, reserved, and peak bytes;
- last completed kernel-level checkpoint name;
- elapsed time and rolling units/second.

For diagnostics using `CUDA_LAUNCH_BLOCKING=1`, call `torch.cuda.synchronize()` after named major operations and emit an event immediately afterward. Do not synchronize every tiny normal-production kernel because it would distort performance.

Required named checkpoints for the barn generation route include, when applicable:

```text
INPUT_VALIDATED
MODEL_LOADED
IMAGE_ENCODED
DENOISE_STEP_<n>_START
DENOISE_STEP_<n>_COMPLETE
CHUNK_<start>_<end>_COMPLETE
LATENT_DECODE_COMPLETE
OCTREE_EXTRACTION_STARTED
OCTREE_EXTRACTION_COMPLETE
MESH_CREATED
GLB_WRITTEN
TEXTURE_STAGE_STARTED
TEXTURE_STAGE_COMPLETE
BLENDER_FRESH_IMPORT_COMPLETE
EXPORT_QA_COMPLETE
```

## Blender integration

Blender subprocesses emit machine-readable stdout markers in addition to normal logs:

```text
LOWVRAM3D_EVENT {json}
```

The supervising process parses these lines and writes canonical telemetry. Blender workers do not write competing top-level status files directly unless running standalone; standalone mode uses the same schema.

Blender operations with known totals must report them:

- importing N objects;
- auditing N meshes;
- rendering view i/N;
- processing component i/N;
- UV chart/preset candidates;
- exporting format i/N.

## Console and user display

The default console prints one concise line when meaningful progress changes and one heartbeat line at most every 30 seconds:

```text
[42.3%] GENERATE_GEOMETRY > denoise 3/5 > chunk 1840/3000 | 11m08s | ETA 8m MEDIUM | GPU 5.1/6.0 GB
```

When progress is indeterminate:

```text
[--.-%] OCTREE_EXTRACTION level=384 | alive 6m22s | last progress 18s ago | ETA unavailable
```

At stage completion:

```text
[58.0%] GENERATE_GEOMETRY completed | artifact=mesh.glb | QA pending | next=TEXTURE
```

At failure:

```text
[BLOCKED] GENERATE_GEOMETRY > denoise 4/5 > chunk 2210/3000
last success=DENOISE_STEP_3_COMPLETE
error=CUDA_DEVICE_SIDE_ASSERT
resume=diagnostic reproduction required; unchanged retry forbidden
```

## Agent-facing status command

Provide one stable command that never needs log parsing by the agent:

```text
python -m lowvram3d.progress status --run <run-root>
```

Output modes:

- default human summary;
- `--json` exact `status.json`;
- `--watch` refresh on change;
- `--events N` last N significant events;
- `--errors` error and safe-resume packet.

A PowerShell wrapper may be added for users:

```text
scripts/show-run-status.ps1 -Run <run-root> -Watch
```

## Resume behavior

On restart:

1. verify run identity and input/config hashes;
2. read plan, status, events, provenance receipts, and artifact hashes;
3. detect terminal versus interrupted stage;
4. preserve old telemetry;
5. append `RETRY_STARTED` or `RUN_RESUMED` with a new attempt ID;
6. resume only from a proven checkpoint;
7. never mark an interrupted operation complete from file existence alone.

## Implementation boundaries

Add one small shared module, expected location:

```text
src/lowvram3d/progress.py
```

Keep it focused on:

- schemas/dataclasses;
- atomic status writes;
- append-only events;
- heartbeat;
- progress/ETA calculation;
- console rendering;
- resume reads.

It must integrate with existing stage receipts and provenance rather than replacing them.

Do not create:

- a new scheduler;
- a new job database;
- a web server;
- a plugin framework;
- a second orchestration engine.

A richer UI can later poll `status.json`; the telemetry contract comes first.

## Acceptance gates

The first implementation is accepted only when fixtures prove:

1. exact 3/5-step progress is represented correctly;
2. nested 1840/3000 chunk progress is represented correctly;
3. overall percent is monotonic within a plan revision;
4. no stage reaches 100% before its operation completes;
5. QA rejection is distinct from process failure;
6. atomic status readers never see truncated JSON;
7. interrupted run becomes stale and retains last checkpoint;
8. resume preserves attempt history;
9. unknown-duration operation reports indeterminate progress without fake ETA;
10. CUDA failure packet records step, chunk, last operation, tensor metadata, and memory;
11. user summary lists done/current/left/next;
12. no source artifact is modified by telemetry.
