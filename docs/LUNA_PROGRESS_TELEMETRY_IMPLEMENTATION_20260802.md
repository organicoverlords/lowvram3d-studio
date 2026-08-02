# Luna bounded task — progress telemetry v1

## Role

Luna Medium executes this specification. It does not redesign the telemetry contract or pipeline architecture.

Authoritative design:

- `docs/PIPELINE_PROGRESS_TELEMETRY_ARCHITECTURE_20260802.md`
- `configs/schemas/pipeline_status_v1.schema.json`

Do not integrate this into the barn process while that process is still running. Seal the active baseline first.

## Milestone T1 — shared telemetry core only

Implement a small module at:

```text
src/lowvram3d/progress.py
```

Required capabilities:

- atomic `status.json` writes;
- append-only `events.jsonl` writes;
- heartbeat worker;
- exact-unit and indeterminate progress modes;
- monotonic weighted overall progress;
- ETA with explicit confidence or no ETA;
- resource snapshot hooks;
- terminal state and proof classification separation;
- resume read preserving attempt history;
- human summary generation.

Keep the file under approximately 500 lines. Split only when necessary by responsibility. Do not create a scheduler, database, service, plugin framework, or new orchestration engine.

Add a CLI entry point compatible with:

```text
python -m lowvram3d.progress status --run <run-root>
python -m lowvram3d.progress status --run <run-root> --json
python -m lowvram3d.progress status --run <run-root> --watch
python -m lowvram3d.progress status --run <run-root> --events 20
python -m lowvram3d.progress status --run <run-root> --errors
```

Add:

```text
scripts/show-run-status.ps1
```

The PowerShell wrapper only invokes the Python CLI. Do not duplicate status logic in PowerShell.

## Milestone T1 tests

Use fixtures only. Tests must prove:

1. stage `3/5` produces exact 60% stage progress;
2. nested chunk `1840/3000` is represented without losing parent step identity;
3. weighted overall progress is monotonic;
4. stage reaches 100% only after completion event;
5. `COMPLETED + REJECTED` is supported;
6. atomic replacement never exposes invalid JSON to a concurrent reader;
7. stale heartbeat detection preserves last checkpoint;
8. resume creates a new attempt while preserving prior events;
9. indeterminate operation has no fake percent or ETA;
10. CUDA error packet records step, chunk, operation, tensor metadata, memory, and safe resume point;
11. generated summary contains DONE, NOW, LEFT, BLOCKER, and NEXT SAFE ACTION;
12. source artifacts remain byte-identical.

Do not touch generation, Blender, texture, rigging, or benchmark code in T1.

Return test results and stop for architecture review.

## Milestone T2 — integrate one barn diagnostic path

Start only after T1 review approval.

Instrument one existing barn generation command without changing its numerical behavior or settings.

Required run-level stages:

```text
INPUT_VALIDATION
MODEL_LOAD
IMAGE_ENCODE
GEOMETRY_GENERATION
MESH_WRITE
FRESH_IMPORT
EXPORT_QA
```

Inside geometry generation, report when supported:

```text
denoise step n/N
chunk start-end/total
octree level
mesh extraction
GLB write
```

For the exact five-step diagnostic workload, record:

- step index/total;
- chunk index/range/total;
- allocated/reserved/peak CUDA memory at step boundaries;
- tensor shape/dtype/device at named boundaries;
- last successful named operation;
- a heartbeat at least every 10 seconds;
- meaningful progress time separately from heartbeat time.

Do not add `torch.cuda.synchronize()` to normal production inner loops. Synchronize only at named boundaries in explicit diagnostic mode.

Required terminal behaviors:

- exact failure location appears in `last_error.json`;
- console prints current step/chunk and last checkpoint;
- status lists all remaining stages after failure;
- unchanged retry is marked forbidden when the same workload failed without a correction;
- successful GLB write still shows `FRESH_IMPORT` and `EXPORT_QA` as remaining.

Run one fixture/smoke execution. Do not rerun the full barn merely to demonstrate telemetry.

## Milestone T3 — broader integration

Not authorized until T1 and T2 are reviewed.

Potential later integration targets:

- stage runner and receipts;
- Blender subprocess event markers;
- texture projection;
- UV/geometry audits;
- rigging and animation;
- multi-asset benchmark orchestrator.

Stop with `WAIT_FOR_SOL_ARCHITECTURE` when existing code does not expose a required progress boundary cleanly. Return the exact boundary and smallest required interface change.

## Source control

For each authorized milestone:

- verify branch and clean worktree;
- run focused tests;
- commit only the milestone scope;
- push;
- verify local equals remote;
- verify clean worktree;
- return changed files, tests, artifacts, and classification.

Use `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED` for results.
