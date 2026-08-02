# Telemetry hardening — logging must never abort generation

## Incident

A `steps=2`, `octree=384` diagnostic run completed diffusion and mesh decode, then failed while printing a diagnostic message with:

```text
OSError: [Errno 22] Invalid argument
```

The mesh existed in memory, but no GLB was written. Therefore:

- `DIFFUSION_STEPS_2=COMPLETED`
- `OCTREE_384_DECODE=COMPLETED_IN_MEMORY`
- `CUDA_RESULT=NOT_PROVEN`
- `GLB_WRITE=NOT_ATTEMPTED_OR_NOT_REACHED`
- `RUN=NOT_PROVEN_LOGGER_ABORT_AFTER_MESH_DECODE`

This is a telemetry/output-path failure, not evidence of a CUDA failure or CUDA success.

## Non-negotiable rule

Telemetry is observational. It may never become a proof-critical dependency for geometry generation, mesh serialization, export, or cleanup.

A console write failure must not:

- terminate generation;
- skip an already-reachable artifact write;
- destroy an in-memory mesh;
- change the numerical path;
- convert a successful compute stage into a false CUDA failure;
- prevent `last_error.json` or terminal status from being written to a file sink.

## Required sink order

For every progress or diagnostic event:

1. normalize the event into a plain JSON-safe dictionary;
2. write the event to the file sink first;
3. atomically update `status.json` when applicable;
4. attempt human-readable console output last;
5. catch console-output exceptions without re-raising into generation code.

The durable file sink is authoritative. Console output is best-effort.

## Safe console emitter

Implement one helper, not scattered `print()` wrappers:

```python
def safe_console_emit(text: object, *, stream=None) -> bool:
    """Best-effort console output. Never raises into generation code."""
```

Required behavior:

- default to `sys.stdout`;
- convert non-string objects with bounded, defensive formatting;
- replace unsupported control characters except `\n`, `\r`, and `\t`;
- avoid passing raw tensors, mesh objects, binary buffers, or enormous `repr()` strings;
- bound console message length while keeping full structured data in `events.jsonl`;
- flush only inside the protected block;
- catch at least `OSError`, `UnicodeError`, `ValueError`, and `BrokenPipeError`;
- on failure, attempt one compact ASCII-only warning to `sys.stderr`;
- if stderr also fails, silently preserve the failure in file telemetry;
- return `False` on console failure and `True` on success;
- never terminate or alter the compute/export path.

Do not repeatedly retry a broken console handle.

After the first console failure in a process:

- set `console_sink_state=DEGRADED`;
- disable verbose console events;
- continue durable file telemetry;
- allow one terminal ASCII summary attempt at process exit.

## Structured payload restrictions

Event payloads must contain summaries rather than arbitrary object representations.

For tensors record only:

- logical name;
- shape;
- dtype;
- device;
- finite/non-finite counts when explicitly measured;
- allocated/reserved/peak CUDA memory.

For meshes record only:

- vertex count;
- face/triangle count;
- component count when available;
- bounds;
- finite status;
- intended output path.

Never print or serialize raw tensor values, mesh arrays, Blender objects, ctypes handles, file handles, generators, or custom object `repr()` output in progress events.

## Artifact-first recovery boundary

After mesh decode succeeds, the worker must establish a protected artifact boundary:

```text
MESH_DECODE_COMPLETE
→ MESH_SUMMARY_CAPTURED
→ GLB_WRITE_ATTEMPTED
→ GLB_WRITE_COMPLETE
```

Telemetry emitted between these operations is non-fatal.

When an in-memory mesh exists and telemetry fails:

1. record `TELEMETRY_CONSOLE_DEGRADED` to the durable sink when possible;
2. continue immediately to GLB write;
3. write the GLB to a temporary candidate path;
4. fsync/close as appropriate;
5. verify non-zero bytes and structural readability;
6. atomically promote or rename to the intended candidate path;
7. only then perform fresh-import validation.

Do not keep a large decoded mesh alive longer merely to format diagnostics.

## Current diagnostic correction

Make one bounded correction to the logger/output path only.

Do not alter:

- source image;
- model or checkpoint;
- seed;
- `steps=2`;
- `octree=384`;
- `num_chunks=3000`;
- decoder settings;
- dtype policy;
- CUDA settings unrelated to the logger.

Before rerunning, add fixture tests that simulate:

1. `sys.stdout.write()` raising `OSError(22, "Invalid argument")`;
2. `flush()` raising `OSError`;
3. stdout failure followed by stderr failure;
4. a message containing unsupported control characters;
5. an enormous object representation;
6. an event containing tensor/mesh summaries only;
7. successful GLB-write callback execution despite console failure;
8. durable `events.jsonl`, `status.json`, and `last_error.json` remaining valid.

## One corrected reproduction

After focused tests pass, rerun exactly the same `steps=2`, `octree=384`, `num_chunks=3000` workload once in a fresh process.

Required classification outcomes:

### Full success

Only when all are present:

- diffusion completes;
- mesh decode completes;
- GLB is written;
- GLB is non-empty and structurally readable;
- fresh Blender import succeeds;
- material/texture state is reported explicitly.

Then classify the compute path separately from final asset QA.

### Logger still aborts

```text
BLOCKED_TELEMETRY_OUTPUT_PATH_NOT_FAIL_SAFE
```

Do not run another workload attempt.

### CUDA fails before mesh decode

Record exact step, chunk, named operation, tensor summaries, and memory. This becomes a CUDA diagnostic result.

### Mesh/GLB write fails after decode

Classify the specific serialization/export blocker; do not call it CUDA.

## Proof fields

The corrected report must include:

- `DIFFUSION_RESULT`
- `MESH_DECODE_RESULT`
- `TELEMETRY_FILE_SINK`
- `TELEMETRY_CONSOLE_SINK`
- `GLB_WRITE_RESULT`
- `GLB_STRUCTURAL_VALIDATION`
- `FRESH_BLENDER_IMPORT`
- `CUDA_RESULT`
- `RUN_CLASSIFICATION`
- `LAST_SUCCESSFUL_OPERATION`
- `NEXT_SAFE_ACTION`

Use `PROVEN`, `REJECTED`, `NOT_PROVEN`, or `BLOCKED` with exact reason codes.
