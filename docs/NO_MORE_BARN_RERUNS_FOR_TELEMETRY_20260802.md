# No more barn reruns for telemetry validation

## User decision

Do not spend another long GPU run on the barn merely to validate logging, telemetry, console hardening, or progress reporting.

The existing barn evidence is sufficient to establish:

- one-step runs through octree 384 can complete and write real GLBs;
- the two-step diagnostic reached diffusion completion and in-memory mesh decode;
- the two-step attempt became `NOT_PROVEN_LOGGER_ABORT_AFTER_MESH_DECODE` because console output aborted before GLB serialization;
- the original five-step CUDA failure remains unresolved;
- the barn remains a useful benchmark, but it is no longer the next expensive execution target.

## Required policy

1. Fix telemetry and console hardening using unit tests, fixtures, mocked sinks, synthetic mesh summaries, and a local artifact-write callback only.
2. Do not rerun the barn at steps 1, 2, 3, 4, or 5 to prove telemetry.
3. Do not rerun the barn solely to reproduce the logger failure.
4. Do not rerun the barn solely to demonstrate progress percentages.
5. Preserve all existing barn outputs, logs, timings, and classifications.
6. Mark the exact five-step CUDA root cause as `NOT_PROVEN_DEFERRED` rather than consuming more time now.
7. The next GPU-heavy benchmark must be another asset, preferably the Lucky Drown casino boat, then turtle or frog diver according to available staged inputs.
8. Apply the telemetry layer to the next new GPU run so instrumentation is validated during useful pipeline work rather than by repeating the barn.

## Telemetry acceptance without a barn rerun

Telemetry T1 may be classified `PROVEN_BY_FIXTURE_TESTS` when tests prove:

- atomic status writes;
- append-only events;
- exact-unit and indeterminate progress;
- nested step/chunk identity;
- heartbeat and stale-process detection;
- console failure does not escape into generation;
- artifact-write callback runs after simulated `OSError(22)`;
- file telemetry remains valid when stdout and stderr fail;
- terminal summary includes done/current/remaining/blocker/next action;
- resume attempts preserve earlier history.

Telemetry T2 should be integrated into the next real asset run without changing that asset's numerical pipeline settings.

## Barn classification retained

```text
BARN_BASELINE=BLOCKED_CUDA_DEVICE_ASSERT_OR_INCOMPLETE_VALIDATION
STEPS_1_OCTREE_64_128_256_384=PROVEN_REDUCED_CANARIES
STEPS_2_DIFFUSION=PROVEN_BY_TRACE
STEPS_2_MESH_DECODE=PROVEN_IN_MEMORY
STEPS_2_GLB_WRITE=NOT_PROVEN_LOGGER_ABORT
ORIGINAL_STEPS_5_ROOT_CAUSE=NOT_PROVEN_DEFERRED
NEXT_BARN_GPU_RUN=NOT_AUTHORIZED
```

## Executor instruction

Luna Medium must not independently decide to rerun the barn. A future barn rerun requires an explicit new user instruction or high-reasoning architecture authorization tied to a materially new hypothesis that cannot be tested on fixtures or another useful benchmark.
