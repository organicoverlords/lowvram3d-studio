# Current production state

**Updated:** 2026-07-31

This file is the short, current source of truth for the active high-resolution pipeline work. Older session notes remain useful history, but this document takes precedence when they conflict.

## Reference success: TurboBird

TurboBird is the user's first real successful result and the current qualitative reference for the project. The user considers it a good result. It proves that the practical route can produce a useful asset on the target Windows 10 / GTX 1660 SUPER 6 GB machine.

Do not redesign the pipeline so aggressively that it can no longer preserve a TurboBird-class result. New validation should catch demonstrated damage, not reject usable geometry merely because a classifier is uncertain.

The exact canonical TurboBird receipt, source identity and comparison statistics still need to be consolidated into the benchmark pack. Until that is done, TurboBird is a **user-validated qualitative reference**, not a fully reproduced quantitative benchmark.

## Shaman anchor run: what actually worked

The full-resolution shaman PNG was verified locally:

- path: `C:\Users\Lauri\Downloads\ChatGPT Image 29.7.2026 klo 20.00.45.png`
- dimensions: `1122x1402`
- SHA-256: `4d23adc758c5b700dd29939e37c043ce61919792b566bdcf13f58b1409d6cf6f`

The prior recorded hash `eccef854f816f446ce2bf2e08559df519adff223b425c1dccc1c0a9b299f13f6` is rejected because it did not match that exact file.

GitHub Actions run `30657995651` proved the following:

1. the self-hosted runner `LOWVRAM3D-KONE` registered and accepted the job;
2. the PNG-to-3D entrypoint ran geometry generation successfully;
3. ingest produced a valid single-mesh GLB with 55,684 faces;
4. topology at the cleanup boundary was manifold: boundary edges `0 -> 0`;
5. the run failed because the component classifier left two tiny visible fragments as `AUDIT_REQUIRED`, not because geometry generation crashed or the mesh was corrupt.

That failure was a pipeline-policy defect: an inconclusive cleanup classifier was being treated as a hard production failure.

## Current cleanup policy

`workers/component_audit_cleanup.py` now follows a preserve-first policy:

- confirmed cleanup with safe topology continues normally;
- audit ambiguity alone is not a hard failure;
- when the only errors are non-convergence and unresolved `AUDIT_REQUIRED` components, and no topology regression is demonstrated, the original generated GLB is copied byte-for-byte to `clean_master.glb`;
- the report sets `manual_review_required=true` and records the original audit errors as warnings;
- real process errors, increased boundary/non-manifold edges, or a changed main component still fail closed.

This matches the TurboBird lesson: preserve a usable generated asset and continue to LOD, texture, rig/export and independent validation. Cleanup confidence is evidence, not permission to discard the only valid mesh.

## Current workflow branches

- `fix/highres-production-pipeline-20260731`: high-resolution geometry, benchmark ordering, component audit, measured LOD selection and pose-policy work.
- `infra/windows-self-hosted-runner-20260731`: trusted Windows worker and shaman PNG-to-3D execution, stacked on the high-resolution branch through PR #3.
- `improve/stage-provenance-20260731`: stage artifact provenance, stacked separately through PR #2.

Do not merge these branches into `main` until the required local evidence is complete.

## Still not proven

- A replacement shaman run has not yet proven the full route through LOD selection, texture, rigging, final export and fresh-process validation.
- The shaman comparison-master GLB has not been recovered and hash-verified.
- The A-pose normalizer exists but is not yet connected to the main postprocess route and has not been visually validated in Blender or Unreal.
- TurboBird's exact source/master hashes and quantitative receipt are not yet committed as a canonical benchmark record.
- A successful GitHub job is not enough by itself: final GLBs, reports and preview renders must be downloaded and inspected.

## Production rules

1. Preserve source and generated masters byte-for-byte.
2. Hard-fail on demonstrated corruption, topology regression, missing required inputs, invalid exports or subprocess failure.
3. Do not hard-fail solely because a quality classifier is uncertain.
4. Record ambiguity as `manual_review_required` and continue with the preserved mesh.
5. Compare every major change against TurboBird and the shaman anchor.
6. Promote a final deliverable only after fresh-process re-import, renders and receipt validation.
7. Rigging or pose normalization must never overwrite a valid unrigged result when they fail.
