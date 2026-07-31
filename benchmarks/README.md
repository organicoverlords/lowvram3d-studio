# Benchmark media

This directory keeps lightweight repository-visible previews and immutable provenance manifests.

The full-resolution canonical source images and large GLB masters remain in the local benchmark pack under `C:\AI\LowVRAM3D-benchmarks`. Production benchmark runs resolve those files by manifest name and verify their SHA-256 values before execution.

A preview is never a generation input. It exists only so reviewers can identify a fixture in GitHub.

## What must fail closed

A run must stop when a required canonical source or comparison master is absent, mismatched, ambiguous in identity, corrupt, or produces a demonstrated topology/export regression. Source identity and artifact integrity are hard gates.

A component classifier being uncertain is **not** proof that the generated mesh is corrupt. When cleanup evidence is inconclusive but topology remains safe, the pipeline preserves the generated master byte-for-byte, continues downstream processing, and records `manual_review_required=true`.

Such a run may produce artifacts for inspection, but it cannot silently claim a benchmark PASS. Its benchmark classification remains `REVIEW_REQUIRED` until the final GLB and renders are inspected and the unresolved components are accepted or rejected.

## Mandatory first gate

`antlered_bird_shaman_anchor` is priority 0. Its full source image and verified comparison master must pass before the bird, panda, or any later fixture can produce an overall benchmark PASS.

The verified local shaman source identity is:

- dimensions: `1122x1402`
- SHA-256: `4d23adc758c5b700dd29939e37c043ce61919792b566bdcf13f58b1409d6cf6f`

The previous `eccef854f816f446ce2bf2e08559df519adff223b425c1dccc1c0a9b299f13f6` hash is rejected because it did not match the user's exact local source file.

## TurboBird reference

TurboBird is the user's first real successful result and the current qualitative reference. Every major geometry, cleanup, LOD, texture or rigging change should be checked against the requirement to preserve a TurboBird-class result.

TurboBird is not yet a fully reproduced quantitative benchmark because its exact canonical source/master identity and complete receipt still need consolidation. Until then, document it as `USER_VALIDATED_QUALITATIVE_REFERENCE`, not `PROVEN_QUANTITATIVE_BENCHMARK`.

See `docs/CURRENT_STATE.md` for the latest proof boundary and production rules.
