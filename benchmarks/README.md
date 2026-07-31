# Benchmark media

This directory keeps lightweight repository-visible previews and immutable provenance manifests.

The full-resolution canonical source images and large GLB masters remain in the local benchmark pack under `C:\AI\LowVRAM3D-benchmarks`. Production benchmark runs resolve those files by manifest name and verify their SHA-256 values before execution.

A preview is never a generation input. It exists only so reviewers can identify a fixture in GitHub. A run must fail closed when the canonical source image or master is absent, mismatched, or ambiguous.

## Mandatory first gate

`antlered_bird_shaman_anchor` is priority 0. Its full source image and verified comparison master must pass before the bird, panda, or any later fixture can produce an overall benchmark PASS.
