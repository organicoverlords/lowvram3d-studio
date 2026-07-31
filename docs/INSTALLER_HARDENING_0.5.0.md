# Execution hardening 0.5.0

Version 0.5.0 changes installation and asset processing from rerunnable scripts into checkpointed workflows.

## Installer checkpoints

Each installer stage writes an atomic JSON checkpoint under:

```text
%LOCALAPPDATA%\LowVRAM3DStudio\install-state\stages
```

A stage is skipped only when its fingerprint matches and its readiness probe still passes. Existing work may be adopted only for stages whose readiness probe is strong and whose outputs are safe across package upgrades, such as pinned source checkouts and isolated Python environments.

Version-sensitive stages are never adopted from an old proof file. Configuration generation, service smoke tests, the ComfyUI bridge, package tests, and desktop shortcuts rerun when their package fingerprint changes.

Large operations are preserved:

- `uv pip install --python <venv-python>` targets the existing environment and reuses uv's cache.
- 3D Gen Studio does not run destructive `npm ci` when its dependency tree is already importable.
- Only `sqlite3` is rebuilt when its native install script was previously skipped.
- Hugging Face downloads use the persistent `HF_HOME` cache and retry only models that are still missing.
- Optional TripoSR, model-cache, or ComfyUI-bridge failure does not block the primary Mini Turbo and deterministic post-process stack. It is recorded as `degraded` and normal resumes do not repeat it; `RETRY-OPTIONAL.cmd` retries degraded optional stages explicitly.
- Model receipts are not trusted alone: an offline Hugging Face cache probe verifies each required snapshot before the model stage is reused.

## Asset-job checkpoints

Every full or post-process job uses one stable job ID. Successful stage artifacts remain in the same directory and are reused after a failure.

Resume is available through three paths:

1. rerun the same failed card in 3D Gen Studio;
2. call `POST /v1/jobs/<job-id>/resume`;
3. use the desktop `Resume Last 3D Job` shortcut.

The worker stores source hashes and canonical processing-setting fingerprints. Resume is rejected when the image, source mesh, asset class, quality, texture resolution, LOD policy, splitting policy, or prompt changed. This prevents a new request from being combined with stale retopo, UV, or bake outputs.

Legacy jobs without fingerprints can be adopted only when their stored profile and options agree with the requested settings.

## Artifact validation

A file existing is not sufficient proof. Reuse validates:

- GLB magic, version, and declared file length;
- JSON parseability;
- PNG/JPEG signatures;
- non-empty remaining artifacts.

A truncated GLB, invalid image payload, failed JSON report, or broken report invalidates that stage and causes only that stage to rerun. GLB validation parses the JSON chunk and checks every chunk boundary against the declared file length.

## Runtime controls

Mini Turbo, MV-Adapter, TripoSR, and proxy generation share one worker-level GPU lock. CPU retopo, UV, analysis, and reporting are not serialized unnecessarily.

Stage failures record:

- command and exit code;
- stdout/stderr locations and a bounded log tail;
- peak process-tree RAM;
- peak total GPU memory;
- failure class: process launch, timeout, VRAM ceiling, CUDA OOM, unsupported CUDA kernel, missing dependency, network interruption, non-zero exit, or missing artifact.

## 3D Gen Studio startup

The pinned 3D Gen Studio revision uses Node for the local backend and SQLite for project storage. The installer installs dependencies with scripts disabled, explicitly rebuilds only SQLite, does not execute Electron's unused postinstall, verifies `require('sqlite3')`, builds the frontend, then launches three temporary services on free ports:

- 3D Gen Studio backend;
- CPU mesh tools;
- LowVRAM worker.

The installed Studio runtime uses dedicated port `8311`, avoiding the user's existing router service on port `3001`. Vite 8 compatibility is checked before the Studio dependency/build stage (Node 20.19+, 22.12+, or a newer compatible major).

The stage passes only when all three health/API endpoints answer and all smoke processes are shut down cleanly.

## Research basis

- uv environment targeting and cache behavior: official uv documentation.
- package-lock installation, install-script policy, and package-specific rebuilds: official npm CLI documentation.
- persistent resumable model cache: official Hugging Face Hub documentation.
- service ports and mesh-tool contracts: pinned 3D Gen Studio source.
- process lifecycle and stream redirection behavior: Microsoft PowerShell documentation.

## Proof boundary

These controls are unit- and integration-tested in the build environment. Windows PowerShell execution and real Blender/MV-Adapter processing still require the target-PC run. A passing installer does not by itself prove visual quality or rig deformation.
