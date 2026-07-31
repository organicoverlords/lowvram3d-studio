# Stage provenance

Every subprocess stage now seals each declared output with a sibling provenance record:

```text
mesh.glb
mesh.glb.provenance-required
mesh.glb.provenance.json
```

The required marker is written **before** the stage starts and the previous provenance record is removed. This makes an old output fail closed when a rerun crashes, times out, exceeds the VRAM limit, or exits before replacing it.

A successful stage records:

- provenance schema version;
- stage name and complete command;
- canonical command fingerprint;
- working directory and explicit environment overrides;
- SHA-256 hashes for existing file arguments used as stage inputs;
- SHA-256 hash for the output artifact;
- the logical artifact name used by the pipeline receipt.

`artifact_is_valid()` still performs the normal GLB, image, or JSON checks. For newly sealed outputs it also verifies the artifact hash and every recorded input hash. A modified source mesh, source image, workflow, script, or output therefore invalidates checkpoint reuse automatically.

Legacy artifacts without a required marker or provenance sidecar remain readable. The next successful stage execution upgrades them to sealed outputs.

## Failure behavior

The order is deliberate:

1. Arm all declared outputs and remove old provenance sidecars.
2. Discover and hash existing file inputs.
3. Run the stage under the existing timeout, RAM, and VRAM supervision.
4. Validate output formats without trusting provenance yet.
5. Write provenance sidecars atomically.
6. Revalidate the outputs with provenance enabled.
7. Mark the stage passed only after all six steps succeed.

A provenance-write or verification failure is reported as `failure_class=provenance_write`. The artifact remains invalid and cannot be silently reused.

## Scope

This protects deterministic artifact identity and stage-input continuity. It does not claim that an output is visually good; fresh-process Blender validation and benchmark renders remain separate gates.
