# 3D Gen Studio / ComfyUI rig integration — reuse before rewrite

The rigging lane uses 3D Gen Studio's existing workflow orchestration instead of adding another independent ComfyUI client.

Pinned Studio reference: `visualbruno/3DGenStudio@4c2c3f8da9cbd4ef04ebf59f79738be7c9d774ad`, `mcp/tools/workflows.js`.

## Existing known solution

3D Gen Studio already exposes MCP tools that cover the integration work we need:

1. `inspect_workflow` — inspect a ComfyUI **API-format** graph and enumerate literal inputs and terminal outputs.
2. `import_workflow` — save the graph to Studio's workflow library and declare mesh/image/string/number/boolean parameters plus result nodes.
3. `run_workflow` — upload a local mesh with `fileInputs` or use a Studio project asset id, submit to the configured ComfyUI, stream progress, persist results, and attach them to graph/kanban nodes.
4. `get_run_status` — poll a workflow that outlives the initial timeout.

`run_workflow` also already handles prompt ids, SSE progress, file upload, asset/version ancestry and result attachment. None of that should be reimplemented in lowvram3d unless the stock Studio path is proven inadequate.

## Rigging sequence

The first executable proof remains upstream ComfyUI-UniRig itself:

`vendor realistic_male_character.glb -> stock MIA FP16 -> vendor preview/deformation evidence`

Only after that succeeds do we import the same stock-node graph into Studio and prove Studio orchestration:

`Studio mesh asset -> Studio run_workflow -> configured ComfyUI -> stock MIA -> rigged result attached back to Studio`

Then repeat on one preserved production humanoid from:

`TRELLIS -> native Stage 6 -> Hunyuan Paint -> Blender QA`

## Workflow-format boundary

The `PozzettiAndrea/ComfyUI-UniRig` repository ships normal ComfyUI UI workflow JSON (`nodes`/`links`). Studio's `inspect_workflow` explicitly consumes ComfyUI **API-format** graphs (`node id -> {class_type, inputs}`). Do not silently invent an algorithmic rewrite here.

For the first stock test, open/run the upstream workflow as shipped in ComfyUI. Once that passes, export the same graph using ComfyUI's API-format workflow export and feed that exported graph to Studio's `inspect_workflow` / `import_workflow`. The exported API graph becomes a provenance artifact next to the pinned upstream UI graph.

The only allowed configuration change in the first 6 GB MIA benchmark is selecting the stock node's supported `fp16` precision setting. No MIA/UniRig model, skeleton, skinning or attention code is modified.

## Promotion gates

Studio integration is not PROVEN because a workflow appears in its library. It must show:

- the upstream MIA graph completed through Studio;
- the output armature and skin weights exist;
- source texture/materials survive;
- peak dedicated VRAM and shared-memory spill are recorded;
- five-pose Blender deformation evidence passes;
- one stock animation is retargeted/applied using the existing Studio/UniRig path;
- the rigged/animated asset is attached back into the Studio project without losing provenance.

Only after one of those stock stages fails visibly or measurably do we diagnose and consider a custom replacement.
