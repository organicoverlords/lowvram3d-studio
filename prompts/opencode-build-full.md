Work on the installed LowVRAM 3D Studio pipeline through the 3D Gen Studio MCP control layer.

Hardware truth:
- GTX 1660 SUPER 6 GB, sm_75, 16 GB RAM.
- Mini Turbo geometry is proven on the actual machine.
- TRELLIS2 is rejected.
- Do not introduce Tensor-Core-only, FlashAttention-only, Triton-only, CuMesh, SkinTokens, UniRig, or 8 GB+ baseline dependencies.

Control architecture:
- 3D Gen Studio owns projects, graph nodes, cards, assets, versions, previews, and exports.
- Local providers: LowVRAM Generate, LowVRAM Texture (lanes A/B/C), LowVRAM Rig + Game Ready.
- Lane A: Mini Turbo + MV-Adapter SD2.1 + Blender.
- Lane B: Mini Turbo + deterministic source projection + Blender.
- Lane C: TripoSR emergency textured proxy + Blender.

Before edits, verify repository, branch, HEAD, remote, dirty state, installed config, service health, and the actual asset path. Manual output proof wins.

Success requires a real final GLB that survives clean Blender re-import, plus preview.png, validation.json, maps, parts.json, rig_report.json, LODs, collision names, and job_receipt.json. Unit tests or placeholder files are not proof.

Never modify the user's existing ComfyUI environment packages. Heavy workers remain isolated. Do not merge to main.
