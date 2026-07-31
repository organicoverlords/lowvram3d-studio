# Hardware-focused research

Target machine:

- GTX 1660 SUPER, 6 GB VRAM, Turing `sm_75`, no Tensor Cores;
- 16 GB system RAM and 32 GB pagefile;
- Windows 10;
- existing ComfyUI Python 3.12 / PyTorch 2.8 CUDA 12.8.

## Accepted

- **Hunyuan3D Mini Turbo geometry:** manually proven on the target machine with an approximately 1.8-million-face bird.
- **Early mesh reduction:** mandatory before UV, maps or rigging. The installer targets 50,000 faces while preserving the high-poly source as an intermediate.
- **MV-Adapter SD2.1 TG2MV:** selected as the preferred texture experiment because it uses Stable Diffusion 2.1 and has an official low-resource path. It is isolated from ComfyUI and stripped of the `nvdiffrast` control-rendering dependency by using Blender renders.
- **Blender CPU processing:** selected for UV, projection, map baking, parts, rigging, collisions, LODs and export.
- **TripoSR:** selected as the second complete alternative because it can generate a baked texture and exposes chunk-size controls.

## Rejected from baseline

- TRELLIS2 and TRELLIS2 GGUF: failed repeatedly in dequantization, shape-slat flow and decoder paths on the real stack.
- Hunyuan Paint: not selected because texture-stage memory and custom renderer compatibility are not sufficiently reliable on this card.
- SkinTokens and UniRig: published/runtime memory requirements exceed this machine.
- PartCrafter/P3-SAM: not a 6 GB baseline.
- FlashAttention, SageAttention, Tensor-Core-specific kernels, CuMesh and unverified precompiled `sm_75` extensions.

## Compatibility rules

- No dependency upgrades inside the working ComfyUI environment.
- Every heavy model has its own virtual environment.
- `TORCH_CUDA_ARCH_LIST=7.5` is used if any extension must be compiled.
- CPU mesh tools are installed with `MESHTOOLS_SKIP_GPU=1`.
- One GPU worker at a time.
- A README or nominal model size is never accepted as runtime proof.
