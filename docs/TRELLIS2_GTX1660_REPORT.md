# TRELLIS2 GGUF on GTX 1660 SUPER — rejected production backend

## Target machine

- NVIDIA GeForce GTX 1660 SUPER, 6 GB VRAM
- Turing compute capability **7.5 / sm_75**
- 15 GB system RAM and 32 GB page file
- Python 3.12 embedded
- PyTorch 2.8.0 + CUDA 12.8
- ComfyUI_Trellis2_GGUF and ComfyUI-GGUF Q4_K_M models

## Proven fixes

1. Restoring ComfyUI-GGUF resolved `GGMLTensor.copy_` / sparse 0D tensor loading failure.
2. Calling `pipeline.load_image_cond_model()` before `pipeline.get_cond()` resolved the missing image-condition model failure.
3. The all-in-one generator completed sparse-structure generation and decoded a 64³ voxel structure.

## Proven failures

- Modular workflow without `--novram`: Windows error 1455 while loading DINOv3.
- Modular workflow with `--novram`: fatal CUDA failure in GGUF dequantization during sparse-structure inference.
- Basic and Advanced all-in-one workflows: fatal CUDA failure while loading the 512 shape-slat flow model.
- Decoder experiments: memory exhaustion during the high-resolution VAE decode.
- Tiled decode, disabled texture generation, zero texture steps, and internal low-VRAM mode did not bypass the failing stage.

## Diagnosis boundaries

The observed failures are consistent with a combination of:

- insufficient VRAM and host-memory headroom;
- high activation and sparse-token memory pressure;
- unstable or unsupported Triton/GGUF CUDA paths on `sm_75`;
- model offload occurring too late to prevent load-time failure.

The available evidence does **not** prove that every fatal `c10_cuda.dll` abort was exclusively an OOM. Fatal CUDA aborts can also result from unsupported kernels, invalid device code, or extension/runtime incompatibility. The earlier `CUDA error 209` evidence strongly supports an independent architecture-compatibility blocker.

The claim that a dense attention matrix alone necessarily exceeds 6 GB is also not established: optimized attention implementations may avoid materializing the full matrix. The total runtime footprint is still incompatible with this machine in the tested stack.

## Decision

**REJECTED as a production backend on this machine and software stack.**

Do not spend further repair cycles on TRELLIS2 GGUF unless one of these materially changes:

- GPU with at least 12 GB VRAM;
- an officially supported `sm_75` inference path with no Triton-only kernels;
- a substantially smaller shape-slat model;
- a proven CPU decoder/offload implementation;
- a different machine with adequate RAM and VRAM.

The production stack remains:

```text
Mini Turbo geometry
  -> early reduction to 30k–50k faces
  -> MV-Adapter / projection-based texturing
  -> Blender finalization and validation
```
