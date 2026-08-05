# MV-Adapter step-1 NaN: root cause and repair

**Classification: `MVADAPTER_STEP1_NAN_FIXED_SMOKE_PROVEN`** — 2026-08-03, branch
`integration/unified-pipeline-v2-20260802`.

## Root cause

On **this** machine — GTX 1660 SUPER (compute capability 7.5), torch 2.8.0+cu126, CUDA 12.6,
cuDNN 91002 — an FP16 convolution executed through cuDNN returns partly non-finite output from
entirely finite inputs. This is a defect of this specific runtime stack. It is **not** a claim that
Turing cuDNN FP16 convolutions are broken in general.

The chain that localised it: all UNet inputs finite → raw UNet output non-finite → first
finite-to-non-finite module `down_blocks.0.resnets.0.conv1` (input `finite_fraction` 1.0,
`abs_max` ≈ 3.82; output ≈ 21.9 % NaN) → isolated `Conv2d` reproducer.

The reproducer now runs at startup as `convolution_self_test`, on the exact shape of that module:

| path | finite fraction | max abs error vs FP32 |
| --- | --- | --- |
| FP16, cuDNN on | **0.7493** | 2.38 |
| FP16, cuDNN off | 1.0 | 0.0015 |
| FP32, cuDNN on | 1.0 | — |

## Repair

Every MV-Adapter UNet forward runs under `torch.backends.cudnn.flags(enabled=False)`. The guard
wraps `unet.forward`, so it covers the reference-image pass and every denoising pass through one
mechanism. It is installed only when the self-test says the FP16 cuDNN path is non-finite or
materially inconsistent with FP32, and only after confirming the fallback it selects is itself
finite and close to FP32.

Untouched: the FP32 VAE, the FP32 condition encoder and the text encoder keep cuDNN — the defect is
FP16-only, and disabling it for them would cost speed for nothing. Also unchanged: UNet FP16
weights, VAE FP32, condition encoder FP32, scheduler configuration, guidance and conditioning
scales, adapter and camera contracts.

## Validation

Two bounded runs, six views, seed 12345, two steps, VAE FP32, condition encoder FP32, UNet FP16
with cuDNN disabled, scheduler and conditioning unchanged.

| | 256 / 2 steps | 384 / 2 steps |
| --- | --- | --- |
| reference UNet output finite | yes | yes |
| step-1 scaled input finite | yes | yes |
| raw step-1 UNet output finite | yes | yes |
| step-1 scheduler output finite | yes | yes |
| step-1 latent finite | yes | yes |
| raw step-2 UNet output finite | yes | yes |
| step-2 scheduler output finite | yes | yes |
| final latent finite | yes | yes |
| VAE output finite | yes | yes |
| six PNGs non-black and non-flat | yes | yes |
| views not all identical | yes (6 distinct hashes) | yes (6 distinct hashes) |
| latent hashes change between steps | yes | yes |
| wall clock | 33.7 s | 44.3 s |
| peak VRAM allocated / reserved | 373.7 / 394.0 MB | 828.8 / 892.0 MB |

Both runs report `status: QA_REJECTED`. That is the production QA gate rejecting an image denoised
for 2 of 20 steps on structural IoU and foreground saturation — it is expected, and it is not part
of the numerical claim. The previous failure mode was six *identical black* PNGs; these are six
distinct images with real content.

## Performance impact

There is no end-to-end cuDNN-enabled baseline to compare against, because that path returns NaN.
What is measurable is the convolution itself
(`proof/benchmarks/20260803-mvadapter-cudnn-convolution-benchmark.json`), batch 6, FP16:

| latent | cuDNN on | cuDNN off | factor |
| --- | --- | --- | --- |
| 32×32 (256 px) | 13.35 ms | 21.61 ms | 1.62× |
| 48×48 (384 px) | 27.68 ms | 63.36 ms | 2.29× |

The cuDNN-on column is 75 % finite at both sizes — the benchmark reproduces the defect a third
time, at production batch size.

## Not authorized by this proof

A 384 / 20-step production run. Both smokes pass, which is the precondition for creating a new
384/20 authorization — creating and running it is a separate, explicitly authorized step.
