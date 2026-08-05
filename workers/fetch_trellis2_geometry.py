"""Fetch the geometry-only TRELLIS.2 GGUF weights for trellis.cpp.

Why TRELLIS.2 at all. Every lever inside the Hunyuan3D lane has now been
measured and eliminated -- diffusion steps, octree resolution, conditioning
views, feature-preserving smoothing, model capacity (0.6B vs 1.1B), triangle
count, and finally surface extraction (marching cubes vs dual marching cubes
produced the same mesh). Getting an identical result from a primal and a dual
extractor says the decoded occupancy field is genuinely smooth where the detail
should be, which puts the blame on the latent representation itself.

Hunyuan3D is a **VecSet** model: shape is encoded as an unordered set of latent
vectors with no spatial grounding, so detail is bounded by latent capacity and
no amount of downstream resolution recovers it. TRELLIS.2 is a **structured
latent** (SLAT) model: local latents live on active voxels that intersect the
surface. That is a different representation, not a bigger dial, which is the
only category of change left worth trying.

Why the C++ runtime rather than ComfyUI. The ComfyUI TRELLIS route was already
attempted on this machine and failed in hardware: `flex_gemm` has no sm_75
kernel, and the spconv fallback died with CUDA error 716 (misaligned address).
trellis.cpp is C++/GGML and pins its 3D convolution decoders to the CPU because
ggml has no CUDA CONV_3D kernel -- that limitation is precisely what routes
around the kernel path that crashes here. It also offers a Vulkan backend as a
second escape.

Only the geometry half of the pipeline is fetched, at q8:

    ss_flow           1.282 GB   sparse-structure flow DiT (1.3B)
    ss_dec            0.137 GB   sparse-structure decoder
    shape_flow_512    1.282 GB   shape SLAT flow, res-512 light path
    shape_dec         0.821 GB   FlexiDualGrid VAE
    dinov3            0.301 GB   image conditioning
                      -------
                      3.82 GB

Excluded on purpose:

- `tex_flow_*` / `tex_dec` -- texturing is already solved by the MV-Adapter
  SD2.1 lane against a local SD2.1. `trellis-cli --no-texture` skips them.
- `shape_flow_1024` -- only the `--res 1024` cascade needs it, and 512 is the
  light path this 6 GB card should be tried on first.
- `birefnet` -- the pipeline's own matte (`workers/pipeline_matte.py`) already
  emits alpha, and trellis.cpp's default `--bg-removal auto` keeps the alpha of
  a pre-matted image. Fetching a second matting model would be redundant and
  would risk the fp16-NaN-on-sm_75 defect this project has hit three times.

q8 rather than f16 or q4: f16 is 6.40 GB for the same five files, q4 is 2.45 GB
but quantises a generator we are trying to evaluate on quality. q8 is close to
lossless and its largest single file is 1.28 GB, which leaves the 6 GB card's
budget for activations rather than weights.

    py fetch_trellis2_geometry.py --output C:\\AI\\trellis-cpp\\models
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

REPO = "ilintar/trellis2-gguf"

#: Basenames of the components a geometry-only res-512 run needs. The repo
#: stores each quantisation in its own folder; the runtime wants them flat in
#: one --models directory, so the folder prefix is stripped on write.
COMPONENTS = ("ss_flow", "ss_dec", "shape_flow_512", "shape_dec", "dinov3")

REQUIRED_FREE_GB = 6.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quant", default="q8", choices=("q4", "q8", "f16"))
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    # WinError 1314: this account cannot create symlinks, and the hub cache
    # defaults to symlinking blobs into place. Verified defect on this machine.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    # huggingface_hub has no default stall timeout; a previous large fetch here
    # hung at "4/8 files" with zero bytes moving for ten minutes.
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

    from huggingface_hub import hf_hub_download

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage(output.drive + "\\").free / 1024 ** 3
    if free_gb < REQUIRED_FREE_GB:
        raise SystemExit(
            f"INSUFFICIENT_DISK: {free_gb:.1f} GB free on {output.drive}, "
            f"need {REQUIRED_FREE_GB}")

    print(f"free before: {free_gb:.1f} GB", flush=True)

    fetched = []
    for name in COMPONENTS:
        # f16 lives at the repo root; the quantised sets live in a subfolder.
        remote = f"{name}.gguf" if args.quant == "f16" else f"{args.quant}/{name}.gguf"
        print(f"  fetching {remote}", flush=True)
        cached = hf_hub_download(repo_id=REPO, filename=remote)
        target = output / f"{name}.gguf"
        # Copy rather than move: the hub cache is content-addressed and shared,
        # and moving out of it leaves the cache believing the file is present.
        shutil.copyfile(cached, target)
        fetched.append({"component": name,
                        "remote": remote,
                        "local": str(target),
                        "gb": round(target.stat().st_size / 1024 ** 3, 3)})

    after_gb = shutil.disk_usage(output.drive + "\\").free / 1024 ** 3
    receipt = {
        "schema_version": "trellis2_geometry_fetch_v1",
        "repo": REPO,
        "quant": args.quant,
        "models_dir": str(output),
        "files": fetched,
        "total_gb": round(sum(f["gb"] for f in fetched), 3),
        "free_gb_after": round(after_gb, 2),
        "excluded": ["tex_flow_512", "tex_flow_1024", "tex_dec",
                     "shape_flow_1024", "birefnet"],
        "usage": ("trellis-cli <image.png> <out.glb> --models <models_dir> "
                  "--res 512 --no-texture"),
        "note": ("geometry only; texturing stays on the MV-Adapter SD2.1 lane. "
                 "Input must be pre-matted (alpha) so --bg-removal auto skips "
                 "BiRefNet, which is not fetched."),
    }
    if args.receipt:
        Path(args.receipt).write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
