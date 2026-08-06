"""Generate geometry from several named views with Hunyuan3D-2mv Turbo.

The Lucky Drown boat came with a reference sheet carrying a 3/4 hero view, plus
labelled FRONT, SIDE, BACK, a cutaway section and a deck plan. Everything
downstream was built from a crop of the hero panel alone -- a strongly oblique
view, which is the worst single view to hand a single-image reconstructor if you
want a hull. The result was a skewed parallelogram hull with a featureless lid
where the deck should be. Texturing was never the problem; the geometry was.

`hunyuan3d-dit-v2-mv-turbo` takes named views rather than one image, and it was
already sitting in the local HF tree, untested. The orthographic FRONT, SIDE and
BACK panels are exactly what it wants, and they were sitting unused in the same
reference sheet the hero crop came from.

From the operator's side this is still one picture in: the panels are extracted
from the sheet automatically, at a common scale, by
`workers/extract_orthographic_panels.py`.

    py workers/hunyuan_mv_shape.py --front f.png --left s.png --back b.png \
       --out boat_mv.glb

View tags come from MVImageProcessorV2: front=0, left=1 (front clockwise 90),
back=2, right=3. Which physical side of an asymmetric subject a given SIDE panel
represents is genuinely ambiguous from the sheet, so it is a flag, not a guess
buried in the code.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

MODEL_ROOT = Path(r"C:\AI\HY3D2\HuggingFaceHub\hunyuan3d-2mv-direct")
SUBFOLDER = "hunyuan3d-dit-v2-mv-turbo"

#: The distilled VAE the Lightning decoder was trained against. It sits in the
#: same local tree as the DiT and was never being loaded -- see load_turbo_vae.
TURBO_VAE_SUBFOLDER = "hunyuan3d-vae-v2-0-turbo"

#: Tencent's own Turbo defaults. Deliberately not tuned before a first result
#: exists -- this project has lost more time to premature tuning than to any
#: model limitation.
STEPS = 5
GUIDANCE = 5.0
OCTREE = 256
CHUNKS = 1500


def load_turbo_vae(pipeline, torch) -> dict:
    """Swap in the distilled VAE that the Lightning decoder actually needs.

    `pipeline.enable_flashvdm()` is supposed to do this itself, but it decides
    which turbo VAE to fetch by looking `model_path.split('/')[-1]` up in a
    three-entry dict of Hugging Face repo names. A local directory is never in
    that dict, so the lookup misses, the swap is skipped **silently**, and the
    Lightning decoder is left running against the undistilled VAE that shipped
    inside the DiT checkpoint.

    That is the whole "FlashVDM crash". The adaptive-KV decoder selects a small
    top-k of latents per query; against a decoder that was not distilled for it
    the resulting occupancy field has no zero crossing, the near-surface
    extraction returns an empty candidate set, and the hierarchy dies on
    `next_points.min(axis=0)` with "Expected reduction dim 0 to have non-zero
    size" -- after the diffusion has already been paid for. Nothing about it is
    specific to this GPU.

    The two configs differ in exactly the places that matter to that decoder:
    `geo_decoder_downsample_ratio` 2 against 1, `geo_decoder_mlp_expand_ratio`
    1 against 4, `geo_decoder_ln_post` false against true.
    """
    from hy3dgen.shapegen.models import ShapeVAE

    before = type(pipeline.vae.geo_decoder).__name__
    pipeline.vae = ShapeVAE.from_pretrained(
        str(MODEL_ROOT), subfolder=TURBO_VAE_SUBFOLDER,
        use_safetensors=True, device="cuda")
    pipeline.vae.to("cuda", dtype=torch.float16)
    return {
        "turbo_vae_subfolder": TURBO_VAE_SUBFOLDER,
        "geo_decoder_before": before,
        "geo_decoder_downsample_ratio":
            int(getattr(pipeline.vae.geo_decoder, "downsample_ratio", 1)),
    }


def run(views: dict, out_path: Path, steps: int, guidance: float,
        octree: int, chunks: int, seed: int,
        flashvdm: bool = False, mc_algo: str = "dmc",
        offload: bool = True, topk_mode: str = "mean") -> dict:
    import torch

    torch.backends.cudnn.enabled = False  # FP16 cuDNN convolution is defective here

    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    started = time.time()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(MODEL_ROOT), subfolder=SUBFOLDER, use_safetensors=True, device="cuda")
    vae_receipt = {"turbo_vae_subfolder": None}
    if flashvdm:
        # Order matters: load the distilled VAE first, then attach the decoder
        # to it, and keep enable_flashvdm's own VAE swap switched off so it
        # cannot quietly undo this or reach for the network.
        vae_receipt = load_turbo_vae(pipeline, torch)
        # topk_mode picks the attention processor the Lightning decoder uses,
        # and the two are not interchangeable on this card. 'mean' gathers a
        # FIXED top-k of latents (1024 of 3072), so every tensor it builds has
        # a static, aligned length. 'merge' gathers a DATA-DEPENDENT set --
        # torch.where(sim > 1e-6) then torch.unique -- whose length is whatever
        # the data says, and at octree 384 that lands on a count the fp16
        # gather kernel cannot address: "CUDA error: misaligned address",
        # raised after the coarse level has already decoded. 'mean' is also the
        # vendor default; 'merge' was chosen here for no reason at all.
        pipeline.enable_flashvdm(topk_mode=topk_mode, mc_algo=mc_algo,
                                 replace_vae=False)

    # The DiT is 4.7 GB fp16 and the conditioner is DINOv2-giant, 40 layers at
    # width 1536. Resident together they do not fit in 6 GB -- and on Windows
    # that does not raise, it silently pages into shared system memory over
    # PCIe. The symptom is a diffusion loop that runs at ~109 s/step with wild
    # per-step jitter (167s, 74s, 82s, 85s, 189s) while the GPU is mostly idle.
    # Offloading holds one stage on the card at a time, which is the difference
    # between a resident model and a thrashing one, not a quality setting.
    #
    # `enable_model_cpu_offload` is copied from diffusers but only half ported,
    # and both gaps have to be closed from out here rather than in the vendor
    # tree. It reads `self.components`, which the diffusers base class provides
    # and this one never defines -- an AttributeError eight seconds in. And it
    # ends with `self.to("cpu")`, which sets `self.device` to cpu, while
    # `__call__` still creates its latents on `self.device`; the class carries a
    # correct `_execution_device` property and does not use it. Restoring
    # `device` after the hooks are attached is what makes the two agree: the
    # accelerate hooks place the modules, `device` only places the tensors.
    if offload:
        pipeline.components = {"conditioner": pipeline.conditioner,
                               "model": pipeline.model,
                               "vae": pipeline.vae}
        pipeline.enable_model_cpu_offload()
        pipeline.device = torch.device("cuda")
    loaded = time.time()

    images = {tag: Image.open(path).convert("RGBA") for tag, path in views.items()}
    mesh = pipeline(
        image=images,
        num_inference_steps=steps,
        guidance_scale=guidance,
        octree_resolution=octree,
        num_chunks=chunks,
        generator=torch.manual_seed(seed),
    )[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    finished = time.time()

    return {
        "schema": "lowvram3d_hunyuan_mv_shape_v1",
        "checkpoint": SUBFOLDER,
        "views": {k: str(v) for k, v in views.items()},
        "mesh_out": str(out_path),
        "steps": steps, "guidance_scale": guidance,
        "octree_resolution": octree, "num_chunks": chunks, "seed": seed,
        "flashvdm": bool(flashvdm),
        "mc_algo": mc_algo if flashvdm else "mc",
        "topk_mode": topk_mode if flashvdm else None,
        "model_cpu_offload": bool(offload),
        **vae_receipt,
        "faces": int(len(mesh.faces)),
        "load_seconds": round(loaded - started, 1),
        "shape_seconds": round(finished - loaded, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", required=True, type=Path)
    parser.add_argument("--left", type=Path, default=None)
    parser.add_argument("--back", type=Path, default=None)
    parser.add_argument("--right", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--guidance", type=float, default=GUIDANCE)
    parser.add_argument("--octree", type=int, default=OCTREE)
    parser.add_argument("--chunks", type=int, default=CHUNKS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--flashvdm", action="store_true",
                        help="Lightning decoder; also loads the distilled turbo VAE")
    parser.add_argument("--mc-algo", default="dmc", choices=("mc", "dmc"),
                        help="surface extractor used with --flashvdm")
    parser.add_argument("--topk-mode", default="mean", choices=("mean", "merge"),
                        help="'merge' builds variable-length gathers and faults here")
    parser.add_argument("--no-offload", action="store_true",
                        help="keep every stage resident; thrashes on 6 GB")
    args = parser.parse_args(argv)

    views = {tag: getattr(args, tag) for tag in ("front", "left", "back", "right")
             if getattr(args, tag) is not None}
    result = run(views, args.out, args.steps, args.guidance,
                 args.octree, args.chunks, args.seed,
                 flashvdm=args.flashvdm, mc_algo=args.mc_algo,
                 offload=not args.no_offload, topk_mode=args.topk_mode)
    args.out.with_suffix(".mv.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
