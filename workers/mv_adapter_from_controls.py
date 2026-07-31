from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

VIEW_ORDER = ("front", "right", "back", "left", "top", "bottom")


def image_tensor(torch, path: Path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--controls-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--base-model", default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--adapter", default="huanngzh/mv-adapter")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=6.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import sys
    sys.path.insert(0, args.repo)
    import torch
    from diffusers import DDPMScheduler
    from mvadapter.models.attention_processor import DecoupledMVRowColSelfAttnProcessor2_0
    from mvadapter.pipelines.pipeline_mvadapter_t2mv_sd import MVAdapterT2MVSDPipeline
    from mvadapter.schedulers.scheduling_shift_snr import ShiftSNRScheduler

    if not torch.cuda.is_available():
        raise RuntimeError("MV-Adapter lane requires CUDA; use projection fallback when unavailable")
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    dtype = torch.float16
    pipe = MVAdapterT2MVSDPipeline.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=args.offline,
    )
    pipe.scheduler = ShiftSNRScheduler.from_scheduler(
        pipe.scheduler,
        shift_mode="interpolated",
        shift_scale=8.0,
        scheduler_class=DDPMScheduler,
    )
    pipe.init_custom_adapter(num_views=6, self_attn_processor=DecoupledMVRowColSelfAttnProcessor2_0)
    pipe.load_custom_adapter(
        args.adapter,
        weight_name="mvadapter_tg2mv_sd21.safetensors",
        local_files_only=args.offline,
    )
    pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("max")
    pipe.enable_model_cpu_offload()
    # _init_custom_adapter builds the T2IAdapter condition encoder at the default float32 and
    # never casts it to the pipeline dtype, so the first conv raises
    # "Input type (c10::Half) and bias type (float) should be the same". It is also not a
    # registered pipeline component, so enable_model_cpu_offload() never places it either;
    # pin it to the compute device explicitly. The encoder is small, so this costs little VRAM.
    if getattr(pipe, "cond_encoder", None) is not None:
        pipe.cond_encoder.to(device="cuda", dtype=dtype)

    controls_dir = Path(args.controls_dir)
    controls = []
    for name in VIEW_ORDER:
        position = image_tensor(torch, controls_dir / f"{name}_position.png")
        normal = image_tensor(torch, controls_dir / f"{name}_normal.png")
        controls.append(torch.cat((position, normal), dim=0))
    control_tensor = torch.stack(controls, dim=0)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    started = time.time()
    result = pipe(
        args.prompt,
        height=512,
        width=512,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        num_images_per_prompt=6,
        control_image=control_tensor,
        control_conditioning_scale=1.0,
        negative_prompt="watermark, text, deformed, noisy, blurry, inconsistent materials, unrelated details",
        generator=generator,
    ).images
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, image in zip(VIEW_ORDER, result):
        image.save(output_dir / f"{name}.png")
    width, height = result[0].size
    contact = Image.new("RGB", (width * len(result), height))
    for index, image in enumerate(result):
        contact.paste(image, (index * width, 0))
    contact.save(output_dir / "contact_sheet.png")
    (output_dir / "worker_receipt.json").write_text(json.dumps({
        "success": True,
        "backend": "mv_adapter_sd21_tg2mv",
        "view_order": VIEW_ORDER,
        "duration_seconds": round(time.time() - started, 2),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "peak_allocated_mb": int(torch.cuda.max_memory_allocated() / 1024 / 1024),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
