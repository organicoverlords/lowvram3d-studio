"""Minimal SD2.1 image-to-multiview camera runtime.

This module reproduces only the camera/control-image and source-image preparation
used by MV-Adapter's official ``inference_i2mv_sd.py``. It deliberately avoids
``mvadapter.utils`` because that package eagerly imports optional texturing
stacks (nvdiffrast and Triton) that are not used by image-to-multiview inference.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


AZIMUTHS = (0, 45, 90, 180, 270, 315)


def build_camera_to_world(num_views: int, device: str):
    import torch
    import torch.nn.functional as F

    if num_views != len(AZIMUTHS):
        raise ValueError(f"camera runtime requires {len(AZIMUTHS)} views")

    azimuth_deg = torch.tensor(
        [value - 90 for value in AZIMUTHS], dtype=torch.float32, device=device
    )
    elevation = torch.zeros(num_views, dtype=torch.float32, device=device)
    distance = torch.full((num_views,), 1.8, dtype=torch.float32, device=device)
    azimuth = azimuth_deg * torch.pi / 180.0

    positions = torch.stack(
        [
            distance * torch.cos(elevation) * torch.cos(azimuth),
            distance * torch.cos(elevation) * torch.sin(azimuth),
            distance * torch.sin(elevation),
        ],
        dim=-1,
    )
    centre = torch.zeros_like(positions)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=device)[None, :].repeat(
        num_views, 1
    )
    lookat = F.normalize(centre - positions, dim=-1)
    right = F.normalize(torch.cross(lookat, world_up, dim=-1), dim=-1)
    up = F.normalize(torch.cross(right, lookat, dim=-1), dim=-1)

    c2w_3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), positions[:, :, None]], dim=-1
    )
    c2w = torch.cat([c2w_3x4, torch.zeros_like(c2w_3x4[:, :1])], dim=1)
    c2w[:, 3, 3] = 1.0
    return c2w


def build_orthographic_control_images(num_views: int, image_size: int, device: str):
    import torch
    import torch.nn.functional as F

    c2w = build_camera_to_world(num_views, device)
    controls = []
    for matrix in c2w:
        world_to_camera = torch.linalg.inv(matrix).clone()
        world_to_camera[1, :] *= -1
        world_to_camera[2, :] *= -1
        rotation = world_to_camera[:3, :3]
        translation = world_to_camera[:3, 3]
        camera_position = -rotation.T @ translation
        view_direction = rotation.T @ torch.tensor(
            [0.0, 0.0, 1.0], dtype=torch.float32, device=device
        )
        camera_position = F.normalize(camera_position, dim=0)
        plucker = torch.cat([view_direction, camera_position])
        controls.append(plucker[:, None, None].repeat(1, image_size, image_size))
    embeds = torch.stack(controls)
    return ((embeds + 1.0) / 2.0).clamp(0, 1)


def preprocess_rgba(image: Image.Image, height: int, width: int) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA"))
    alpha = rgba[..., 3] > 0
    y, x = np.where(alpha)
    if not len(x):
        raise RuntimeError("source image alpha is fully transparent")
    y0, y1 = max(int(y.min()) - 1, 0), min(int(y.max()) + 1, rgba.shape[0])
    x0, x1 = max(int(x.min()) - 1, 0), min(int(x.max()) + 1, rgba.shape[1])
    subject = rgba[y0:y1, x0:x1]
    subject_h, subject_w = subject.shape[:2]
    if subject_h > subject_w:
        resized_h = int(height * 0.9)
        resized_w = max(1, int(subject_w * resized_h / subject_h))
    else:
        resized_w = int(width * 0.9)
        resized_h = max(1, int(subject_h * resized_w / subject_w))
    subject = np.asarray(Image.fromarray(subject).resize((resized_w, resized_h)))
    canvas = np.zeros((height, width, 4), dtype=np.uint8)
    start_y = (height - resized_h) // 2
    start_x = (width - resized_w) // 2
    canvas[start_y : start_y + resized_h, start_x : start_x + resized_w] = subject
    normalised = canvas.astype(np.float32) / 255.0
    rgb = normalised[..., :3] * normalised[..., 3:4] + (1.0 - normalised[..., 3:4]) * 0.5
    return Image.fromarray((rgb * 255.0).clip(0, 255).astype(np.uint8))


def prepare_reference_image(path: Path, height: int, width: int) -> Image.Image:
    image = Image.open(path)
    if image.mode == "RGBA" or "transparency" in image.info:
        return preprocess_rgba(image, height, width)
    return image.convert("RGB")


def _expected_reference_processor_names(pipe: Any) -> tuple[str, ...]:
    names = {
        str(getattr(processor, "name"))
        for processor in pipe.unet.attn_processors.values()
        if bool(getattr(processor, "use_ref", False))
        and getattr(processor, "name", None) is not None
    }
    if not names:
        raise RuntimeError("MV-Adapter has no reference-enabled attention processors")
    return tuple(sorted(names))


def install_reference_cache_relay(pipe: Any) -> dict[str, Any]:
    """Preserve the proven reference cache until the first denoising UNet call.

    On the installed MV-Adapter/Diffusers combination the reference UNet pass
    populates all custom-attention cache entries, but the pipeline later supplies
    an empty ``ref_hidden_states`` dictionary to denoising. This relay stores only
    the reference-enabled entries immediately after the reference pass and
    reconstructs the exact batch expansion intended by the upstream pipeline.

    It never fabricates features: missing expected entries fail closed. A
    non-empty upstream cache is validated and passed through unchanged.
    """

    existing = getattr(pipe, "_lowvram3d_reference_cache_relay_state", None)
    if isinstance(existing, dict):
        return existing

    import torch

    expected_names = _expected_reference_processor_names(pipe)
    original_forward = pipe.unet.forward
    raw_cache: dict[str, Any] = {}
    state: dict[str, Any] = {
        "installed": True,
        "expected_reference_count": len(expected_names),
        "expected_reference_names": list(expected_names),
        "reference_cache_observed": False,
        "reference_cache_count": 0,
        "reference_cache_missing": [],
        "denoise_cache_count_before": None,
        "denoise_cache_count_after": None,
        "relay_injected": False,
        "relay_mode": "pending",
        "num_views": None,
        "classifier_free_guidance": None,
        "expanded_batch_sizes": {},
    }

    def relay_forward(*args, **kwargs):
        cross_kwargs = kwargs.get("cross_attention_kwargs")
        if not isinstance(cross_kwargs, dict):
            return original_forward(*args, **kwargs)

        cache_target = cross_kwargs.get("cache_hidden_states")
        if isinstance(cache_target, dict):
            result = original_forward(*args, **kwargs)
            missing = [name for name in expected_names if name not in cache_target]
            state["reference_cache_observed"] = True
            state["reference_cache_count"] = len(cache_target)
            state["reference_cache_missing"] = missing
            if missing:
                raise RuntimeError(
                    "reference UNet cache is missing required MV entries: "
                    + ", ".join(missing[:4])
                )
            raw_cache.clear()
            raw_cache.update({name: cache_target[name] for name in expected_names})
            state["relay_mode"] = "reference_cache_captured"
            print(
                f"REFERENCE_CACHE_RELAY_CAPTURED={len(raw_cache)}",
                flush=True,
            )
            return result

        if "ref_hidden_states" in cross_kwargs:
            supplied = cross_kwargs.get("ref_hidden_states")
            if not isinstance(supplied, dict):
                raise RuntimeError("denoising ref_hidden_states is not a dictionary")

            state["denoise_cache_count_before"] = len(supplied)
            if supplied:
                missing = [name for name in expected_names if name not in supplied]
                if missing:
                    raise RuntimeError(
                        "upstream denoising cache is incomplete: "
                        + ", ".join(missing[:4])
                    )
                state["denoise_cache_count_after"] = len(supplied)
                state["relay_mode"] = "upstream_cache_passthrough"
                return original_forward(*args, **kwargs)

            if not raw_cache:
                raise RuntimeError(
                    "denoising reference cache is empty and no captured cache is available"
                )

            num_views = int(cross_kwargs.get("num_views", len(AZIMUTHS)))
            if num_views != len(AZIMUTHS):
                raise RuntimeError(
                    f"unexpected MV view count for cache relay: {num_views}"
                )

            do_cfg = bool(pipe.do_classifier_free_guidance)
            expanded: dict[str, Any] = {}
            batch_sizes: dict[str, int] = {}
            for name in expected_names:
                value = raw_cache[name]
                if not isinstance(value, torch.Tensor):
                    raise RuntimeError(f"reference cache entry is not a tensor: {name}")
                if value.shape[0] != 1:
                    raise RuntimeError(
                        f"unexpected reference cache batch for {name}: {value.shape[0]}"
                    )
                value = value.repeat_interleave(num_views, dim=0)
                if do_cfg:
                    value = torch.cat([torch.zeros_like(value), value], dim=0)
                expanded[name] = value
                batch_sizes[name] = int(value.shape[0])

            cross_kwargs["ref_hidden_states"] = expanded
            raw_cache.clear()
            state["relay_injected"] = True
            state["relay_mode"] = "captured_cache_injected"
            state["num_views"] = num_views
            state["classifier_free_guidance"] = do_cfg
            state["denoise_cache_count_after"] = len(expanded)
            state["expanded_batch_sizes"] = batch_sizes
            print(
                f"REFERENCE_CACHE_RELAY_INJECTED={len(expanded)}",
                flush=True,
            )

        return original_forward(*args, **kwargs)

    pipe.unet.forward = relay_forward
    pipe._lowvram3d_reference_cache_relay_state = state
    return state


def run_i2mv_pipeline(
    pipe: Any,
    *,
    source_image: Path,
    text: str,
    negative_prompt: str,
    height: int,
    width: int,
    steps: int,
    seed: int,
    device: str = "cuda",
):
    import torch

    relay_state = install_reference_cache_relay(pipe)
    print(
        f"REFERENCE_CACHE_RELAY_EXPECTED={relay_state['expected_reference_count']}",
        flush=True,
    )

    control_images = build_orthographic_control_images(len(AZIMUTHS), width, device)
    reference = prepare_reference_image(source_image, height, width)
    generator = torch.Generator(device=device).manual_seed(seed)
    images = pipe(
        text,
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=3.0,
        num_images_per_prompt=len(AZIMUTHS),
        control_image=control_images,
        control_conditioning_scale=1.0,
        reference_image=reference,
        reference_conditioning_scale=1.0,
        negative_prompt=negative_prompt,
        cross_attention_kwargs={"scale": 1.0},
        generator=generator,
    ).images
    return images, reference
