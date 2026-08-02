"""Minimal SD2.1 image-to-multiview camera runtime.

This module reproduces only the camera/control-image and source-image preparation
used by MV-Adapter's official ``inference_i2mv_sd.py``.  It deliberately avoids
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
