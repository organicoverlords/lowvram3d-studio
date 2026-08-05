from __future__ import annotations

from pathlib import Path
import sys

import pytest


torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from run_mv_adapter_fp32_canary_direct import install_fp32_vae_boundaries  # noqa: E402


class FakeVAE:
    def __init__(self) -> None:
        self.dtype = torch.float16
        self.encode_seen = None
        self.decode_seen = None

    def to(self, *, dtype):
        self.dtype = dtype
        return self

    def encode(self, sample, *args, **kwargs):
        self.encode_seen = sample.dtype
        return sample + 1.0

    def decode(self, latents, *args, **kwargs):
        self.decode_seen = latents.dtype
        return (latents - 1.0,)


class FakePipe:
    def __init__(self) -> None:
        self.vae = FakeVAE()

    def prepare_image_latents(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator=None,
        add_noise=True,
    ):
        encoded = self.vae.encode(image.to(device=device, dtype=dtype))
        return encoded


def test_fp32_vae_boundaries_cast_inputs_and_restore_latent_dtype() -> None:
    pipe = FakePipe()

    receipt = install_fp32_vae_boundaries(pipe, torch)

    image = torch.zeros((1, 3, 8, 8), dtype=torch.float16)
    latents = pipe.prepare_image_latents(
        image,
        torch.zeros(1),
        1,
        1,
        torch.float16,
        "cpu",
        None,
        add_noise=False,
    )

    assert pipe.vae.dtype == torch.float32
    assert pipe.vae.encode_seen == torch.float32
    assert latents.dtype == torch.float16

    pipe.vae.decode(torch.zeros((1, 4, 8, 8), dtype=torch.float16))
    assert pipe.vae.decode_seen == torch.float32

    assert receipt == {
        "vae_parameter_dtype": "torch.float32",
        "vae_encode_input_dtype": "torch.float32",
        "vae_decode_input_dtype": "torch.float32",
        "reference_latent_return_dtype": "pipeline_requested_dtype",
    }
