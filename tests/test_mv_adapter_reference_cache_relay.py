from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from workers.mv_adapter_i2mv_camera_runtime import install_reference_cache_relay


class FakeProcessor:
    def __init__(self, name: str, *, use_ref: bool) -> None:
        self.name = name
        self.use_ref = use_ref


class FakeUnet:
    def __init__(self) -> None:
        self.attn_processors = {
            "required.processor": FakeProcessor("required.processor", use_ref=True),
            "cache-only.processor": FakeProcessor("cache-only.processor", use_ref=False),
        }
        self.seen_denoise_cache = None

    def forward(self, *args, **kwargs):
        cross = kwargs.get("cross_attention_kwargs") or {}
        if "cache_hidden_states" in cross:
            target = cross["cache_hidden_states"]
            target["required.processor"] = torch.ones((1, 4, 8))
            target["cache-only.processor"] = torch.ones((1, 4, 8))
        elif "ref_hidden_states" in cross:
            self.seen_denoise_cache = cross["ref_hidden_states"]
        return (torch.zeros((1, 4, 8)),)


class FakePipe:
    def __init__(self) -> None:
        self.unet = FakeUnet()
        self._guidance_scale = 3.0

    @property
    def do_classifier_free_guidance(self) -> bool:
        return self._guidance_scale > 1.0


def test_reference_cache_relay_rebuilds_only_required_entries() -> None:
    pipe = FakePipe()
    state = install_reference_cache_relay(pipe)

    raw_cache = {}
    pipe.unet.forward(
        cross_attention_kwargs={
            "cache_hidden_states": raw_cache,
            "use_mv": False,
            "use_ref": False,
        }
    )

    denoise_kwargs = {
        "num_views": 6,
        "ref_hidden_states": {},
    }
    pipe.unet.forward(cross_attention_kwargs=denoise_kwargs)

    assert set(raw_cache) == {"required.processor", "cache-only.processor"}
    assert set(denoise_kwargs["ref_hidden_states"]) == {"required.processor"}
    assert denoise_kwargs["ref_hidden_states"]["required.processor"].shape[0] == 12
    assert pipe.unet.seen_denoise_cache is denoise_kwargs["ref_hidden_states"]
    assert state["relay_injected"] is True
    assert state["reference_cache_count"] == 2
    assert state["denoise_cache_count_before"] == 0
    assert state["denoise_cache_count_after"] == 1


def test_reference_cache_relay_fails_closed_when_required_key_missing() -> None:
    pipe = FakePipe()
    install_reference_cache_relay(pipe)

    original = pipe.unet.forward

    def incomplete_reference(*args, **kwargs):
        cross = kwargs.get("cross_attention_kwargs") or {}
        if "cache_hidden_states" in cross:
            cross["cache_hidden_states"]["cache-only.processor"] = torch.ones((1, 4, 8))
            return (torch.zeros((1, 4, 8)),)
        return original(*args, **kwargs)

    # Reinstall on a fresh pipeline whose underlying forward omits the required key.
    broken = FakePipe()
    broken.unet.forward = incomplete_reference
    install_reference_cache_relay(broken)

    with pytest.raises(RuntimeError, match="missing required MV entries"):
        broken.unet.forward(
            cross_attention_kwargs={
                "cache_hidden_states": {},
                "use_mv": False,
                "use_ref": False,
            }
        )
