"""Static model registry for the GTX 1660 SUPER 6 GB validation lane.

Size and compatibility fields are planning data, not proof. A model becomes eligible for
pipeline use only after a local benchmark receipt records successful inference, peak VRAM,
wall time, output validation, and process cleanup.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from .contracts import ContractError


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    upstream: str
    role: str
    parameters_millions: float | None
    weight_disk_mib: int | None
    license: str
    runtime: str
    precision: str
    status: str
    independent_of_moge: bool
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="qwen3.5-2b-supervisor",
        upstream="Qwen/Qwen3.5-2B",
        role="primary visual supervisor and structured router",
        parameters_millions=2000,
        weight_disk_mib=4570,
        license="Apache-2.0",
        runtime="local OpenAI-compatible server; quantized weights required",
        precision="Q4/Q5 candidate",
        status="BENCHMARK_REQUIRED",
        independent_of_moge=True,
        notes="Full BF16 weights are not the 6 GB target; cap context and image count.",
    ),
    ModelSpec(
        model_id="minicpm-v-4.6-secondary",
        upstream="openbmb/MiniCPM-V-4.6",
        role="independent second-opinion visual judge",
        parameters_millions=1300,
        weight_disk_mib=None,
        license="model-card license review required",
        runtime="Transformers, llama.cpp, or local OpenAI-compatible server",
        precision="GPTQ/AWQ/GGUF Q4 candidate",
        status="BENCHMARK_REQUIRED",
        independent_of_moge=True,
        notes="Run sequentially after the primary model; never keep both resident.",
    ),
    ModelSpec(
        model_id="florence2-base-ft-labels",
        upstream="microsoft/Florence-2-base-ft",
        role="captions, OCR, boxes, region labels, dense region proposals",
        parameters_millions=230,
        weight_disk_mib=463,
        license="MIT",
        runtime="Transformers in isolated labels environment",
        precision="FP16 candidate",
        status="PREFERRED_SPECIALIST",
        independent_of_moge=True,
    ),
    ModelSpec(
        model_id="edgetam-masks",
        upstream="facebookresearch/EdgeTAM",
        role="prompted masks from points and Florence boxes",
        parameters_millions=None,
        weight_disk_mib=56,
        license="Apache-2.0",
        runtime="official PyTorch implementation in isolated segmentation environment",
        precision="FP16 candidate",
        status="PREFERRED_SPECIALIST",
        independent_of_moge=True,
    ),
    ModelSpec(
        model_id="da3-small-depth-check",
        upstream="depth-anything/DA3-SMALL",
        role="independent depth and camera consistency cross-check",
        parameters_millions=34.3,
        weight_disk_mib=137,
        license="Apache-2.0 code; verify checkpoint card before production",
        runtime="Depth Anything 3 Python API in isolated geometry-validator environment",
        precision="FP16 candidate",
        status="PREFERRED_SPECIALIST",
        independent_of_moge=True,
    ),
    ModelSpec(
        model_id="moge2-vits-normal-baseline",
        upstream="Ruicheng/moge-2-vits-normal",
        role="fast point/depth/normal/FOV implementation and regression check",
        parameters_millions=35,
        weight_disk_mib=141,
        license="MIT",
        runtime="existing pinned MoGe environment or ONNX",
        precision="FP16",
        status="ALREADY_MEASURED_ON_TARGET",
        independent_of_moge=False,
        notes="Useful for transform and implementation regressions, not independent truth.",
    ),
    ModelSpec(
        model_id="qwen3-vl-embedding-2b-memory",
        upstream="Qwen/Qwen3-VL-Embedding-2B",
        role="phase-two visual failure retrieval and duplicate-retry detection",
        parameters_millions=2000,
        weight_disk_mib=4260,
        license="Apache-2.0",
        runtime="separate sequential embedding process",
        precision="quantized candidate",
        status="PHASE_TWO",
        independent_of_moge=True,
    ),
    ModelSpec(
        model_id="moge3-watch",
        upstream="microsoft/MoGe MoGe-3",
        role="future fine-detail geometry candidate",
        parameters_millions=None,
        weight_disk_mib=None,
        license="not yet reviewable",
        runtime="not released",
        precision="unknown",
        status="WATCHLIST_NO_WEIGHTS",
        independent_of_moge=False,
    ),
)


def get_model(model_id: str) -> ModelSpec:
    for spec in MODELS:
        if spec.model_id == model_id:
            return spec
    raise ContractError(f"unknown model id: {model_id}")


def eligible_models(role_contains: str | None = None) -> tuple[ModelSpec, ...]:
    blocked = {"WATCHLIST_NO_WEIGHTS"}
    models: Iterable[ModelSpec] = (item for item in MODELS if item.status not in blocked)
    if role_contains:
        needle = role_contains.lower()
        models = (item for item in models if needle in item.role.lower())
    return tuple(models)
