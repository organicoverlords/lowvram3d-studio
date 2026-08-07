"""Fail-closed rigging policy for the low-VRAM character pipeline.

This module deliberately contains no backend imports. It is safe to evaluate on a
CPU-only coordinator before any large model is loaded.

Policy rule: use known upstream implementations unchanged first. Only after a
stock backend is measured and rejected may this lane introduce a patched or
custom runtime for the same job. The working image->mesh->paint foundation is
not replaced by rigging experiments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


HUMANOID_TYPES = frozenset({"avatar", "humanoid"})
CREATURE_TYPES = frozenset({"creature", "quadruped", "flying_creature"})
STATIC_TYPES = frozenset({"static_prop", "building", "room", "scene", "level", "environment_piece"})
MECHANICAL_TYPES = frozenset({"vehicle", "mechanical"})

DEFORMATION_POSES = (
    "rest_pose",
    "elbow_bend",
    "knee_bend",
    "hip_crouch",
    "shoulder_raise",
)

PIPELINE_ORDER = (
    "preserve_textured_lod0",
    "rig_and_skin",
    "static_rig_qa",
    "deformation_qa",
    "animation_retarget",
    "engine_export",
    "skeletal_lod_generation",
)


@dataclass(frozen=True, slots=True)
class RiggingPlan:
    asset_type: str
    rig_kind: str
    backend: str
    fallback_backends: tuple[str, ...]
    attention_backend: str | None
    dtype: str | None
    vram_ceiling_mb: int
    segmentation_before_rig: bool
    preserve_textured_lod0: bool
    generate_lods_after_rig: bool
    required_deformation_poses: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["fallback_backends"] = list(self.fallback_backends)
        data["required_deformation_poses"] = list(self.required_deformation_poses)
        return data


def _normalise(value: str | None) -> str:
    return (value or "").strip().lower().replace("-", "_")


def build_rigging_plan(
    asset_type: str,
    *,
    rig_kind: str = "auto",
    vram_ceiling_mb: int = 5600,
    preferred_backend: str | None = None,
) -> RiggingPlan:
    """Select a known upstream backend without loading a model.

    Default order intentionally avoids writing our own rigger first:
    * humanoids -> stock Make-It-Animatable through ComfyUI-UniRig;
    * arbitrary organic creatures -> stock UniRig through ComfyUI-UniRig;
    * mechanical assets -> existing rigid hierarchy route;
    * static assets -> no rig.

    Puppeteer remains an explicit experiment because making it work on sm75 may
    require replacing its upstream FlashAttention-2 assumption. That patch is
    not attempted until the stock general-rigging route has been measured.
    """
    asset = _normalise(asset_type)
    kind = _normalise(rig_kind) or "auto"
    preferred = _normalise(preferred_backend) or None
    if vram_ceiling_mb <= 0:
        raise ValueError("vram_ceiling_mb must be positive")

    if preferred:
        if preferred not in {"mia", "puppeteer", "unirig", "legacy_rigid", "none"}:
            raise ValueError(f"unknown rigging backend: {preferred_backend!r}")
        backend = preferred
        if backend == "puppeteer":
            attention = "sdpa"
            dtype = "fp16"
        elif backend in {"mia", "unirig"}:
            attention = "sdpa" if backend == "unirig" else None
            dtype = "fp16"
        else:
            attention = None
            dtype = None
        fallbacks: tuple[str, ...] = ()
        reasons = ("Explicit backend override; promotion gates still apply.",)
    elif asset in STATIC_TYPES and kind not in {"humanoid", "creature", "mechanical"}:
        backend = "none"
        attention = None
        dtype = None
        fallbacks = ()
        reasons = ("Asset profile is static; do not add an unnecessary armature.",)
    elif asset in MECHANICAL_TYPES or kind == "mechanical":
        backend = "legacy_rigid"
        attention = None
        dtype = None
        fallbacks = ()
        reasons = ("Rigid/mechanical motion should not pay the neural skinning cost.",)
    elif asset in HUMANOID_TYPES or kind == "humanoid":
        backend = "mia"
        attention = None
        dtype = "fp16"
        fallbacks = ("unirig",)
        reasons = (
            "Test stock Make-It-Animatable via ComfyUI-UniRig before writing a custom humanoid runtime.",
            "Keep stock UniRig as the known fallback; no silent backend substitution.",
        )
    else:
        backend = "unirig"
        attention = "sdpa"
        dtype = "fp16"
        fallbacks = ()
        reasons = (
            "Test stock ComfyUI-UniRig for arbitrary organic assets first.",
            "Puppeteer+SDPA is a later challenger only if the known route fails its measured gates.",
        )

    return RiggingPlan(
        asset_type=asset or "unknown",
        rig_kind=kind,
        backend=backend,
        fallback_backends=fallbacks,
        attention_backend=attention,
        dtype=dtype,
        vram_ceiling_mb=int(vram_ceiling_mb),
        segmentation_before_rig=False,
        preserve_textured_lod0=True,
        generate_lods_after_rig=True,
        required_deformation_poses=DEFORMATION_POSES if backend not in {"none", "legacy_rigid"} else (),
        reasons=reasons,
    )


def pipeline_stage_order(plan: RiggingPlan) -> tuple[str, ...]:
    if plan.backend == "none":
        return ("preserve_textured_lod0", "engine_export")
    if plan.backend == "legacy_rigid":
        return (
            "preserve_textured_lod0",
            "rig_and_skin",
            "static_rig_qa",
            "animation_retarget",
            "engine_export",
            "skeletal_lod_generation",
        )
    return PIPELINE_ORDER


def evaluate_rig_promotion(report: Mapping[str, Any], plan: RiggingPlan) -> tuple[bool, list[str]]:
    """Evaluate machine-readable rig evidence and fail closed."""
    if plan.backend == "none":
        return True, []

    failures: list[str] = []
    if not bool(report.get("armature_present")):
        failures.append("armature_missing")
    if not bool(report.get("skin_weights_present")):
        failures.append("skin_weights_missing")
    if plan.preserve_textured_lod0 and not bool(report.get("materials_preserved")):
        failures.append("materials_not_preserved")

    peak = report.get("peak_vram_mb")
    if peak is not None:
        try:
            if int(peak) > plan.vram_ceiling_mb:
                failures.append("vram_ceiling_exceeded")
        except (TypeError, ValueError):
            failures.append("invalid_peak_vram")

    if plan.required_deformation_poses:
        poses = report.get("deformation_poses") or {}
        for pose in plan.required_deformation_poses:
            entry = poses.get(pose)
            passed = entry is True or (isinstance(entry, Mapping) and entry.get("passed") is True)
            if not passed:
                failures.append(f"deformation_pose_failed:{pose}")

    return not failures, failures


def needs_segmentation_recovery(report: Mapping[str, Any]) -> bool:
    """Segmentation is a recovery action, not a mandatory pre-rig stage."""
    return bool(report.get("weight_bleed_detected") or report.get("rigid_accessory_requires_isolation"))
