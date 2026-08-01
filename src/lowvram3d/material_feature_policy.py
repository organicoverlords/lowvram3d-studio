"""Deterministic material-feature and low-VRAM soft-surface policy.

Visual inference may propose regions, but production decisions are made here.
Colour alone never enables emission, transparency, fur, hair, fire or toxic
materials. Missing evidence produces an ordinary PBR fallback rather than a
fabricated enhancement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

MATERIAL_CATEGORIES = frozenset({
    "skin", "cloth", "leather", "fur", "hair", "feather", "wood", "bark",
    "foliage", "stone", "bone_or_horn", "metal", "rust", "ceramic", "glass",
    "crystal", "wet_surface", "slime", "emissive", "subsurface",
    "masked_soft", "translucent",
})

MATERIAL_FAMILIES = (
    "OpaqueLit",
    "MaskedSoft",
    "TranslucentSpecial",
    "UnlitOrVFX",
)

SOFT_SURFACE_CATEGORIES = frozenset({"fur", "hair", "feather", "masked_soft"})
TRANSLUCENT_CATEGORIES = frozenset({"glass", "crystal", "translucent"})
EMISSIVE_CATEGORIES = frozenset({"emissive"})


class MaterialManifestError(ValueError):
    """Raised when a feature manifest is structurally unsafe."""


@dataclass(frozen=True)
class MaterialFeature:
    id: str
    category: str
    subtype: str
    confidence: float
    uv_mask: str | None = None
    direction_map: str | None = None
    source_views: tuple[str, ...] = ()
    evidence_types: tuple[str, ...] = ()
    auto_enable: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "MaterialFeature":
        if not isinstance(raw, dict):
            raise MaterialManifestError("each material feature must be an object")
        identifier = str(raw.get("id", "")).strip()
        if not identifier:
            raise MaterialManifestError("material feature id is required")
        category = str(raw.get("category", "")).strip().lower()
        if category not in MATERIAL_CATEGORIES:
            raise MaterialManifestError(
                f"feature {identifier!r} has unsupported category {category!r}"
            )
        subtype = str(raw.get("subtype", category)).strip().lower() or category
        confidence = _unit(raw.get("confidence", 0.0), f"feature {identifier}.confidence")
        source_views = tuple(sorted({str(value) for value in raw.get("source_views", []) if value}))
        evidence_types = tuple(sorted({
            str(value).strip().lower()
            for value in raw.get("evidence_types", [])
            if str(value).strip()
        }))
        return cls(
            id=identifier,
            category=category,
            subtype=subtype,
            confidence=confidence,
            uv_mask=str(raw["uv_mask"]) if raw.get("uv_mask") else None,
            direction_map=str(raw["direction_map"]) if raw.get("direction_map") else None,
            source_views=source_views,
            evidence_types=evidence_types,
            auto_enable=bool(raw.get("auto_enable", False)),
            metadata=dict(raw.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_views"] = list(self.source_views)
        result["evidence_types"] = list(self.evidence_types)
        return result


@dataclass(frozen=True)
class FeatureDecision:
    feature_id: str
    category: str
    status: str
    material_family: str
    enabled: bool
    confidence: float
    reason_codes: tuple[str, ...]
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_id": self.feature_id,
            "category": self.category,
            "status": self.status,
            "material_family": self.material_family,
            "enabled": self.enabled,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
            "parameters": self.parameters,
        }


def _unit(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MaterialManifestError(f"{field_name} must be numeric") from exc
    if not 0.0 <= number <= 1.0:
        raise MaterialManifestError(f"{field_name} must be in [0, 1]")
    return number


def load_material_manifest(source: str | Path | dict[str, Any] | None) -> dict[str, Any]:
    if source is None or source == "":
        return {
            "schema_version": SCHEMA_VERSION,
            "features": [],
            "status": "missing_optional_manifest",
        }
    if isinstance(source, dict):
        raw = source
    else:
        path = Path(source)
        if not path.is_file():
            return {
                "schema_version": SCHEMA_VERSION,
                "features": [],
                "status": "missing_optional_manifest",
                "missing_path": str(path),
            }
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise MaterialManifestError(f"invalid JSON in {path}: {exc}") from exc
    if int(raw.get("schema_version", SCHEMA_VERSION)) != SCHEMA_VERSION:
        raise MaterialManifestError("unsupported material-feature schema version")
    features_raw = raw.get("features") or []
    if not isinstance(features_raw, list):
        raise MaterialManifestError("features must be a list")
    features = [MaterialFeature.from_raw(item) for item in features_raw]
    identifiers = [feature.id for feature in features]
    if len(identifiers) != len(set(identifiers)):
        raise MaterialManifestError("material feature ids must be unique")
    return {
        "schema_version": SCHEMA_VERSION,
        "features": features,
        "status": str(raw.get("status", "provided")),
        "source_sha256": raw.get("source_sha256"),
        "metadata": dict(raw.get("metadata") or {}),
    }


def material_family(category: str) -> str:
    if category in SOFT_SURFACE_CATEGORIES:
        return "MaskedSoft"
    if category in TRANSLUCENT_CATEGORIES:
        return "TranslucentSpecial"
    if category in EMISSIVE_CATEGORIES:
        return "OpaqueLit"
    return "OpaqueLit"


def _default_parameters(feature: MaterialFeature) -> dict[str, Any]:
    category = feature.category
    if category == "cloth":
        return {"roughness_floor": 0.55, "sheen_weight": 0.18}
    if category == "leather":
        return {"roughness_floor": 0.42, "coat_weight_max": 0.12}
    if category == "skin":
        return {"subsurface_weight_max": 0.16, "roughness_floor": 0.38}
    if category in {"wood", "bark"}:
        return {"metallic": 0.0, "roughness_floor": 0.58}
    if category == "metal":
        return {"metallic": 1.0, "roughness_floor": 0.22}
    if category == "rust":
        return {"metallic_max": 0.18, "roughness_floor": 0.62}
    if category == "bone_or_horn":
        return {"metallic": 0.0, "roughness_floor": 0.36, "subsurface_weight_max": 0.05}
    if category in {"wet_surface", "slime"}:
        return {"roughness_min": 0.12, "coat_weight_max": 0.35}
    if category in TRANSLUCENT_CATEGORIES:
        return {"transmission_weight": 1.0, "ior": 1.45}
    if category == "emissive":
        return {"emission_strength_default": 2.0, "preserve_base_detail": True}
    if category in SOFT_SURFACE_CATEGORIES:
        return {"use_cards": True, "dense_groom": False}
    return {}


def decide_feature(feature: MaterialFeature) -> FeatureDecision:
    family = material_family(feature.category)
    reasons: list[str] = []
    parameters = _default_parameters(feature)

    if not feature.uv_mask:
        reasons.append("MATERIAL_UV_MASK_MISSING")

    independent_evidence = len(set(feature.evidence_types))
    if feature.category in {"emissive", "translucent", "fur", "hair", "slime"}:
        if independent_evidence < 2:
            reasons.append("MATERIAL_INDEPENDENT_EVIDENCE_INSUFFICIENT")

    colour_only = set(feature.evidence_types).issubset({"colour", "brightness", "saturation"})
    if colour_only and feature.category in {"emissive", "slime", "translucent"}:
        reasons.append("MATERIAL_COLOUR_ONLY_EVIDENCE_REJECTED")

    if feature.confidence < 0.70:
        status = "rejected"
        enabled = False
        reasons.append("MATERIAL_CONFIDENCE_TOO_LOW")
    elif feature.confidence < 0.88:
        status = "proposal_only"
        enabled = False
        reasons.append("MATERIAL_REVIEW_REQUIRED")
    elif not feature.auto_enable:
        status = "proposal_only"
        enabled = False
        reasons.append("MATERIAL_AUTO_ENABLE_FALSE")
    elif reasons:
        status = "rejected"
        enabled = False
    else:
        status = "enabled"
        enabled = True
        reasons.append("MATERIAL_POLICY_PASSED")

    return FeatureDecision(
        feature_id=feature.id,
        category=feature.category,
        status=status,
        material_family=family,
        enabled=enabled,
        confidence=feature.confidence,
        reason_codes=tuple(dict.fromkeys(reasons)),
        parameters=parameters,
    )


def soft_surface_budget(profile_name: str, *, hero: bool = False) -> dict[str, Any]:
    """Return card budgets sized for the GTX 1660 SUPER baseline."""
    if hero:
        return {
            "mode": "hero_opt_in",
            "lod_card_limits": {"0": 5000, "1": 1800, "2": 500, "3": 0},
            "dense_groom_allowed": False,
        }
    if profile_name in {"humanoid", "humanoid_complex_accessories", "flying_creature"}:
        limits = {"0": 2000, "1": 900, "2": 300, "3": 0}
    elif profile_name == "quadruped":
        limits = {"0": 2400, "1": 1000, "2": 350, "3": 0}
    else:
        limits = {"0": 1000, "1": 400, "2": 120, "3": 0}
    return {
        "mode": "standard_low_vram",
        "lod_card_limits": limits,
        "dense_groom_allowed": False,
    }


def build_material_plan(
    source: str | Path | dict[str, Any] | None,
    *,
    profile_name: str,
    hero_soft_surfaces: bool = False,
    maximum_material_slots: int = 4,
) -> dict[str, Any]:
    manifest = load_material_manifest(source)
    features: list[MaterialFeature] = manifest["features"]
    decisions = [decide_feature(feature) for feature in features]
    enabled = [decision for decision in decisions if decision.enabled]
    families = sorted({decision.material_family for decision in enabled})
    slot_budget_ok = len(families) <= maximum_material_slots
    failures = [] if slot_budget_ok else ["MATERIAL_SLOT_BUDGET_EXCEEDED"]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "planned" if features else "ordinary_pbr_fallback",
        "semantic_source_status": manifest["status"],
        "feature_count": len(features),
        "enabled_feature_count": len(enabled),
        "features": [feature.to_dict() for feature in features],
        "decisions": [decision.to_dict() for decision in decisions],
        "material_families": families or ["OpaqueLit"],
        "maximum_material_slots": maximum_material_slots,
        "material_slot_budget_ok": slot_budget_ok,
        "failure_codes": failures,
        "soft_surface_budget": soft_surface_budget(
            profile_name,
            hero=hero_soft_surfaces,
        ),
    }
