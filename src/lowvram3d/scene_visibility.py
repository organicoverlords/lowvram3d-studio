"""Visibility classification and conservative hidden-world completion policy."""

from __future__ import annotations

from typing import Any, Mapping


VISIBILITY_CLASSES = ("fully_visible", "partially_occluded", "silhouette_only", "distant", "unseen_backside", "reflection", "transparent", "ambiguous")
COMPLETION_POLICIES = ("exact_visible_surface", "conservative_extrusion", "symmetry_completion", "modular_continuation", "terrain_continuation", "simple_closed_back", "visual_shell_only", "gameplay_boundary", "unresolved")


def classify_visibility(region: Mapping[str, Any]) -> str:
    explicit = str(region.get("visibility", ""))
    if explicit in VISIBILITY_CLASSES:
        return explicit
    representation = str(region.get("representation", ""))
    if representation in {"visual_shell", "source_visible_shell"}:
        return "fully_visible"
    if region.get("silhouette_only"):
        return "silhouette_only"
    if region.get("distant"):
        return "distant"
    if region.get("transparent"):
        return "transparent"
    return "partially_occluded" if region.get("interactive") or region.get("walkable") else "ambiguous"


def completion_policy(region: Mapping[str, Any], visibility: str | None = None) -> str:
    visibility = visibility or classify_visibility(region)
    explicit = str(region.get("completion_policy", ""))
    if explicit in COMPLETION_POLICIES:
        return explicit
    representation = str(region.get("representation", ""))
    if representation in {"visual_shell", "source_visible_shell"}:
        return "visual_shell_only"
    semantic = str(region.get("semantic_class") or region.get("layer_type") or "").lower()
    if visibility == "fully_visible":
        return "exact_visible_surface"
    if semantic in {"terrain", "ground", "cliff", "ground_surface"}:
        return "terrain_continuation"
    if visibility == "unseen_backside":
        return "simple_closed_back"
    if region.get("interactive") or region.get("walkable"):
        return "gameplay_boundary"
    return "conservative_extrusion"


def build_visibility_manifest(spec: Mapping[str, Any]) -> dict[str, Any]:
    records = []
    for index, region in enumerate(spec.get("regions", [])):
        visibility = classify_visibility(region)
        records.append({
            "region_id": str(region.get("id", f"region_{index + 1:03d}")),
            "visibility": visibility,
            "completion_policy": completion_policy(region, visibility),
            "source_visible_evidence": visibility in {"fully_visible", "silhouette_only", "partially_occluded", "distant"},
            "hidden_geometry_inferred": visibility in {"partially_occluded", "unseen_backside", "ambiguous"},
            "gameplay_only": bool(region.get("interactive") or region.get("walkable")),
            "decorative": bool("decorative" in {str(tag).lower() for tag in region.get("tags", [])}),
            "confidence": max(0.0, min(1.0, float(region.get("confidence", 0.5)))),
        })
    return {"schema_version": "visibility_manifest_v1", "classification": "PROVEN", "records": records, "full_reconstruction_claim": False, "policy": "source-visible fidelity with conservative bounded completion"}
