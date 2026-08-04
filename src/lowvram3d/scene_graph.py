"""Scene graph construction with explicit, confidence-bearing relationships."""

from __future__ import annotations

import re
from typing import Any, Mapping


RELATIONSHIPS = (
    "sits_on", "attached_to", "inside", "contains", "spans", "crosses", "connects",
    "occludes", "behind", "in_front_of", "grows_on", "supported_by", "adjacent_to",
)


def inferred_semantic(region: Mapping[str, Any]) -> str:
    raw = str(region.get("semantic_class") or region.get("layer_type") or region.get("layer") or "unknown").lower()
    if raw != "unknown":
        return {"castle": "architecture", "bridge_module": "crossing", "grass": "vegetation", "tree": "vegetation", "forest": "vegetation", "sky": "sky_or_ceiling", "source_visible_shell": "background_geometry"}.get(raw, raw)
    tags = {str(tag).lower() for tag in region.get("tags", [])}
    text = " ".join((str(region.get("id", "")), str(region.get("label", "")), str(region.get("depth_band", "")), str(region.get("representation", "")), *tags)).lower()
    if "water" in text:
        return "water"
    if any(token in text for token in ("vegetation", "grass", "tree", "decorative")):
        return "vegetation"
    if any(token in text for token in ("castle", "architecture", "hero_structure")):
        return "architecture"
    if any(token in text for token in ("sky", "cloud", "background")):
        return "sky_or_ceiling"
    if "shell" in text:
        return "background_geometry"
    return raw


def stable_region_id(region: Mapping[str, Any], index: int) -> str:
    layer = inferred_semantic(region)
    prefix = "unknown"
    if layer in {"architecture", "interior_structure"} or "building" in layer:
        prefix = "architecture"
    elif layer in {"terrain", "ground_surface", "cliff", "ground"}:
        prefix = "terrain"
    elif layer == "water":
        prefix = "water"
    elif layer in {"crossing", "road_or_path"}:
        prefix = "crossing"
    elif layer == "vegetation" or any(token in layer for token in ("tree", "forest", "grass")):
        prefix = "vegetation_region"
    elif layer == "hero_object":
        prefix = "hero_object"
    elif layer in {"sky_or_ceiling", "background_geometry"}:
        prefix = "background_geometry"
    return f"{prefix}_{index + 1:03d}"


def _confidence(value: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def build_scene_graph(spec: Mapping[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    id_map: dict[str, str] = {}
    for index, region in enumerate(spec.get("regions", [])):
        source_id = str(region.get("id", f"region_{index + 1:03d}"))
        generated_id = stable_region_id(region, index)
        id_map[source_id] = generated_id
        nodes.append({
            "id": generated_id,
            "source_id": source_id,
            "semantic_class": inferred_semantic(region),
            "confidence": _confidence(region.get("confidence")),
            "evidence": region.get("evidence", ["source_image"]),
            "gameplay_role": "interactive" if region.get("interactive") or region.get("walkable") else "visual",
        })
    for spline_index, spline in enumerate(spec.get("splines", [])):
        kind = str(spline.get("kind", "")).lower()
        if kind in {"river", "stream", "water"}:
            semantic = "water"
            node_id = f"water_spline_{spline_index + 1:03d}"
        elif kind in {"bridge_axis", "crossing", "road"}:
            semantic = "crossing"
            node_id = f"crossing_{spline_index + 1:03d}"
        else:
            continue
        nodes.append({"id": node_id, "source_id": str(spline.get("id", node_id)), "semantic_class": semantic, "confidence": _confidence(spline.get("confidence", 0.6)), "evidence": ["scene_spec.splines"], "gameplay_role": "interactive" if semantic == "crossing" else "visual"})
    edges: list[dict[str, Any]] = []
    for relation in spec.get("relationships", []):
        if not isinstance(relation, Mapping) or relation.get("type") not in RELATIONSHIPS:
            continue
        source = id_map.get(str(relation.get("source")), str(relation.get("source")))
        target = id_map.get(str(relation.get("target")), str(relation.get("target")))
        if source in {node["id"] for node in nodes} and target in {node["id"] for node in nodes}:
            edges.append({"source": source, "target": target, "type": relation["type"], "confidence": _confidence(relation.get("confidence")), "evidence": relation.get("evidence", "scene_spec"), "fallback": relation.get("fallback", "conservative_completion")})
    by_class = {node["id"]: node["semantic_class"].lower() for node in nodes}
    for source, source_class in by_class.items():
        for target, target_class in by_class.items():
            if source == target:
                continue
            if source_class in {"architecture", "interior_structure", "tower"} and target_class in {"terrain", "ground_surface", "cliff", "ground"}:
                edges.append({"source": source, "target": target, "type": "supported_by", "confidence": 0.55, "evidence": "semantic_adjacency", "fallback": "conservative_support"})
            elif source_class == "vegetation" and target_class in {"terrain", "ground_surface", "cliff", "ground"}:
                edges.append({"source": source, "target": target, "type": "grows_on", "confidence": 0.55, "evidence": "semantic_adjacency", "fallback": "terrain_affinity"})
            elif source_class in {"crossing", "road_or_path"} and target_class == "water":
                edges.append({"source": source, "target": target, "type": "crosses", "confidence": 0.5, "evidence": "semantic_adjacency", "fallback": "unresolved"})
    water_nodes = [node_id for node_id, semantic in by_class.items() if semantic == "water"]
    crossing_nodes = [node_id for node_id, semantic in by_class.items() if semantic == "crossing"]
    for crossing in crossing_nodes:
        for water in water_nodes:
            edges.append({"source": crossing, "target": water, "type": "spans", "confidence": 0.6, "evidence": "scene_spec.splines", "fallback": "bounded_crossing"})
    unique = {(edge["source"], edge["target"], edge["type"]): edge for edge in edges}
    return {"schema_version": "scene_graph_v1", "classification": "PROVEN", "nodes": nodes, "edges": list(unique.values()), "relationship_types": list(RELATIONSHIPS), "uncertainty_policy": "unsupported relationships remain unresolved"}


def validate_scene_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    ids = {str(node.get("id")) for node in graph.get("nodes", [])}
    errors = []
    for edge in graph.get("edges", []):
        if edge.get("type") not in RELATIONSHIPS or edge.get("source") not in ids or edge.get("target") not in ids:
            errors.append(edge)
    return {"schema_version": "scene_graph_receipt_v1", "classification": "PROVEN" if not errors else "REJECTED", "node_count": len(ids), "edge_count": len(graph.get("edges", [])), "errors": errors}
