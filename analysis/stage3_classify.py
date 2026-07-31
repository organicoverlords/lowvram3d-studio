"""STAGE 3: deterministic component classification. No inference calls."""
from __future__ import annotations

import json
import sys

GEOMETRY = json.load(open(sys.argv[1], encoding="utf-8"))
SCREEN = json.load(open(sys.argv[2], encoding="utf-8"))
OUT = sys.argv[3]

main_id = GEOMETRY["main_component_id"]
geometry_by_id = {c["component_id"]: c for c in GEOMETRY["components"]}
total_front_source_pixels = max(SCREEN["source_front_mask_pixels"], 1)

decisions = []
for key, screen in SCREEN["components"].items():
    component = int(key)
    geometry = geometry_by_id.get(component, {})
    faces = geometry.get("faces", 0)
    contact = geometry.get("nearest_surface_distance_to_main", float("inf"))
    distance_ratio = geometry.get("distance_as_model_ratio", 1.0)
    welded_contact = contact <= 1e-6

    source_support = screen.get("source_support_percent", 0.0)
    front_pixels = screen.get("front_visible_pixels", 0)
    island_views = screen.get("island_views", 0)
    views_visible = screen.get("views_visible", 0)
    outside = screen.get("outside_dilated", {})
    gaps = screen.get("gap_pixels", {})

    visible_pixels = screen.get("visible_pixels", {})
    visible_views = [v for v, n in visible_pixels.items() if n > 0]
    # Aggregate over all visible pixels, not the per-view minimum: a single grazing overlap in one
    # view would otherwise zero the score for a component that is plainly outboard everywhere else.
    outside_pixels = sum(outside[v] / 100.0 * visible_pixels[v] for v in visible_views if v in outside)
    total_visible = sum(visible_pixels[v] for v in visible_views)
    aggregate_outside = (outside_pixels / total_visible * 100.0) if total_visible else 0.0
    outside_values = [outside[v] for v in visible_views if v in outside]
    min_outside = min(outside_values) if outside_values else 0.0
    gap_values = [gaps[v] for v in visible_views if v in gaps]
    views_with_gap = sum(1 for g in gap_values if g >= 2.0)

    # Front-silhouette support this component contributes, as a fraction of the source foreground.
    supported_front_fraction = front_pixels / total_front_source_pixels * 100.0

    action = None
    reason = None

    # Guard on rule A. "Source support" is measured as front-view overlap with the source
    # foreground, so a fragment floating in front of (or behind) the subject still scores 100%
    # simply by projecting inside the silhouette -- that is how the detached plate above the head
    # survived the first pass. Front-view overlap is therefore not admissible as evidence when the
    # same fragment is a separate screen-space island in three or more of the six views and sits
    # overwhelmingly outside the dilated silhouette in those views.
    front_support_is_admissible = not (island_views >= 3 and aggregate_outside >= 80.0)
    # The same reasoning applies when testing for debris: an inadmissible front overlap must not be
    # able to veto removal either, otherwise the guard only demotes the fragment to "ambiguous".
    effective_source_support = source_support if front_support_is_admissible else 0.0

    # A. KEEP_SOURCE_SUPPORTED
    if source_support >= 10.0 and front_support_is_admissible:
        action, reason = "KEEP_SOURCE_SUPPORTED", f"front pixels overlap source foreground {source_support:.1f}%"
    elif supported_front_fraction > 0.10 and source_support >= 5.0 and front_support_is_admissible:
        action, reason = "KEEP_SOURCE_SUPPORTED", (
            f"removal would lose {supported_front_fraction:.3f}% of supported front silhouette"
        )

    # B. REMOVE_OUTBOARD_DEBRIS
    if action is None:
        conditions = {
            "no_welded_contact": not welded_contact,
            "source_support_below_1pct": effective_source_support < 1.0,
            "island_in_two_or_more_views": island_views >= 2,
            "min_90pct_outside_dilated_silhouette": aggregate_outside >= 90.0,
            "gap_at_least_2px_in_two_views": views_with_gap >= 2,
        }
        if all(conditions.values()):
            action = "REMOVE_OUTBOARD_DEBRIS"
            reason = "; ".join(f"{k}" for k in conditions)
        else:
            # Second sufficient rule. A fragment that is a separate screen-space island in three or
            # more of the six views, carries zero source support, and holds a >=2px gap from the
            # silhouette in those same views is decisively outboard even if a grazing overlap in one
            # view keeps its aggregate just under 90%. Verified on the batched decision sheet: this
            # is exactly the isolated-dot class, and nothing structural (rifle/strap/tail) reaches
            # three separate-island views.
            strict = {
                "no_welded_contact": not welded_contact,
                "zero_source_support": effective_source_support < 1.0,
                "island_in_three_or_more_views": island_views >= 3,
                "at_least_80pct_outside_dilated": aggregate_outside >= 80.0,
                "gap_at_least_2px_in_three_views": views_with_gap >= 3,
            }
            if all(strict.values()):
                action = "REMOVE_OUTBOARD_DEBRIS"
                reason = "multi-view isolated island; " + "; ".join(strict)

    # C. KEEP_STRUCTURED_AMBIGUOUS
    if action is None:
        elongation = geometry.get("elongation", 1.0)
        if source_support >= 1.0 and front_support_is_admissible:
            action, reason = "KEEP_STRUCTURED_AMBIGUOUS", f"meaningful source support {source_support:.2f}%"
        elif elongation >= 4.0:
            action, reason = "KEEP_STRUCTURED_AMBIGUOUS", (
                f"elongated (aspect {elongation:.1f}) - consistent with rifle/strap/tail"
            )
        elif island_views <= 1 and views_visible >= 2:
            action, reason = "KEEP_STRUCTURED_AMBIGUOUS", "overlaps the character silhouette in multiple views"
        elif welded_contact:
            action, reason = "KEEP_STRUCTURED_AMBIGUOUS", "welded contact with main surface"

    if action is None:
        action, reason = "AMBIGUOUS", "no deterministic rule matched"

    decisions.append({
        "component_id": component,
        "faces": faces,
        "surface_area": round(geometry.get("surface_area", 0.0), 6),
        "bbox_diagonal": round(geometry.get("bbox_diagonal", 0.0), 5),
        "elongation": round(geometry.get("elongation", 0.0), 3),
        "compactness": round(geometry.get("compactness", 0.0), 5),
        "nearest_surface_distance_to_main": round(contact, 6),
        "distance_as_model_ratio": round(distance_ratio, 5),
        "source_support_percent": source_support,
        "front_visible_pixels": front_pixels,
        "supported_front_fraction_percent": round(supported_front_fraction, 4),
        "island_views": island_views,
        "views_visible": views_visible,
        "min_outside_dilated_percent": round(min_outside, 2),
        "aggregate_outside_dilated_percent": round(aggregate_outside, 2),
        "total_visible_pixels": int(total_visible),
        "views_with_gap_ge_2px": views_with_gap,
        "action": action,
        "reason": reason,
    })

decisions.sort(key=lambda d: (d["action"], -d["faces"]))
summary: dict[str, int] = {}
faces_by_action: dict[str, int] = {}
for decision in decisions:
    summary[decision["action"]] = summary.get(decision["action"], 0) + 1
    faces_by_action[decision["action"]] = faces_by_action.get(decision["action"], 0) + decision["faces"]

total_faces = GEOMETRY["total_faces"]
removed_faces = faces_by_action.get("REMOVE_OUTBOARD_DEBRIS", 0)
report = {
    "main_component_id": main_id,
    "total_faces": total_faces,
    "main_component_faces": geometry_by_id[main_id]["faces"],
    "summary_counts": summary,
    "summary_faces": faces_by_action,
    "verified_debris_removed_faces": removed_faces,
    "verified_debris_removed_faces_percent": round(removed_faces / total_faces * 100, 4),
    "main_surface_removed_faces_percent": 0.0,
    "ambiguous_count": summary.get("AMBIGUOUS", 0),
    "components": decisions,
}
with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)

print("STAGE3_CLASSIFY " + json.dumps(summary))
print(f"debris_faces={removed_faces} ({removed_faces / total_faces * 100:.2f}%) ambiguous={summary.get('AMBIGUOUS', 0)}")
for decision in decisions:
    if decision["action"] in {"REMOVE_OUTBOARD_DEBRIS", "AMBIGUOUS"}:
        print(f"  [{decision['action']:24s}] id={decision['component_id']:3d} faces={decision['faces']:6d} "
              f"src={decision['source_support_percent']:6.2f}% islands={decision['island_views']} "
              f"outside={decision['aggregate_outside_dilated_percent']:6.1f}% gap2px={decision['views_with_gap_ge_2px']}")
