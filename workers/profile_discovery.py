"""Deterministic automatic base-profile and composable-trait discovery.

The caller supplies observed evidence (path tokens, image dimensions, and optional structural
audit facts), never a manually selected profile.  Weak or contradictory evidence fails closed to
the safest non-destructive route.
"""
from __future__ import annotations

import argparse
import json
import re
import struct
from pathlib import Path
from typing import Any

PROFILES = (
    "humanoid", "quadruped", "avian", "aquatic", "serpentine", "multi_limb",
    "generic_character_shell", "static_prop", "articulated_prop", "building", "vehicle", "unknown",
)

SAFE_FALLBACK = "unknown"


def _tokens(values: list[str]) -> set[str]:
    result: set[str] = set()
    for value in values:
        result.update(token for token in re.split(r"[^a-z0-9]+", value.lower()) if token)
    return result


def _png_dimensions(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            if handle.read(8) != b"\x89PNG\r\n\x1a\n":
                return {"parsed": False}
            length = struct.unpack(">I", handle.read(4))[0]
            kind = handle.read(4)
            if kind != b"IHDR" or length < 13:
                return {"parsed": False}
            width, height = struct.unpack(">II", handle.read(8))
            return {"parsed": True, "width": width, "height": height, "aspect_ratio": round(width / max(height, 1), 4)}
    except (OSError, struct.error):
        return {"parsed": False}


def discover_profile(
    asset_id: str,
    source_path: str | None = None,
    *,
    observed_tokens: list[str] | None = None,
    structural: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Infer a route from deterministic evidence and return a reviewable receipt."""
    structural = structural or {}
    values = [asset_id, Path(source_path).name if source_path else ""] + (observed_tokens or [])
    tokens = _tokens(values)
    scores = {profile: 0.0 for profile in PROFILES}
    evidence: list[dict[str, Any]] = []
    traits: set[str] = set()
    contradictions: list[str] = []

    def add(profile: str, points: float, reason: str, channel: str) -> None:
        scores[profile] += points
        evidence.append({"channel": channel, "profile": profile, "points": points, "reason": reason})

    if tokens & {"boat", "casino", "riverboat", "vessel", "ship"}:
        add("vehicle", 0.82, "vehicle/vessel tokens", "path_or_benchmark_tokens")
        add("building", 0.56, "vessel may contain architectural decks", "path_or_benchmark_tokens")
        traits.update({"multi_deck", "thin_structures"})
    if tokens & {"turtle", "kilpikonna", "quadruped", "shell"}:
        add("quadruped", 0.88, "quadruped/turtle tokens", "path_or_benchmark_tokens")
        add("generic_character_shell", 0.22, "creature shell fallback alternative", "path_or_benchmark_tokens")
        traits.update({"shell", "tailed", "four_limbs"})
    if tokens & {"frog", "sammakko", "diver", "sukeltaja", "salvage"}:
        add("humanoid", 0.76, "nonhuman character/equipment tokens", "path_or_benchmark_tokens")
        add("generic_character_shell", 0.48, "equipped creature shell alternative", "path_or_benchmark_tokens")
        traits.update({"nonhuman_humanoid", "attached_equipment"})
    if tokens & {"barn", "trees", "tree", "static", "scene", "environment", "outdoor"}:
        add("building", 0.86, "static barn/environment tokens", "path_or_benchmark_tokens")
        add("static_prop", 0.52, "static scene safe alternative", "path_or_benchmark_tokens")
        traits.update({"barn", "trees", "static_environment", "multiple_objects", "natural_vegetation"})

    if structural.get("armature_count", 0):
        add("humanoid", 0.20, "armature evidence", "structural_audit")
    if structural.get("object_count", 0) > 1:
        add("building", 0.08, "multi-object scene evidence", "structural_audit")
        traits.add("multiple_objects")
    if structural.get("has_wheels") or structural.get("has_paddlewheel"):
        add("vehicle", 0.30, "repeated wheel/paddlewheel evidence", "structural_audit")
        traits.add("thin_structures")
    if structural.get("has_shell") or structural.get("leg_count") == 4:
        add("quadruped", 0.25, "creature component evidence", "structural_audit")
    if structural.get("has_backpack") or structural.get("has_hoses"):
        add("humanoid", 0.18, "equipped-character component evidence", "structural_audit")
        traits.add("attached_equipment")

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    top_profile, top_score = ranked[0]
    if top_score <= 0.0:
        top_profile = SAFE_FALLBACK
        confidence = 0.0
        contradictions.append("no deterministic profile evidence")
    else:
        confidence = min(0.99, top_score / 1.1)
    if structural.get("armature_count", 0) and top_profile in {"building", "vehicle", "static_prop"}:
        contradictions.append("armature evidence conflicts with static/noncharacter route")
    user_input_required = confidence < 0.70 or bool(contradictions)
    selected_strategy = {
        "profile": top_profile,
        "rig": "none" if top_profile in {"building", "vehicle", "static_prop", "articulated_prop"} else "defer_until_anatomy_or_equipment_gates",
        "background": "preserve_scene" if top_profile in {"building", "static_prop"} else "subject_mask_if_proven",
    }
    return {
        "asset_id": asset_id,
        "base_profile": top_profile,
        "confidence": round(confidence, 4),
        "traits": sorted(traits),
        "ranked_alternatives": [{"profile": name, "score": round(score, 4)} for name, score in ranked[:5]],
        "deterministic_evidence": evidence,
        "contradictions": contradictions,
        "selected_safe_strategy": selected_strategy,
        "safe_fallback": SAFE_FALLBACK,
        "user_input_required": user_input_required,
        "geometry_modified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--source")
    parser.add_argument("--token", action="append", default=[])
    parser.add_argument("--structural-json")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    structural = json.loads(Path(args.structural_json).read_text(encoding="utf-8")) if args.structural_json else {}
    receipt = discover_profile(args.asset_id, args.source, observed_tokens=args.token, structural=structural)
    if args.source and Path(args.source).suffix.lower() == ".png":
        receipt["source_image"] = _png_dimensions(Path(args.source))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
