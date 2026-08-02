"""Deterministic first-pass routing for image-to-world jobs.

This classifier consumes measurable preprocessing signals. It deliberately
avoids an LLM dependency so route selection is reproducible and testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import ContractError, ImageWorldRoute, RouteDecision


@dataclass(frozen=True)
class InputSignals:
    has_alpha: bool = False
    transparent_border_fraction: float = 0.0
    foreground_edge_touch_fraction: float = 1.0
    board_plane_confidence: float = 0.0
    top_down_confidence: float = 0.0
    horizon_confidence: float = 0.0
    perspective_strength: float = 0.0
    sky_fraction: float = 0.0
    water_fraction: float = 0.0
    semantic_group_count: int = 1
    occlusion_fraction: float = 0.0
    layout_coverage: float = 0.0

    def validate(self) -> None:
        for name in (
            "transparent_border_fraction",
            "foreground_edge_touch_fraction",
            "board_plane_confidence",
            "top_down_confidence",
            "horizon_confidence",
            "perspective_strength",
            "sky_fraction",
            "water_fraction",
            "occlusion_fraction",
            "layout_coverage",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ContractError(f"{name} must be between zero and one")
        if self.semantic_group_count < 1:
            raise ContractError("semantic_group_count must be at least one")


def classify_input(signals: InputSignals, *, review_threshold: float = 0.52) -> RouteDecision:
    """Classify an input into one of the four supported reconstruction routes."""

    signals.validate()
    if not 0.0 <= review_threshold <= 1.0:
        raise ContractError("review_threshold must be between zero and one")

    group_complexity = min(1.0, max(0.0, (signals.semantic_group_count - 1) / 4.0))
    alpha_strength = 1.0 if signals.has_alpha else 0.0
    border_isolation = signals.transparent_border_fraction
    contained_foreground = 1.0 - signals.foreground_edge_touch_fraction
    no_horizon = 1.0 - signals.horizon_confidence
    no_alpha = 1.0 - alpha_strength

    raw = {
        ImageWorldRoute.ISOLATED_ASSET: (
            0.44 * alpha_strength
            + 0.28 * border_isolation
            + 0.18 * contained_foreground
            + 0.10 * (1.0 - group_complexity)
        ),
        ImageWorldRoute.DIORAMA_MAP: (
            0.36 * signals.board_plane_confidence
            + 0.28 * signals.top_down_confidence
            + 0.18 * no_horizon
            + 0.18 * signals.layout_coverage
        ),
        ImageWorldRoute.PERSPECTIVE_VISTA: (
            0.30 * signals.horizon_confidence
            + 0.25 * signals.perspective_strength
            + 0.16 * signals.sky_fraction
            + 0.12 * signals.water_fraction
            + 0.17 * no_alpha
        ),
        ImageWorldRoute.COMPOSITE_SCENE: (
            0.34 * group_complexity
            + 0.24 * signals.occlusion_fraction
            + 0.18 * signals.horizon_confidence
            + 0.14 * signals.perspective_strength
            + 0.10 * no_alpha
        ),
    }

    total = sum(raw.values())
    if total <= 1e-12:
        normalized = {route: 0.25 for route in ImageWorldRoute}
    else:
        normalized = {route: value / total for route, value in raw.items()}

    tie_order = {
        ImageWorldRoute.ISOLATED_ASSET: 0,
        ImageWorldRoute.DIORAMA_MAP: 1,
        ImageWorldRoute.PERSPECTIVE_VISTA: 2,
        ImageWorldRoute.COMPOSITE_SCENE: 3,
    }
    selected = max(normalized, key=lambda route: (normalized[route], -tie_order[route]))
    confidence = normalized[selected]
    return RouteDecision(
        selected=selected,
        confidence=confidence,
        alternatives=normalized,
        manual_review_required=confidence < review_threshold,
        reasons=_reasons(selected, signals),
    )


def _reasons(route: ImageWorldRoute, signals: InputSignals) -> tuple[str, ...]:
    if route is ImageWorldRoute.ISOLATED_ASSET:
        reasons = ["meaningful source alpha" if signals.has_alpha else "contained foreground"]
        if signals.transparent_border_fraction >= 0.25:
            reasons.append("transparent border separates the subject")
        return tuple(reasons)
    if route is ImageWorldRoute.DIORAMA_MAP:
        reasons = ["planar board or map evidence"]
        if signals.top_down_confidence >= 0.5:
            reasons.append("top-down layout is visible")
        return tuple(reasons)
    if route is ImageWorldRoute.PERSPECTIVE_VISTA:
        reasons = ["perspective landscape evidence"]
        if signals.horizon_confidence >= 0.5:
            reasons.append("horizon is present")
        if signals.water_fraction >= 0.1:
            reasons.append("water provides a world-up constraint")
        return tuple(reasons)
    return ("multiple occluding semantic groups require scene decomposition",)
