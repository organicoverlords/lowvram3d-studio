from __future__ import annotations

import copy

import pytest

from lowvram3d.scene_splines import build_scene_spline_spec, validate_scene_spline_spec


def test_bounded_spec_is_deterministic_and_exact() -> None:
    first = build_scene_spline_spec()
    second = build_scene_spline_spec()
    assert first == second
    assert validate_scene_spline_spec(first)["scene_spline_spec_valid"]
    river = next(item for item in first["splines"] if item["id"] == "river_main")
    bridge = next(item for item in first["splines"] if item["id"] == "bridge_axis_main")
    assert river["points_m"] == [[-30.0, 45.0, 0.0], [0.0, 50.0, 0.0], [30.0, 58.0, 0.0]]
    assert river["width_m"] == 12.0
    assert river["exclusion_radius_m"] == 3.0
    assert bridge["points_m"] == [[-10.0, 31.0, 4.0], [10.0, 31.0, 4.0]]
    assert bridge["width_m"] == 3.0


@pytest.mark.parametrize("mutation", [
    lambda spec: spec["splines"].pop(),
    lambda spec: spec["splines"][0].update({"width_m": 0}),
    lambda spec: spec["splines"][0].update({"points_m": [[0, 0, 0]]}),
    lambda spec: spec["splines"][0].update({"tags": ["water"]}),
])
def test_invalid_spline_spec_fails_closed(mutation) -> None:
    spec = copy.deepcopy(build_scene_spline_spec())
    mutation(spec)
    result = validate_scene_spline_spec(spec)
    assert result["scene_spline_spec_valid"] is False
    assert result["errors"]
