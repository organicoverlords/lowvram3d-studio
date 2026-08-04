from __future__ import annotations

from pathlib import Path

from PIL import Image

from lowvram3d.scene_visual_validation import build_offset_validation, compare_source_view, repair_history


def _image(path: Path, colour: tuple[int, int, int]) -> None:
    Image.new("RGB", (16, 16), colour).save(path)


def test_source_view_validation_reports_colour_defect(tmp_path: Path):
    source = tmp_path / "source.png"
    render = tmp_path / "render.png"
    _image(source, (220, 80, 30))
    _image(render, (20, 20, 20))
    result = compare_source_view(source, render)
    assert result["classification"] == "REJECTED"
    assert result["defects"]
    assert repair_history(result["defects"])["max_attempts_per_defect"] == 1


def test_offset_validation_fails_closed_without_views():
    result = build_offset_validation([])
    assert result["classification"] == "NOT_PROVEN"
    assert result["bounded_360_claim"] is False
