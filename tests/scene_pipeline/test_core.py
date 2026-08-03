from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from workers.scene_pipeline.core import (
    compare_reload_receipt,
    ground_placement,
    image_input_receipt,
    require_dedicated_content_root,
    uniform_scale,
    upright_axis_conversion,
    validate_collision,
    validate_material_texture_binding,
)


def test_image_hash_and_dimensions_are_recorded(tmp_path: Path) -> None:
    source = tmp_path / "scene.png"
    source.write_bytes(b"fixture")
    receipt = image_input_receipt(source, dimensions=(1280, 720), mode="RGB")
    assert receipt["sha256"] == hashlib.sha256(b"fixture").hexdigest()
    assert receipt["dimensions"] == [1280, 720]


def test_dedicated_content_root_rejects_existing_root() -> None:
    assert require_dedicated_content_root("/Game/AgentProof/Smoke/") == "/Game/AgentProof/Smoke/"
    with pytest.raises(ValueError, match="FORBIDDEN"):
        require_dedicated_content_root("/Game/Maps/", forbidden=("/Game/Maps/",))


def test_scale_upright_and_grounding() -> None:
    assert uniform_scale((2, 4, 8), 16) == 2
    assert upright_axis_conversion("Y")["rotation_required"]
    assert ground_placement(-3.5, 0) == 3.5


def test_material_and_image_collision_contracts() -> None:
    assert validate_material_texture_binding({
        "material": "/Game/M",
        "texture": "/Game/T",
        "bound_to_base_color": True,
    })["passed"]
    assert validate_collision({"applicability": "NOT_APPLICABLE_IMAGE_TO_SCENE"})["passed"]


def test_reload_receipt_detects_reference_loss() -> None:
    before = {"map": "/Game/M", "image_texture": "/Game/T", "image_surface": "Surface", "camera_primary": "A", "camera_secondary": "B"}
    assert compare_reload_receipt(before, dict(before))["passed"]
    after = dict(before)
    after["image_texture"] = None
    assert not compare_reload_receipt(before, after)["passed"]
