from __future__ import annotations

import pytest

from lowvram3d.source_view_policy import capture_contract_for, measure_image_file, validate_capture_contract, validate_capture_evidence, validate_image_evidence, validate_material_audit


def _capture(**overrides):
    record = {
        "camera_label": "Castlegrounds_Camera_Source",
        "player_camera_used": False,
        "slate_capture": False,
        "editor_ui_visible": False,
        "width": 1448,
        "height": 1086,
        "fov_deg": 66.50838470458984,
        "projection": "perspective",
        "source_shell_visible": True,
        "preview_material_visible": False,
        "proxy_occluding_shell": False,
    }
    record.update(overrides)
    return record


def test_named_camera_contract_is_proven_only_for_direct_capture():
    contract = capture_contract_for()
    assert validate_capture_contract(_capture(), contract)["classification"] == "PROVEN"
    rejected = validate_capture_contract(_capture(camera_label="PlayerStart", player_camera_used=True), contract)
    assert rejected["classification"] == "REJECTED"
    assert {item["code"] for item in rejected["defects"]} >= {"NAMED_CAMERA_NOT_USED", "PLAYER_CAMERA_DEPENDENCY"}


def test_capture_contract_rejects_wrong_aspect_and_preview_material():
    contract = capture_contract_for()
    result = validate_capture_contract(_capture(width=532, height=540, preview_material_visible=True), contract)
    assert result["classification"] == "REJECTED"
    assert {item["code"] for item in result["defects"]} >= {"DIMENSIONS_MISMATCH", "ASPECT_RATIO_MISMATCH", "PREVIEW_MATERIAL_VISIBLE"}


def test_capture_resolution_must_be_exact_four_three():
    with pytest.raises(ValueError):
        capture_contract_for(960, 720 + 1)


def test_black_and_constant_images_fail_closed():
    result = validate_image_evidence({"dimensions_px": [1448, 1086], "non_dark_fraction": 0.0, "stddev_rgb": [0, 0, 0], "sha256": "x"})
    assert result["classification"] == "REJECTED"
    assert {item["code"] for item in result["defects"]} == {"NEARLY_BLACK_CAPTURE", "LOW_IMAGE_VARIANCE"}


def test_dark_outline_capture_fails_mean_luminance_gate():
    result = validate_image_evidence({"dimensions_px": [1448, 1086], "non_dark_fraction": 0.12, "mean_luminance": 3.0, "stddev_rgb": [2, 3, 6], "sha256": "x"})
    assert result["classification"] == "REJECTED"
    assert {item["code"] for item in result["defects"]} == {"LOW_MEAN_LUMINANCE"}


def test_material_audit_rejects_preview_and_lit_slots():
    result = validate_material_audit({"slots": [{"slot": 0, "path": "/Engine/EngineMaterials/WorldGridMaterial", "engine_placeholder": True, "preview_material": True, "unlit": False}]})
    assert result["classification"] == "REJECTED"
    assert {item["code"] for item in result["defects"]} == {"PLACEHOLDER_MATERIAL", "SOURCE_MATERIAL_NOT_UNLIT"}


def test_measure_image_file_rejects_nearly_black_render(tmp_path):
    from PIL import Image

    path = tmp_path / "black.png"
    Image.new("RGB", (32, 24), (1, 1, 2)).save(path)
    result = measure_image_file(path)
    assert result["classification"] == "REJECTED"
    assert "NEARLY_BLACK_CAPTURE" in {item["code"] for item in result["defects"]}


def test_capture_evidence_requires_camera_pixels_and_materials(tmp_path):
    from PIL import Image

    path = tmp_path / "valid.png"
    pixels = [(20 + (x * 7) % 220, 30 + (y * 9) % 200, 40 + ((x + y) * 11) % 180) for y in range(24) for x in range(32)]
    image = Image.new("RGB", (32, 24))
    image.putdata(pixels)
    image.save(path)
    record = _capture(width=32, height=24)
    contract = capture_contract_for(32, 24)
    material_audit = {"slots": [{"slot": 0, "path": "/Game/Proof/M_SourceProjection", "engine_placeholder": False, "preview_material": False, "unlit": True}]}
    result = validate_capture_evidence(record, path, contract, material_audit)
    assert result["classification"] == "PROVEN"
