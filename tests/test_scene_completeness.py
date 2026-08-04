from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lowvram3d.scene_completeness import audit_scene_completeness


ROOT = Path(__file__).resolve().parents[1]


def _spec():
    return json.loads((ROOT / "configs/scene/castlegrounds_scene_spec_v1.json").read_text(encoding="utf-8"))


def test_audit_accounts_for_every_required_region() -> None:
    result = audit_scene_completeness(_spec())
    ids = {item["region_id"] for item in result["regions"]}
    assert ids == {"source_shell", "castle_core", "water_channels", "vegetation", "sky"}
    assert "source_shell" not in result["unresolved_regions"]
    assert result["classification"] == "PARTIAL"


def test_shell_does_not_count_as_editable_geometry() -> None:
    result = audit_scene_completeness(_spec())
    distant = next(item for item in result["regions"] if item["region_id"] == "source_shell")
    assert distant["representation"] == "VISUAL_SHELL"
    assert distant["source_shell_only"] is True
    assert distant["independent_geometry"] is False


def test_proven_receipts_promote_all_regions() -> None:
    receipts = {name: {"classification": "PROVEN", "lighthouse": name == "architecture"} for name in ("terrain", "architecture", "water", "bridge", "vegetation", "environment")}
    result = audit_scene_completeness(_spec(), receipts)
    assert result["classification"] == "PROVEN"
    assert result["unresolved_regions"] == []


def test_invalid_spec_fails_closed() -> None:
    spec = copy.deepcopy(_spec())
    spec["scene_id"] = "BAD"
    with pytest.raises(ValueError, match="SceneSpec invalid"):
        audit_scene_completeness(spec)
