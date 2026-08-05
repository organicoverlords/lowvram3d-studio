"""Regression tests for the corrected raw-index -> semantic camera permutation."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "workers"))

from apply_camera_permutation import CONTROL_SUFFIXES, apply_permutation  # noqa: E402
from prove_bar_repair import file_prefix  # noqa: E402
from render_control_bundle_texture import file_prefix as sheet_file_prefix  # noqa: E402

PROVEN_RAW_TO_SEMANTIC = {0: "left", 1: "rear", 2: "right", 3: "front", 4: "top", 5: "bottom"}
# What the raw builder writes before anything is relabelled.
RAW_PREFIXES = ("horizontal_0", "horizontal_1", "horizontal_2", "horizontal_3", "top", "bottom")
DIRECTIONS = ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, -1.0, 0.0),
              (1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))
AZIMUTHS = (-90.0, 0.0, 90.0, 180.0, 90.0, 90.0)
ELEVATIONS = (0.0, 0.0, 0.0, 0.0, 89.99, -89.99)
# The label each raw index carried before the user review, i.e. what must not survive.
BUILDER_GUESS = ("front", "right", "rear", "left", "top", "bottom")


@pytest.fixture()
def bundle(tmp_path: Path) -> Path:
    views = []
    for index, prefix in enumerate(RAW_PREFIXES):
        direction = np.asarray(DIRECTIONS[index], dtype=float)
        views.append({
            "index": index,
            "semantic_name": prefix,
            "proven_semantic": BUILDER_GUESS[index],
            "azimuth_deg": AZIMUTHS[index],
            "elevation_deg": ELEVATIONS[index],
            "camera_direction": direction.tolist(),
            "camera_position": (-direction * 1.8).tolist(),
            "camera_up": [0.0, 0.0, 1.0],
        })
        for suffix in CONTROL_SUFFIXES.values():
            path = tmp_path / f"{prefix}{suffix}"
            if suffix.endswith(".png"):
                Image.new("L", (4, 4), 255).save(path)
            else:
                np.save(path, np.zeros((4, 4), dtype=np.float32))
    (tmp_path / "camera_contract.json").write_text(
        json.dumps({"schema": "lowvram3d_mvadapter_camera_contract_v1",
                    "view_count": 6, "views": views}, indent=2), encoding="utf-8")
    return tmp_path


def _applied(bundle: Path) -> dict:
    return apply_permutation(bundle, PROVEN_RAW_TO_SEMANTIC,
                             {"semantic_source": "USER_VISION_REVIEW_20260803",
                              "evidence": ["raw3 shows the muzzle"]})


def test_raw3_is_front_and_raw1_is_rear(bundle: Path):
    contract = _applied(bundle)
    assert contract["raw_to_semantic"]["3"] == "front"
    assert contract["raw_to_semantic"]["1"] == "rear"
    assert contract["semantic_to_raw"]["front"] == 3
    assert contract["semantic_to_raw"]["rear"] == 1


def test_side_and_vertical_pairs_are_unchanged(bundle: Path):
    contract = _applied(bundle)
    assert contract["raw_to_semantic"]["2"] == "right"
    assert contract["raw_to_semantic"]["0"] == "left"
    assert contract["raw_to_semantic"]["4"] == "top"
    assert contract["raw_to_semantic"]["5"] == "bottom"


def test_every_semantic_pair_is_opposed(bundle: Path):
    contract = _applied(bundle)
    for key in ("front_rear_direction_dot", "left_right_direction_dot",
                "top_bottom_direction_dot"):
        assert contract[key] == pytest.approx(-1.0, abs=1e-3)


def test_a_non_opposed_permutation_is_rejected(bundle: Path):
    """front and rear on the same axis sign must fail rather than be recorded."""
    broken = dict(PROVEN_RAW_TO_SEMANTIC)
    broken[1], broken[2] = "right", "rear"   # rear now shares the left/right axis
    with pytest.raises(RuntimeError, match="CAMERA_PERMUTATION_NOT_OPPOSED"):
        apply_permutation(bundle, broken, {"semantic_source": "test", "evidence": []})


def test_a_non_bijective_permutation_is_rejected(bundle: Path):
    duplicated = dict(PROVEN_RAW_TO_SEMANTIC)
    duplicated[3] = "rear"
    with pytest.raises(RuntimeError, match="CAMERA_PERMUTATION_NOT_A_BIJECTION"):
        apply_permutation(bundle, duplicated, {"semantic_source": "test", "evidence": []})


def test_raw_order_and_control_arrays_are_untouched(bundle: Path):
    contract = _applied(bundle)
    assert [int(view["index"]) for view in contract["views"]] == list(range(6))
    assert contract["raw_order_preserved"] is True
    assert contract["control_arrays_rewritten"] is False
    for view in contract["views"]:
        # The label moved; the files did not.
        assert view["control_file_prefix"] == RAW_PREFIXES[int(view["index"])]
        assert (bundle / view["control_mask_filename"]).is_file()


def test_relabelling_records_provenance_instead_of_renaming(bundle: Path):
    contract = _applied(bundle)
    assert contract["superseded_classification"] == "PANDA_CAMERA_SEMANTICS_PREVIOUS_CONTRACT_REJECTED"
    assert contract["superseded_index_semantics"] == {str(i): BUILDER_GUESS[i] for i in range(6)}
    for view in contract["views"]:
        assert view["superseded_semantic"] == BUILDER_GUESS[int(view["index"])]
        assert view["raw_semantic_name"] == RAW_PREFIXES[int(view["index"])]
        assert view["semantic_source"] == "USER_VISION_REVIEW_20260803"
    # No historical control file was renamed away.
    for prefix in RAW_PREFIXES:
        assert (bundle / f"{prefix}_mask.png").is_file()


def test_readers_resolve_files_through_the_contract_not_the_label(bundle: Path):
    contract = _applied(bundle)
    for view in contract["views"]:
        assert view["semantic_name"] != view["control_file_prefix"] or view["index"] >= 4
        assert file_prefix(view) == view["control_file_prefix"]
        assert sheet_file_prefix(view) == view["control_file_prefix"]


def test_no_worker_carries_a_hardcoded_six_view_name_tuple():
    """A positional tuple of the six labels is exactly how the wrong mapping got baked in."""
    pattern = re.compile(r"""["']front["']\s*,\s*["']right["']""")
    for name in ("apply_camera_permutation.py", "render_control_bundle_texture.py",
                 "prove_bar_repair.py", "prove_camera_semantics.py"):
        source = (ROOT / "workers" / name).read_text(encoding="utf-8")
        assert not pattern.search(source), f"{name} hardcodes an ordered view-name tuple"


def test_shipped_contract_matches_the_proven_permutation():
    config = json.loads(
        (ROOT / "configs" / "texture"
         / "gpu_panda_mvadapter_ig2mv_sd21_repaired_384x2_20260803.json").read_text(encoding="utf-8"))
    semantics = config["camera_semantics"]
    assert semantics["semantic_to_raw"] == {"front": 3, "right": 2, "rear": 1,
                                            "left": 0, "top": 4, "bottom": 5}
    assert semantics["raw_to_semantic"] == {str(k): v for k, v in PROVEN_RAW_TO_SEMANTIC.items()}
    assert semantics["superseded_raw_to_semantic"]["1"] == "front"
    assert semantics["superseded_raw_to_semantic"]["3"] == "rear"
    assert config["primary"]["steps"] == 2
    assert config["primary"]["resolution"] == 384
    assert config["primary"]["seed"] == 12345


def test_old_production_run_is_not_described_as_visually_passed():
    config = json.loads(
        (ROOT / "configs" / "texture"
         / "gpu_panda_mvadapter_ig2mv_sd21_production_384_20260803.json").read_text(encoding="utf-8"))
    result = config["production_result"]
    assert result["classification"] == "MVADAPTER_NUMERICALLY_PROVEN_OLD_CONTROLS_UNPROMOTED"
    assert result["visual_qa"] != "PROVEN"
    assert result["superseded_classification"] == "MVADAPTER_FULL_RUN_PROVEN"
    assert result["unpromoted_reasons"]
