from __future__ import annotations

import copy
import math

import pytest

from lowvram3d.asset_profiles import PROFILES
from lowvram3d.thin_feature_anchors import (
    AnchorReceiptValidationError,
    SIX_ORTHOGRAPHIC_VIEWS,
    discover_thin_feature_anchors,
    parse_anchor_receipt,
    serialize_anchor_receipt,
    validate_anchor_receipt,
)
from thin_feature_anchor_fixtures import (
    attached_narrow_strip,
    detached_singleton,
    occluded_singleton,
    ordinary_body,
)


SOURCE_HASH = "7" * 64
PROFILE = PROFILES["humanoid_complex_accessories"]


def discover(mesh):
    return discover_thin_feature_anchors(
        mesh,
        source_mesh_sha256=SOURCE_HASH,
        profile=PROFILE,
    )


def test_uses_production_six_view_contract() -> None:
    receipt = discover(ordinary_body())
    assert [view["name"] for view in receipt["view_set"]] == [
        "front",
        "right",
        "back",
        "left",
        "top",
        "bottom",
    ]
    assert len(SIX_ORTHOGRAPHIC_VIEWS) == 6


def test_ordinary_body_does_not_create_anchor() -> None:
    receipt = discover(ordinary_body())
    assert receipt["anchors"] == []


def test_detached_singleton_with_silhouette_support_is_retained() -> None:
    receipt = discover(detached_singleton())
    assert len(receipt["anchors"]) == 1
    anchor = receipt["anchors"][0]
    assert anchor["candidate_kind"] == "detached_component"
    assert anchor["supported_views"]
    assert max(
        support["exclusive_pixels"] for support in anchor["per_view_support"].values()
    ) > 0


def test_seed_and_bounds_coordinates_are_normalized_floats() -> None:
    anchor = discover(detached_singleton())["anchors"][0]
    assert all(type(value) is float for seed in anchor["seeds"] for value in seed)
    bounds = anchor["bounds_normalized"]
    assert all(type(value) is float for value in bounds["min"] + bounds["max"])
    assert all(-1.0 <= value <= 1.0 for seed in anchor["seeds"] for value in seed)
    assert all(-1.0 <= value <= 1.0 for value in bounds["min"] + bounds["max"])

    decoded = parse_anchor_receipt(
        serialize_anchor_receipt(discover(detached_singleton())),
        expected_source_mesh_sha256=SOURCE_HASH,
    )
    decoded_anchor = decoded["anchors"][0]
    assert decoded_anchor["seeds"] == anchor["seeds"]
    assert decoded_anchor["bounds_normalized"] == bounds


def test_receipt_persists_clean_source_normalization_frame() -> None:
    receipt = discover(detached_singleton())
    frame = receipt["discovery"]["normalization_frame"]
    assert all(type(value) is float for value in frame["center"])
    assert all(type(value) is float for value in frame["bounds_min"] + frame["bounds_max"])
    assert type(frame["diagonal"]) is float and frame["diagonal"] > 0.0
    assert all(
        math.isclose(center, (low + high) * 0.5, abs_tol=1.0e-7)
        for center, low, high in zip(frame["center"], frame["bounds_min"], frame["bounds_max"])
    )


def test_normalization_frame_validation_rejects_changed_extrema() -> None:
    receipt = discover(detached_singleton())
    receipt["discovery"]["normalization_frame"]["bounds_max"][0] += 0.25
    with pytest.raises(AnchorReceiptValidationError, match="normalization_frame"):
        validate_anchor_receipt(receipt)


def test_integer_coordinate_units_are_rejected_in_receipts() -> None:
    receipt = discover(detached_singleton())
    receipt["anchors"][0]["seeds"][0][0] = 1
    with pytest.raises(AnchorReceiptValidationError, match="normalized floats"):
        validate_anchor_receipt(receipt)


def test_detached_candidate_without_silhouette_support_is_not_registered() -> None:
    receipt = discover(occluded_singleton())
    assert receipt["discovery"]["candidate_counts"]["detached"] == 1
    assert receipt["anchors"] == []


def test_attached_narrow_strip_is_discovered_from_main_component() -> None:
    receipt = discover(attached_narrow_strip())
    attached = [
        anchor for anchor in receipt["anchors"]
        if anchor["candidate_kind"] == "attached_protrusion"
    ]
    assert attached
    assert any(anchor["supported_views"] for anchor in attached)


def test_ids_and_receipt_bytes_ignore_object_ordering() -> None:
    normal = discover(detached_singleton(reverse_objects=False))
    reversed_objects = discover(detached_singleton(reverse_objects=True))
    assert [item["anchor_id"] for item in normal["anchors"]] == [
        item["anchor_id"] for item in reversed_objects["anchors"]
    ]
    assert serialize_anchor_receipt(normal) == serialize_anchor_receipt(reversed_objects)


def test_serialization_round_trips_canonical_bytes() -> None:
    encoded = serialize_anchor_receipt(discover(attached_narrow_strip()))
    decoded = parse_anchor_receipt(encoded, expected_source_mesh_sha256=SOURCE_HASH)
    assert serialize_anchor_receipt(decoded) == encoded


def test_validation_reports_missing_required_fields() -> None:
    receipt = discover(ordinary_body())
    del receipt["view_set"]
    with pytest.raises(AnchorReceiptValidationError, match="missing required fields: view_set"):
        validate_anchor_receipt(receipt)


def test_validation_reports_source_hash_mismatch() -> None:
    receipt = discover(ordinary_body())
    with pytest.raises(AnchorReceiptValidationError, match="source-hash mismatch"):
        validate_anchor_receipt(receipt, expected_source_mesh_sha256="8" * 64)


def test_validation_reports_malformed_schema() -> None:
    receipt = copy.deepcopy(discover(ordinary_body()))
    receipt["schema_version"] = {"major": 1}
    with pytest.raises(AnchorReceiptValidationError, match="malformed schema_version"):
        validate_anchor_receipt(receipt)


def test_parse_reports_malformed_json() -> None:
    with pytest.raises(AnchorReceiptValidationError, match="malformed anchor receipt JSON"):
        parse_anchor_receipt(b"{not-json")
