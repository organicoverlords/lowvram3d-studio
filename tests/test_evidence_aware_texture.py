import numpy as np

from lowvram3d.texture_provenance import (
    EvidenceState,
    FrequencyAuthority,
    Lineage,
    create_empty_atlas_provenance,
    create_empty_triangle_provenance,
    raw_rgb_allowed,
    validate_evidence_invariants,
)
from workers.assign_texture_sources import assign
from workers.complete_unobserved_surfaces import complete
from workers.fuse_texture_evidence import fuse
from workers.atlas_raster import injectivity


def test_raw_rgb_requires_all_direct_gates_and_allows_validated_generated_views():
    assert raw_rgb_allowed(
        evidence_state=np.array([EvidenceState.DIRECT_OBSERVED, EvidenceState.UNOBSERVED,
                                 EvidenceState.GENERATED_OBSERVED]),
        frequency_authority=np.array([FrequencyAuthority.FULL] * 3),
        visible=np.ones(3, bool), facing=np.ones(3), face_id_match=np.ones(3, bool),
        source_mask_valid=np.ones(3, bool), coherent_assignment=np.ones(3, bool),
    ).tolist() == [True, False, True]


def test_unobserved_lineage_cannot_claim_full_frequency():
    p = create_empty_atlas_provenance(2)
    p["evidence_state"][0] = EvidenceState.PROCEDURAL_COMPLETION
    p["frequency_authority"][0] = FrequencyAuthority.LOW_ONLY
    p["lineage"][0] = Lineage.ORIGINAL_FACE
    assert validate_evidence_invariants(p)["passed"] is False
    p["lineage"][0] = 0
    assert validate_evidence_invariants(p)["passed"] is True


def test_completion_uses_region_palette_not_raw_donor_pixels():
    result, report = complete(
        np.array([[255, 0, 0], [0, 255, 0]], np.uint8),
        np.array([EvidenceState.DIRECT_OBSERVED, EvidenceState.UNOBSERVED], np.uint8),
        np.array([1, 1], np.int32),
    )
    assert result["rgb"][1].tolist() == [255, 0, 0]
    assert report["raw_rgb_donors"] == 0
    assert result["frequency_authority"][1] == FrequencyAuthority.LOW_ONLY


def test_assignment_never_labels_unobserved_triangles():
    labels, confidence, report = assign(
        {"evidence_state": np.array([[1, 4], [4, 4]], np.uint8),
         "confidence": np.array([[0.8, 0.0], [0.0, 0.0]], np.float32)},
        np.array([[0, 1, 2], [2, 3, 0]], np.int32),
    )
    assert labels.tolist() == [0, -1]
    assert confidence[1] == 0
    assert report["unobserved_receive_no_view"] is True


def test_positive_area_uv_overlap_is_a_hard_conflict():
    uv = np.array([[0.1, 0.1], [0.8, 0.1], [0.1, 0.8],
                   [0.1, 0.1], [0.8, 0.1], [0.1, 0.8]], np.float32)
    triangles = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    assert injectivity(uv, triangles, 64)["injective"] is False


def test_frequency_fusion_preserves_direct_and_keeps_completion_low_only():
    direct = np.array([[[255, 0, 0], [0, 0, 0]]], np.uint8)
    completion = np.array([[[0, 0, 0], [4, 5, 6]]], np.uint8)
    output, report = fuse(direct, completion,
                          np.array([[EvidenceState.DIRECT_OBSERVED,
                                     EvidenceState.PROCEDURAL_COMPLETION]], np.uint8))
    assert output[0, 0].tolist() == [255, 0, 0]
    assert output[0, 1].tolist() == [4, 5, 6]
    assert report["unobserved_full_frequency_texels"] == 0
