import numpy as np

from lowvram3d.texture_provenance import Lineage, SourceClass, create_empty_atlas_provenance, create_empty_triangle_provenance, direct_projection_lineage, merge_face_refinement_lineage, propagate_donor_lineage, rasterize_triangle_lineage_to_atlas, validate_no_forbidden_lineage


def test_donor_lineage_is_transitive():
    a=create_empty_triangle_provenance(3); b=create_empty_triangle_provenance(3)
    direct_projection_lineage(a,[0],SourceClass.ORIGINAL_FACE); propagate_donor_lineage(b,[1],a,[0]); propagate_donor_lineage(a,[2],b,[1])
    assert a["lineage"][2] & Lineage.ORIGINAL_FACE
    assert a["lineage"][2] & Lineage.DONOR_TRANSFER


def test_face_refinement_does_not_erase_original_face():
    p=create_empty_triangle_provenance(1); direct_projection_lineage(p,[0],SourceClass.ORIGINAL_FACE); merge_face_refinement_lineage(p,[0])
    assert p["lineage"][0] & Lineage.ORIGINAL_FACE
    assert p["lineage"][0] & Lineage.FACE_REFINEMENT


def test_rasterized_lineage_and_forbidden_gate():
    p=create_empty_triangle_provenance(2); direct_projection_lineage(p,[1],SourceClass.ORIGINAL_FACE); atlas=rasterize_triangle_lineage_to_atlas(p,np.array([[1,0],[0,-1]],np.int32)); result=validate_no_forbidden_lineage({"lineage":atlas["lineage"].reshape(-1)},Lineage.ORIGINAL_FACE)
    assert atlas["lineage"][0,0] & Lineage.ORIGINAL_FACE
    assert not result["passed"]
