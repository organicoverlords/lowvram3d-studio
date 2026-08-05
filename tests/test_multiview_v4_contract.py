from lowvram3d.view_generation_contract import ViewRequest, ViewResult, contract_hash
from plan_next_semantic_view import candidate_score


def test_view_contract_is_backend_neutral_and_stable():
    request=ViewRequest("shaman-rear-001","rear",{"azimuth_degrees":180},"cam","mesh","uv","prov","GENERATED_REAR",31.4)
    assert contract_hash(request.to_dict())==contract_hash(request.to_dict())
    result=ViewResult("shaman-rear-001","manual_registered_view","rear.png","rear.png","cam","GENERATED_REAR",.8,1,"manual","hash")
    assert result.to_dict()["status"]=="GENERATED_VIEW_ACCEPTED"


def test_planner_score_is_normalized_and_non_exterior_views_are_not_competitors():
    rear={"view_name":"rear","visible_pixels":100,"expected_new_semantic_coverage_percent":80,"donor_coverage_percent":0,"component_prior_coverage_percent":80,"global_prior_coverage_percent":0,"protected_face_overlap":0,"forbidden_face_lineage_exposure":0,"original_source_coverage_percent":10,"meaningful_exterior":True}
    bottom=dict(rear,view_name="bottom",visible_pixels=1000,meaningful_exterior=False)
    assert -1.0 < candidate_score(rear, 1000) < 1.0
    assert bottom["meaningful_exterior"] is False
