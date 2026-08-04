from lowvram3d.view_generation_contract import ViewRequest, ViewResult, contract_hash


def test_view_contract_is_backend_neutral_and_stable():
    request=ViewRequest("shaman-rear-001","rear",{"azimuth_degrees":180},"cam","mesh","uv","prov","GENERATED_REAR",31.4)
    assert contract_hash(request.to_dict())==contract_hash(request.to_dict())
    result=ViewResult("shaman-rear-001","manual_registered_view","rear.png","rear.png","cam","GENERATED_REAR",.8,1,"manual","hash")
    assert result.to_dict()["status"]=="GENERATED_VIEW_ACCEPTED"
