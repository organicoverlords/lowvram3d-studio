from lowvram3d.vision_qa.contracts import VisionQaPacket
from lowvram3d.vision_qa.prompting import build_user_prompt, select_image_artifacts


def packet():
    return VisionQaPacket.from_dict({
        "schema": "vision_qa_packet_v1",
        "packet_id": "p1",
        "stage": "geometry",
        "artifacts": [
            {"artifact_id": "wire", "kind": "wireframe", "path": "wire.png"},
            {"artifact_id": "source", "kind": "source", "path": "source.png"},
            {"artifact_id": "log", "kind": "log", "path": "run.log"},
        ],
        "hard_gates": [{"name": "REPROJECTION", "status": "PROVEN"}],
    })


def test_source_image_is_prioritized():
    selected = select_image_artifacts(packet().artifacts)
    assert selected[0].artifact_id == "source"
    assert all(item.artifact_id != "log" for item in selected)


def test_prompt_contains_action_whitelist_and_no_paths():
    prompt = build_user_prompt(packet())
    assert "allowed_actions" in prompt
    assert "repair_mesh_coverage_from_saved_arrays" in prompt
    assert '"path"' not in prompt
