import pytest

from lowvram3d.vision_qa.contracts import ContractError
from lowvram3d.vision_qa.specialists import validate_specialist_output


def base(task, observations=None, metrics=None):
    return {
        "schema": "vision_specialist_output_v1",
        "model_id": "test-model",
        "task": task,
        "source_artifact_id": "source",
        "observations": observations or [],
        "metrics": metrics or {},
        "warnings": [],
    }


def test_label_box_is_normalized_and_positive_area():
    output = base("labels", [{"label": "face", "confidence": 0.9, "bbox_norm_xyxy": [0.1, 0.2, 0.4, 0.6]}])
    assert validate_specialist_output(output)["task"] == "labels"


def test_bad_label_box_fails_closed():
    output = base("labels", [{"label": "face", "confidence": 0.9, "bbox_norm_xyxy": [0.4, 0.2, 0.1, 0.6]}])
    with pytest.raises(ContractError, match="non-positive"):
        validate_specialist_output(output)


def test_mask_requires_artifact_and_valid_area():
    output = base("masks", [{
        "label": "face", "confidence": 0.95, "area_fraction": 0.12,
        "bbox_norm_xyxy": [0.1, 0.1, 0.5, 0.5], "mask_artifact_id": "face-mask",
    }])
    validate_specialist_output(output)


def test_depth_requires_finite_valid_fraction():
    validate_specialist_output(base("depth", metrics={"valid_fraction": 0.97, "aligned_l1": 0.04}))
    with pytest.raises(ContractError, match="finite"):
        validate_specialist_output(base("depth", metrics={"valid_fraction": float("nan")}))


def test_pose_points_are_normalized():
    output = base("pose", [{"joint": "left_wrist", "confidence": 0.8, "point_norm_xy": [0.25, 0.75]}])
    validate_specialist_output(output)


def test_embedding_requires_one_positive_dimension():
    output = base("embedding", [{"dimension": 2048, "embedding_artifact_id": "embedding"}])
    validate_specialist_output(output)
