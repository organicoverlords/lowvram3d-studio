import json
from pathlib import Path

import pytest

from workers.comfyui_gpu_texture_job import (
    Binding,
    ComfyError,
    apply_binding,
    build_attempt_workflow,
    classify_error,
    read_bindings,
)


def workflow_fixture():
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
        "2": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 1}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
    }


def test_read_bindings_and_apply_binding():
    bindings = read_bindings(
        {
            "bindings": {
                "source": {"node": 1, "input": "image"},
                "width": {"node": "2", "input": "width"},
            }
        }
    )
    workflow = workflow_fixture()
    apply_binding(workflow, bindings["source"], "uploaded/source.png")
    apply_binding(workflow, bindings["width"], 512)
    assert workflow["1"]["inputs"]["image"] == "uploaded/source.png"
    assert workflow["2"]["inputs"]["width"] == 512


def test_build_attempt_workflow_does_not_mutate_template():
    template = workflow_fixture()
    bindings = {
        "source": Binding("1", "image"),
        "width": Binding("2", "width"),
        "height": Binding("2", "height"),
        "seed": Binding("3", "seed"),
        "prompt": Binding("4", "text"),
    }
    result = build_attempt_workflow(
        template,
        bindings,
        {
            "source": "uploaded.png",
            "width": 512,
            "height": 512,
            "seed": 42,
            "prompt": "true rear view",
        },
    )
    assert template["1"]["inputs"]["image"] == "old.png"
    assert result["1"]["inputs"]["image"] == "uploaded.png"
    assert result["2"]["inputs"] == {"width": 512, "height": 512}
    assert result["3"]["inputs"]["seed"] == 42
    assert result["4"]["inputs"]["text"] == "true rear view"


def test_missing_bound_node_or_input_fails_closed():
    workflow = workflow_fixture()
    with pytest.raises(ComfyError, match="BOUND_NODE_MISSING"):
        apply_binding(workflow, Binding("999", "image"), "x")
    with pytest.raises(ComfyError, match="BOUND_INPUT_MISSING"):
        apply_binding(workflow, Binding("1", "missing"), "x")


def test_invalid_binding_schema_fails_closed():
    with pytest.raises(ComfyError, match="INVALID_BINDING"):
        read_bindings({"bindings": {"source": {"node": "1"}}})
    with pytest.raises(ComfyError, match="BINDINGS_MUST_BE_OBJECT"):
        read_bindings({"bindings": []})


def test_oom_classification_is_specific():
    assert classify_error("torch.cuda.OutOfMemoryError: Allocation on device").__name__ == "ComfyOOM"
    assert classify_error("missing custom node").__name__ == "ComfyError"
