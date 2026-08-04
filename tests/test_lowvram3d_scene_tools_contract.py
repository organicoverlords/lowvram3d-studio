from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "unreal" / "LowVRAM3DSceneTools"
PYTHON = PLUGIN / "Content" / "Python" / "lowvram3d_scene_tools.py"


def test_dedicated_scene_tools_plugin_declares_required_editor_dependencies():
    manifest = json.loads((PLUGIN / "LowVRAM3DSceneTools.uplugin").read_text(encoding="utf-8"))
    names = {item["Name"] for item in manifest["Plugins"]}
    assert {"PythonScriptPlugin", "EditorScriptingUtilities", "ModelContextProtocol", "ToolsetRegistry", "MovieRenderPipeline", "SequencerScripting", "LevelSequenceEditor"} <= names


def test_scene_tools_are_allowlisted_and_have_no_arbitrary_execution_entrypoint():
    text = PYTHON.read_text(encoding="utf-8")
    required = {"get_camera_contract", "capture_named_camera_fast", "create_camera_cut_sequence", "render_named_camera_mrq", "apply_visual_capture_profile", "restore_capture_visibility", "audit_actor_materials", "audit_source_shell_projection", "validate_capture_image"}
    assert required <= {name for name in required if name in text}
    assert "execute_python" not in text
    assert "execute_console_command" not in text
    assert "terminate_process" not in text


def test_fast_capture_is_named_camera_and_render_target_only():
    text = PYTHON.read_text(encoding="utf-8")
    assert "SceneCapture2D" in text
    assert "create_render_target2d" in text
    assert "export_render_target" in text
    assert "AutomationLibrary.take_high_res_screenshot" not in text
    assert "get_level_viewport_camera_info" not in text
    assert "player_camera_used" in text
