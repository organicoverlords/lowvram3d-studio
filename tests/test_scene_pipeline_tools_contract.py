"""CPU-only contract checks for the narrow official MCP toolset extension."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CPP = ROOT / "unreal" / "ScenePipelineTools" / "Source" / "ScenePipelineTools" / "ScenePipelineTools.cpp"
HEADER = ROOT / "unreal" / "ScenePipelineTools" / "Source" / "ScenePipelineTools" / "ScenePipelineTools.h"


def test_allowlisted_tools_and_workflow_are_explicit() -> None:
    text = CPP.read_text(encoding="utf-8")
    assert 'diagnostic_cube_v1' in text
    assert 'scene_pipeline_start_job' in text
    assert 'scene_pipeline_get_job_status' in text
    assert 'scene_pipeline_cancel_job' in text
    assert 'UNKNOWN_WORKFLOW' in text


def test_source_map_protection_is_fail_closed() -> None:
    text = CPP.read_text(encoding="utf-8")
    assert 'SOURCE_MAP_PROTECTION_REJECTED' in text
    assert 'Castlegrounds' in text
    assert '/Game/AgentProof/MCP/L_MCP_Diagnostic' in text


def test_tick_state_machine_contains_required_order_and_no_retry_policy() -> None:
    text = CPP.read_text(encoding="utf-8")
    order = [
        'MAP_CREATED',
        'MAP_READY',
        'SAVE_REQUESTED',
        'SAVED',
        'RELOAD_REQUESTED',
        'RELOADED',
        'CUBE_SPAWNED',
        'VALIDATED',
    ]
    positions = [text.index(value) for value in order]
    assert positions == sorted(positions)
    assert 'never_auto_retry_mutation' in text
    assert 'switch (ActiveJob->Phase++)' in text
    assert 'TEXT("COMPLETED")' in text


def test_receipts_are_atomic_and_terminal_receipt_is_written() -> None:
    text = CPP.read_text(encoding="utf-8")
    assert 'TEXT(".tmp")' in text
    assert 'WriteJsonAtomic' in text
    assert 'TEXT("accepted.json")' in text
    assert 'TEXT("preflight.json")' in text
    assert 'TEXT("final.json")' in text


def test_remote_input_cannot_execute_python_console_or_process_commands() -> None:
    text = CPP.read_text(encoding="utf-8")
    assert 'execute_python' not in text
    assert 'execute_console_command' not in text
    assert 'TerminateProcess' not in text
    assert 'CreateProcess' not in text


def test_toolset_is_registered_with_existing_toolset_registry() -> None:
    header = HEADER.read_text(encoding="utf-8")
    source = CPP.read_text(encoding="utf-8")
    assert 'UToolsetDefinition' in header
    assert 'RegisterToolsetClass' in source
    assert 'ModelContextProtocol' in (ROOT / "unreal" / "ScenePipelineTools" / "ScenePipelineTools.uplugin").read_text(encoding="utf-8")
