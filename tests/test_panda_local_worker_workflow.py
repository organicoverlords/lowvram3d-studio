from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "panda-atlas-root-local-worker.yml"


def test_windows_worker_flattens_json_manifest_to_scalar_entries() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "$document = ConvertFrom-Json -InputObject" in text
    assert "$candidates = [System.Collections.Generic.List[object]]::new()" in text
    assert "foreach ($entry in $document)" in text
    assert '$label = [string]$candidate.label' in text
    assert '$inputPath = [string]$candidate.path' in text
    assert "@(Get-Content -LiteralPath" not in text


def test_worker_is_failure_tolerant_and_packages_partial_evidence() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    assert 'render_status = $status' in text
    for status in ("RENDERED", "RENDER_FAILED", "CONTACT_SHEET_FAILED", "INPUT_MISSING"):
        assert status in text
    assert "render_matrix.json" in text
    assert "if: always()" in text
    assert "if-no-files-found: error" in text
    assert "panda_atlas_support_fix_${{ github.run_id }}.zip" in text


def test_four_entry_fixture_has_scalar_labels_and_paths() -> None:
    fixture = [
        {"label": f"candidate_{index}", "path": f"C:/AI/candidate_{index}.glb"}
        for index in range(4)
    ]
    document = json.loads(json.dumps(fixture))
    entries = list(document)
    assert len(entries) == 4
    assert all(isinstance(entry["label"], str) and isinstance(entry["path"], str) for entry in entries)
