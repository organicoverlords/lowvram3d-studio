from pathlib import Path
import json


def test_blender_handoff_contract_documents_non_promotion(tmp_path: Path):
    report = {
        "promotion_allowed": False,
        "requirements": ["proof render before Unreal export"],
    }
    path = tmp_path / "handoff.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["promotion_allowed"] is False
    assert "proof render before Unreal export" in loaded["requirements"]
