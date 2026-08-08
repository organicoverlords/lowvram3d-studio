from pathlib import Path

import pytest

from lowvram3d.vision_qa.contracts import ContractError, EvidenceArtifact, EvidenceKind
from lowvram3d.vision_qa.evidence import assert_stage_complete, seal_artifacts, sha256_file


def test_seal_artifact_hashes_file(tmp_path: Path):
    path = tmp_path / "evidence.txt"
    path.write_text("proof", encoding="utf-8")
    artifact = EvidenceArtifact("receipt", EvidenceKind.RECEIPT, path.name)
    sealed = seal_artifacts([artifact], root=tmp_path)
    assert sealed[0].sha256 == sha256_file(path)


def test_missing_required_file_fails_closed(tmp_path: Path):
    artifact = EvidenceArtifact("source", EvidenceKind.SOURCE, "missing.png")
    with pytest.raises(ContractError, match="missing"):
        seal_artifacts([artifact], root=tmp_path)


def test_geometry_pack_requires_diagnostic_views():
    artifacts = [EvidenceArtifact("source", EvidenceKind.SOURCE, "source.png")]
    with pytest.raises(ContractError, match="incomplete"):
        assert_stage_complete("geometry", artifacts)


def test_unknown_stage_fails_closed():
    with pytest.raises(ContractError, match="unknown"):
        assert_stage_complete("magic", [])
