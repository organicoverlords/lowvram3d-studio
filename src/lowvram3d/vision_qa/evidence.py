"""Evidence-pack construction and stage completeness validation."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from .contracts import ContractError, EvidenceArtifact, EvidenceKind


STAGE_REQUIREMENTS: dict[str, frozenset[EvidenceKind]] = {
    "geometry": frozenset({
        EvidenceKind.SOURCE,
        EvidenceKind.UNLIT,
        EvidenceKind.SILHOUETTE,
        EvidenceKind.WIREFRAME,
        EvidenceKind.DEPTH,
        EvidenceKind.NORMAL,
        EvidenceKind.METRICS,
    }),
    "uv": frozenset({
        EvidenceKind.UV_ATLAS,
        EvidenceKind.UV_SEAMS,
        EvidenceKind.METRICS,
        EvidenceKind.RECEIPT,
    }),
    "texture": frozenset({
        EvidenceKind.SOURCE,
        EvidenceKind.BEAUTY,
        EvidenceKind.UNLIT,
        EvidenceKind.ALBEDO,
        EvidenceKind.UV_ATLAS,
        EvidenceKind.MASK,
        EvidenceKind.METRICS,
    }),
    "rig": frozenset({
        EvidenceKind.BEAUTY,
        EvidenceKind.SILHOUETTE,
        EvidenceKind.WIREFRAME,
        EvidenceKind.METRICS,
    }),
    "export": frozenset({EvidenceKind.BEAUTY, EvidenceKind.RECEIPT, EvidenceKind.LOG}),
}


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def seal_artifacts(
    artifacts: Iterable[EvidenceArtifact],
    *,
    root: str | Path | None = None,
    require_files: bool = True,
) -> tuple[EvidenceArtifact, ...]:
    root_path = Path(root).resolve() if root is not None else None
    sealed: list[EvidenceArtifact] = []
    seen: set[str] = set()
    for artifact in artifacts:
        if artifact.artifact_id in seen:
            raise ContractError(f"duplicate artifact id: {artifact.artifact_id}")
        seen.add(artifact.artifact_id)
        path = Path(artifact.path)
        resolved = path if path.is_absolute() else ((root_path / path) if root_path else path)
        if require_files and artifact.required and not resolved.is_file():
            raise ContractError(f"required evidence is missing: {resolved}")
        digest = sha256_file(resolved) if resolved.is_file() else artifact.sha256
        sealed.append(EvidenceArtifact(
            artifact_id=artifact.artifact_id,
            kind=artifact.kind,
            path=artifact.path,
            sha256=digest,
            view=artifact.view,
            required=artifact.required,
            description=artifact.description,
            metadata=dict(artifact.metadata),
        ))
    return tuple(sealed)


def missing_stage_evidence(stage: str, artifacts: Iterable[EvidenceArtifact]) -> set[EvidenceKind]:
    required = STAGE_REQUIREMENTS.get(stage)
    if required is None:
        raise ContractError(f"unknown vision QA stage: {stage}")
    present = {artifact.kind for artifact in artifacts if artifact.required}
    return set(required - present)


def assert_stage_complete(stage: str, artifacts: Iterable[EvidenceArtifact]) -> None:
    missing = missing_stage_evidence(stage, artifacts)
    if missing:
        names = ", ".join(sorted(item.value for item in missing))
        raise ContractError(f"{stage} evidence pack is incomplete; missing: {names}")
