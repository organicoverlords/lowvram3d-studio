from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping, Sequence


PROVENANCE_SCHEMA = 1
PROVENANCE_SUFFIX = ".provenance.json"
PROVENANCE_REQUIRED_SUFFIX = ".provenance-required"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def provenance_path(path: Path) -> Path:
    return path.with_name(path.name + PROVENANCE_SUFFIX)


def provenance_required_path(path: Path) -> Path:
    return path.with_name(path.name + PROVENANCE_REQUIRED_SUFFIX)


def _resolved(path: Path, cwd: Path | None = None) -> Path:
    candidate = path if path.is_absolute() or cwd is None else cwd / path
    return candidate.resolve(strict=False)


def fingerprint_command_inputs(
    command: Sequence[str],
    cwd: Path,
    artifact_paths: Mapping[str, str],
) -> dict[str, str]:
    """Hash existing file arguments before a stage starts.

    The executable itself is deliberately excluded: hashing a multi-gigabyte Blender or Python
    installation on every stage would dominate short jobs. Script files, source images, meshes,
    workflow JSON, checkpoints passed as files, and other existing file arguments are included.
    Directory arguments are excluded because model/cache trees can be enormous and mutable.
    """
    outputs = {_resolved(Path(raw), cwd) for raw in artifact_paths.values()}
    fingerprints: dict[str, str] = {}
    for token in command[1:]:
        if not token or token.startswith("-"):
            continue
        candidate = _resolved(Path(token), cwd)
        if candidate in outputs or not candidate.is_file():
            continue
        key = str(candidate)
        if key not in fingerprints:
            fingerprints[key] = sha256_file(candidate)
    return dict(sorted(fingerprints.items()))


def stage_command_fingerprint(
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str] | None,
    artifact_paths: Mapping[str, str],
    input_fingerprints: Mapping[str, str],
) -> str:
    payload = {
        "schema": PROVENANCE_SCHEMA,
        "command": [str(item) for item in command],
        "cwd": str(cwd.resolve(strict=False)),
        "env": dict(sorted((str(k), str(v)) for k, v in (env or {}).items())),
        "artifacts": dict(sorted((str(k), str(_resolved(Path(v), cwd))) for k, v in artifact_paths.items())),
        "inputs": dict(sorted(input_fingerprints.items())),
    }
    return canonical_sha256(payload)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def arm_artifact_provenance(artifact_paths: Mapping[str, str]) -> None:
    """Make old outputs fail closed while a declared stage is being rerun."""
    for raw in artifact_paths.values():
        artifact = Path(raw)
        artifact.parent.mkdir(parents=True, exist_ok=True)
        provenance_path(artifact).unlink(missing_ok=True)
        _write_json_atomic(
            provenance_required_path(artifact),
            {"schema": PROVENANCE_SCHEMA, "status": "required"},
        )


def write_artifact_provenance(
    *,
    stage: str,
    command: Sequence[str],
    cwd: Path,
    env: Mapping[str, str] | None,
    artifact_paths: Mapping[str, str],
    input_fingerprints: Mapping[str, str],
    command_fingerprint: str,
) -> tuple[dict[str, str], dict[str, str]]:
    artifact_fingerprints: dict[str, str] = {}
    provenance_files: dict[str, str] = {}
    for logical_name, raw in artifact_paths.items():
        artifact = Path(raw)
        artifact_hash = sha256_file(artifact)
        artifact_fingerprints[logical_name] = artifact_hash
        sidecar = provenance_path(artifact)
        payload = {
            "schema": PROVENANCE_SCHEMA,
            "status": "complete",
            "stage": stage,
            "logical_name": logical_name,
            "artifact": str(artifact.resolve(strict=False)),
            "artifact_sha256": artifact_hash,
            "command": [str(item) for item in command],
            "command_sha256": command_fingerprint,
            "cwd": str(cwd.resolve(strict=False)),
            "env": dict(sorted((str(k), str(v)) for k, v in (env or {}).items())),
            "inputs": dict(sorted(input_fingerprints.items())),
        }
        _write_json_atomic(sidecar, payload)
        provenance_files[logical_name] = str(sidecar)
    return artifact_fingerprints, provenance_files


def artifact_provenance_is_valid(path: Path) -> bool:
    required = provenance_required_path(path)
    sidecar = provenance_path(path)
    if not required.exists() and not sidecar.exists():
        return True  # Legacy artifacts remain readable; new stages arm provenance before running.
    if not sidecar.is_file():
        return False
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8-sig"))
        if payload.get("schema") != PROVENANCE_SCHEMA or payload.get("status") != "complete":
            return False
        if payload.get("artifact_sha256") != sha256_file(path):
            return False
        inputs = payload.get("inputs", {})
        if not isinstance(inputs, dict):
            return False
        for raw, expected_hash in inputs.items():
            source = Path(raw)
            if not source.is_file() or sha256_file(source) != expected_hash:
                return False
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
