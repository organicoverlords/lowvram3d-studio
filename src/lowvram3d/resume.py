from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import JobReceipt


CONTRACT_VERSION = 1


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_contract(kind: str, source: Path, parameters: dict[str, Any]) -> dict[str, Any]:
    if not source.is_file() or source.stat().st_size == 0:
        raise RuntimeError(f"Cannot create {kind} resume contract: source file is missing or empty: {source}")
    normalized = json.loads(json.dumps(parameters, sort_keys=True, default=str))
    source_data = {
        "sha256": sha256_file(source),
        "size_bytes": source.stat().st_size,
        "suffix": source.suffix.lower(),
    }
    body = {
        "version": CONTRACT_VERSION,
        "kind": kind,
        "source": source_data,
        "parameters": normalized,
    }
    body["fingerprint"] = canonical_hash(body)
    return body


def store_or_validate_contract(
    receipt: JobReceipt,
    key: str,
    candidate: dict[str, Any],
    *,
    allow_adopt: bool = True,
) -> None:
    existing = receipt.parameters.get(key)
    if existing is None:
        if not allow_adopt:
            raise RuntimeError(f"Resume contract '{key}' is missing; start a new job instead of reusing unknown outputs.")
        receipt.parameters[key] = candidate
        return
    if not isinstance(existing, dict) or existing.get("fingerprint") != candidate.get("fingerprint"):
        previous = existing.get("fingerprint", "invalid") if isinstance(existing, dict) else "invalid"
        current = candidate.get("fingerprint", "invalid")
        raise RuntimeError(
            f"Resume contract mismatch for {key}: existing={previous[:12]} requested={current[:12]}. "
            "The source file or processing settings changed. Start a new job rather than mixing incompatible stage outputs."
        )


def find_existing_input(job_dir: Path, *, prefix: str = "") -> Path | None:
    input_dir = job_dir / "input"
    if not input_dir.is_dir():
        return None
    candidates = [
        path for path in input_dir.iterdir()
        if path.is_file() and path.stat().st_size > 0 and not path.name.startswith("resume_candidate_")
    ]
    if prefix:
        preferred = [path for path in candidates if path.name.startswith(prefix)]
        if preferred:
            candidates = preferred
    return min(candidates, key=lambda path: path.stat().st_mtime, default=None)
