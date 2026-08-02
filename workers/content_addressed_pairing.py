"""Fixture-safe content-addressed inventory and image/model pairing.

This worker deliberately knows nothing about the user's live collection.  Callers pass one or
more roots explicitly; tests use temporary fixture roots.  A file's SHA-256 is its identity, while
paths are recorded as observations that may change between inventories.
"""
from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Iterable

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MODEL_EXTENSIONS = {".glb", ".gltf", ".fbx", ".obj", ".ply", ".blend", ".stl"}
PACKAGE_EXTENSIONS = {".zip"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | MODEL_EXTENSIONS | PACKAGE_EXTENSIONS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _png_pixels(path: Path) -> tuple[int, int, list[int]] | None:
    """Decode the small 8-bit RGB/RGBA PNG subset used by fixture tests."""
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos = 8
    width = height = bit_depth = color_type = None
    raw = bytearray()
    while pos + 12 <= len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        kind = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            if bit_depth != 8 or color_type not in (2, 6) or interlace:
                return None
        elif kind == b"IDAT":
            raw.extend(body)
        elif kind == b"IEND":
            break
    if not width or not height or bit_depth != 8 or color_type not in (2, 6):
        return None
    channels = 3 if color_type == 2 else 4
    row_size = width * channels
    decoded = zlib.decompress(bytes(raw))
    rows: list[list[int]] = []
    cursor = 0
    previous = [0] * row_size
    for _ in range(height):
        filter_type = decoded[cursor]
        cursor += 1
        current = list(decoded[cursor : cursor + row_size])
        cursor += row_size
        for i in range(row_size):
            left = current[i - channels] if i >= channels else 0
            up = previous[i]
            up_left = previous[i - channels] if i >= channels else 0
            if filter_type == 1:
                current[i] = (current[i] + left) & 255
            elif filter_type == 2:
                current[i] = (current[i] + up) & 255
            elif filter_type == 3:
                current[i] = (current[i] + ((left + up) // 2)) & 255
            elif filter_type == 4:
                estimate = left + up - up_left
                distances = (abs(estimate - left), abs(estimate - up), abs(estimate - up_left))
                current[i] = (current[i] + (left, up, up_left)[distances.index(min(distances))]) & 255
            elif filter_type != 0:
                return None
        rows.append(current)
        previous = current
    grayscale: list[int] = []
    for row in rows:
        for i in range(0, len(row), channels):
            grayscale.append((299 * row[i] + 587 * row[i + 1] + 114 * row[i + 2]) // 1000)
    return width, height, grayscale


def perceptual_hash(path: Path) -> str | None:
    decoded = _png_pixels(path) if path.suffix.lower() == ".png" else None
    if decoded is None:
        return None
    width, height, pixels = decoded
    sample: list[int] = []
    for y in range(8):
        for x in range(8):
            sx = min(width - 1, (x * width) // 8)
            sy = min(height - 1, (y * height) // 8)
            sample.append(pixels[sy * width + sx])
    mean = sum(sample) / len(sample)
    return "".join("1" if value > mean else "0" for value in sample)


def _hamming(left: str | None, right: str | None) -> int | None:
    if not left or not right or len(left) != len(right):
        return None
    return sum(a != b for a, b in zip(left, right))


def _kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in PACKAGE_EXTENSIONS:
        return "package"
    return "model"


def _iter_supported(roots: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                found.append(path)
    return sorted(set(found), key=lambda item: str(item).lower())


def _cached_record(path: Path, cache: dict[str, Any]) -> dict[str, Any]:
    key = str(path.resolve())
    stat = path.stat()
    old = cache.get(key)
    if old and old.get("bytes") == stat.st_size and old.get("mtime_ns") == stat.st_mtime_ns:
        record = dict(old)
        record["cache_hit"] = True
        return record
    record = {
        "asset_id": None,
        "path": key,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_file(path),
        "kind": _kind(path),
        "extension": path.suffix.lower(),
        "cache_hit": False,
    }
    record["asset_id"] = f"sha256:{record['sha256']}"
    if record["kind"] == "image":
        record["perceptual_hash"] = perceptual_hash(path)
        decoded = _png_pixels(path) if path.suffix.lower() == ".png" else None
        if decoded:
            record["dimensions"] = {"width": decoded[0], "height": decoded[1], "channels": 1}
    return record


def inventory_snapshot(roots: Iterable[Path], cache: dict[str, Any] | None = None) -> dict[str, Any]:
    cache = cache if cache is not None else {}
    records = [_cached_record(path, cache) for path in _iter_supported(roots)]
    now = {record["path"]: record for record in records}
    cache.clear()
    cache.update(now)
    return {
        "roots": [str(Path(root).resolve()) for root in roots],
        "files": records,
        "file_count": len(records),
        "aggregate_bytes": sum(record["bytes"] for record in records),
        "identities": sorted(record["asset_id"] for record in records),
    }


def stability_report(snapshot_a: dict[str, Any], snapshot_b: dict[str, Any]) -> dict[str, Any]:
    stable = (
        snapshot_a.get("file_count") == snapshot_b.get("file_count")
        and snapshot_a.get("aggregate_bytes") == snapshot_b.get("aggregate_bytes")
        and snapshot_a.get("identities") == snapshot_b.get("identities")
    )
    return {
        "classification": "STABLE" if stable else "PROVISIONAL_SNAPSHOT_MOVING_DATASET",
        "stable": stable,
        "file_count_equal": snapshot_a.get("file_count") == snapshot_b.get("file_count"),
        "aggregate_bytes_equal": snapshot_a.get("aggregate_bytes") == snapshot_b.get("aggregate_bytes"),
        "identities_equal": snapshot_a.get("identities") == snapshot_b.get("identities"),
    }


def _load_sidecar(model: dict[str, Any]) -> dict[str, Any]:
    path = Path(model["path"])
    for sidecar in (Path(str(path) + ".json"), path.with_suffix(".json")):
        if sidecar.exists():
            try:
                value = json.loads(sidecar.read_text(encoding="utf-8"))
                return value if isinstance(value, dict) else {}
            except (OSError, ValueError):
                return {}
    return {}


def _tokens(path: str) -> set[str]:
    return {part for part in Path(path).stem.lower().replace("-", "_").split("_") if len(part) > 2}


def _score(image: dict[str, Any], model: dict[str, Any]) -> tuple[float, dict[str, Any], list[str]]:
    sidecar = _load_sidecar(model)
    evidence: dict[str, Any] = {}
    contradictions: list[str] = []
    score = 0.0
    source = sidecar.get("source_image") or sidecar.get("input_image")
    if isinstance(source, str) and Path(source).name.lower() == Path(image["path"]).name.lower():
        score += 0.96
        evidence["exact_provenance"] = "model sidecar explicitly names image"
    overlap = _tokens(image["path"]) & _tokens(model["path"])
    if overlap:
        score += min(0.18, 0.06 * len(overlap))
        evidence["filename_tokens"] = sorted(overlap)
    image_hash = image.get("perceptual_hash")
    model_image_hash = sidecar.get("source_perceptual_hash")
    distance = _hamming(image_hash, model_image_hash)
    if distance is not None and distance <= 8:
        score += 0.18
        evidence["visual_hash"] = {"hamming_distance": distance}
    if not evidence:
        contradictions.append("no independent evidence channel")
    return min(score, 1.0), evidence, contradictions


def pair_images_to_models(images: list[dict[str, Any]], models: list[dict[str, Any]]) -> dict[str, Any]:
    proposals: list[dict[str, Any]] = []
    for image in images:
        ranked = []
        for model in models:
            score, evidence, contradictions = _score(image, model)
            ranked.append((score, model, evidence, contradictions))
        ranked.sort(key=lambda item: (-item[0], item[1]["asset_id"]))
        for index, (score, model, evidence, contradictions) in enumerate(ranked):
            if score >= 0.92 and "exact_provenance" in evidence and not contradictions:
                classification = "PROVEN_HIGH_CONFIDENCE_PAIR"
            elif score >= 0.80:
                classification = "PROVISIONAL_PAIR_REVIEW_REQUIRED"
            elif score >= 0.55:
                classification = "AMBIGUOUS_TOP_CANDIDATES"
            else:
                classification = "UNPAIRED"
            proposals.append({
                "image_group_id": f"sha256:{image['sha256']}",
                "model_asset_id": model["asset_id"],
                "confidence": round(score, 4),
                "evidence_by_channel": evidence,
                "contradictions": contradictions,
                "runner_up_candidates": [item[1]["asset_id"] for item in ranked[index + 1 : index + 3]],
                "review_required": classification != "PROVEN_HIGH_CONFIDENCE_PAIR",
                "classification": classification,
            })
    return {"proposals": proposals}


def group_images(images: list[dict[str, Any]], max_hamming: int = 4) -> list[dict[str, Any]]:
    groups: list[list[dict[str, Any]]] = []
    for image in images:
        placed = False
        for group in groups:
            if any(_hamming(image.get("perceptual_hash"), member.get("perceptual_hash")) is not None and _hamming(image.get("perceptual_hash"), member.get("perceptual_hash")) <= max_hamming for member in group):
                group.append(image)
                placed = True
                break
        if not placed:
            groups.append([image])
    return [{"image_group_id": f"image-group-{index:04d}", "members": [item["asset_id"] for item in group]} for index, group in enumerate(groups)]


def duplicate_exports(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_hash: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        by_hash.setdefault(model["sha256"], []).append(model)
    return [{"asset_id": f"sha256:{digest}", "paths": [item["path"] for item in group]} for digest, group in by_hash.items() if len(group) > 1]


def build_fixture_report(root: Path, cache: dict[str, Any] | None = None) -> dict[str, Any]:
    snapshot = inventory_snapshot([root], cache)
    images = [record for record in snapshot["files"] if record["kind"] == "image"]
    models = [record for record in snapshot["files"] if record["kind"] == "model"]
    return {
        "snapshot": snapshot,
        "image_groups": group_images(images),
        "pairing": pair_images_to_models(images, models),
        "duplicate_exports": duplicate_exports(models),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
