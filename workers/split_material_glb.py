"""Split selected triangle runs into a second material without re-exporting geometry."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from uv_exact_validate import _accessor


def read_glb(path: Path) -> tuple[dict, bytes]:
    data = path.read_bytes()
    json_len = struct.unpack_from("<I", data, 12)[0]
    json_start = 20
    gltf = json.loads(data[json_start:json_start + json_len])
    offset = json_start + ((json_len + 3) // 4) * 4
    bin_len, kind = struct.unpack_from("<II", data, offset)
    if kind != 0x004E4942:
        raise RuntimeError("GLB has no BIN chunk")
    return gltf, data[offset + 8:offset + 8 + bin_len]


def append_aligned(blob: bytearray, payload: bytes) -> tuple[int, int]:
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    blob.extend(payload)
    return offset, len(payload)


def add_index_accessor(gltf: dict, blob: bytearray, values: np.ndarray) -> int:
    values = np.asarray(values, dtype="<u2").reshape(-1)
    offset, length = append_aligned(blob, values.tobytes(order="C"))
    gltf.setdefault("bufferViews", []).append({"buffer": 0, "byteOffset": offset, "byteLength": length, "target": 34963})
    gltf.setdefault("accessors", []).append({
        "bufferView": len(gltf["bufferViews"]) - 1,
        "componentType": 5123,
        "count": int(values.size),
        "type": "SCALAR",
        "min": [int(values.min())] if values.size else [0],
        "max": [int(values.max())] if values.size else [0],
    })
    return len(gltf["accessors"]) - 1


def hash_arrays(gltf: dict, blob: bytes) -> dict[str, str]:
    positions = None
    uvs = None
    indices = []
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attrs = primitive.get("attributes", {})
            if "POSITION" not in attrs or "indices" not in primitive:
                continue
            if positions is None:
                positions = _accessor(gltf, blob, attrs["POSITION"]).astype("<f4")
            if uvs is None and "TEXCOORD_0" in attrs:
                uvs = _accessor(gltf, blob, attrs["TEXCOORD_0"]).astype("<f4")
            indices.append(_accessor(gltf, blob, primitive["indices"]).astype("<u2").reshape(-1, 3))
    digest = lambda array: hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest()
    all_indices = np.concatenate(indices) if indices else np.empty((0, 3), dtype="<u2")
    return {"positions_sha256": digest(positions), "uv_sha256": digest(uvs), "indices_sha256": digest(all_indices), "triangles": int(len(all_indices))}


def split(input_path: Path, safe_path: Path, atlas_path: Path, output_path: Path, report_path: Path) -> dict:
    gltf, original_blob = read_glb(input_path)
    safe_ids = set(int(v) for v in np.load(safe_path).reshape(-1).tolist())
    if not safe_ids or min(safe_ids) < 0:
        raise RuntimeError("safe triangle list is empty or invalid")
    if len(gltf.get("meshes", [])) != 1 or len(gltf["meshes"][0].get("primitives", [])) != 1:
        raise RuntimeError("bounded splitter expects one indexed mesh primitive")
    original = hash_arrays(gltf, original_blob)
    new_gltf = copy.deepcopy(gltf)
    new_blob = bytearray(original_blob)

    image_offset, image_length = append_aligned(new_blob, atlas_path.read_bytes())
    new_gltf.setdefault("bufferViews", []).append({"buffer": 0, "byteOffset": image_offset, "byteLength": image_length})
    image_view = len(new_gltf["bufferViews"]) - 1
    new_gltf.setdefault("images", []).append({"bufferView": image_view, "mimeType": "image/png", "name": atlas_path.stem})
    image_index = len(new_gltf["images"]) - 1
    new_gltf.setdefault("textures", []).append({"sampler": 0, "source": image_index})
    texture_index = len(new_gltf["textures"]) - 1
    safe_material = copy.deepcopy(new_gltf["materials"][0])
    safe_material["name"] = "ShamanPBR_SurfaceSafe"
    safe_material.setdefault("pbrMetallicRoughness", {})["baseColorTexture"] = {"index": texture_index}
    new_gltf["materials"].append(safe_material)
    safe_material_index = len(new_gltf["materials"]) - 1

    primitive = new_gltf["meshes"][0]["primitives"][0]
    old_indices = _accessor(gltf, original_blob, primitive["indices"]).astype(np.uint16).reshape(-1, 3)
    runs = []
    start = 0
    current_safe = 0 in safe_ids
    for triangle in range(1, len(old_indices) + 1):
        next_safe = triangle in safe_ids if triangle < len(old_indices) else None
        if next_safe is None or next_safe != current_safe:
            runs.append((start, triangle, current_safe))
            start = triangle
            current_safe = bool(next_safe) if next_safe is not None else current_safe
    if start < len(old_indices):
        runs.append((start, len(old_indices), current_safe))
    # The loop above closes the final run through next_safe=None; keep only valid ranges.
    runs = [(a, b, safe) for a, b, safe in runs if b > a]
    new_primitives = []
    for start, end, is_safe in runs:
        item = copy.deepcopy(primitive)
        item["indices"] = add_index_accessor(new_gltf, new_blob, old_indices[start:end].reshape(-1))
        item["material"] = safe_material_index if is_safe else int(primitive.get("material", 0))
        new_primitives.append(item)
    if sum(end - start for start, end, _ in runs) != len(old_indices):
        raise RuntimeError("primitive run partition does not cover all triangles")
    new_gltf["meshes"][0]["primitives"] = new_primitives
    new_gltf["buffers"][0]["byteLength"] = len(new_blob)

    json_bytes = json.dumps(new_gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = bytes(new_blob) + b"\x00" * ((4 - len(new_blob) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes + struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes)
    after_gltf, after_blob = read_glb(output_path)
    after = hash_arrays(after_gltf, after_blob)
    report = {
        "schema": "direct_material_split_v1",
        "input": str(input_path),
        "output": str(output_path),
        "safe_triangle_count": len(safe_ids),
        "primitive_run_count": len(runs),
        "original_fingerprint": original,
        "output_fingerprint": after,
        "geometry_uv_index_unchanged": original == after,
        "existing_attribute_buffers_reused": True,
        "safe_atlas": str(atlas_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--safe-triangles", required=True)
    p.add_argument("--safe-atlas", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()
    report = split(Path(args.input), Path(args.safe_triangles), Path(args.safe_atlas), Path(args.output), Path(args.report))
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
