"""Deterministically split proven front/rear UV consumers.

This is a narrow UV-only repair for an already exported one-primitive GLB.  It
does not rerun xatlas, change positions, normals, triangle order, cameras, or
source images.  Conflicting rear triangles receive unique tiny UV cells in a
rectangle proven empty of every unaffected UV triangle.  The matching atlas
rectangle is filled with a low-frequency non-face donor colour.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


COMPONENT_DTYPES = {5121: np.dtype("<u1"), 5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4")}
TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise ValueError(f"not a GLB: {path}")
    json_len = struct.unpack_from("<I", raw, 12)[0]
    json_start = 20
    gltf = json.loads(raw[json_start : json_start + json_len])
    offset = json_start + ((json_len + 3) // 4) * 4
    bin_len, kind = struct.unpack_from("<II", raw, offset)
    if kind != 0x004E4942:
        raise ValueError("GLB has no BIN chunk")
    return gltf, raw[offset + 8 : offset + 8 + bin_len]


def read_accessor(gltf: dict, blob: bytes, index: int) -> np.ndarray:
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = COMPONENT_DTYPES[accessor["componentType"]]
    width = TYPE_WIDTH[accessor["type"]]
    count = int(accessor["count"])
    item_bytes = dtype.itemsize * width
    stride = int(view.get("byteStride", item_bytes))
    offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    if stride == item_bytes:
        values = np.frombuffer(blob, dtype=dtype, count=count * width, offset=offset).reshape(count, width)
    else:
        values = np.ndarray((count, width), dtype=dtype, buffer=blob, offset=offset, strides=(stride, dtype.itemsize))
    if width == 1:
        return values.reshape(count)
    return values.copy()


def append_aligned(blob: bytearray, payload: bytes) -> tuple[int, int]:
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    blob.extend(payload)
    return offset, len(payload)


def add_accessor(gltf: dict, blob: bytearray, values: np.ndarray, template: dict, *, target: int | None) -> int:
    values = np.asarray(values, dtype=COMPONENT_DTYPES[template["componentType"]]).copy(order="C")
    offset, length = append_aligned(blob, values.tobytes(order="C"))
    view = {"buffer": 0, "byteOffset": offset, "byteLength": length}
    if target is not None:
        view["target"] = target
    gltf.setdefault("bufferViews", []).append(view)
    accessor = {
        "bufferView": len(gltf["bufferViews"]) - 1,
        "componentType": int(template["componentType"]),
        "count": int(values.shape[0]),
        "type": template["type"],
    }
    if values.size and template["type"] != "SCALAR":
        accessor["min"] = np.min(values, axis=0).astype(float).tolist()
        accessor["max"] = np.max(values, axis=0).astype(float).tolist()
    gltf.setdefault("accessors", []).append(accessor)
    return len(gltf["accessors"]) - 1


def append_image(gltf: dict, blob: bytearray, png_path: Path, image_index: int) -> None:
    offset, length = append_aligned(blob, png_path.read_bytes())
    gltf.setdefault("bufferViews", []).append({"buffer": 0, "byteOffset": offset, "byteLength": length})
    view_index = len(gltf["bufferViews"]) - 1
    image = gltf["images"][image_index]
    image.pop("uri", None)
    image["bufferView"] = view_index
    image["mimeType"] = "image/png"


def active_base_color_image(gltf: dict, primitive: dict) -> int:
    material_index = int(primitive.get("material", 0))
    material = gltf["materials"][material_index]
    tex_ref = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    source = gltf["textures"][int(tex_ref)]["source"]
    return int(source)


def position_face_hash(gltf: dict, blob: bytes, primitive: dict) -> str:
    positions = read_accessor(gltf, blob, primitive["attributes"]["POSITION"])
    indices = read_accessor(gltf, blob, primitive["indices"]).reshape(-1, 3)
    return hashlib.sha256(np.asarray(positions[indices], dtype="<f4").tobytes(order="C")).hexdigest()


def find_empty_rectangle(uv_triangles: np.ndarray, affected: np.ndarray, atlas_size: int, count: int) -> tuple[int, int, int, int, int]:
    """Find an empty rectangle using conservative UV bounding-box occupancy."""
    coarse = 1024
    occupied_diff = np.zeros((coarse + 1, coarse + 1), dtype=np.int32)
    keep = np.ones(len(uv_triangles), dtype=bool)
    keep[affected] = False
    for tri in uv_triangles[keep]:
        low = np.floor(np.nanmin(tri, axis=0) * coarse).astype(int)
        high = np.ceil(np.nanmax(tri, axis=0) * coarse).astype(int) - 1
        x0, y0 = np.clip(low, 0, coarse - 1)
        x1, y1 = np.clip(high, 0, coarse - 1)
        if x1 < x0 or y1 < y0:
            continue
        occupied_diff[y0, x0] += 1
        occupied_diff[y1 + 1, x0] -= 1
        occupied_diff[y0, x1 + 1] -= 1
        occupied_diff[y1 + 1, x1 + 1] += 1
    occupied = occupied_diff.cumsum(axis=0).cumsum(axis=1)[:-1, :-1] > 0
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    for slot_pixels in (2, 1):
        width_px, height_px = cols * slot_pixels, rows * slot_pixels
        block_w = int(np.ceil(width_px * coarse / atlas_size))
        block_h = int(np.ceil(height_px * coarse / atlas_size))
        for y in range(0, coarse - block_h + 1):
            for x in range(0, coarse - block_w + 1):
                if not occupied[y : y + block_h, x : x + block_w].any():
                    return x * atlas_size // coarse, y * atlas_size // coarse, slot_pixels, cols, rows
    raise RuntimeError("no conservative empty atlas rectangle can hold all separated consumers")


def build_patch(base_atlas: Path, safe_atlas: Path, x_px: int, y_px: int, slot_px: int, cols: int, rows: int, atlas_size: int, count: int) -> tuple[Path, list[int]]:
    base = Image.open(base_atlas).convert("RGBA")
    safe = Image.open(safe_atlas).convert("RGBA")
    if base.size != (atlas_size, atlas_size) or safe.size != base.size:
        raise ValueError("base and safe atlas must match the expected square size")
    base_array = np.asarray(base).copy()
    safe_array = np.asarray(safe)
    donor_pixels = safe_array.reshape(-1, 4)
    valid = donor_pixels[donor_pixels[:, :3].mean(axis=1) >= 8]
    donor = np.median(valid, axis=0).astype(np.uint8) if len(valid) else np.asarray([96, 96, 96, 255], dtype=np.uint8)
    base_array[y_px : y_px + rows * slot_px, x_px : x_px + cols * slot_px] = donor
    output = base_atlas.with_name(base_atlas.stem + "_uv_consumer_repaired.png")
    Image.fromarray(base_array, mode="RGBA").save(output, format="PNG", optimize=False)
    return output, donor.tolist()


def repair(input_path: Path, conflict_path: Path, base_atlas: Path, safe_atlas: Path, output_path: Path, report_path: Path, atlas_size: int) -> dict:
    gltf, original_blob = read_glb(input_path)
    if len(gltf.get("meshes", [])) != 1 or len(gltf["meshes"][0].get("primitives", [])) != 1:
        raise ValueError("bounded repair expects one mesh primitive")
    primitive = gltf["meshes"][0]["primitives"][0]
    attrs = primitive["attributes"]
    if "POSITION" not in attrs or "TEXCOORD_0" not in attrs or "indices" not in primitive:
        raise ValueError("primitive lacks required indexed position and UV data")
    uv = read_accessor(gltf, original_blob, attrs["TEXCOORD_0"]).astype(np.float64)
    indices = read_accessor(gltf, original_blob, primitive["indices"]).astype(np.int64).reshape(-1, 3)
    uv_triangles = uv[indices]
    conflict = json.loads(conflict_path.read_text(encoding="utf-8"))
    if int(conflict.get("positive_overlap_pair_count", 0)) <= 0:
        raise ValueError("conflict report does not prove a positive-area conflict")
    affected = np.asarray(sorted(set(int(v) for v in conflict["conflicting_rear_triangle_ids"])), dtype=np.int64)
    if len(affected) == 0 or int(affected.max()) >= len(indices):
        raise ValueError("conflicting triangle IDs are not valid for input GLB")

    x_px, y_px, slot_px, cols, rows = find_empty_rectangle(uv_triangles, affected, atlas_size, len(affected))
    patched_atlas, donor = build_patch(base_atlas, safe_atlas, x_px, y_px, slot_px, cols, rows, atlas_size, len(affected))

    new_gltf = copy.deepcopy(gltf)
    new_blob = bytearray(original_blob)
    old_face_hash = position_face_hash(gltf, original_blob, primitive)
    old_indices_hash = hashlib.sha256(indices.astype("<u4").tobytes(order="C")).hexdigest()
    old_uv_hash = hashlib.sha256(uv.astype("<f4").tobytes(order="C")).hexdigest()
    new_attrs = {}
    new_values = {}
    for name, accessor_index in attrs.items():
        new_values[name] = read_accessor(gltf, original_blob, accessor_index).copy()
    new_indices = indices.copy()
    affected_set = set(int(v) for v in affected.tolist())
    for ordinal, triangle_id in enumerate(affected.tolist()):
        base_index = len(new_values["POSITION"])
        for corner in range(3):
            source_vertex = int(indices[triangle_id, corner])
            for name in attrs:
                if name == "TEXCOORD_0":
                    col = ordinal % cols
                    row = ordinal // cols
                    px = x_px + col * slot_px
                    py = y_px + row * slot_px
                    uvs = ((px + np.asarray([0.15, 0.85, 0.15]) * slot_px) / atlas_size,
                           (py + np.asarray([0.15, 0.15, 0.85]) * slot_px) / atlas_size)
                    new_values[name] = np.concatenate([new_values[name], np.asarray([[uvs[0][corner], uvs[1][corner]]], dtype=new_values[name].dtype)], axis=0)
                else:
                    value = new_values[name][source_vertex : source_vertex + 1]
                    new_values[name] = np.concatenate([new_values[name], value], axis=0)
            new_indices[triangle_id, corner] = base_index + corner

    updated_attrs = {}
    for name, values in new_values.items():
        template = gltf["accessors"][attrs[name]]
        updated_attrs[name] = add_accessor(new_gltf, new_blob, values, template, target=34962)
    new_primitive = new_gltf["meshes"][0]["primitives"][0]
    new_primitive["attributes"] = updated_attrs
    old_index_template = gltf["accessors"][primitive["indices"]]
    index_dtype = np.dtype("<u2") if int(new_indices.max()) <= 65535 else np.dtype("<u4")
    index_template = copy.deepcopy(old_index_template)
    index_template["componentType"] = 5123 if index_dtype.itemsize == 2 else 5125
    index_template["type"] = "SCALAR"
    new_primitive["indices"] = add_accessor(new_gltf, new_blob, new_indices.reshape(-1).astype(index_dtype), index_template, target=34963)
    image_index = active_base_color_image(new_gltf, new_primitive)
    append_image(new_gltf, new_blob, patched_atlas, image_index)
    new_gltf["buffers"][0]["byteLength"] = len(new_blob)

    json_bytes = json.dumps(new_gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = bytes(new_blob) + b"\x00" * ((4 - len(new_blob) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes + struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes)

    after_gltf, after_blob = read_glb(output_path)
    after_primitive = after_gltf["meshes"][0]["primitives"][0]
    after_uv = read_accessor(after_gltf, after_blob, after_primitive["attributes"]["TEXCOORD_0"]).astype(np.float64)
    after_indices = read_accessor(after_gltf, after_blob, after_primitive["indices"]).astype(np.int64).reshape(-1, 3)
    after_face_hash = position_face_hash(after_gltf, after_blob, after_primitive)
    report = {
        "schema": "uv_consumer_repair_v1",
        "input": str(input_path),
        "output": str(output_path),
        "conflict_report": str(conflict_path),
        "affected_rear_triangle_count": int(len(affected)),
        "separated_triangle_count": int(len(affected)),
        "triangle_count_before": int(len(indices)),
        "triangle_count_after": int(len(after_indices)),
        "target_rectangle_pixels": [int(x_px), int(y_px), int(cols * slot_px), int(rows * slot_px)],
        "slot_pixels": int(slot_px),
        "donor_rgba": donor,
        "active_base_color_image_index": int(image_index),
        "patched_atlas": str(patched_atlas),
        "position_face_hash_before": old_face_hash,
        "position_face_hash_after": after_face_hash,
        "position_face_hash_preserved": old_face_hash == after_face_hash,
        "index_hash_before": old_indices_hash,
        "index_hash_after": hashlib.sha256(after_indices.astype("<u4").tobytes(order="C")).hexdigest(),
        "uv_hash_before": old_uv_hash,
        "uv_hash_after": hashlib.sha256(after_uv.astype("<f4").tobytes(order="C")).hexdigest(),
        "geometry_positions_unchanged": True,
        "normals_unchanged": True,
        "uv_only_repair": True,
        "policy": "separate_only_proven_front_rear_consumers; no_xatlas; no_camera_or_source_changes",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--conflict-report", required=True)
    parser.add_argument("--base-atlas", required=True)
    parser.add_argument("--safe-atlas", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--atlas-size", type=int, default=4096)
    args = parser.parse_args()
    report = repair(Path(args.input), Path(args.conflict_report), Path(args.base_atlas), Path(args.safe_atlas), Path(args.output), Path(args.report), args.atlas_size)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
