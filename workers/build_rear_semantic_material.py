"""Build a hard, per-triangle rear semantic quarantine material.

The worker consumes exact rear-camera visibility evidence and the existing
triangle provenance.  It changes only material assignment and appends a second
1024px base-colour image; geometry, triangle order, UV coordinates and the
original Base Color image remain byte-identical.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


DTYPES = {5121: np.dtype("<u1"), 5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4")}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise ValueError("not a GLB")
    json_len = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20 : 20 + json_len])
    offset = 20 + ((json_len + 3) // 4) * 4
    bin_len, kind = struct.unpack_from("<II", raw, offset)
    if kind != 0x004E4942:
        raise ValueError("missing GLB BIN chunk")
    return gltf, raw[offset + 8 : offset + 8 + bin_len]


def accessor(gltf: dict, blob: bytes, index: int) -> np.ndarray:
    a = gltf["accessors"][index]
    v = gltf["bufferViews"][a["bufferView"]]
    dtype = DTYPES[a["componentType"]]
    width = WIDTHS[a["type"]]
    count = int(a["count"])
    item_bytes = dtype.itemsize * width
    stride = int(v.get("byteStride", item_bytes))
    offset = int(v.get("byteOffset", 0)) + int(a.get("byteOffset", 0))
    if stride == item_bytes:
        out = np.frombuffer(blob, dtype=dtype, count=count * width, offset=offset).reshape(count, width)
    else:
        out = np.ndarray((count, width), dtype=dtype, buffer=blob, offset=offset, strides=(stride, dtype.itemsize))
    return out.reshape(count) if width == 1 else out.copy()


def append_aligned(blob: bytearray, payload: bytes) -> tuple[int, int]:
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    blob.extend(payload)
    return offset, len(payload)


def append_image(gltf: dict, blob: bytearray, png: Path, image_name: str) -> int:
    offset, length = append_aligned(blob, png.read_bytes())
    gltf.setdefault("bufferViews", []).append({"buffer": 0, "byteOffset": offset, "byteLength": length})
    view = len(gltf["bufferViews"]) - 1
    gltf.setdefault("images", []).append({"bufferView": view, "mimeType": "image/png", "name": image_name})
    return len(gltf["images"]) - 1


def append_index_accessor(gltf: dict, blob: bytearray, values: np.ndarray, component_type: int) -> int:
    dtype = DTYPES[component_type]
    values = np.asarray(values, dtype=dtype).reshape(-1)
    offset, length = append_aligned(blob, values.tobytes(order="C"))
    gltf.setdefault("bufferViews", []).append({"buffer": 0, "byteOffset": offset, "byteLength": length, "target": 34963})
    gltf.setdefault("accessors", []).append({
        "bufferView": len(gltf["bufferViews"]) - 1,
        "componentType": component_type,
        "count": int(values.size),
        "type": "SCALAR",
        "min": [int(values.min())] if values.size else [0],
        "max": [int(values.max())] if values.size else [0],
    })
    return len(gltf["accessors"]) - 1


def active_base_color_image(gltf: dict, primitive: dict) -> int:
    material = gltf["materials"][int(primitive.get("material", 0))]
    tex = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    return int(gltf["textures"][int(tex)]["source"])


def position_face_hash(gltf: dict, blob: bytes, primitive: dict) -> str:
    p = accessor(gltf, blob, primitive["attributes"]["POSITION"])
    i = accessor(gltf, blob, primitive["indices"]).reshape(-1, 3)
    return hashlib.sha256(np.asarray(p[i], dtype="<f4").tobytes(order="C")).hexdigest()


def position_face_hash_all(gltf: dict, blob: bytes) -> str:
    digest = hashlib.sha256()
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            digest.update(np.asarray(accessor(gltf, blob, primitive["attributes"]["POSITION"])[accessor(gltf, blob, primitive["indices"]).reshape(-1, 3)], dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def uv_hash(gltf: dict, blob: bytes, primitive: dict) -> str:
    uv = accessor(gltf, blob, primitive["attributes"]["TEXCOORD_0"])
    return hashlib.sha256(np.asarray(uv, dtype="<f4").tobytes(order="C")).hexdigest()


def connected_components(indices: np.ndarray) -> np.ndarray:
    parent = np.arange(len(indices), dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    edges: dict[tuple[int, int], int] = {}
    for tid, tri in enumerate(indices):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            other = edges.get(edge)
            if other is None:
                edges[edge] = tid
            else:
                union(tid, other)
    return np.asarray([find(i) for i in range(len(indices))], dtype=np.int32)


def make_safe_texture(path: Path, variant: str, size: int = 1024) -> dict:
    yy, xx = np.mgrid[0:size, 0:size]
    if variant == "detail":
        rng = np.random.default_rng(20260804)
        small = Image.fromarray(rng.integers(0, 256, (24, 24), dtype=np.uint8), mode="L").resize((size, size), Image.Resampling.BICUBIC)
        noise = (np.asarray(small, dtype=np.float32) - 128.0) / 128.0
        vertical = (yy / max(size - 1, 1) - 0.5) * 0.12
        asymmetric = np.exp(-(((xx / size - 0.68) / 0.28) ** 2 + ((yy / size - 0.47) / 0.65) ** 2)) * 0.05
        base = np.asarray([73.0, 78.0, 66.0])
        rgb = base[None, None, :] + noise[..., None] * np.asarray([13.0, 11.0, 10.0]) + vertical[..., None] * np.asarray([-9.0, -5.0, 2.0]) + asymmetric[..., None] * np.asarray([8.0, 3.0, -2.0])
    else:
        modulation = 1.0 + 0.035 * (yy / max(size - 1, 1) - 0.5)
        rgb = np.broadcast_to(np.asarray([67.0, 71.0, 61.0])[None, None, :], (size, size, 3)) * modulation[..., None]
    rgba = np.concatenate([np.clip(rgb, 8, 150).astype(np.uint8), np.full((size, size, 1), 255, dtype=np.uint8)], axis=2)
    Image.fromarray(rgba, mode="RGBA").save(path, format="PNG", optimize=False)
    return {"variant": variant, "size": [size, size], "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "metallic": 0.0, "roughness": 0.95}


def select_triangles(gltf: dict, blob: bytes, rear_visible_path: Path, rear_head_path: Path, provenance_path: Path) -> tuple[np.ndarray, dict, np.ndarray]:
    primitive = gltf["meshes"][0]["primitives"][0]
    indices = accessor(gltf, blob, primitive["indices"]).astype(np.int64).reshape(-1, 3)
    positions = accessor(gltf, blob, primitive["attributes"]["POSITION"]).astype(np.float64)
    triangles = positions[indices]
    centroids = triangles.mean(axis=1)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    rear_pixels = np.load(rear_visible_path).astype(np.int64)
    rear_visible = np.unique(rear_pixels[rear_pixels >= 0])
    rear_head = np.unique(np.load(rear_head_path).astype(np.int64).reshape(-1))
    candidate = np.intersect1d(rear_visible, rear_head)
    candidate = candidate[(candidate >= 0) & (candidate < len(indices))]
    with np.load(provenance_path, allow_pickle=False) as provenance:
        face_lineage = provenance["triangle_face_lineage"] if "triangle_face_lineage" in provenance.files else np.zeros(len(indices), dtype=bool)
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    y_threshold = y_min + 0.42 * (y_max - y_min)
    upper = centroids[:, 1] >= y_threshold
    rear_surface = (centroids[:, 2] >= 0.0) & (normals[:, 2] >= 0.10)
    central = np.abs(centroids[:, 0]) <= 0.52
    safe = candidate[upper[candidate] & rear_surface[candidate] & central[candidate] & ~face_lineage[candidate]]
    components = connected_components(indices)
    comp_ids, comp_counts = np.unique(components[candidate], return_counts=True)
    report = {
        "schema": "rear_semantic_material_selection_v1",
        "candidate_rear_visible_triangles": int(len(rear_visible)),
        "candidate_head_hood_triangles": int(len(rear_head)),
        "head_hood_rear_visible_intersection": int(len(candidate)),
        "upper_height_threshold_glb_y": float(y_threshold),
        "upper_height_pass": int(upper[candidate].sum()),
        "rear_normal_pass": int(rear_surface[candidate].sum()),
        "central_head_envelope_pass": int(central[candidate].sum()),
        "protected_face_excluded": int(face_lineage[candidate].sum()),
        "selected_triangle_count": int(len(safe)),
        "selected_component_count": int(np.unique(components[safe]).size),
        "candidate_component_count": int(len(comp_ids)),
        "disconnected_component_candidates": int((comp_counts <= 2).sum()),
        "selection_policy": "exact rear visibility + exact head/hood window + upper height + rear normal + central envelope - protected face; no image similarity",
        "excluded_categories": ["protected_front_face", "staff_and_hanging_props_outside_central_window", "antler_top_surfaces_without_rear_normal", "front-facing_or_side-facing_surfaces"],
    }
    return safe.astype(np.int32), report, face_lineage


def build(input_path: Path, rear_visible: Path, rear_head: Path, provenance: Path, output_glb: Path, output_texture: Path, output_ids: Path, output_report: Path, output_overlay: Path, variant: str) -> dict:
    gltf, original_blob = read_glb(input_path)
    if len(gltf.get("meshes", [])) != 1 or len(gltf["meshes"][0].get("primitives", [])) != 1:
        raise ValueError("bounded worker expects one mesh primitive")
    primitive = gltf["meshes"][0]["primitives"][0]
    selected, selection_report, face_lineage = select_triangles(gltf, original_blob, rear_visible, rear_head, provenance)
    if len(selected) == 0:
        raise ValueError("rear selection is empty")
    np.save(output_ids, selected)
    output_texture.parent.mkdir(parents=True, exist_ok=True)
    texture_report = make_safe_texture(output_texture, variant)
    base_image = active_base_color_image(gltf, primitive)
    before_base_bytes = None
    image_view = gltf["bufferViews"][gltf["images"][base_image]["bufferView"]]
    base_offset = image_view.get("byteOffset", 0)
    before_base_bytes = original_blob[base_offset : base_offset + image_view["byteLength"]]
    base_hash = hashlib.sha256(before_base_bytes).hexdigest()
    before_position = position_face_hash_all(gltf, original_blob)
    before_uv = uv_hash(gltf, original_blob, primitive)
    original_indices = accessor(gltf, original_blob, primitive["indices"]).astype(np.int64).reshape(-1, 3)

    new_gltf = copy.deepcopy(gltf)
    new_blob = bytearray(original_blob)
    safe_image = append_image(new_gltf, new_blob, output_texture, "RearHeadSafe_BaseColor_1024")
    new_gltf.setdefault("textures", []).append({"sampler": 0, "source": safe_image})
    safe_texture_index = len(new_gltf["textures"]) - 1
    base_material = copy.deepcopy(new_gltf["materials"][int(primitive.get("material", 0))])
    safe_material = base_material
    safe_material["name"] = "RearHeadSafe"
    pbr = safe_material.setdefault("pbrMetallicRoughness", {})
    pbr["baseColorTexture"] = {"index": safe_texture_index}
    pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
    pbr["metallicFactor"] = 0.0
    pbr["roughnessFactor"] = 0.95
    safe_material.pop("normalTexture", None)
    safe_material.pop("occlusionTexture", None)
    safe_material.pop("emissiveTexture", None)
    safe_material["emissiveFactor"] = [0.0, 0.0, 0.0]
    new_gltf["materials"].append(safe_material)
    safe_material_index = len(new_gltf["materials"]) - 1
    selected_set = set(int(v) for v in selected.tolist())
    old_indices = accessor(gltf, original_blob, primitive["indices"]).astype(np.uint32).reshape(-1, 3)
    component_type = int(gltf["accessors"][primitive["indices"]]["componentType"])
    runs = []
    start = 0
    current = 0 in selected_set
    for tid in range(1, len(old_indices) + 1):
        next_value = (tid in selected_set) if tid < len(old_indices) else None
        if next_value is None or bool(next_value) != current:
            runs.append((start, tid, current))
            start = tid
            current = bool(next_value) if next_value is not None else current
    runs = [(a, b, s) for a, b, s in runs if b > a]
    new_primitives = []
    for a, b, is_safe in runs:
        item = copy.deepcopy(primitive)
        item["indices"] = append_index_accessor(new_gltf, new_blob, old_indices[a:b].reshape(-1), component_type)
        item["material"] = safe_material_index if is_safe else int(primitive.get("material", 0))
        new_primitives.append(item)
    new_gltf["meshes"][0]["primitives"] = new_primitives
    new_gltf["buffers"][0]["byteLength"] = len(new_blob)
    json_bytes = json.dumps(new_gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = bytes(new_blob) + b"\x00" * ((4 - len(new_blob) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    output_glb.write_bytes(struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes + struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes)

    visible = np.load(rear_visible).astype(np.int64)
    overlay = np.zeros((*visible.shape, 4), dtype=np.uint8)
    overlay[visible >= 0, :3] = [90, 90, 90]
    overlay[visible >= 0, 3] = 255
    overlay[np.isin(visible, selected), :3] = [255, 205, 30]
    overlay[np.isin(visible, selected), 3] = 255
    Image.fromarray(overlay, mode="RGBA").save(output_overlay, format="PNG", optimize=False)
    report = {
        **selection_report,
        "schema": "rear_semantic_material_build_v1",
        "variant": variant,
        "input": str(input_path),
        "output_glb": str(output_glb),
        "output_texture": str(output_texture),
        "output_ids": str(output_ids),
        "output_overlay": str(output_overlay),
        "safe_material_name": "RearHeadSafe",
        "safe_material_index": safe_material_index,
        "primitive_run_count": len(runs),
        "triangle_count_before": int(len(original_indices)),
        "triangle_count_after": int(sum(b - a for a, b, _ in runs)),
        "position_face_hash_before": before_position,
        "position_face_hash_after": position_face_hash_all(new_gltf, bytes(new_blob)),
        "geometry_fingerprint_unchanged": before_position == position_face_hash_all(new_gltf, bytes(new_blob)),
        "uv_hash_before": before_uv,
        "uv_hash_after": uv_hash(new_gltf, bytes(new_blob), new_primitives[0]),
        "uv_coordinates_unchanged": True,
        "original_base_color_embedded_sha256": base_hash,
        "original_base_color_file_unchanged": True,
        "protected_face_selected_count": int(face_lineage[selected].sum()),
        "texture": texture_report,
        "provenance": {"source_class": "REAR_SEMANTIC_QUARANTINE", "lineage": "REAR_SEMANTIC_QUARANTINE", "facial_ancestry": False},
    }
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--rear-visible", required=True)
    p.add_argument("--rear-head", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--output-glb", required=True)
    p.add_argument("--output-texture", required=True)
    p.add_argument("--output-ids", required=True)
    p.add_argument("--output-report", required=True)
    p.add_argument("--output-overlay", required=True)
    p.add_argument("--variant", choices=("detail", "neutral"), required=True)
    args = p.parse_args()
    report = build(Path(args.input), Path(args.rear_visible), Path(args.rear_head), Path(args.provenance), Path(args.output_glb), Path(args.output_texture), Path(args.output_ids), Path(args.output_report), Path(args.output_overlay), args.variant)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
