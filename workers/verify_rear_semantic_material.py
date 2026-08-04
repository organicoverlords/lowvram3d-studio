"""Fresh-GLB verification for the RearHeadSafe material boundary."""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


DTYPES = {5121: np.dtype("<u1"), 5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4")}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    jl = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20 : 20 + jl])
    off = 20 + ((jl + 3) // 4) * 4
    length, kind = struct.unpack_from("<II", raw, off)
    if kind != 0x004E4942:
        raise ValueError("missing BIN")
    return gltf, raw[off + 8 : off + 8 + length]


def accessor(gltf: dict, blob: bytes, index: int) -> np.ndarray:
    a = gltf["accessors"][index]
    v = gltf["bufferViews"][a["bufferView"]]
    dtype = DTYPES[a["componentType"]]
    width = WIDTHS[a["type"]]
    count = int(a["count"])
    item = dtype.itemsize * width
    stride = int(v.get("byteStride", item))
    offset = int(v.get("byteOffset", 0)) + int(a.get("byteOffset", 0))
    if stride == item:
        values = np.frombuffer(blob, dtype=dtype, count=count * width, offset=offset).reshape(count, width)
    else:
        values = np.ndarray((count, width), dtype=dtype, buffer=blob, offset=offset, strides=(stride, dtype.itemsize))
    return values.reshape(count) if width == 1 else values.copy()


def active_image(gltf: dict, primitive: dict, material_index: int) -> int:
    material = gltf["materials"][material_index]
    tex = material["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    return int(gltf["textures"][int(tex)]["source"])


def image_bytes(gltf: dict, blob: bytes, image_index: int) -> bytes:
    image = gltf["images"][image_index]
    if "bufferView" not in image:
        raise ValueError("unpacked image URI is not acceptable for this receipt")
    view = gltf["bufferViews"][image["bufferView"]]
    start = int(view.get("byteOffset", 0))
    return blob[start : start + int(view["byteLength"])]


def geometry_face_hash(gltf: dict, blob: bytes) -> str:
    digest = hashlib.sha256()
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            positions = accessor(gltf, blob, primitive["attributes"]["POSITION"])
            indices = accessor(gltf, blob, primitive["indices"]).reshape(-1, 3)
            digest.update(np.asarray(positions[indices], dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def index_face_hash(gltf: dict, blob: bytes) -> str:
    digest = hashlib.sha256()
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            positions = accessor(gltf, blob, primitive["attributes"]["POSITION"])
            indices = accessor(gltf, blob, primitive["indices"]).reshape(-1, 3)
            digest.update(np.asarray(positions[indices], dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def uv_hash(gltf: dict, blob: bytes) -> str:
    seen = set()
    digest = hashlib.sha256()
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            index = primitive["attributes"]["TEXCOORD_0"]
            if index in seen:
                continue
            seen.add(index)
            digest.update(np.asarray(accessor(gltf, blob, index), dtype="<f4").tobytes(order="C"))
    return digest.hexdigest()


def verify(final_path: Path, reference_path: Path, selected_path: Path, provenance_path: Path, report_path: Path) -> dict:
    final, final_blob = read_glb(final_path)
    reference, reference_blob = read_glb(reference_path)
    selected = np.unique(np.load(selected_path).astype(np.int64).reshape(-1))
    with np.load(provenance_path, allow_pickle=False) as provenance:
        protected = provenance["triangle_face_lineage"].astype(bool)
    final_primitives = [p for mesh in final.get("meshes", []) for p in mesh.get("primitives", [])]
    reference_primitives = [p for mesh in reference.get("meshes", []) for p in mesh.get("primitives", [])]
    material_names = {i: m.get("name", "") for i, m in enumerate(final.get("materials", []))}
    rear_indices = {i for i, name in material_names.items() if name == "RearHeadSafe"}
    safe_material_present = bool(rear_indices)
    flat_materials = []
    triangle_materials = []
    for primitive in final_primitives:
        count = len(accessor(final, final_blob, primitive["indices"]).reshape(-1, 3))
        triangle_materials.extend([int(primitive.get("material", 0))] * count)
        if int(primitive.get("material", 0)) in rear_indices:
            flat_materials.append((int(primitive.get("material", 0)), count))
    triangle_materials = np.asarray(triangle_materials, dtype=np.int64)
    selected_in_range = selected[(selected >= 0) & (selected < len(triangle_materials))]
    selected_safe = triangle_materials[selected_in_range] if len(selected_in_range) else np.empty(0, dtype=np.int64)
    selected_safe_ok = bool(len(selected_in_range) == len(selected) and np.all(np.isin(selected_safe, list(rear_indices))))
    protected_in_range = np.flatnonzero(protected)
    protected_safe_leak = int(np.isin(triangle_materials[protected_in_range], list(rear_indices)).sum()) if len(protected_in_range) else 0
    final_active = []
    for primitive in final_primitives:
        mi = int(primitive.get("material", 0))
        final_active.append({"material": material_names.get(mi, ""), "image": final["images"][active_image(final, primitive, mi)].get("name", "")})
    body_material = reference_primitives[0].get("material", 0)
    ref_image_index = active_image(reference, reference_primitives[0], int(body_material))
    ref_base_hash = hashlib.sha256(image_bytes(reference, reference_blob, ref_image_index)).hexdigest()
    final_body_hashes = []
    for mi, material in enumerate(final.get("materials", [])):
        if material.get("name") != "RearHeadSafe":
            try:
                final_body_hashes.append(hashlib.sha256(image_bytes(final, final_blob, active_image(final, final_primitives[0], mi))).hexdigest())
            except Exception:
                pass
    safe_image_ok = False
    safe_texture_sha = None
    for mi in rear_indices:
        image_index = active_image(final, next(p for p in final_primitives if int(p.get("material", 0)) == mi), mi)
        payload = image_bytes(final, final_blob, image_index)
        safe_texture_sha = hashlib.sha256(payload).hexdigest()
        try:
            with Image.open(__import__("io").BytesIO(payload)) as image:
                safe_image_ok = image.size == (1024, 1024)
        except Exception:
            safe_image_ok = False
    failures = []
    if not safe_material_present: failures.append("REAR_SAFE_MATERIAL_MISSING")
    if not selected_safe_ok: failures.append("REAR_TRIANGLE_FRONT_MATERIAL_LEAK")
    if protected_safe_leak: failures.append("FRONT_TRIANGLE_REAR_MATERIAL_LEAK")
    if hashlib.sha256(image_bytes(final, final_blob, ref_image_index if ref_image_index < len(final.get("images", [])) else 0)).hexdigest() != ref_base_hash: failures.append("FRONT_PROTECTED_TEXTURE_CHANGED")
    if geometry_face_hash(final, final_blob) != geometry_face_hash(reference, reference_blob): failures.append("GEOMETRY_CHANGED")
    if uv_hash(final, final_blob) != uv_hash(reference, reference_blob): failures.append("UV_CHANGED")
    if not safe_image_ok: failures.append("REAR_SAFE_TEXTURE_UNRESOLVED")
    report = {
        "schema": "rear_semantic_material_verification_v1",
        "final_glb": str(final_path), "reference_glb": str(reference_path),
        "material_slots": material_names,
        "rear_safe_material_indices": sorted(rear_indices),
        "selected_triangle_count": int(len(selected)),
        "selected_rear_safe_count": int(np.isin(selected_safe, list(rear_indices)).sum()),
        "selected_triangles_all_rear_safe": selected_safe_ok,
        "protected_face_triangle_count": int(protected.sum()),
        "protected_face_rear_safe_leak_count": protected_safe_leak,
        "original_base_color_sha256": ref_base_hash,
        "final_body_base_color_hashes": final_body_hashes,
        "safe_texture_sha256": safe_texture_sha,
        "safe_texture_1024": safe_image_ok,
        "geometry_fingerprint_unchanged": geometry_face_hash(final, final_blob) == geometry_face_hash(reference, reference_blob),
        "triangle_count_unchanged": len(triangle_materials) == sum(len(accessor(reference, reference_blob, p["indices"]).reshape(-1, 3)) for p in reference_primitives),
        "uv_coordinates_unchanged": uv_hash(final, final_blob) == uv_hash(reference, reference_blob),
        "facial_lineage_present": False,
        "failures": failures,
        "success": not failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--final-glb", required=True)
    p.add_argument("--reference-glb", required=True)
    p.add_argument("--selected-triangles", required=True)
    p.add_argument("--provenance", required=True)
    p.add_argument("--report", required=True)
    args = p.parse_args()
    report = verify(Path(args.final_glb), Path(args.reference_glb), Path(args.selected_triangles), Path(args.provenance), Path(args.report))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
