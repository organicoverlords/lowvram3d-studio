"""One CPU-only, material-aware color recovery pass for the chart-separated panda.

The mesh is treated as immutable.  The worker reconstructs atlas colors from the existing
registered source, face-ID/depth bundle, and triangle provenance, then appends a replacement
PNG to the GLB without touching any geometry, index, UV, or ownership buffer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from mesh_io import read_glb, triangle_components

REGIONS = ("fur", "clothing", "rifle", "backpack", "tail")
REGION_IDS = {name: index for index, name in enumerate(REGIONS)}
LAB_EPS = 1e-5
CONFIDENCE_THRESHOLD = 0.20
HIGH_CONFIDENCE = 0.45
DONOR_K = 32
WELD = 4e-4


def _read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise RuntimeError(f"not a GLB: {path}")
    json_length = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20 : 20 + json_length])
    offset = 20 + json_length + ((4 - json_length % 4) % 4)
    bin_length, kind = struct.unpack_from("<II", raw, offset)
    if kind != 0x004E4942:
        raise RuntimeError("GLB has no BIN chunk")
    return gltf, raw[offset + 8 : offset + 8 + bin_length]


def _accessor_bytes(gltf: dict, blob: bytes, index: int) -> bytes:
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]
    component = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}[accessor["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    stride = view.get("byteStride") or component * width
    if stride == component * width:
        return blob[start : start + count * stride]
    return b"".join(blob[start + row * stride : start + row * stride + component * width] for row in range(count))


def immutable_buffer_hashes(path: Path) -> dict:
    gltf, blob = _read_glb(path)
    position = bytearray()
    normals = bytearray()
    uv = bytearray()
    indices = bytearray()
    for mesh in gltf.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            attrs = primitive.get("attributes", {})
            if "POSITION" in attrs:
                position.extend(_accessor_bytes(gltf, blob, attrs["POSITION"]))
            if "NORMAL" in attrs:
                normals.extend(_accessor_bytes(gltf, blob, attrs["NORMAL"]))
            if "TEXCOORD_0" in attrs:
                uv.extend(_accessor_bytes(gltf, blob, attrs["TEXCOORD_0"]))
            if "indices" in primitive:
                indices.extend(_accessor_bytes(gltf, blob, primitive["indices"]))

    def digest(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    return {
        "position_bytes_sha256": digest(bytes(position)),
        "normal_bytes_sha256": digest(bytes(normals)),
        "uv_bytes_sha256": digest(bytes(uv)),
        "index_bytes_sha256": digest(bytes(indices)),
        "geometry_uv_index_sha256": digest(bytes(position + normals + uv + indices)),
    }


def _project(points: np.ndarray, direction: np.ndarray, ortho: float) -> np.ndarray:
    axis = int(np.argmax(np.abs(direction)))
    ua, va = (0, 2) if axis == 1 else ((1, 2) if axis == 0 else (0, 1))
    flip_u = -1.0 if direction[axis] > 0 else 1.0
    return np.stack(
        [points[:, ua] * flip_u / ortho + 0.5, 0.5 - points[:, va] / ortho], axis=1
    )


def _source_material_masks(source_rgb: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, dict]:
    """Create adaptive source masks; coordinates are relative to the detected foreground bbox."""
    hsv = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    foreground = alpha > 0.35
    ys, xs = np.nonzero(foreground)
    if len(xs) == 0:
        raise RuntimeError("SOURCE_FOREGROUND_EMPTY")
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    nx = (np.arange(source_rgb.shape[1])[None, :] - x0) / max(x1 - x0, 1)
    ny = (np.arange(source_rgb.shape[0])[:, None] - y0) / max(y1 - y0, 1)
    warm = ((h < 55) | (h > 340)) & (s > 0.16) & (v > 0.12)
    dark_neutral = (s < 0.32) & (v < 0.64)
    olive = (h >= 45) & (h <= 145) & (s > 0.10) & (v < 0.70)

    # Priority is deliberate: equipment is removed from the broad fur/clothing remainder.
    tail = foreground & (nx > 0.60) & (ny > 0.45) & warm
    rifle_line = np.abs(ny - (1.52 * nx - 0.15)) < 0.115
    rifle = foreground & (nx > 0.27) & (nx < 0.76) & (ny > 0.28) & rifle_line & dark_neutral
    backpack = foreground & (nx > 0.55) & (nx < 0.90) & (ny > 0.18) & (ny < 0.72) & (olive | dark_neutral) & ~tail & ~rifle
    clothing = foreground & ~tail & ~rifle & ~backpack & (olive | ((s > 0.08) & (v < 0.58)))
    fur = foreground & ~tail & ~rifle & ~backpack & ~clothing

    labels = np.full(foreground.shape, -1, np.int8)
    labels[fur] = REGION_IDS["fur"]
    labels[clothing] = REGION_IDS["clothing"]
    labels[rifle] = REGION_IDS["rifle"]
    labels[backpack] = REGION_IDS["backpack"]
    labels[tail] = REGION_IDS["tail"]
    # Any foreground pixel that did not meet a narrow class rule remains fur rather than empty.
    labels[foreground & (labels < 0)] = REGION_IDS["fur"]
    counts = {name: int(np.count_nonzero(labels == idx)) for name, idx in REGION_IDS.items()}
    return labels, {"bbox": [x0, y0, x1 + 1, y1 + 1], "pixel_counts": counts}


def _face_id_match(face_id: np.ndarray, x: int, y: int, triangle_id: int, radius: int = 2) -> bool:
    y0, y1 = max(0, y - radius), min(face_id.shape[0], y + radius + 1)
    x0, x1 = max(0, x - radius), min(face_id.shape[1], x + radius + 1)
    return bool(np.any(face_id[y0:y1, x0:x1] == int(triangle_id)))


def _raster_owner(uv_triangles: np.ndarray, size: int) -> np.ndarray:
    owner = np.full((size, size), -1, np.int32)
    px = uv_triangles * float(size - 1)
    for triangle_id, a in enumerate(px):
        x0, y0 = np.maximum(np.floor(a.min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(a.max(0)).astype(int), size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        fx, fy = xs + 0.5, ys + 0.5
        (ax, ay), (bx, by), (cx, cy) = a
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        wa = ((by - cy) * (fx - cx) + (cx - bx) * (fy - cy)) / den
        wb = ((cy - ay) * (fx - cx) + (ax - cx) * (fy - cy)) / den
        inside = (wa >= -1e-4) & (wb >= -1e-4) & (1.0 - wa - wb >= -1e-4)
        owner[ys[inside], xs[inside]] = triangle_id
    return owner


def _append_image_glb(input_glb: Path, output_glb: Path, png: bytes) -> None:
    gltf, original_blob = _read_glb(input_glb)
    images = gltf.get("images", [])
    if len(images) != 1 or "bufferView" not in images[0]:
        raise RuntimeError("expected exactly one embedded image bufferView")
    blob = bytearray(original_blob)
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    blob.extend(png)
    gltf.setdefault("bufferViews", []).append({"buffer": 0, "byteOffset": offset, "byteLength": len(png)})
    images[0]["bufferView"] = len(gltf["bufferViews"]) - 1
    gltf["buffers"][0]["byteLength"] = len(blob)
    json_bytes = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = bytes(blob) + b"\x00" * ((4 - len(blob) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    output_glb.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
        + struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes
    )


def recover(args: argparse.Namespace) -> dict:
    input_glb = Path(args.input_glb)
    before_hashes = immutable_buffer_hashes(input_glb)
    positions, _normals, uv, tris = read_glb(input_glb)
    positions = positions.astype(np.float64)
    uv_triangles = uv[tris].astype(np.float64)

    bundle = np.load(args.bundle_npz)
    if len(bundle["tris"]) != len(tris):
        raise RuntimeError("BUNDLE_TRIANGLE_ORDER_MISMATCH")
    verts = bundle["verts"].astype(np.float64)
    bundle_tris = bundle["tris"].astype(np.int64)
    bundle_uv_triangles = bundle["uvs"].astype(np.float64)
    if not np.allclose(uv_triangles, bundle_uv_triangles, atol=0.0, rtol=0.0):
        raise RuntimeError("BUNDLE_UV_ORDER_OR_VALUE_MISMATCH")
    normals = bundle["normals"].astype(np.float64)
    visible = np.asarray(bundle["vis_front"], dtype=bool)
    face_id = np.asarray(bundle["face_id_front"], dtype=np.int32)
    direction = bundle["view_locs"][0].astype(np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    ortho = float(bundle["ortho_scale"])

    provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
    observed = np.asarray(np.load(args.observed), dtype=bool)
    confidence = np.asarray(provenance["winning_confidence"], dtype=np.float64)
    face_matched = np.asarray(provenance["face_id_matched"], dtype=bool)
    masked_valid = np.asarray(provenance["masked_valid"], dtype=bool)
    rear_dominant = np.asarray(provenance["rear_dominant"], dtype=bool)
    if len(observed) != len(tris):
        raise RuntimeError("OBSERVED_TRIANGLE_ORDER_MISMATCH")

    source_bgra = cv2.imread(str(Path(args.source_view)), cv2.IMREAD_UNCHANGED)
    if source_bgra is None or source_bgra.shape[:2] != face_id.shape:
        raise RuntimeError("SOURCE_VIEW_OR_FACE_ID_DIMENSION_MISMATCH")
    source_rgb = cv2.cvtColor(source_bgra[:, :, :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    source_alpha = source_bgra[:, :, 3].astype(np.float32) / 255.0 if source_bgra.shape[2] == 4 else np.ones(face_id.shape, np.float32)
    source_labels, source_region_report = _source_material_masks(source_rgb, source_alpha)
    facial = cv2.imread(str(Path(args.facial_mask)), cv2.IMREAD_GRAYSCALE)
    if facial is None or facial.shape != face_id.shape:
        raise RuntimeError("FACIAL_MASK_MISSING_OR_DIMENSION_MISMATCH")

    centroids = verts[bundle_tris].mean(axis=1)
    screen = _project(centroids, direction, ortho)
    sx = np.clip((screen[:, 0] * (source_rgb.shape[1] - 1)).astype(int), 0, source_rgb.shape[1] - 1)
    sy = np.clip((screen[:, 1] * (source_rgb.shape[0] - 1)).astype(int), 0, source_rgb.shape[0] - 1)
    source_valid = (
        observed & visible & face_matched & masked_valid & (confidence >= CONFIDENCE_THRESHOLD)
        & (source_alpha[sy, sx] > 0.35)
    )
    source_tri_colour = np.zeros((len(tris), 3), np.float32)
    source_tri_colour[source_valid] = source_rgb[sy[source_valid], sx[source_valid]]
    tri_region = np.full(len(tris), -1, np.int8)
    tri_region[source_valid] = source_labels[sy[source_valid], sx[source_valid]]
    tri_facial = source_valid & (facial[sy, sx] > 0)
    high_conf = source_valid & (confidence >= HIGH_CONFIDENCE)

    # Existing atlas is used as the current-color side of each region transfer.
    base_bgr = cv2.imread(str(Path(args.basecolor)), cv2.IMREAD_COLOR)
    if base_bgr is None or base_bgr.shape[:2] != face_id.shape:
        raise RuntimeError("BASECOLOR_DIMENSION_MISMATCH")
    base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    size = base_rgb.shape[0]
    owner = _raster_owner(uv_triangles, size)
    owner_region = np.full(owner.shape, -1, np.int8)
    valid_owner = owner >= 0
    owner_region[valid_owner] = tri_region[owner[valid_owner]]
    atlas = base_rgb.copy()
    observed_mask = np.zeros(owner.shape, bool)
    synthesized_mask = np.zeros(owner.shape, bool)
    confidence_atlas = np.zeros(owner.shape, np.float32)

    # Per-region Lab affine transfer for observed source pixels.  Source colors are then written
    # through the existing face-ID/visibility mapping, never from a mirrored or wrapped view.
    region_transfers: dict[str, dict] = {}
    for region_name, region_id in REGION_IDS.items():
        triangle_ids = np.flatnonzero(source_valid & (tri_region == region_id))
        if len(triangle_ids) == 0:
            region_transfers[region_name] = {"samples": 0, "status": "NOT_PROVEN"}
            continue
        uv_centres = uv_triangles[triangle_ids].mean(axis=1)
        ux = np.clip((uv_centres[:, 0] * (size - 1)).astype(int), 0, size - 1)
        uy = np.clip((uv_centres[:, 1] * (size - 1)).astype(int), 0, size - 1)
        current = base_rgb[uy, ux]
        target = source_tri_colour[triangle_ids]
        cur_lab = cv2.cvtColor(current.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        tgt_lab = cv2.cvtColor(target.reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
        mean_cur, mean_tgt = cur_lab.mean(axis=0), tgt_lab.mean(axis=0)
        std_cur, std_tgt = cur_lab.std(axis=0), tgt_lab.std(axis=0)
        scale = std_tgt / np.maximum(std_cur, LAB_EPS)
        region_transfers[region_name] = {
            "samples": int(len(triangle_ids)),
            "mean_current_lab": mean_cur.round(4).tolist(),
            "mean_source_lab": mean_tgt.round(4).tolist(),
            "scale_lab": scale.round(4).tolist(),
            "status": "PROVEN",
        }

        for triangle_id in triangle_ids.tolist():
            a = uv_triangles[triangle_id] * float(size - 1)
            x0, y0 = np.maximum(np.floor(a.min(0)).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(a.max(0)).astype(int), size - 1)
            if x1 < x0 or y1 < y0:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            fx, fy = xs + 0.5, ys + 0.5
            (ax, ay), (bx, by), (cx, cy) = a
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-12:
                continue
            wa = ((by - cy) * (fx - cx) + (cx - bx) * (fy - cy)) / den
            wb = ((cy - ay) * (fx - cx) + (ax - cx) * (fy - cy)) / den
            wc = 1.0 - wa - wb
            inside = (wa >= -1e-4) & (wb >= -1e-4) & (wc >= -1e-4)
            if not inside.any():
                continue
            world = (wa[..., None] * verts[bundle_tris[triangle_id, 0]]
                     + wb[..., None] * verts[bundle_tris[triangle_id, 1]]
                     + wc[..., None] * verts[bundle_tris[triangle_id, 2]])
            flat_world = world[inside]
            src_screen = _project(flat_world, direction, ortho)
            qx = np.clip((src_screen[:, 0] * (source_rgb.shape[1] - 1)).astype(int), 0, source_rgb.shape[1] - 1)
            qy = np.clip((src_screen[:, 1] * (source_rgb.shape[0] - 1)).astype(int), 0, source_rgb.shape[0] - 1)
            valid = (
                (src_screen[:, 0] >= 0) & (src_screen[:, 0] <= 1)
                & (src_screen[:, 1] >= 0) & (src_screen[:, 1] <= 1)
                & (source_alpha[qy, qx] > 0.35)
                & np.array([_face_id_match(face_id, int(x), int(y), triangle_id) for x, y in zip(qx, qy)])
            )
            if not valid.any():
                continue
            dest_x, dest_y = xs[inside][valid], ys[inside][valid]
            sample = source_rgb[qy[valid], qx[valid]]
            lab = cv2.cvtColor(base_rgb[dest_y, dest_x].reshape(1, -1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
            corrected = (lab - mean_cur) * scale + mean_tgt
            corrected = np.clip(corrected, [0, 0, 0], [100, 255, 255]).astype(np.float32)
            corrected_rgb = cv2.cvtColor(corrected.reshape(1, -1, 3), cv2.COLOR_LAB2RGB).reshape(-1, 3)
            # Direct source color is the source-aware fallback if the current atlas was empty/black.
            current_luma = base_rgb[dest_y, dest_x].mean(axis=1)
            corrected_rgb[current_luma < 0.02] = sample[current_luma < 0.02]
            existing_conf = confidence_atlas[dest_y, dest_x]
            win = confidence[triangle_id] >= existing_conf
            if win.any():
                atlas[dest_y[win], dest_x[win]] = corrected_rgb[win]
                confidence_atlas[dest_y[win], dest_x[win]] = confidence[triangle_id]
                observed_mask[dest_y[win], dest_x[win]] = True

    # 3D same-component, material-aware donor propagation. No UV-neighbour search is used.
    edge1 = verts[bundle_tris[:, 1]] - verts[bundle_tris[:, 0]]
    edge2 = verts[bundle_tris[:, 2]] - verts[bundle_tris[:, 0]]
    tri_normals = np.cross(edge1, edge2)
    tri_normals /= np.maximum(np.linalg.norm(tri_normals, axis=1, keepdims=True), 1e-12)
    component, _ = triangle_components(verts, bundle_tris, WELD)
    synth_colour = np.zeros((len(tris), 3), np.float32)
    synth_region = np.full(len(tris), -1, np.int8)
    donor_ids_all = np.flatnonzero(high_conf & (tri_region >= 0))
    unresolved = []
    donor_count = 0
    if len(donor_ids_all) == 0:
        raise RuntimeError("NO_HIGH_CONFIDENCE_MATERIAL_DONORS")
    for component_id in np.unique(component):
        members = np.flatnonzero(component == component_id)
        donors = members[np.isin(members, donor_ids_all)]
        targets = members[~source_valid[members]]
        if not len(targets):
            continue
        if not len(donors):
            unresolved.extend(targets.tolist())
            continue
        tree = cKDTree(centroids[donors])
        k = min(DONOR_K, len(donors))
        distance, nearest = tree.query(centroids[targets], k=k)
        if k == 1:
            distance, nearest = distance[:, None], nearest[:, None]
        candidate_ids = donors[nearest]
        dot = np.einsum("ijk,ik->ij", tri_normals[candidate_ids], tri_normals[targets])
        weights = np.clip(dot, 0.0, None) / np.maximum(distance, 1e-6)
        for row, target_id in enumerate(targets.tolist()):
            ids = candidate_ids[row]
            w = weights[row]
            allowed = w > 0
            if rear_dominant[target_id]:
                allowed &= ~tri_facial[ids]
            if not allowed.any():
                unresolved.append(target_id)
                continue
            ids, w = ids[allowed], w[allowed]
            region_weights = np.zeros(len(REGIONS), np.float64)
            np.add.at(region_weights, tri_region[ids], w)
            region_id = int(np.argmax(region_weights))
            chosen = tri_region[ids] == region_id
            if not chosen.any():
                unresolved.append(target_id)
                continue
            w2 = w[chosen]
            w2 /= max(float(w2.sum()), 1e-12)
            synth_colour[target_id] = (source_tri_colour[ids[chosen]] * w2[:, None]).sum(axis=0)
            synth_region[target_id] = region_id
            donor_count += 1

    if unresolved:
        raise RuntimeError(f"UNRESOLVED_MATERIAL_REGIONS:{len(unresolved)}")

    # Fill every non-observed atlas ownership with its 3D donor color.
    target = valid_owner & ~observed_mask
    target_ids = owner[target]
    valid_synth = (target_ids >= 0) & (synth_region[target_ids] >= 0)
    if valid_synth.any():
        atlas[target][valid_synth] = synth_colour[target_ids[valid_synth]]
        synthesized_mask[target] = valid_synth

    # One-pixel dilation/feather per material label and ownership chart only.
    for region_id in range(len(REGIONS)):
        mask = valid_owner & (owner_region == region_id)
        painted = mask & (observed_mask | synthesized_mask)
        if not painted.any():
            continue
        kernel = np.ones((3, 3), np.uint8)
        local_mask = cv2.dilate(painted.astype(np.uint8), kernel) > 0
        fill = mask & ~painted & local_mask
        if fill.any():
            for channel in range(3):
                source_channel = np.where(painted, atlas[..., channel], 0.0).astype(np.float32)
                weights = painted.astype(np.float32)
                num = cv2.blur(source_channel, (3, 3))
                den = cv2.blur(weights, (3, 3))
                values = num / np.maximum(den, 1e-6)
                atlas[..., channel][fill] = values[fill]

    output_atlas = Path(args.output_atlas)
    output_atlas.parent.mkdir(parents=True, exist_ok=True)
    encoded = cv2.imencode(".png", cv2.cvtColor(np.clip(atlas * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))[1].tobytes()
    output_atlas.write_bytes(encoded)
    output_glb = Path(args.output_glb)
    _append_image_glb(input_glb, output_glb, encoded)
    after_hashes = immutable_buffer_hashes(output_glb)

    cv2.imwrite(str(Path(args.observed_mask)), (observed_mask.astype(np.uint8) * 255))
    cv2.imwrite(str(Path(args.synthesized_mask)), (synthesized_mask.astype(np.uint8) * 255))
    palette = np.asarray([[170, 100, 70], [70, 125, 80], [80, 100, 120], [100, 70, 155], [190, 135, 50]], np.uint8)
    region_image = np.zeros((*owner_region.shape, 3), np.uint8)
    for region_id in range(len(REGIONS)):
        region_image[owner_region == region_id] = palette[region_id]
    cv2.imwrite(str(Path(args.region_mask)), cv2.cvtColor(region_image, cv2.COLOR_RGB2BGR))

    report = {
        "schema": "material_aware_color_recovery_v1",
        "classification": "PROVEN_CPU_ONLY_PASS",
        "input_glb": str(input_glb),
        "output_glb": str(output_glb),
        "input_atlas": str(args.basecolor),
        "output_atlas": str(output_atlas),
        "regions": list(REGIONS),
        "source_region_detection": source_region_report,
        "region_transfers_lab": region_transfers,
        "observed_triangle_count": int(source_valid.sum()),
        "high_confidence_donor_triangle_count": int(donor_ids_all.size),
        "synthesized_triangle_count": int(donor_count),
        "observed_atlas_pixels": int(observed_mask.sum()),
        "synthesized_atlas_pixels": int(synthesized_mask.sum()),
        "remaining_unpainted_owned_pixels": int((valid_owner & ~(observed_mask | synthesized_mask)).sum()),
        "rear_dominant_triangle_count": int(rear_dominant.sum()),
        "rear_facial_provenance_count": int(len(provenance.get("illegal_rear_facial_triangle_ids", []))),
        "rear_facial_donors_used": 0,
        "material_donor_policy": "same welded connected component; 3D centroid KD-tree; normal-compatible; facial donors barred for rear-dominant targets",
        "atlas_policy": "one-pixel dilation inside owner/material region only",
        "immutable_hashes_before": before_hashes,
        "immutable_hashes_after": after_hashes,
        "geometry_uv_index_hashes_unchanged": before_hashes == after_hashes,
        "masks": {"observed": str(args.observed_mask), "synthesized": str(args.synthesized_mask), "regions": str(args.region_mask)},
        "gpu_used": False,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glb", required=True)
    parser.add_argument("--bundle-npz", required=True)
    parser.add_argument("--source-view", required=True)
    parser.add_argument("--facial-mask", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--observed", required=True)
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--output-atlas", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--observed-mask", required=True)
    parser.add_argument("--synthesized-mask", required=True)
    parser.add_argument("--region-mask", required=True)
    args = parser.parse_args()
    report = recover(args)
    print(
        f"MATERIAL_AWARE_RECOVERY observed_triangles={report['observed_triangle_count']} "
        f"synthesized_triangles={report['synthesized_triangle_count']} "
        f"observed_pixels={report['observed_atlas_pixels']} synthesized_pixels={report['synthesized_atlas_pixels']} "
        f"immutable={report['geometry_uv_index_hashes_unchanged']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
