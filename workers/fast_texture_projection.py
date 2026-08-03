"""Vectorised single-view texture projection, without Blender in the hot path.

The Blender route this replaces walks `mesh.polygons` in Python calling `scene.ray_cast` per
polygon, then touches every atlas texel through `bpy.types.Image.pixels`. On the 763,850-triangle
ship at a 4096 atlas that ran 190 minutes without finishing an export. Nothing about the job needs
Blender: it is a rasterisation problem, and NumPy does it in one pass over arrays.

Shape of the work:

  * fit an orthographic camera to the conditioning matte by searching yaw/pitch/roll for the pose
    whose projected silhouette best matches the mask, then keep that pose for the projection. The
    mesh comes out of the generator in its own frame, so the camera has to be recovered rather
    than assumed.
  * scatter-rasterise the mesh into image space once to get a depth buffer, for occlusion.
  * scatter-rasterise the UV triangles into the atlas once to get per-texel ownership and
    barycentric weights.
  * interpolate each owned texel's world position and normal, project it, gate it, and sample.

Rasterisation is by barycentric supersampling in area-based tiers rather than a per-triangle Python
loop: triangles are bucketed by how many texels they cover and each bucket is rasterised as one
vectorised batch. Small triangles dominate here -- roughly 1.5 texels each -- so the first tier
carries almost everything.

Gates on every texel: inside image bounds, source matte says foreground, triangle faces the camera,
and the sample agrees with the depth buffer. Texels that fail the matte gate are recorded
separately as rejected background rather than being painted, which is what stops plate colour from
reaching the hull.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np

from mesh_io import read_glb
from projection_input_contract import enforce, validate

# Area tiers, in atlas texels, and the supersampling grid used for each.
TIERS = ((2.0, 4), (16.0, 8), (128.0, 16), (1024.0, 40), (float("inf"), 96))
DEPTH_TOLERANCE_SCALE = 0.004
FRONT_FACING_MIN = 0.10


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- GLB helpers
def _read_glb(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise RuntimeError(f"not a GLB: {path}")
    json_length = struct.unpack_from("<I", raw, 12)[0]
    gltf = json.loads(raw[20:20 + json_length])
    offset = 20 + json_length + ((4 - json_length % 4) % 4)
    bin_length, kind = struct.unpack_from("<II", raw, offset)
    if kind != 0x004E4942:
        raise RuntimeError("GLB has no BIN chunk")
    return gltf, raw[offset + 8:offset + 8 + bin_length]


def _accessor_bytes(gltf: dict, blob: bytes, index: int) -> bytes:
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]
    component = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}[accessor["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[accessor["type"]]
    stride = view.get("byteStride") or component * width
    if stride == component * width:
        return blob[start:start + count * stride]
    return b"".join(blob[start + row * stride:start + row * stride + component * width]
                    for row in range(count))


def immutable_buffer_hashes(path: Path) -> dict:
    """Position/normal/UV/index byte digests, so texture binding can be proven non-destructive."""
    gltf, blob = _read_glb(path)
    position, normals, uv, indices = bytearray(), bytearray(), bytearray(), bytearray()
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
    return {
        "position_bytes_sha256": sha256_bytes(bytes(position)),
        "normal_bytes_sha256": sha256_bytes(bytes(normals)),
        "uv_bytes_sha256": sha256_bytes(bytes(uv)),
        "index_bytes_sha256": sha256_bytes(bytes(indices)),
        "geometry_uv_index_sha256": sha256_bytes(bytes(position + normals + uv + indices)),
    }


def bind_texture(input_glb: Path, output_glb: Path, png: bytes) -> int:
    """Append the atlas and bind it as baseColorTexture, leaving every geometry buffer alone."""
    gltf, original = _read_glb(input_glb)
    blob = bytearray(original)
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    blob.extend(png)
    gltf.setdefault("bufferViews", []).append(
        {"buffer": 0, "byteOffset": offset, "byteLength": len(png)})
    view_index = len(gltf["bufferViews"]) - 1

    images = gltf.setdefault("images", [])
    images.append({"bufferView": view_index, "mimeType": "image/png", "name": "basecolor"})
    image_index = len(images) - 1
    samplers = gltf.setdefault("samplers", [])
    if not samplers:
        samplers.append({"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497})
    textures = gltf.setdefault("textures", [])
    textures.append({"sampler": 0, "source": image_index})
    texture_index = len(textures) - 1

    materials = gltf.setdefault("materials", [])
    if not materials:
        materials.append({"name": "projected"})
        for mesh in gltf.get("meshes", []):
            for primitive in mesh.get("primitives", []):
                primitive.setdefault("material", 0)
    bound = 0
    for material in materials:
        pbr = material.setdefault("pbrMetallicRoughness", {})
        pbr["baseColorTexture"] = {"index": texture_index}
        pbr["baseColorFactor"] = [1.0, 1.0, 1.0, 1.0]
        pbr.setdefault("metallicFactor", 0.0)
        pbr.setdefault("roughnessFactor", 0.85)
        bound += 1

    gltf["buffers"][0]["byteLength"] = len(blob)
    json_bytes = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = bytes(blob) + b"\x00" * ((4 - len(blob) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    output_glb.parent.mkdir(parents=True, exist_ok=True)
    output_glb.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
        + struct.pack("<II", len(bin_bytes), 0x004E4942) + bin_bytes)
    return bound


# --------------------------------------------------------------------------- rasterisation
def _barycentric_grid(steps: int) -> np.ndarray:
    """Sample points inside the unit triangle, as (S, 2) barycentric (wa, wb) pairs."""
    ticks = (np.arange(steps) + 0.5) / steps
    a, b = np.meshgrid(ticks, ticks, indexing="ij")
    a, b = a.ravel(), b.ravel()
    keep = (a + b) <= 1.0
    a, b = a[keep], b[keep]
    if a.size == 0:  # steps == 1 degenerates; fall back to the centroid
        a = np.array([1 / 3]); b = np.array([1 / 3])
    return np.stack((a, b), axis=1)


def _tier_split(areas: np.ndarray) -> list[tuple[np.ndarray, int]]:
    groups, low = [], -1.0
    for high, steps in TIERS:
        selected = np.flatnonzero((areas > low) & (areas <= high))
        if selected.size:
            groups.append((selected, steps))
        low = high
    return groups


def rasterise_atlas(uv: np.ndarray, triangles: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-texel triangle ownership and barycentric weights, by tiered supersampling."""
    owner = np.full((size, size), -1, np.int32)
    weights = np.zeros((size, size, 2), np.float32)

    px = uv[triangles] * float(size)          # (T, 3, 2) in texel units
    edge1 = px[:, 1] - px[:, 0]
    edge2 = px[:, 2] - px[:, 0]
    areas = 0.5 * np.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])

    for selected, steps in _tier_split(areas):
        bary = _barycentric_grid(steps)                       # (S, 2)
        for start in range(0, selected.size, 200_000):        # bound peak memory, not per-triangle
            batch = selected[start:start + 200_000]
            origin = px[batch, 0][:, None, :]                 # (B, 1, 2)
            e1 = edge1[batch][:, None, :]
            e2 = edge2[batch][:, None, :]
            wa = bary[None, :, 0, None]
            wb = bary[None, :, 1, None]
            points = origin + e1 * wa + e2 * wb               # (B, S, 2)
            xs = np.clip(points[..., 0].astype(np.int32), 0, size - 1)
            ys = np.clip(points[..., 1].astype(np.int32), 0, size - 1)
            ids = np.repeat(batch[:, None], bary.shape[0], axis=1)
            owner[ys.ravel(), xs.ravel()] = ids.ravel()
            weights[ys.ravel(), xs.ravel(), 0] = np.broadcast_to(
                bary[None, :, 0], ids.shape).ravel()
            weights[ys.ravel(), xs.ravel(), 1] = np.broadcast_to(
                bary[None, :, 1], ids.shape).ravel()
    return owner, weights


def depth_buffer(screen: np.ndarray, depth: np.ndarray, triangles: np.ndarray,
                 height: int, width: int) -> np.ndarray:
    """Nearest-surface depth per source pixel, by the same tiered scatter."""
    buffer = np.full((height, width), np.inf, np.float32)
    tri_screen = screen[triangles]                             # (T, 3, 2)
    tri_depth = depth[triangles]                               # (T, 3)
    e1 = tri_screen[:, 1] - tri_screen[:, 0]
    e2 = tri_screen[:, 2] - tri_screen[:, 0]
    areas = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])

    for selected, steps in _tier_split(areas):
        bary = _barycentric_grid(steps)
        for start in range(0, selected.size, 200_000):
            batch = selected[start:start + 200_000]
            wa = bary[None, :, 0]
            wb = bary[None, :, 1]
            wc = 1.0 - wa - wb
            points = (tri_screen[batch, 0][:, None, :] * wc[..., None]
                      + tri_screen[batch, 1][:, None, :] * wa[..., None]
                      + tri_screen[batch, 2][:, None, :] * wb[..., None])
            zs = (tri_depth[batch, 0][:, None] * wc
                  + tri_depth[batch, 1][:, None] * wa
                  + tri_depth[batch, 2][:, None] * wb)
            xs = points[..., 0].astype(np.int32)
            ys = points[..., 1].astype(np.int32)
            inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
            np.minimum.at(buffer, (ys[inside], xs[inside]), zs[inside].astype(np.float32))
    return buffer


# --------------------------------------------------------------------------- camera
def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ rx @ ry


def fit_to_mask(rotated: np.ndarray, mask: np.ndarray) -> tuple[float, np.ndarray]:
    """Uniform scale and offset mapping the projected silhouette bbox onto the mask bbox."""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        raise RuntimeError("PROJECTION_MASK_EMPTY")
    target = np.array([cols[0], rows[0]], np.float64)
    target_size = np.array([cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1], np.float64)
    low = rotated[:, :2].min(axis=0)
    span = np.maximum(rotated[:, :2].max(axis=0) - low, 1e-9)
    scale = float(np.min(target_size / span))
    centred = target + (target_size - span * scale) * 0.5
    return scale, centred - low * scale


def project(vertices: np.ndarray, matrix: np.ndarray, scale: float,
            offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotated = vertices @ matrix.T
    screen = rotated[:, :2] * scale + offset
    return screen, rotated[:, 2]


def fit_camera(vertices: np.ndarray, triangles: np.ndarray, mask: np.ndarray,
               coarse_step: float = 15.0) -> dict:
    """Search yaw/pitch/roll for the pose whose silhouette best matches the conditioning matte."""
    height, width = mask.shape
    probe = 192
    small = cv2.resize(mask.astype(np.uint8), (probe, probe), interpolation=cv2.INTER_NEAREST) > 0
    centre = vertices.mean(axis=0)
    centred = vertices - centre

    # Score against a rasterised silhouette, not a vertex splat. A splat leaves interior holes on
    # a mesh whose vertices are unevenly dense, which caps the achievable IoU and biases the search
    # toward whichever pose happens to splat most evenly rather than the pose that actually matches.
    stride = max(1, triangles.shape[0] // 60000)
    faces = triangles[::stride]
    bary = _barycentric_grid(4)

    def score(yaw, pitch, roll):
        matrix = rotation(yaw, pitch, roll)
        rotated = centred @ matrix.T
        try:
            scale, offset = fit_to_mask(rotated, small)
        except RuntimeError:
            return -1.0
        xy = rotated[:, :2] * scale + offset
        corners = xy[faces]
        wa = bary[None, :, 0, None]
        wb = bary[None, :, 1, None]
        points = (corners[:, 0][:, None, :] * (1.0 - wa - wb)
                  + corners[:, 1][:, None, :] * wa
                  + corners[:, 2][:, None, :] * wb)
        xs = np.clip(points[..., 0].astype(np.int32), 0, probe - 1).ravel()
        ys = np.clip(points[..., 1].astype(np.int32), 0, probe - 1).ravel()
        hit = np.zeros((probe, probe), bool)
        hit[ys, xs] = True
        union = np.count_nonzero(hit | small)
        return float(np.count_nonzero(hit & small) / union) if union else 0.0

    best = (-1.0, 0.0, 0.0, 0.0)
    for yaw in np.arange(0.0, 360.0, coarse_step):
        for pitch in (-30.0, -15.0, 0.0, 15.0, 30.0):
            for roll in (-15.0, 0.0, 15.0):
                value = score(yaw, pitch, roll)
                if value > best[0]:
                    best = (value, yaw, pitch, roll)
    _iou, yaw, pitch, roll = best
    for _refine in range(2):
        step = coarse_step / 3.0
        for dy in (-step, 0.0, step):
            for dp in (-step, 0.0, step):
                for dr in (-step, 0.0, step):
                    value = score(yaw + dy, pitch + dp, roll + dr)
                    if value > best[0]:
                        best = (value, yaw + dy, pitch + dp, roll + dr)
        _iou, yaw, pitch, roll = best
        coarse_step = step

    matrix = rotation(yaw, pitch, roll)
    rotated = centred @ matrix.T
    scale, offset = fit_to_mask(rotated, mask)
    return {"yaw": round(yaw, 3), "pitch": round(pitch, 3), "roll": round(roll, 3),
            "silhouette_iou": round(best[0], 4), "scale": float(scale),
            "offset": [float(v) for v in offset], "centre": [float(v) for v in centre],
            "matrix": matrix, "image_size": [int(width), int(height)]}


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="UV'd GLB; geometry and UVs are never modified")
    parser.add_argument("--source-image", required=True, help="cleaned RGBA conditioning")
    parser.add_argument("--alpha-mask", required=True)
    parser.add_argument("--original-plate", required=True)
    parser.add_argument("--mask-method", default="BIREFNET_HARD_MASK")
    parser.add_argument("--conditioning", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--atlas-size", type=int, default=2048)
    parser.add_argument("--padding-px", type=int, default=1)
    parser.add_argument("--skip-contract", action="store_true",
                        help="benchmark harness only; never for a production atlas")
    args = parser.parse_args()

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    started_all = time.time()

    # ---- input contract -------------------------------------------------------------
    contract = None
    if not args.skip_contract:
        contract = validate(Path(args.source_image), Path(args.alpha_mask),
                            Path(args.conditioning) if args.conditioning else None,
                            Path(args.original_plate), args.mask_method)
        (root / "projection_input_contract.json").write_text(
            json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        print(f"PROJECTION_CONTRACT {contract['classification']} "
              f"failures={contract['failures']}", flush=True)
        enforce(contract)

    # ---- inputs ---------------------------------------------------------------------
    started = time.time()
    positions, normals, uv, triangles = read_glb(Path(args.mesh))
    positions = np.asarray(positions, np.float64)
    triangles = np.asarray(triangles, np.int64)
    if uv is None or len(uv) == 0:
        raise RuntimeError("mesh has no TEXCOORD_0; run the UV stage first")
    uv = np.asarray(uv, np.float64)
    before_hashes = immutable_buffer_hashes(Path(args.mesh))

    rgba = cv2.imread(str(args.source_image), cv2.IMREAD_UNCHANGED)
    if rgba.shape[2] == 3:
        rgba = cv2.cvtColor(rgba, cv2.COLOR_BGR2BGRA)
    source_rgb = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
    source_alpha = rgba[..., 3].astype(np.float32) / 255.0
    source_mask = source_alpha > 0.5
    height, width = source_mask.shape
    timings["load"] = time.time() - started

    # ---- camera ---------------------------------------------------------------------
    started = time.time()
    camera = fit_camera(positions, triangles, source_mask)
    timings["camera_fit"] = time.time() - started
    print(f"PROJECTION_CAMERA yaw={camera['yaw']} pitch={camera['pitch']} "
          f"roll={camera['roll']} iou={camera['silhouette_iou']}", flush=True)

    centre = np.asarray(camera["centre"])
    matrix = camera["matrix"]
    screen, depth = project(positions - centre, matrix, camera["scale"],
                            np.asarray(camera["offset"]))

    # ---- depth buffer ---------------------------------------------------------------
    started = time.time()
    zbuffer = depth_buffer(screen.astype(np.float32), depth.astype(np.float32),
                           triangles, height, width)
    timings["depth_buffer"] = time.time() - started

    # ---- atlas ownership ------------------------------------------------------------
    started = time.time()
    size = args.atlas_size
    owner, weights = rasterise_atlas(uv, triangles, size)
    timings["atlas_raster"] = time.time() - started
    owned = owner >= 0
    print(f"PROJECTION_ATLAS owned_texels={int(owned.sum())} "
          f"({owned.mean() * 100:.2f}% of atlas)", flush=True)

    # ---- per-texel sampling ---------------------------------------------------------
    started = time.time()
    ys, xs = np.nonzero(owned)
    tri = owner[ys, xs]
    wa = weights[ys, xs, 0].astype(np.float64)
    wb = weights[ys, xs, 1].astype(np.float64)
    wc = 1.0 - wa - wb

    corners = positions[triangles[tri]]
    world = (corners[:, 0] * wc[:, None] + corners[:, 1] * wa[:, None]
             + corners[:, 2] * wb[:, None])
    face_normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    face_normal /= np.maximum(np.linalg.norm(face_normal, axis=1, keepdims=True), 1e-12)

    rotated = (world - centre) @ matrix.T
    sx = rotated[:, 0] * camera["scale"] + camera["offset"][0]
    sy = rotated[:, 1] * camera["scale"] + camera["offset"][1]
    sz = rotated[:, 2]

    view_normal = face_normal @ matrix.T
    facing = -view_normal[:, 2]

    in_bounds = (sx >= 0) & (sx < width - 1) & (sy >= 0) & (sy < height - 1)
    ix = np.clip(sx.astype(np.int32), 0, width - 1)
    iy = np.clip(sy.astype(np.int32), 0, height - 1)

    matte_ok = source_mask[iy, ix]
    front_ok = facing > FRONT_FACING_MIN
    extent = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    tolerance = extent * DEPTH_TOLERANCE_SCALE
    nearest = zbuffer[iy, ix]
    depth_ok = np.isfinite(nearest) & (sz <= nearest + tolerance)

    accepted = in_bounds & matte_ok & front_ok & depth_ok
    rejected_background = in_bounds & front_ok & depth_ok & ~matte_ok

    # bilinear sample, only where accepted
    fx = np.clip(sx, 0, width - 1.001)
    fy = np.clip(sy, 0, height - 1.001)
    x0 = fx.astype(np.int32); y0 = fy.astype(np.int32)
    tx = (fx - x0)[:, None]; ty = (fy - y0)[:, None]
    x1 = np.minimum(x0 + 1, width - 1); y1 = np.minimum(y0 + 1, height - 1)
    colour = ((source_rgb[y0, x0] * (1 - tx) + source_rgb[y0, x1] * tx) * (1 - ty)
              + (source_rgb[y1, x0] * (1 - tx) + source_rgb[y1, x1] * tx) * ty)
    timings["sampling"] = time.time() - started

    atlas = np.zeros((size, size, 3), np.float32)
    observed = np.zeros((size, size), bool)
    rejected = np.zeros((size, size), bool)
    confidence = np.zeros((size, size), np.float32)
    depth_map = np.zeros((size, size), np.float32)

    atlas[ys[accepted], xs[accepted]] = colour[accepted]
    observed[ys[accepted], xs[accepted]] = True
    rejected[ys[rejected_background], xs[rejected_background]] = True
    confidence[ys[accepted], xs[accepted]] = np.clip(facing[accepted], 0.0, 1.0)
    margin = np.clip(1.0 - np.abs(sz - nearest) / max(tolerance, 1e-9), 0.0, 1.0)
    depth_map[ys[in_bounds], xs[in_bounds]] = margin[in_bounds].astype(np.float32)

    # ---- fill unobserved owned texels from nearest observed colour -------------------
    started = time.time()
    synthesized = owned & ~observed
    if observed.any() and synthesized.any():
        # Distance transform on the complement gives, for every texel, the nearest observed texel.
        _dist, labels = cv2.distanceTransformWithLabels(
            (~observed).astype(np.uint8), cv2.DIST_L2, 3,
            labelType=cv2.DIST_LABEL_PIXEL)
        oy, ox = np.nonzero(observed)
        lookup = np.zeros(labels.max() + 1, np.int64)
        lookup[labels[oy, ox]] = np.arange(oy.size)
        picked = lookup[labels[synthesized]]
        atlas[synthesized] = atlas[oy[picked], ox[picked]]
        confidence[synthesized] = 0.0
    # bleed a little past chart edges so bilinear filtering never reaches the gutter
    if args.padding_px > 0 and owned.any():
        kernel = np.ones((2 * args.padding_px + 1,) * 2, np.uint8)
        grown = cv2.dilate(owned.astype(np.uint8), kernel) > 0
        gutter = grown & ~owned
        if gutter.any() and (observed.any() or synthesized.any()):
            filled = owned
            _d2, labels2 = cv2.distanceTransformWithLabels(
                (~filled).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            fy_, fx_ = np.nonzero(filled)
            lookup2 = np.zeros(labels2.max() + 1, np.int64)
            lookup2[labels2[fy_, fx_]] = np.arange(fy_.size)
            picked2 = lookup2[labels2[gutter]]
            atlas[gutter] = atlas[fy_[picked2], fx_[picked2]]
    timings["fill"] = time.time() - started

    # ---- write ----------------------------------------------------------------------
    started = time.time()
    atlas_u8 = np.clip(atlas, 0, 255).astype(np.uint8)
    atlas_path = root / "basecolor.png"
    cv2.imwrite(str(atlas_path), cv2.cvtColor(atlas_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(root / "observed_mask.png"), observed.astype(np.uint8) * 255)
    cv2.imwrite(str(root / "synthesized_mask.png"), synthesized.astype(np.uint8) * 255)
    cv2.imwrite(str(root / "rejected_background_mask.png"), rejected.astype(np.uint8) * 255)
    cv2.imwrite(str(root / "confidence.png"), (confidence * 255).astype(np.uint8))
    cv2.imwrite(str(root / "depth_consistency.png"), (depth_map * 255).astype(np.uint8))

    textured = root / "textured.glb"
    bound = bind_texture(Path(args.mesh), textured, atlas_path.read_bytes())
    after_hashes = immutable_buffer_hashes(textured)
    timings["write_and_bind"] = time.time() - started
    timings["total"] = time.time() - started_all

    receipt = {
        "schema": "fast_texture_projection_v1",
        "backend": "numpy_vectorised_rasteriser",
        "blender_used": False,
        "scene_ray_cast_calls": 0,
        "bpy_pixel_accesses": 0,
        "mesh": str(args.mesh),
        "mesh_sha256": sha256(Path(args.mesh)),
        "source_image": str(args.source_image),
        "source_image_sha256": sha256(Path(args.source_image)),
        "alpha_mask_sha256": sha256(Path(args.alpha_mask)),
        "input_contract": contract,
        "atlas_size": size,
        "padding_px": args.padding_px,
        "camera": {k: v for k, v in camera.items() if k != "matrix"},
        "triangles": int(len(triangles)),
        "vertices": int(len(positions)),
        "atlas": {
            "owned_texels": int(owned.sum()),
            "owned_fraction": round(float(owned.mean()), 6),
            "observed_texels": int(observed.sum()),
            "observed_fraction_of_owned": round(float(observed.sum()) / max(int(owned.sum()), 1), 6),
            "synthesized_texels": int(synthesized.sum()),
            "rejected_background_texels": int(rejected.sum()),
        },
        "gates": {
            "in_bounds": int(in_bounds.sum()),
            "matte_foreground": int((in_bounds & matte_ok).sum()),
            "front_facing": int((in_bounds & front_ok).sum()),
            "depth_consistent": int((in_bounds & depth_ok).sum()),
            "accepted": int(accepted.sum()),
            "rejected_as_background": int(rejected_background.sum()),
            "depth_tolerance": round(float(tolerance), 8),
            "front_facing_min": FRONT_FACING_MIN,
        },
        "artifacts": {
            "basecolor": str(atlas_path),
            "basecolor_sha256": sha256(atlas_path),
            "observed_mask": str(root / "observed_mask.png"),
            "synthesized_mask": str(root / "synthesized_mask.png"),
            "rejected_background_mask": str(root / "rejected_background_mask.png"),
            "confidence": str(root / "confidence.png"),
            "depth_consistency": str(root / "depth_consistency.png"),
            "textured_glb": str(textured),
            "textured_glb_sha256": sha256(textured),
        },
        "materials_bound": bound,
        "immutable_buffers_before": before_hashes,
        "immutable_buffers_after": after_hashes,
        "geometry_uv_index_preserved": before_hashes["geometry_uv_index_sha256"] == after_hashes["geometry_uv_index_sha256"],
        "timings_seconds": {k: round(v, 3) for k, v in timings.items()},
    }
    (root / "projection_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                  encoding="utf-8")
    print(f"FAST_PROJECTION_DONE total={timings['total']:.1f}s "
          f"observed={receipt['atlas']['observed_texels']} "
          f"geometry_preserved={receipt['geometry_uv_index_preserved']} {textured}", flush=True)


if __name__ == "__main__":
    main()
