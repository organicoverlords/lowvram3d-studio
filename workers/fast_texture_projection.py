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
def push_pull_fill(colour: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Spread observed colour into unobserved texels smoothly, not in cells.

    The fill this replaces gave every unobserved texel the colour of its nearest
    observed one. Texels sharing a nearest neighbour form a Voronoi cell, so the
    result is a mosaic of hard-edged flat plates -- and since a single view of the
    castle observed 9.9% of the atlas while 48.1% needed filling, the plates
    covered most of the asset. Measured on that atlas: 15.8% of texels sat on a
    hard edge.

    That is the same shape of defect as the nearest-neighbour texture lookup in
    render_textured_views.py, one stage earlier: a nearest-neighbour operation
    manufacturing edges that are not in the data. The difference is that this one
    is baked into the atlas, so a correct renderer shows it faithfully.

    Push-pull instead. Pull: repeatedly halve the premultiplied colour and its
    weight, so coarse levels carry an average of whatever was observed nearby.
    Push: walk back up, and wherever a level is short of weight, take the
    upsampled coarser estimate for the remainder. Colour therefore diffuses
    outward from observed texels at a rate set by distance, with no cell
    boundaries anywhere. Cost is O(texels) and runs in well under a second at
    2048.

    `colour` must be premultiplied by `weight`; `weight` is 1 where observed.
    """
    colours, weights = [colour], [weight]
    while min(colours[-1].shape[:2]) > 2:
        colours.append(cv2.pyrDown(colours[-1]))
        weights.append(cv2.pyrDown(weights[-1]))
    for level in range(len(colours) - 1, 0, -1):
        height, width = colours[level - 1].shape[:2]
        coarse_c = cv2.pyrUp(colours[level], dstsize=(width, height))
        coarse_w = cv2.pyrUp(weights[level], dstsize=(width, height))
        # How much of this texel is still unaccounted for. Saturating at 1 keeps
        # an already-covered texel from being pulled back toward the blur.
        short = 1.0 - np.clip(weights[level - 1], 0.0, 1.0)
        colours[level - 1] = colours[level - 1] + coarse_c * short[..., None]
        weights[level - 1] = weights[level - 1] + coarse_w * short
    return colours[0] / np.maximum(weights[0], 1e-6)[..., None]


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
#: Model space has +Y up. Image space has row indices increasing DOWNWARD. Every
#: consumer of `rotation()` maps the rotated (x, y) straight onto (col, row), so
#: without this the silhouette is rasterised upside down and the camera search
#: scores an inverted mesh against an upright matte.
#:
#: It did not fail loudly, which is why it survived. The search still returns a
#: pose -- the least-bad tilt that hides the inversion -- so the symptom is a flat
#: score landscape rather than an error. Measured on the Mini Turbo shaman, whose
#: silhouette is strongly yaw-dependent:
#:
#:     as shipped   best IoU 0.585 at yaw 225   (facing away, and 0.475-0.585
#:                                               across the whole 360 sweep)
#:     Y negated    best IoU 0.748 at yaw 330   (near front, as it should be)
#:
#: Applied here rather than at the three call sites so the search, the vertex
#: projection and the texel sampling cannot disagree. Only the y row changes, so
#: `view_normal[:, 2]` and therefore the front-facing gate are untouched.
IMAGE_SPACE = np.diag([1.0, -1.0, 1.0])


def rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    cp, sp = np.cos(np.radians(pitch)), np.sin(np.radians(pitch))
    cr, sr = np.cos(np.radians(roll)), np.sin(np.radians(roll))
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return IMAGE_SPACE @ rz @ rx @ ry


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


def canonical_orientation(matrix: np.ndarray) -> dict:
    """Where the painted face ends up, and the spin that brings it to +Z.

    A generator has no notion of "front": Mini Turbo returned this shaman facing
    152 degrees away from +Z, so a viewer's front camera showed its unpainted
    back and the good texture sat on the "back" thumbnail. Nothing is wrong with
    the asset -- only with which way it happens to point.

    Silhouette IoU cannot settle it. A figure's front and back silhouettes are
    near mirrors, and on this shaman they scored 0.689 against 0.560 -- close
    enough that the wrong one wins on a different subject.

    The camera that painted the atlas already knows. A texel was accepted when
    `-(n @ matrix.T)[2] > 0`, so the painted hemisphere is centred on `-matrix[2]`
    in object space. That is a fact about where the paint went, not a guess about
    shape, and it is the only orientation signal in the pipeline that cannot be
    fooled by a symmetric outline.

    Reported rather than applied: this worker guarantees `geometry_preserved`,
    so rotating here would break its own contract.
    """
    front = -np.asarray(matrix)[2]
    front = front / max(float(np.linalg.norm(front)), 1e-12)
    # Yaw about +Y carrying `front` onto +Z, in the same convention as the ry
    # block of `rotation()`: x' = x cos a + z sin a, z' = -x sin a + z cos a.
    degrees = float(-np.degrees(np.arctan2(front[0], front[2])))
    return {
        "painted_face_direction": [round(float(v), 4) for v in front],
        "rotate_about_y_degrees": round(degrees, 2),
        "note": ("apply this Y rotation to put the painted face at +Z; "
                 "derived from the projection camera, not from the silhouette"),
    }


def project(vertices: np.ndarray, matrix: np.ndarray, scale: float,
            offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rotated = vertices @ matrix.T
    screen = rotated[:, :2] * scale + offset
    return screen, rotated[:, 2]


#: Below this the two candidates are genuinely indistinguishable to the feature
#: model and the tie is NOT broken -- the fitted yaw is kept and the receipt says
#: so, rather than a coin flip being dressed up as a decision. Measured
#: separation on the assets whose answer is known independently: red panda 0.194
#: (wrong yaw rejected), sky whale see receipt. A threshold this far below those
#: only fires on a real tie.
FORE_AFT_MIN_SEPARATION = 0.05

#: Source of the tie-break. DINOv2 is self-supervised and its features are
#: dominated by shape and part layout rather than colour, which is the invariance
#: needed to match a painted illustration to a grey clay render. The weights are
#: in the local HF cache and this runs on CPU.
FORE_AFT_MODEL = "facebook/dinov2-large"


def _break_fore_aft_tie(centred: np.ndarray, triangles: np.ndarray,
                        mask: np.ndarray, yaw: float, pitch: float,
                        roll: float, source_rgb: np.ndarray | None = None) -> float:
    """Return whichever of `yaw` and `yaw + 180` actually faces the source.

    Renders the bare geometry from both, embeds both plus the source with
    DINOv2, and keeps the higher cosine similarity. Touches no atlas, no
    projection and no silhouette, so it cannot inherit the error it exists to
    catch. If anything is unavailable or the two candidates are within
    FORE_AFT_MIN_SEPARATION, the fitted yaw is returned unchanged: this is a
    tie-breaker, not a second opinion that overrides a clear fit.
    """
    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModel
    except Exception:  # pragma: no cover - environment without transformers
        return yaw

    size = 384
    # Every triangle. Decimating here produced a speckled render full of holes --
    # at 400k triangles the stride was 3, so two thirds of the surface was
    # missing -- and DINOv2 scored the two candidates 0.223 and 0.232, a
    # separation of 0.009 that would have declined to break any tie. The same
    # comparison on a complete render separates them by 0.157. Two renders cost
    # about a minute; a tie-breaker that cannot see the geometry costs an asset.
    faces = triangles

    def clay(candidate: float) -> "Image.Image":
        matrix = rotation(candidate, pitch, roll)
        rotated = centred @ matrix.T
        span = float(np.abs(rotated[:, :2]).max()) or 1.0
        scale = (size * 0.44) / span
        # No Y flip here. `rotation()` already returns an image-space matrix --
        # that is what IMAGE_SPACE is for -- so the projector's own screen path
        # does `rotated[:, :2] * scale + offset` and nothing more. Adding a flip
        # would render the clay upside down and hand DINOv2 two inverted
        # candidates, which is the same class of bug as the fit_to_mask Y
        # negation that this file was already fixed for once.
        xy = rotated[:, :2] * scale + size * 0.5
        image = np.ones((size, size), np.float64)
        zbuffer = np.full((size, size), np.inf)
        tri = xy[faces]
        depth = rotated[faces][:, :, 2]
        edge1 = rotated[faces[:, 1]] - rotated[faces[:, 0]]
        edge2 = rotated[faces[:, 2]] - rotated[faces[:, 0]]
        face_normals = np.cross(edge1, edge2)
        lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
        face_normals /= np.maximum(lengths, 1e-12)
        # Dark enough to separate from the white field. The obvious ramp,
        # 0.65 * |n| + 0.35, sends camera-facing surfaces to 1.0 -- exactly the
        # background -- so the subject dissolves into it and only the grazing rim
        # survives. The background is white because the source is composited onto
        # white; the object must not be.
        shade = np.abs(face_normals[:, 2]) * 0.55 + 0.18
        for index in np.argsort(-depth.mean(axis=1)):
            t = tri[index]
            x0, y0 = np.floor(t.min(axis=0)).astype(int)
            x1, y1 = np.ceil(t.max(axis=0)).astype(int) + 1
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, size), min(y1, size)
            if x1 <= x0 or y1 <= y0:
                continue
            ys, xs = np.mgrid[y0:y1, x0:x1]
            px, py = xs + 0.5, ys + 0.5
            ax, ay = t[0]
            bx, by = t[1]
            cx, cy = t[2]
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if abs(area) < 1e-12:
                continue
            w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
            w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
            inside = (w0 >= 0) & (w1 >= 0) & ((1.0 - w0 - w1) >= 0)
            if not inside.any():
                continue
            z = w0 * depth[index, 0] + w1 * depth[index, 1] + (1.0 - w0 - w1) * depth[index, 2]
            window = zbuffer[y0:y1, x0:x1]
            write = inside & (z < window)
            if not write.any():
                continue
            window[write] = z[write]
            image[y0:y1, x0:x1][write] = shade[index]
        return Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8)).convert("RGB")

    try:
        processor = AutoImageProcessor.from_pretrained(FORE_AFT_MODEL)
        model = AutoModel.from_pretrained(FORE_AFT_MODEL).eval()
        # The SOURCE IMAGE, not its silhouette. A binary mask throws away the
        # only evidence that separates a front from a back -- a silhouette is
        # what the fit already used, and reusing it here would rebuild the tie
        # rather than break it. Composited onto white because the clay renders
        # are on white, so the background cannot be what gets matched.
        if source_rgb is None:
            return yaw
        rgb = np.clip(source_rgb, 0, 255).astype(np.uint8)
        flat = np.where(mask[..., None], rgb, 255).astype(np.uint8)
        source = Image.fromarray(flat).convert("RGB")
        batch = [source, clay(yaw), clay(yaw + 180.0)]
        inputs = processor(images=batch, return_tensors="pt")
        with torch.no_grad():
            vectors = model(**inputs).last_hidden_state[:, 0].cpu().numpy()
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        kept, flipped = float(vectors[1] @ vectors[0]), float(vectors[2] @ vectors[0])
    except Exception:  # pragma: no cover - model unavailable at runtime
        return yaw

    if abs(kept - flipped) < FORE_AFT_MIN_SEPARATION:
        return yaw
    return yaw if kept >= flipped else (yaw + 180.0) % 360.0


def fit_camera(vertices: np.ndarray, triangles: np.ndarray, mask: np.ndarray,
               coarse_step: float = 15.0, source_rgb: np.ndarray | None = None) -> dict:
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

    def refine(seed, step):
        """Local silhouette refinement around `seed`, returned as (iou, y, p, r)."""
        best_local = (score(*seed), *seed)
        for _pass in range(2):
            _iou, y, p, r = best_local
            for dy in (-step, 0.0, step):
                for dp in (-step, 0.0, step):
                    for dr in (-step, 0.0, step):
                        value = score(y + dy, p + dp, r + dr)
                        if value > best_local[0]:
                            best_local = (value, y + dy, p + dp, r + dr)
            step /= 3.0
        return best_local

    best = (-1.0, 0.0, 0.0, 0.0)
    for yaw in np.arange(0.0, 360.0, coarse_step):
        for pitch in (-30.0, -15.0, 0.0, 15.0, 30.0):
            for roll in (-15.0, 0.0, 15.0):
                value = score(yaw, pitch, roll)
                if value > best[0]:
                    best = (value, yaw, pitch, roll)
    best = refine(best[1:], coarse_step / 3.0)
    _iou, yaw, pitch, roll = best

    # The silhouette objective cannot tell front from back, and its confidence is
    # not a warning. An orthographic silhouette from direction u and from -u is
    # the same set for a convex body, and a hooded character in a ghillie suit is
    # very nearly convex: the muzzle protrudes too little to break the tie. So
    # the search sees two near-equal optima and initialisation picks one.
    #
    # The red panda lost that coin flip at IoU 0.915 -- a number that reads as
    # certainty and means nothing -- and its photograph was projected onto the
    # back of its head. Every check downstream then agreed, including the
    # paint-coverage audit, which reports where paint landed and so confirms a
    # backwards projection instead of catching it.
    #
    # Breaking the tie needs evidence the silhouette does not contain, and the
    # only such evidence is shape: a face protrudes, a hood does not. So compare
    # the SOURCE IMAGE against a shaded render of the bare geometry at yaw and at
    # yaw+180, and keep the better. Both candidates are expressed in this
    # module's own rotation convention, so nothing has to be translated between
    # frames -- which is itself a class of bug this file has already had.
    chosen = _break_fore_aft_tie(centred, triangles, mask, yaw, pitch, roll,
                                 source_rgb=source_rgb)
    flipped = abs(((chosen - yaw) + 180.0) % 360.0 - 180.0) > 1.0
    if flipped:
        # Yaw is not the only thing that reverses. A tilt of +13.3 degrees seen
        # from the front is -13.3 seen from behind, and the same goes for roll,
        # so carrying the fitted pitch across the flip tips the model the wrong
        # way and the photograph lands on the right hemisphere but misregistered
        # -- which looked, on the panda's first corrected run, like a smeared
        # double exposure rather than a face.
        #
        # So DINOv2 chooses the hemisphere and the silhouette refines inside it.
        # That division is the point: the silhouette objective is degenerate
        # ONLY across the fore-aft flip. Within one hemisphere it is exactly the
        # right tool, and it is far cheaper and more precise than the feature
        # model for the last few degrees.
        # The EXACT antipode of the fitted pose, derived rather than searched.
        #
        #   rotation(y, p, r) = IMAGE_SPACE @ rz(r) @ rx(p) @ ry(y)
        #
        # Reversing the view direction while keeping up is left-multiplication by
        # ry(180) = diag(-1, 1, -1). That is diagonal, so it commutes with
        # IMAGE_SPACE; conjugating it through rz and rx flips their signs, since
        # ry(180) negates both the x and z axes. What comes out is
        #
        #   IMAGE_SPACE @ rz(-r) @ rx(-p) @ ry(y + 180)
        #
        # so the antipode is exactly (yaw + 180, -pitch, -roll).
        #
        # Re-searching instead of deriving was a mistake worth recording. In the
        # CORRECT hemisphere the silhouette optimum is broad and shallow -- 0.78
        # to 0.84 against a sharp 0.915 on the back -- so refinement wandered to a
        # different pose on every run: 151.7, 183.3, 202.5, and once to 115 when
        # the window was wide. A flat objective does not become trustworthy
        # because it is being optimised locally. The fitted pose is
        # well-determined; its antipode is therefore well-determined too, and is
        # the answer.
        yaw, pitch, roll = (chosen, -pitch, -roll)
        best = (score(yaw, pitch, roll), yaw, pitch, roll)
        _iou = best[0]

    matrix = rotation(yaw, pitch, roll)
    rotated = centred @ matrix.T
    scale, offset = fit_to_mask(rotated, mask)
    return {"yaw": round(yaw, 3), "pitch": round(pitch, 3), "roll": round(roll, 3),
            "silhouette_iou": round(best[0], 4), "scale": float(scale),
            "offset": [float(v) for v in offset], "centre": [float(v) for v in centre],
            "matrix": matrix, "image_size": [int(width), int(height)],
            "fore_aft_tie_broken_by": "dinov2_geometry_vs_source",
            "fore_aft_flipped": bool(flipped)}


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
    parser.add_argument("--fill", choices=("pushpull", "nearest"), default="pushpull",
                        help="how unobserved owned texels are filled. nearest is the "
                             "original and lays down Voronoi plates wherever coverage "
                             "is low, which on the castle was most of the surface")
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
    camera = fit_camera(positions, triangles, source_mask, source_rgb=source_rgb)
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

    # ---- fill unobserved owned texels ------------------------------------------------
    # (push_pull_fill is defined at module scope; see its docstring for why the
    #  nearest-neighbour fill it replaces produced plates.)
    started = time.time()
    synthesized = owned & ~observed
    if observed.any() and synthesized.any():
        if args.fill == "nearest":
            # Every unobserved texel copies its nearest observed texel. The regions
            # that share a nearest neighbour are Voronoi cells, so the fill comes out
            # as hard-edged flat plates -- and on the castle only 9.9% of the atlas
            # was observed against 48.1% synthesized, which put plates over most of
            # the asset. Kept only so the two fills can be compared on one mesh.
            _dist, labels = cv2.distanceTransformWithLabels(
                (~observed).astype(np.uint8), cv2.DIST_L2, 3,
                labelType=cv2.DIST_LABEL_PIXEL)
            oy, ox = np.nonzero(observed)
            lookup = np.zeros(labels.max() + 1, np.int64)
            lookup[labels[oy, ox]] = np.arange(oy.size)
            picked = lookup[labels[synthesized]]
            atlas[synthesized] = atlas[oy[picked], ox[picked]]
        else:
            atlas[synthesized] = push_pull_fill(
                atlas.astype(np.float32) * observed[..., None],
                observed.astype(np.float32))[synthesized]
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
        "fill": args.fill,
        "camera": {k: v for k, v in camera.items() if k != "matrix"},
        "canonical_orientation": canonical_orientation(camera["matrix"]),
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
