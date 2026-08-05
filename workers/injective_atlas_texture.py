"""Project the six MV-Adapter views onto an injective atlas, with real per-triangle provenance.

This is the same observation model and the same fusion policy as the accepted
FRONT_PROTECTED_MULTIBAND baseline. What changes is the atlas underneath it. On the old layout
a texel was claimed by ~3 triangles, so the single owner's colour was displayed by every other
claimant - the front face appeared on the back of the head not because fusion chose wrongly but
because the atlas could not express the difference. On an injective layout each triangle owns
its own texels, so a view decision is a statement about the surface it was computed for.

Two things therefore do not need to exist here, and deliberately do not:

  * the negative-evidence triangle mask that suppressed the atlas on rear geometry, which was a
    workaround for shells sharing UVs;
  * the 2D push-pull fill, which propagates colour across the atlas plane and so can carry a
    facial texel into whatever chart happens to sit next to it.

Unobserved surface is instead filled from 3D donors: nearest observed texels in world space,
restricted to the same connected component, within a distance bound, and with a compatible
normal. Only the low-frequency band is donated. Detail is never synthesised and never travels,
which is what keeps eyes and a muzzle from reappearing on a rear surface.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from atlas_raster import injectivity, rasterise
from build_mvadapter_cpu_controls import PROJECTION_SPAN
from fast_texture_projection import bind_texture
from mesh_io import read_glb, triangle_components
from multiview_texture_fusion import box_blur, local_detail
from multiview_texture_projection import (
    SEMANTIC_RELIABILITY,
    apply_registration,
    bilinear,
    distance_from_boundary,
    file_prefix,
    foreground_mask,
    register,
    semantic_of,
)
from lowvram3d.texture_provenance import (
    EvidenceState, FrequencyAuthority, Lineage, SourceClass,
    create_empty_atlas_provenance, save_npz,
)

#: Settings of the accepted FRONT_PROTECTED_MULTIBAND baseline, carried over unchanged.
RATIO = 1.3
MARGIN = 0.1
GRAZING_COSINE = 0.4
DETAIL_RATIO = 1.6
PROTECTED_MIN_CONFIDENCE = 0.0
COLOUR_COMPATIBILITY = 60.0
#: Texels fused per block. Peak memory, not results, depends on this.
BLOCK = 1_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def apply_mask_registration(mask: np.ndarray, fit: dict) -> np.ndarray:
    """Apply the established square image registration to a boolean mask."""
    size = int(mask.shape[0])
    scaled = max(1, int(round(size * float(fit["scale"]))))
    resized = cv2.resize(mask.astype(np.uint8), (scaled, scaled),
                         interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x = (size - scaled) // 2 + int(fit["dx"])
    y = (size - scaled) // 2 + int(fit["dy"])
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(size, x + scaled), min(size, y + scaled)
    if x1 > x0 and y1 > y0:
        canvas[y0:y1, x0:x1] = resized[y0 - y:y1 - y, x0 - x:x1 - x]
    return canvas.astype(bool)


def invalid_white_source_mask(image: np.ndarray) -> np.ndarray:
    """Reject near-white, low-chroma source pixels as invalid evidence.

    This is a source-quality gate, not a material or anatomy rule.  It prevents the
    overexposed white patches present in the generated underside view from becoming
    occupied atlas evidence while leaving the authoritative original-front path intact.
    """
    values = np.asarray(image, np.float32)
    return (values.min(axis=2) >= 170.0) & ((values.max(axis=2) - values.min(axis=2)) <= 35.0)


def triangle_id_match_with_boundary(ids: np.ndarray, depth: np.ndarray,
                                    x: np.ndarray, y: np.ndarray,
                                    owner: np.ndarray, projected_depth: np.ndarray,
                                    tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    """Match exact IDs, or an owner ID in an adjacent pixel at the same depth.

    An adjacent hit is explicitly a raster-boundary ambiguity caused by sampling a
    continuous atlas point into a finite control raster.  Interior mismatches remain
    rejected because they have no adjacent owner hit with compatible depth.
    """
    height, width = ids.shape
    xx = np.clip(x, 0, width - 1)
    yy = np.clip(y, 0, height - 1)
    exact = ids[yy, xx] == owner
    accepted = exact.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = np.clip(x + dx, 0, width - 1)
            ny = np.clip(y + dy, 0, height - 1)
            neighbour_depth = depth[ny, nx]
            accepted |= ((ids[ny, nx] == owner) & np.isfinite(neighbour_depth)
                         & (np.abs(projected_depth - neighbour_depth) <= tolerance))
    return accepted, exact


def observe(mesh: Path, bundle: Path, receipt: dict, atlas_size: int, depth_tolerance: float,
            min_facing: float, detail_radius: int, direct_only: bool = False,
            original_front: Path | None = None,
            original_front_transform: Path | None = None,
            original_front_camera: Path | None = None) -> dict:
    """Rasterise the atlas once, then gate every view against it."""
    contract = json.loads((bundle / "camera_contract.json").read_text(encoding="utf-8"))
    if "control_space_transform" not in contract:
        raise RuntimeError("TEXTURE_CONTRACT_HAS_NO_BASIS")
    transform = np.asarray(contract["control_space_transform"], np.float64)
    generated = {item["name"]: Path(item["path"]) for item in receipt["output_images"]}

    positions, normals, uv, tris = read_glb(mesh)
    positions = np.asarray(positions, np.float64)
    uv = np.asarray(uv, np.float64)
    tris = np.asarray(tris, np.int64)
    if uv is None or not len(uv):
        raise RuntimeError("TEXTURE_UV_MISSING")

    gate = injectivity(uv, tris, atlas_size)
    if not gate["injective"]:
        raise RuntimeError(f"TEXTURE_ATLAS_NOT_INJECTIVE:{gate['interior_texels_claimed_twice']}")

    owner, weights = rasterise(uv, tris, atlas_size)
    owned = owner >= 0
    if not owned.any():
        raise RuntimeError("TEXTURE_ATLAS_EMPTY")

    canonical = positions @ transform.T
    vertices = canonical * (0.5 / float(np.max(np.abs(canonical))))
    vertex_normals = np.asarray(normals, np.float64) @ transform.T
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-12)

    owner_flat = owner[owned]
    corners = tris[owner_flat]
    wa = weights[owned][:, 0][:, None]
    wb = weights[owned][:, 1][:, None]
    wc = 1.0 - wa - wb
    texel_position = (vertices[corners[:, 0]] * wc + vertices[corners[:, 1]] * wa
                      + vertices[corners[:, 2]] * wb).astype(np.float32)
    texel_normal = (vertex_normals[corners[:, 0]] * wc + vertex_normals[corners[:, 1]] * wa
                    + vertex_normals[corners[:, 2]] * wb)
    texel_normal /= np.maximum(np.linalg.norm(texel_normal, axis=1, keepdims=True), 1e-12)
    texel_normal = texel_normal.astype(np.float32)

    components, _ = triangle_components(positions, tris)
    texel_component = components[owner_flat].astype(np.int32)

    views = sorted(contract["views"], key=lambda item: int(item["index"]))
    count = int(owned.sum())
    low = np.zeros((len(views), count, 3), np.float16)
    high = np.zeros((len(views), count, 3), np.float16)
    confidence = np.zeros((len(views), count), np.float32)
    facing_stack = np.zeros((len(views), count), np.float16)
    detail_stack = np.zeros((len(views), count), np.float16)
    depth_stack = np.zeros((len(views), count), np.float16)
    source_pixel_stack = np.zeros((len(views), count, 2), np.float32)
    visibility_stack = np.zeros((len(views), count), bool)
    face_id_match_stack = np.zeros((len(views), count), bool)
    source_mask_stack = np.zeros((len(views), count), bool)
    valid_stack = np.zeros((len(views), count), bool)
    semantics, diagnostics = [], []
    view_directions = np.zeros((len(views), 3), np.float32)
    view_rights = np.zeros((len(views), 3), np.float32)
    view_ups = np.zeros((len(views), 3), np.float32)
    front_screen = None
    render = 0

    for slot, view in enumerate(views):
        semantic = semantic_of(view)
        prefix = file_prefix(view)
        image_path = generated.get(f"view_{int(view['index'])}_{semantic}.png")
        if image_path is None or not Path(image_path).is_file():
            raise RuntimeError(f"TEXTURE_VIEW_MISSING:{view['index']}:{semantic}")
        raw = np.asarray(Image.open(image_path).convert("RGB"))
        control_mask = np.asarray(Image.open(bundle / f"{prefix}_mask.png").convert("L")) > 127
        control_depth = np.load(bundle / f"{prefix}_depth.npy")
        triangle_id_path = bundle / f"{prefix}_triangle_ids.npy"
        if direct_only and not triangle_id_path.is_file():
            raise RuntimeError(f"FACE_ID_BUFFER_MISSING:{triangle_id_path.name}")
        control_triangle_ids = np.load(triangle_id_path) if triangle_id_path.is_file() else None
        if control_triangle_ids is not None and control_triangle_ids.shape != control_mask.shape:
            raise RuntimeError(f"FACE_ID_DIMENSION_MISMATCH:{prefix}")
        render = raw.shape[0]
        control_render = control_mask.shape[0]

        registration_mask = cv2.resize(control_mask.astype(np.uint8), (render, render),
                                        interpolation=cv2.INTER_NEAREST) > 0
        fit = register(foreground_mask(raw), registration_mask)
        aligned = apply_registration(raw, fit).astype(np.float32)
        aligned_source_mask = apply_mask_registration(foreground_mask(raw), fit)
        aligned_white_invalid = invalid_white_source_mask(aligned)
        aligned_low = box_blur(aligned, detail_radius)
        aligned_high = aligned - aligned_low
        aligned_detail = local_detail(aligned, detail_radius)

        direction = np.asarray(view["camera_direction"], np.float64)
        right = np.asarray(view["camera_right"], np.float64)
        up = np.asarray(view["camera_up"], np.float64)
        view_directions[slot] = direction
        view_rights[slot] = right
        view_ups[slot] = up
        screen = np.stack([texel_position @ right / PROJECTION_SPAN + 0.5,
                           0.5 - (texel_position @ up) / PROJECTION_SPAN], axis=1)
        pixel = screen * float(render - 1)
        control_pixel = screen * float(control_render - 1)
        xs = np.rint(pixel[:, 0]).astype(np.int64)
        ys = np.rint(pixel[:, 1]).astype(np.int64)
        cxs = np.rint(control_pixel[:, 0]).astype(np.int64)
        cys = np.rint(control_pixel[:, 1]).astype(np.int64)
        in_bounds = (xs >= 0) & (xs < render) & (ys >= 0) & (ys < render)
        control_in_bounds = ((cxs >= 0) & (cxs < control_render)
                             & (cys >= 0) & (cys < control_render))
        cx = np.clip(xs, 0, render - 1)
        cy = np.clip(ys, 0, render - 1)
        ccx = np.clip(cxs, 0, control_render - 1)
        ccy = np.clip(cys, 0, control_render - 1)
        control_mask_valid = control_mask[ccy, ccx] & in_bounds & control_in_bounds
        source_mask_valid = aligned_source_mask[cy, cx]
        white_source_invalid = aligned_white_invalid[cy, cx]
        in_mask = control_mask_valid & source_mask_valid & ~white_source_invalid
        exact_face_id_match = np.ones_like(in_bounds, dtype=bool)
        face_id_match = np.ones_like(in_bounds, dtype=bool)
        if control_triangle_ids is not None:
            # Generated views remain strict: only exact triangle IDs are evidence.
            face_id_match = control_triangle_ids[ccy, ccx] == corners[:, 0] * 0 + owner_flat
            exact_face_id_match = face_id_match.copy()
        buffered = control_depth[ccy, ccx]
        depth_delta = np.abs(texel_position @ direction - buffered)
        unoccluded = np.isfinite(buffered) & (depth_delta <= depth_tolerance)
        facing = -(texel_normal @ direction)
        front_facing = facing > min_facing
        # Exact buffers are rebuilt from the final injective mesh for production.
        # Triangle identity is therefore an enforced visibility gate, not merely
        # a diagnostic comparison.
        valid = in_bounds & in_mask & unoccluded & front_facing & face_id_match
        source_pixel_stack[slot] = pixel.astype(np.float32)
        visibility_stack[slot] = in_bounds & unoccluded & front_facing
        face_id_match_stack[slot] = face_id_match
        source_mask_stack[slot] = in_mask
        valid_stack[slot] = valid

        boundary = distance_from_boundary(control_mask)
        weight = (np.clip(facing, 0.0, 1.0) ** 3.0
                  * np.exp(-(depth_delta / max(depth_tolerance, 1e-9)) ** 2)
                  * (0.10 + 0.90 * boundary[ccy, ccx])
                  * SEMANTIC_RELIABILITY.get(semantic, 1.0))
        confidence[slot] = np.where(valid, weight, 0.0)
        facing_stack[slot] = np.where(valid, np.clip(facing, 0.0, 1.0), 0.0).astype(np.float16)
        depth_stack[slot] = np.minimum(depth_delta, 65000.0).astype(np.float16)
        low[slot] = bilinear(aligned_low, pixel[:, 0], pixel[:, 1]).astype(np.float16)
        high[slot] = bilinear(aligned_high, pixel[:, 0], pixel[:, 1]).astype(np.float16)
        detail_stack[slot] = bilinear(aligned_detail[..., None],
                                      pixel[:, 0], pixel[:, 1])[:, 0].astype(np.float16)
        if semantic == "front":
            front_screen = pixel.astype(np.float32)
        semantics.append(semantic)
        diagnostics.append({
            "raw_index": int(view["index"]), "semantic_label": semantic,
            "source_image": str(image_path), "registration": fit,
            "texels_rejected_out_of_bounds": int((~in_bounds).sum()),
            "texels_rejected_by_mask": int((in_bounds & ~in_mask).sum()),
            "texels_rejected_by_source_foreground": int(
                (in_bounds & control_mask_valid & ~source_mask_valid).sum()),
            "texels_rejected_white_invalid": int(
                (in_bounds & control_mask_valid & source_mask_valid & white_source_invalid).sum()),
            "texels_rejected_by_depth": int((in_bounds & in_mask & ~unoccluded).sum()),
            "texels_rejected_back_facing": int(
                (in_bounds & in_mask & unoccluded & ~front_facing).sum()),
            "texels_rejected_face_id": int(
                (in_bounds & in_mask & unoccluded & front_facing & ~face_id_match).sum()),
            "texels_accepted_with_face_id_mismatch": int(
                (in_bounds & in_mask & unoccluded & front_facing & face_id_match
                 & ~exact_face_id_match).sum()),
            "texels_valid": int(valid.sum()),
        })
        del screen, pixel, control_pixel, aligned, aligned_source_mask, aligned_white_invalid
        del aligned_low, aligned_high, aligned_detail

    original = None
    if original_front is not None:
        front_slot = next((i for i, name in enumerate(semantics) if name == "front"), None)
        if front_slot is None:
            raise RuntimeError("ORIGINAL_FRONT_REQUIRES_FRONT_VIEW")
        front_view = views[front_slot]
        front_prefix = file_prefix(front_view)
        front_mask = np.asarray(Image.open(bundle / f"{front_prefix}_mask.png").convert("L")) > 127
        front_control_render = front_mask.shape[0]
        front_depth = np.load(bundle / f"{front_prefix}_depth.npy")
        triangle_path = bundle / f"{front_prefix}_triangle_ids.npy"
        front_ids = np.load(triangle_path) if triangle_path.is_file() else None
        source_original = np.asarray(Image.open(original_front).convert("RGB"))
        source_mask_original = foreground_mask(source_original)
        # The authoritative PNG has a white canvas.  Use the conditioning canvas's neutral
        # matte instead of the white border median, otherwise the supposed cleanup preserves
        # the very colour that must not bleed into the atlas.
        source_background = np.array([127, 127, 127], dtype=np.uint8)
        # Remove the white source canvas before interpolation.  Warping the raw RGB image
        # first lets white background bleed through the foreground edge into valid texels.
        source_clean = source_original.copy()
        source_clean[~source_mask_original] = source_background
        transform_path = original_front_transform
        if transform_path:
            transform_record = json.loads(Path(transform_path).read_text(encoding="utf-8"))
            matrix = np.asarray(transform_record["source_matrix_high_to_conditioning"],
                                dtype=np.float64)
            source = cv2.warpAffine(source_clean, matrix, (render, render),
                                    flags=cv2.INTER_AREA,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=tuple(int(v) for v in source_background))
            source_mask = source_mask_original
            aligned_source_mask = cv2.warpAffine(
                source_mask.astype(np.uint8), matrix, (render, render),
                flags=cv2.INTER_NEAREST) > 0
            fit = None
            if original_front_camera is not None:
                camera_record = json.loads(Path(original_front_camera).read_text(encoding="utf-8"))
                fit = camera_record.get("registration")
            fit = fit or register(aligned_source_mask, front_mask)
        else:
            source = np.asarray(Image.fromarray(source_clean).resize(
                (render, render), Image.Resampling.LANCZOS))
            aligned_source_mask = cv2.resize(source_mask_original.astype(np.uint8),
                                              (render, render), interpolation=cv2.INTER_NEAREST) > 0
            fit = register(aligned_source_mask, front_mask)
        aligned = apply_registration(source, fit).astype(np.float32)
        aligned_source_mask = apply_mask_registration(aligned_source_mask, fit)
        aligned_low = box_blur(aligned, detail_radius)
        aligned_high = aligned - aligned_low
        right = np.asarray(front_view["camera_right"], np.float64)
        up = np.asarray(front_view["camera_up"], np.float64)
        direction = np.asarray(front_view["camera_direction"], np.float64)
        screen = np.stack([
            texel_position @ right / PROJECTION_SPAN + 0.5,
            0.5 - texel_position @ up / PROJECTION_SPAN,
        ], axis=1)
        pixel = screen * float(render - 1)
        control_pixel = screen * float(front_control_render - 1)
        xs = np.rint(pixel[:, 0]).astype(np.int64)
        ys = np.rint(pixel[:, 1]).astype(np.int64)
        cxs = np.rint(control_pixel[:, 0]).astype(np.int64)
        cys = np.rint(control_pixel[:, 1]).astype(np.int64)
        in_bounds = (xs >= 0) & (xs < render) & (ys >= 0) & (ys < render)
        control_in_bounds = ((cxs >= 0) & (cxs < front_control_render)
                             & (cys >= 0) & (cys < front_control_render))
        cx = np.clip(xs, 0, render - 1)
        cy = np.clip(ys, 0, render - 1)
        ccx = np.clip(cxs, 0, front_control_render - 1)
        ccy = np.clip(cys, 0, front_control_render - 1)
        mask_valid = (front_mask[ccy, ccx] & aligned_source_mask[cy, cx]
                      & in_bounds & control_in_bounds)
        buffered = front_depth[ccy, ccx]
        depth_delta = np.abs(texel_position @ direction - buffered)
        visible = np.isfinite(buffered) & (depth_delta <= depth_tolerance)
        facing = -(texel_normal @ direction)
        front_facing = facing > min_facing
        face_match = np.ones_like(in_bounds, dtype=bool)
        exact_face_match = np.ones_like(in_bounds, dtype=bool)
        if front_ids is not None:
            face_match, exact_face_match = triangle_id_match_with_boundary(
                front_ids, front_depth, cxs, cys, owner_flat,
                texel_position @ direction, depth_tolerance)
        valid = in_bounds & mask_valid & visible & front_facing & face_match
        boundary = distance_from_boundary(front_mask)
        confidence_front = (np.clip(facing, 0.0, 1.0) ** 3.0
                            * np.exp(-(depth_delta / max(depth_tolerance, 1e-9)) ** 2)
                            * (0.10 + 0.90 * boundary[ccy, ccx]))
        original = {
            "slot": int(front_slot), "valid": valid,
            "colour": bilinear(aligned, pixel[:, 0], pixel[:, 1]),
            "low": bilinear(aligned_low, pixel[:, 0], pixel[:, 1]),
            "high": bilinear(aligned_high, pixel[:, 0], pixel[:, 1]),
            "source_pixel": pixel.astype(np.float32),
            "visibility": visible & front_facing, "face_id_match": face_match,
            "exact_face_id_match": exact_face_match,
            "source_mask_valid": mask_valid, "facing": np.clip(facing, 0.0, 1.0),
            "confidence": np.where(valid, confidence_front, 0.0),
            "registration": fit, "source_image": str(original_front),
            "source_transform": str(transform_path) if transform_path else None,
        }

    return {
        "owner": owner, "owned": owned, "atlas_size": atlas_size, "tris": tris, "uv": uv,
        "low": low, "high": high, "confidence": confidence, "facing": facing_stack,
        "detail": detail_stack, "depth_delta": depth_stack,
        "source_pixel": source_pixel_stack, "visibility": visibility_stack,
        "face_id_match": face_id_match_stack, "source_mask_valid": source_mask_stack,
        "valid": valid_stack,
        "original_front": original,
        "texel_position": texel_position, "texel_normal": texel_normal,
        "texel_component": texel_component, "front_screen": front_screen,
        "semantics": semantics, "diagnostics": diagnostics, "render_size": render,
        "view_directions": view_directions, "view_rights": view_rights, "view_ups": view_ups,
        "injectivity": gate, "triangle_count": int(len(tris)),
    }


def protected_face(cache: dict, region_config: Path | None) -> list[dict]:
    """Transfer the configured front-view region onto atlas texels through the front camera."""
    if region_config is None:
        return []
    import protected_region

    config = protected_region.load(Path(region_config))
    if int(config["source_image_size"]) != cache["render_size"]:
        raise RuntimeError(
            f"TEXTURE_REGION_SIZE_MISMATCH:{config['source_image_size']}:{cache['render_size']}")
    records = []
    for name, record in protected_region.build_masks(config, cache["render_size"]).items():
        owner_semantic = record["owner_semantic"]
        if owner_semantic not in cache["semantics"]:
            raise RuntimeError(f"TEXTURE_REGION_OWNER_UNKNOWN:{owner_semantic}")
        if owner_semantic != "front" or cache["front_screen"] is None:
            raise RuntimeError("TEXTURE_REGION_OWNER_NOT_FRONT")
        slot = cache["semantics"].index(owner_semantic)
        weight = bilinear(record["weight"][..., None].astype(np.float32),
                          cache["front_screen"][:, 0], cache["front_screen"][:, 1])[:, 0]
        records.append({
            "name": name, "owner_slot": slot, "owner_semantic": owner_semantic,
            "forbidden_slots": [cache["semantics"].index(s)
                                for s in record["forbidden_owner_semantics"]
                                if s in cache["semantics"]],
            "weight": np.where(cache["confidence"][slot] > 0, weight, 0.0).astype(np.float32),
        })
    return records


def fuse(cache: dict, regions: list[dict]) -> dict:
    """Winner-take-all ownership, detail from the owner alone, low frequency may blend."""
    views, count = cache["confidence"].shape
    colour = np.zeros((count, 3), np.float32)
    ownership = np.full(count, -1, np.int16)
    observed = np.zeros(count, bool)
    decisive = np.zeros(count, bool)
    protected = np.zeros(count, bool)
    blend_count = np.zeros(count, np.int32)

    for start in range(0, count, BLOCK):
        stop = min(start + BLOCK, count)
        block = slice(start, stop)
        width = stop - start
        conf = cache["confidence"][:, block].astype(np.float32).copy()
        face = cache["facing"][:, block].astype(np.float32)
        # A grazing observation may still supply colour, but must not win a detailed surface.
        conf[(face < GRAZING_COSINE) & (conf > 0)] *= 0.05

        order = np.argsort(-conf, axis=0)
        columns = np.arange(width)
        leader, runner = order[0], order[1]
        leader_conf = conf[leader, columns]
        runner_conf = conf[runner, columns]
        seen = leader_conf > 0

        detail = cache["detail"][:, block].astype(np.float32)
        take_all = seen & (
            (leader_conf / np.maximum(runner_conf, 1e-9) >= RATIO)
            | ((leader_conf - runner_conf) >= MARGIN)
            | (detail[leader, columns] / np.maximum(detail[runner, columns], 1e-9) >= DETAIL_RATIO)
            | (runner_conf <= 0)
            | (face[runner, columns] < GRAZING_COSINE))

        own = np.where(seen, leader, -1)
        guarded = np.zeros(width, bool)
        for record in regions:
            slot = record["owner_slot"]
            eligible = ((record["weight"][block] > 0.5)
                        & (cache["confidence"][slot, block] >= PROTECTED_MIN_CONFIDENCE))
            own[eligible] = slot
            take_all |= eligible
            guarded |= eligible
            for forbidden in record["forbidden_slots"]:
                conf[forbidden, eligible] = 0.0

        block_low = cache["low"][:, block].astype(np.float32)
        leader_low = block_low[np.maximum(own, 0), columns]
        leader_high = cache["high"][:, block].astype(np.float32)[np.maximum(own, 0), columns]

        distance = np.linalg.norm(block_low - leader_low[None], axis=2)
        compatible = (conf > 0) & (distance <= COLOUR_COMPATIBILITY)
        compatible[np.maximum(own, 0), columns] = seen
        blend = np.where(compatible, conf ** 2, 0.0)
        total = blend.sum(axis=0)
        blended_low = np.where((total > 0)[:, None],
                               np.einsum("vn,vnc->nc", blend, block_low)
                               / np.maximum(total, 1e-12)[:, None],
                               leader_low)
        # Detail comes from the owner alone; that is the whole point of the split.
        colour[block] = np.where(seen[:, None], blended_low + leader_high, 0.0)
        ownership[block] = own
        observed[block] = seen
        decisive[block] = take_all
        protected[block] = guarded
        blend_count[block] = compatible.sum(axis=0)
        del conf, face, detail, block_low, leader_low, leader_high, distance, compatible, blend

    original = cache.get("original_front")
    original_mask = np.zeros(count, bool)
    if original is not None:
        original_mask = np.asarray(original["valid"], bool)
        if original_mask.any():
            colour[original_mask] = np.asarray(original["colour"], np.float32)[original_mask]
            ownership[original_mask] = int(original["slot"])
            observed[original_mask] = True
            decisive[original_mask] = True
            protected[original_mask] = True
    return {"colour": colour, "ownership": ownership, "observed": observed,
            "decisive": decisive, "protected": protected, "original_front": original_mask,
            "blend_count": blend_count}


def donor_fill(cache: dict, fused: dict, blur_radius: int, max_distance_fraction: float,
               min_normal_dot: float, neighbours: int) -> dict:
    """Fill unobserved texels from 3D donors, low frequency only.

    A 2D fill on the atlas plane has no idea what surface it is crossing. In world space the
    constraints are the ones that actually matter: a donor must belong to the same connected
    component, lie close by, and face a compatible direction. Only the low-frequency band is
    carried, so no eye, nostril or muzzle edge can ever be donated onto a rear or side surface.
    """
    from scipy.spatial import cKDTree

    observed = fused["observed"]
    colour = fused["colour"]
    missing = ~observed
    result = {"unobserved_texels": int(missing.sum()), "donated_texels": 0,
              "unresolved_texels": int(missing.sum()), "max_distance": 0.0,
              "min_normal_dot": float(min_normal_dot), "neighbours": int(neighbours),
              "low_frequency_blur_radius_texels": int(blur_radius),
              "high_frequency_donated": False, "material_prior_texels": 0}
    fused["completion_mask"] = np.zeros_like(observed)
    fused["material_prior_mask"] = np.zeros_like(observed)
    if not missing.any() or not observed.any():
        if missing.any():
            colour[missing] = np.array([96.0, 96.0, 96.0], np.float32)
            fused["completion_mask"][missing] = True
            fused["material_prior_mask"][missing] = True
            result["donated_texels"] = 0
            result["unresolved_texels"] = 0
            result["material_prior_texels"] = int(missing.sum())
        return result

    position = cache["texel_position"]
    extent = float(np.linalg.norm(position.max(axis=0) - position.min(axis=0)))
    limit = extent * max_distance_fraction
    result["max_distance"] = float(limit)

    # Donate the low-frequency band only. The blur is normalised by observed coverage so
    # unowned texels contribute no darkening, and its radius is deliberately far wider than
    # the fusion detail radius: at this width an eye or a nostril is no longer an eye or a
    # nostril, only the local average colour of the surface it sat on.
    size = cache["atlas_size"]
    kernel = (2 * blur_radius + 1, 2 * blur_radius + 1)
    plane = np.zeros((size, size, 3), np.float32)
    plane[cache["owned"]] = np.where(observed[:, None], colour, 0.0)
    coverage = np.zeros((size, size), np.float32)
    coverage[cache["owned"]] = observed.astype(np.float32)
    blurred = cv2.blur(plane, kernel)
    weightsum = cv2.blur(coverage, kernel)
    donor_low = np.where(weightsum[cache["owned"]][:, None] > 1e-6,
                         blurred[cache["owned"]] / np.maximum(weightsum[cache["owned"]], 1e-6)[:, None],
                         0.0).astype(np.float32)
    del plane, coverage, blurred, weightsum

    source = np.flatnonzero(observed)
    target = np.flatnonzero(missing)
    tree = cKDTree(position[source])
    normal = cache["texel_normal"]
    component = cache["texel_component"]

    donated = np.zeros(target.size, bool)
    for start in range(0, target.size, 200_000):
        rows = target[start:start + 200_000]
        distance, index = tree.query(position[rows], k=neighbours,
                                     distance_upper_bound=limit, workers=-1)
        distance = np.atleast_2d(distance)
        index = np.atleast_2d(index)
        valid = np.isfinite(distance) & (index < source.size)
        candidate = source[np.clip(index, 0, source.size - 1)]
        valid &= component[candidate] == component[rows][:, None]
        valid &= np.einsum("ijk,ik->ij", normal[candidate], normal[rows]) >= min_normal_dot
        weight = np.where(valid, 1.0 / np.maximum(distance, 1e-6) ** 2, 0.0)
        total = weight.sum(axis=1)
        usable = total > 0
        if usable.any():
            mixed = np.einsum("ij,ijc->ic", weight[usable], donor_low[candidate[usable]])
            colour[rows[usable]] = mixed / total[usable][:, None]
            donated[start:start + rows.size][usable] = True

    unresolved = target[~donated]
    if unresolved.size:
        global_prior = np.median(colour[source], axis=0).astype(np.float32)
        colour[unresolved] = global_prior
    result["donated_texels"] = int(donated.sum())
    result["material_prior_texels"] = int(unresolved.size)
    result["unresolved_texels"] = 0
    fused["completion_mask"] = np.zeros_like(observed)
    fused["completion_mask"][target[donated]] = True
    fused["material_prior_mask"] = np.zeros_like(observed)
    fused["material_prior_mask"][unresolved] = True
    return result


def write_atlas(cache: dict, fused: dict, donor: dict, output: Path, padding: int) -> dict:
    """Compose the atlas image and bleed only into the packer's own gutter."""
    size = cache["atlas_size"]
    owned = cache["owned"]
    atlas = np.zeros((size, size, 3), np.float32)
    atlas[owned] = np.clip(fused["colour"], 0.0, 255.0)

    max_gutter_distance = 0.0
    if padding > 0:
        kernel = np.ones((2 * padding + 1,) * 2, np.uint8)
        grown = cv2.dilate(owned.astype(np.uint8), kernel) > 0
        gutter = grown & ~owned
        if gutter.any():
            _distance, labels = cv2.distanceTransformWithLabels(
                (~owned).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            oy, ox = np.nonzero(owned)
            lookup = np.zeros(labels.max() + 1, np.int64)
            lookup[labels[oy, ox]] = np.arange(oy.size)
            picked = lookup[labels[gutter]]
            atlas[gutter] = atlas[oy[picked], ox[picked]]
            max_gutter_distance = float(_distance[gutter].max())

    image = np.clip(atlas, 0, 255).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return {
        "atlas": str(output), "atlas_sha256": sha256(output),
        "owned_texels": int(owned.sum()),
        "observed_texels": int(fused["observed"].sum()),
        "donated_texels": donor["donated_texels"],
        "unresolved_texels": donor["unresolved_texels"],
        "gutter_padding_texels": int(padding),
        "max_gutter_donor_distance_texels": round(max_gutter_distance, 3),
        "packer_padding_texels": 4,
        "cross_island_fill": False,
        "cross_island_note": ("The 2D bleed writes only into unowned gutter texels, taking the "
                              "nearest owned texel; no owned texel of any chart ever receives "
                              "colour from another chart through the atlas plane. Unobserved "
                              "owned texels are filled only by the 3D donor pass, which is "
                              "constrained in world space by connected component, distance and "
                              "normal, and carries the low-frequency band only."),
    }


def write_per_texel_evidence(cache: dict, fused: dict, output_dir: Path) -> dict:
    """Persist exact winning samples; triangle state is never broadcast to the atlas."""
    size = int(cache["atlas_size"])
    owned2d = np.asarray(cache["owned"], bool)
    owner = np.asarray(cache["owner"], np.int32)
    direct = np.asarray(fused["observed"], bool)
    owned = np.ones(direct.size, dtype=bool)
    completion = np.asarray(fused.get("completion_mask", np.zeros_like(direct)), bool)
    material_prior = np.asarray(fused.get("material_prior_mask", np.zeros_like(direct)), bool)
    unresolved = owned & ~direct & ~completion & ~material_prior
    slot = np.asarray(fused["ownership"], np.int16)
    original_front = np.asarray(fused.get("original_front", np.zeros_like(direct)), bool)
    source_view = np.where(direct, slot, -1).astype(np.int16)
    source_pixel = np.full((direct.size, 2), -1, np.int32)
    facing = np.zeros(direct.size, np.float32)
    confidence = np.zeros(direct.size, np.float32)
    visibility = np.zeros(direct.size, bool)
    face_id_match = np.zeros(direct.size, bool)
    source_mask_valid = np.zeros(direct.size, bool)
    for view in range(cache["confidence"].shape[0]):
        chosen = direct & ~original_front & (slot == view)
        if not chosen.any():
            continue
        flat = np.flatnonzero(chosen)
        xy = np.rint(cache["source_pixel"][view, flat]).astype(np.int32)
        xy[:, 0] = np.clip(xy[:, 0], 0, cache["render_size"] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, cache["render_size"] - 1)
        source_pixel[flat] = xy
        facing[flat] = cache["facing"][view, flat]
        confidence[flat] = cache["confidence"][view, flat]
        visibility[flat] = cache["visibility"][view, flat]
        face_id_match[flat] = cache["face_id_match"][view, flat]
        source_mask_valid[flat] = cache["source_mask_valid"][view, flat]
    if original_front.any():
        original = cache["original_front"]
        flat = np.flatnonzero(original_front)
        xy = np.rint(original["source_pixel"][flat]).astype(np.int32)
        xy[:, 0] = np.clip(xy[:, 0], 0, cache["render_size"] - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, cache["render_size"] - 1)
        source_pixel[flat] = xy
        facing[flat] = original["facing"][flat]
        confidence[flat] = original["confidence"][flat]
        visibility[flat] = original["visibility"][flat]
        face_id_match[flat] = original["face_id_match"][flat]
        source_mask_valid[flat] = original["source_mask_valid"][flat]
    _owner, weights = rasterise(cache["uv"], cache["tris"], size)
    weights = weights[owned2d]
    bary = np.zeros((direct.size, 3), np.float32)
    bary[:, 0] = 1.0 - weights.reshape(-1, 2)[:, 0] - weights.reshape(-1, 2)[:, 1]
    bary[:, 1:] = weights.reshape(-1, 2)
    direct2d = np.zeros((size, size), dtype=bool)
    completion2d = np.zeros((size, size), dtype=bool)
    unresolved2d = np.zeros((size, size), dtype=bool)
    original2d = np.zeros((size, size), dtype=bool)
    material_prior2d = np.zeros((size, size), dtype=bool)
    direct2d[owned2d] = direct
    completion2d[owned2d] = completion
    unresolved2d[owned2d] = unresolved
    original2d[owned2d] = original_front
    material_prior2d[owned2d] = material_prior
    prov = create_empty_atlas_provenance(size, size)
    prov["triangle_id"] = owner
    prov["source_view"][owned2d] = source_view
    prov["primary_view"][owned2d] = source_view
    prov["source_pixel"][owned2d] = source_pixel
    prov["barycentric"][owned2d] = bary
    prov["visibility"][owned2d] = visibility
    prov["facing"][owned2d] = facing
    prov["face_id_match"][owned2d] = face_id_match
    prov["source_mask_valid"][owned2d] = source_mask_valid
    prov["confidence"][owned2d] = confidence
    prov["uv_occupied_mask"] = owned2d
    prov["atlas_occupied_mask"] = owned2d
    prov["uv_occupied"] = owned2d
    prov["direct_observed_texel_mask"] = direct2d
    prov["direct_observed"] = direct2d
    prov["protected"] = original2d
    prov["protected_mask"] = original2d
    prov["procedural_completion_mask"] = completion2d
    prov["procedural_completion"] = completion2d
    prov["unresolved_mask"] = unresolved2d
    prov["unresolved"] = unresolved2d
    prov["unobserved_surface_mask"] = unresolved2d | completion2d | material_prior2d
    prov["unobserved_surface"] = unresolved2d | completion2d | material_prior2d
    prov["material_prior_mask"] = material_prior2d
    prov["material_prior"] = material_prior2d
    prov["evidence_state"][direct2d & ~original2d] = np.uint8(EvidenceState.GENERATED_OBSERVED)
    prov["evidence_state"][original2d] = np.uint8(EvidenceState.DIRECT_OBSERVED)
    prov["evidence_state"][completion2d] = np.uint8(EvidenceState.PROCEDURAL_COMPLETION)
    prov["evidence_state"][unresolved2d] = np.uint8(EvidenceState.UNRESOLVED)
    generated_classes = {
        "front": SourceClass.GENERATED_FRONT,
        "left": SourceClass.GENERATED_SIDE,
        "right": SourceClass.GENERATED_SIDE,
        "rear": SourceClass.GENERATED_REAR,
    }
    occupied_indices = np.flatnonzero(owned2d.reshape(-1))
    for view, semantic in enumerate(cache["semantics"]):
        chosen = direct & (slot == view)
        if chosen.any():
            prov["source_class"].reshape(-1)[occupied_indices[chosen]] = np.uint8(
                generated_classes.get(semantic, SourceClass.ORIGINAL_NONFACE))
    prov["source_class"][completion2d] = np.uint8(SourceClass.COMPONENT_PRIOR)
    prov["source_class"][material_prior2d] = np.uint8(SourceClass.GLOBAL_PRIOR)
    prov["source_class"][original2d] = np.uint8(SourceClass.ORIGINAL_SOURCE)
    for view, semantic in enumerate(cache["semantics"]):
        chosen = direct & (slot == view)
        if not chosen.any():
            continue
        lineage = {
            "front": Lineage.GENERATED_FRONT,
            "left": Lineage.GENERATED_SIDE,
            "right": Lineage.GENERATED_SIDE,
            "rear": Lineage.GENERATED_REAR,
        }.get(semantic, Lineage.ORIGINAL_NONFACE)
        prov["lineage"].reshape(-1)[occupied_indices[chosen]] = np.uint16(lineage)
        prov["lineage_bits"].reshape(-1)[occupied_indices[chosen]] = np.uint16(lineage)
    prov["lineage"][completion2d] = np.uint16(Lineage.COMPONENT_PRIOR)
    prov["lineage_bits"][completion2d] = np.uint16(Lineage.COMPONENT_PRIOR)
    prov["lineage"][material_prior2d] = np.uint16(Lineage.GLOBAL_PRIOR)
    prov["lineage_bits"][material_prior2d] = np.uint16(Lineage.GLOBAL_PRIOR)
    prov["lineage"][original2d] = np.uint16(Lineage.ORIGINAL_SOURCE)
    prov["lineage_bits"][original2d] = np.uint16(Lineage.ORIGINAL_SOURCE)
    prov["frequency_authority"][direct2d] = np.uint8(FrequencyAuthority.FULL)
    prov["frequency_authority"][completion2d] = np.uint8(FrequencyAuthority.LOW_ONLY)
    prov["frequency_authority"][material_prior2d] = np.uint8(FrequencyAuthority.LOW_ONLY)
    prov["completion_method"][direct2d] = "registered_multiview_direct_projection"
    prov["completion_method"][original2d] = "original_source_front_projection"
    prov["completion_method"][completion2d] = "constrained_3d_low_frequency_donor"
    prov["completion_method"][material_prior2d] = "material_prior"
    prov["completion_method"][unresolved2d] = "unresolved"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(output_dir / "atlas_provenance.npz", prov)
    names = {
        "atlas_owner_triangle": owner, "uv_occupied_mask": owned2d,
        "direct_observed_texel_mask": direct2d, "visible_source_gap_mask": np.zeros_like(direct2d),
        "unobserved_surface_mask": unresolved2d | completion2d | material_prior2d,
        "procedural_completion_mask": completion2d, "material_prior_mask": material_prior2d,
        "unresolved_mask": unresolved2d,
        "direct_visibility": prov["visibility"], "direct_face_id_match": prov["face_id_match"],
        "direct_source_view": prov["source_view"],
        "direct_source_pixel": prov["source_pixel"],
        "direct_source_mask_valid": prov["source_mask_valid"],
        "direct_triangle_id": np.where(direct2d, owner, -1),
    }
    for name, value in names.items():
        np.save(output_dir / f"{name}.npy", value)
    # Generated legacy control-ID mismatches are retained as provenance rather
    # than treated as missing samples; the actual owner triangle comes from the
    # exact UV rasterizer and visibility is already gated above.  Authoritative
    # original-front samples remain strict.
    generated_direct = direct & ~original_front
    direct_sample_valid = direct & visibility & source_mask_valid & (source_view.reshape(-1) >= 0)
    direct_sample_valid[original_front] &= face_id_match[original_front]
    generated_triangle_id_mismatch = generated_direct & ~face_id_match
    return {"direct_texels": int(direct.sum()), "completion_texels": int(completion.sum()),
            "unresolved_texels": int(unresolved.sum()), "direct_texels_without_source_sample": 0,
            "direct_texels_without_measured_gates": int((direct & ~direct_sample_valid).sum()),
            "generated_texels_with_triangle_id_mismatch": int(generated_triangle_id_mismatch.sum()),
            "provenance": str(output_dir / "atlas_provenance.npz")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--views-receipt", required=True)
    parser.add_argument("--original-front", default="",
                        help="authoritative original front image; never a model render")
    parser.add_argument("--original-front-transform", default="",
                        help="high-resolution source-to-conditioning affine receipt")
    parser.add_argument("--original-front-camera", default="",
                        help="bounded source-to-front-control registration receipt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--region-config", default="")
    parser.add_argument("--atlas-size", type=int, default=4096)
    parser.add_argument("--depth-tolerance", type=float, default=0.010)
    parser.add_argument("--min-facing-cosine", type=float, default=0.20)
    parser.add_argument("--detail-radius", type=int, default=3)
    parser.add_argument("--padding-px", type=int, default=2)
    parser.add_argument("--donor-blur-radius", type=int, default=16)
    parser.add_argument("--donor-max-distance-fraction", type=float, default=0.030)
    parser.add_argument("--donor-min-normal-dot", type=float, default=0.50)
    parser.add_argument("--donor-neighbours", type=int, default=16)
    parser.add_argument("--direct-only", action="store_true",
                        help="forbid all observed-to-unobserved donor transfer")
    parser.add_argument("--output-basename", default="",
                        help="artifact stem; defaults to the mesh stem without a UV suffix")
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    mesh = Path(args.mesh)
    basename = args.output_basename or mesh.stem
    for suffix in ("_rewrapped", "_uv", "_lod0_uv"):
        if basename.endswith(suffix):
            basename = basename[: -len(suffix)]
            break
    receipt = json.loads(Path(args.views_receipt).read_text(encoding="utf-8"))
    timings = {}

    started = time.time()
    original_front = Path(args.original_front) if args.original_front else None
    cache = observe(mesh, Path(args.bundle), receipt, args.atlas_size, args.depth_tolerance,
                    args.min_facing_cosine, args.detail_radius, args.direct_only, original_front,
                    Path(args.original_front_transform) if args.original_front_transform else None,
                    Path(args.original_front_camera) if args.original_front_camera else None)
    timings["observe"] = time.time() - started
    print(f"TEXTURE_OBSERVE owned={int(cache['owned'].sum())} "
          f"injective={cache['injectivity']['injective']} {timings['observe']:.0f}s", flush=True)

    started = time.time()
    regions = protected_face(cache, Path(args.region_config) if args.region_config else None)
    fused = fuse(cache, regions)
    timings["fuse"] = time.time() - started
    print(f"TEXTURE_FUSE observed={int(fused['observed'].sum())} {timings['fuse']:.0f}s",
          flush=True)

    started = time.time()
    donor = donor_fill(cache, fused, args.donor_blur_radius, args.donor_max_distance_fraction,
                       args.donor_min_normal_dot, args.donor_neighbours)
    donor["direct_projection_only"] = bool(args.direct_only)
    timings["donor_fill"] = time.time() - started
    print(f"TEXTURE_DONOR donated={donor['donated_texels']} "
          f"unresolved={donor['unresolved_texels']} {timings['donor_fill']:.0f}s", flush=True)

    started = time.time()
    basecolor = root / f"{basename}_basecolor.png"
    atlas_info = write_atlas(cache, fused, donor, basecolor, args.padding_px)
    per_texel = write_per_texel_evidence(cache, fused, Path(args.output_dir))
    textured = root / f"{basename}_textured.glb"
    bound = bind_texture(mesh, textured, basecolor.read_bytes(),
                         np.ones(cache["triangle_count"], bool), wrap=33071)
    timings["write"] = time.time() - started

    semantics = cache["semantics"]
    shares = {}
    for slot, semantic in enumerate(semantics):
        owned_by = int((fused["ownership"] == slot).sum())
        shares[semantic] = round(100.0 * owned_by / max(int(fused["observed"].sum()), 1), 4)

    report = {
        "schema": "injective_atlas_texture_v1",
        "mesh": str(mesh), "mesh_sha256": sha256(mesh),
        "bundle": str(args.bundle), "views_receipt": str(args.views_receipt),
        "original_front": str(original_front) if original_front else None,
        "region_config": args.region_config or None,
        "atlas_size": args.atlas_size,
        "settings": {
            "mode": "FRONT_PROTECTED_MULTIBAND", "ratio": RATIO, "margin": MARGIN,
            "grazing_cosine": GRAZING_COSINE, "detail_ratio": DETAIL_RATIO,
            "protected_min_confidence": PROTECTED_MIN_CONFIDENCE,
            "colour_compatibility": COLOUR_COMPATIBILITY,
            "depth_tolerance": args.depth_tolerance,
            "min_facing_cosine": args.min_facing_cosine,
            "detail_radius": args.detail_radius,
            "regularisation": "none",
            "direct_projection_only": bool(args.direct_only),
        },
        "atlas_injectivity": cache["injectivity"],
        "per_view": cache["diagnostics"],
        "ownership_share_percent": shares,
        "atlas": atlas_info,
        "donor_fill": donor,
        "textured_glb": str(textured), "textured_glb_sha256": sha256(textured),
        "materials_bound": bound,
        "triangles": cache["triangle_count"],
        "provenance": {
            "per_triangle": True,
            "per_texel": per_texel,
            "negative_evidence_triangle_mask_used": False,
            "push_pull_2d_fill_used": False,
            "atlas_wrapping": "CLAMP_TO_EDGE",
            "neural_regeneration": False,
            "camera_remapping": False,
            "original_front_direct_projection": bool(original_front),
            "protected_original_front_texels": int(np.asarray(fused.get("original_front", [])).sum()),
            "unobserved_raw_image_rgb_texels": 0 if args.direct_only else None,
            "unobserved_full_frequency_texels": 0 if args.direct_only else None,
        },
        "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
    }
    (root / "injective_texture_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"INJECTIVE_TEXTURE_DONE observed={atlas_info['observed_texels']} "
          f"donated={atlas_info['donated_texels']} unresolved={atlas_info['unresolved_texels']} "
          f"{textured}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
