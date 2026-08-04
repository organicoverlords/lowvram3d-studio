"""Detail-preserving multiview fusion: winner-take-all ownership and multiband blending.

``multiview_texture_projection`` blends every colour-compatible observation by a confidence
product. That is right for low-frequency colour and wrong for detail: a grazing side sample
of the face is colour-compatible with the frontal one, so averaging them dilutes small
high-contrast features like an eye or a nose.

Here the observation stack is built once and then fused several ways from the same cache, so
a bounded grid of variants costs one projection pass rather than one per variant. Ownership
is decided per texel by confidence ratio, absolute margin, grazing rejection and configured
protected regions; detail is then taken from the owner alone while low-frequency colour may
still blend for seam continuity.

Asset-specific geometry - which region of which view must never be outvoted - is loaded from
a protected-region config. No asset coordinate appears in this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_mvadapter_cpu_controls import PROJECTION_SPAN
from fast_texture_projection import bind_texture, immutable_buffer_hashes, rasterise_atlas
from mesh_io import read_glb, triangle_components
from multiview_texture_projection import (
    SEMANTIC_RELIABILITY,
    apply_registration,
    bilinear,
    distance_from_boundary,
    file_prefix,
    foreground_mask,
    push_pull_fill,
    register,
    semantic_of,
)

MODES = ("WINNER_ONLY", "WINNER_HIGH_FREQUENCY", "FRONT_PROTECTED_MULTIBAND")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def textured_triangle_mask(cache: dict, observed: np.ndarray) -> np.ndarray:
    """Mark triangles whose own UV texels have a direct valid source observation.

    The GLB can contain overlapping UV surfaces.  A colour synthesized for one owner chart
    must not activate the same atlas on an unobserved overlapping triangle, especially a rear
    neutral shell.  Material assignment therefore follows direct triangle provenance, not just
    whether an atlas texel eventually received a filled colour.
    """
    mask = np.zeros(len(cache["tris"]), dtype=bool)
    owner_triangles = cache["owner"][cache["owned"]]
    observed = np.asarray(observed, dtype=bool)
    if observed.shape != owner_triangles.shape:
        raise ValueError("observed shape does not match atlas ownership")
    direct = np.asarray(cache["confidence"], dtype=np.float32) > 0
    if direct.shape[1] != owner_triangles.size:
        raise ValueError("confidence shape does not match atlas ownership")
    source_views = np.zeros((len(cache["tris"]), direct.shape[0]), dtype=bool)
    for slot in range(direct.shape[0]):
        source_views[np.unique(owner_triangles[direct[slot]]), slot] = True
    mask[np.any(source_views, axis=1)] = True

    # Overlapping UV shells are the failure mode this mask exists to prevent.  A rear/side
    # triangle whose own view did not provide a direct sample must not inherit a frontal atlas
    # texel just because another shell owns that UV coordinate.  The visible-triangle arrays
    # are generated with the existing controls and are treated as negative evidence only; they
    # do not alter camera semantics or create new views.
    semantics = cache["semantics"]
    for semantic in ("rear", "left", "right"):
        if semantic not in cache.get("visible_triangles", {}):
            continue
        slot = semantics.index(semantic)
        visible = cache["visible_triangles"][semantic]
        mask[visible & ~source_views[:, slot]] = False
    return mask


def box_blur(image: np.ndarray, radius: int) -> np.ndarray:
    """Separable box blur over an HxWxC image, used for the low/high frequency split."""
    if radius <= 0:
        return image.astype(np.float32)
    padded = np.pad(image.astype(np.float32), ((radius, radius), (radius, radius), (0, 0)),
                    mode="edge")
    cumulative = np.cumsum(padded, axis=0)
    cumulative = np.concatenate([np.zeros((1,) + cumulative.shape[1:], np.float32), cumulative])
    size = 2 * radius + 1
    vertical = (cumulative[size:] - cumulative[:-size]) / size
    cumulative = np.cumsum(vertical, axis=1)
    cumulative = np.concatenate(
        [np.zeros((cumulative.shape[0], 1, cumulative.shape[2]), np.float32), cumulative], axis=1)
    return (cumulative[:, size:] - cumulative[:, :-size]) / size


def local_detail(image: np.ndarray, radius: int) -> np.ndarray:
    """Per-pixel local high-frequency energy, used to compare how much detail a view holds."""
    grey = image.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    low = box_blur(grey[..., None], radius)[..., 0]
    return box_blur(np.abs(grey - low)[..., None], radius)[..., 0]


def build_cache(mesh: Path, bundle: Path, receipt: dict, atlas_size: int,
                depth_tolerance: float, min_facing: float, detail_radius: int) -> dict:
    """Project every view once and keep the per-texel observation stack."""
    contract = json.loads((bundle / "camera_contract.json").read_text(encoding="utf-8"))
    if "control_space_transform" not in contract:
        raise RuntimeError("FUSION_CONTRACT_HAS_NO_BASIS")
    transform = np.asarray(contract["control_space_transform"], dtype=np.float64)
    generated = {item["name"]: Path(item["path"]) for item in receipt["output_images"]}

    positions, normals, uv, tris = read_glb(mesh)
    if uv is None:
        raise RuntimeError("FUSION_UV_MISSING")
    canonical = positions.astype(np.float64) @ transform.T
    vertices = canonical * (0.5 / float(np.max(np.abs(canonical))))
    vertex_normals = np.asarray(normals, np.float64) @ transform.T
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-12)

    owner, weights = rasterise_atlas(uv, tris, atlas_size)
    owned = owner >= 0
    if not owned.any():
        raise RuntimeError("FUSION_ATLAS_EMPTY")
    components, _ = triangle_components(positions, tris)
    surface_labels = np.full((atlas_size, atlas_size), -1, np.int32)
    surface_labels[owned] = components[owner[owned]]
    corners = tris[owner[owned]]
    wa = weights[owned][:, 0][:, None]
    wb = weights[owned][:, 1][:, None]
    wc = 1.0 - wa - wb
    texel_position = (vertices[corners[:, 0]] * wc + vertices[corners[:, 1]] * wa
                      + vertices[corners[:, 2]] * wb)
    texel_normal = (vertex_normals[corners[:, 0]] * wc + vertex_normals[corners[:, 1]] * wa
                    + vertex_normals[corners[:, 2]] * wb)
    texel_normal /= np.maximum(np.linalg.norm(texel_normal, axis=1, keepdims=True), 1e-12)

    views = sorted(contract["views"], key=lambda item: int(item["index"]))
    count = int(owned.sum())
    low = np.zeros((len(views), count, 3), np.float32)
    high = np.zeros((len(views), count, 3), np.float32)
    confidence = np.zeros((len(views), count), np.float32)
    depth_delta_stack = np.full((len(views), count), np.inf, np.float32)
    facing_stack = np.zeros((len(views), count), np.float32)
    detail_stack = np.zeros((len(views), count), np.float32)
    screen_stack = np.zeros((len(views), count, 2), np.float32)
    semantics, diagnostics = [], []
    visible_triangles = {}

    for slot, view in enumerate(views):
        index = int(view["index"])
        semantic = semantic_of(view)
        prefix = file_prefix(view)
        visible_path = bundle / f"{prefix}_visible_triangles.npy"
        if visible_path.is_file():
            visible = np.asarray(np.load(visible_path), dtype=bool)
            if visible.shape != (len(tris),):
                raise RuntimeError(f"FUSION_VISIBLE_TRIANGLE_SHAPE:{semantic}")
            visible_triangles[semantic] = visible
        image_path = generated.get(f"view_{index}_{semantic}.png")
        if image_path is None or not Path(image_path).is_file():
            raise RuntimeError(f"FUSION_VIEW_MISSING:{index}:{semantic}")
        raw = np.asarray(Image.open(image_path).convert("RGB"))
        control_mask = np.asarray(Image.open(bundle / f"{prefix}_mask.png").convert("L")) > 127
        control_depth = np.load(bundle / f"{prefix}_depth.npy")
        render = raw.shape[0]
        fit = register(foreground_mask(raw), control_mask)
        aligned = apply_registration(raw, fit).astype(np.float32)
        aligned_low = box_blur(aligned, detail_radius)
        aligned_high = aligned - aligned_low
        aligned_detail = local_detail(aligned, detail_radius)

        direction = np.asarray(view["camera_direction"], np.float64)
        right = np.asarray(view["camera_right"], np.float64)
        up = np.asarray(view["camera_up"], np.float64)
        screen = np.stack([texel_position @ right / PROJECTION_SPAN + 0.5,
                           0.5 - (texel_position @ up) / PROJECTION_SPAN], axis=1)
        depth = texel_position @ direction
        pixel = screen * float(render - 1)
        xs = np.rint(pixel[:, 0]).astype(np.int64)
        ys = np.rint(pixel[:, 1]).astype(np.int64)
        in_bounds = (xs >= 0) & (xs < render) & (ys >= 0) & (ys < render)
        cx = np.clip(xs, 0, render - 1)
        cy = np.clip(ys, 0, render - 1)
        in_mask = control_mask[cy, cx] & in_bounds
        buffered = control_depth[cy, cx]
        depth_delta = np.abs(depth - buffered)
        depth_delta_stack[slot] = depth_delta.astype(np.float32)
        unoccluded = np.isfinite(buffered) & (depth_delta <= depth_tolerance)
        facing = -(texel_normal @ direction)
        front_facing = facing > min_facing
        valid = in_bounds & in_mask & unoccluded & front_facing

        boundary = distance_from_boundary(control_mask)
        weight = (np.clip(facing, 0.0, 1.0) ** 3.0
                  * np.exp(-(depth_delta / max(depth_tolerance, 1e-9)) ** 2)
                  * (0.10 + 0.90 * boundary[cy, cx])
                  * SEMANTIC_RELIABILITY.get(semantic, 1.0))
        confidence[slot] = np.where(valid, weight, 0.0)
        facing_stack[slot] = np.where(valid, np.clip(facing, 0.0, 1.0), 0.0)
        low[slot] = bilinear(aligned_low, pixel[:, 0], pixel[:, 1])
        high[slot] = bilinear(aligned_high, pixel[:, 0], pixel[:, 1])
        detail_stack[slot] = bilinear(aligned_detail[..., None], pixel[:, 0], pixel[:, 1])[:, 0]
        screen_stack[slot] = pixel
        semantics.append(semantic)
        diagnostics.append({
            "raw_index": index, "semantic_label": semantic, "source_image": str(image_path),
            "registration": fit,
            "texels_rejected_out_of_bounds": int((~in_bounds).sum()),
            "texels_rejected_by_mask": int((in_bounds & ~in_mask).sum()),
            "texels_rejected_by_depth": int((in_bounds & in_mask & ~unoccluded).sum()),
            "texels_rejected_back_facing": int(
                (in_bounds & in_mask & unoccluded & ~front_facing).sum()),
            "texels_valid": int(valid.sum()),
        })

    return {
        "owner": owner, "owned": owned, "atlas_size": atlas_size,
        "low": low, "high": high, "confidence": confidence, "depth_delta": depth_delta_stack,
        "facing": facing_stack,
        "detail": detail_stack, "screen": screen_stack, "semantics": semantics,
        "diagnostics": diagnostics, "contract": contract, "uv": uv, "tris": tris,
        "render_size": render, "surface_labels": surface_labels,
        "visible_triangles": visible_triangles,
    }


def protected_weights(cache: dict, region_config: Path | None) -> dict[str, dict]:
    """Transfer configured image-space regions onto atlas texels via their owner view."""
    if region_config is None:
        return {}
    import protected_region
    config = protected_region.load(Path(region_config))
    size = int(config["source_image_size"])
    if size != cache["render_size"]:
        raise RuntimeError(f"FUSION_REGION_SIZE_MISMATCH:{size}:{cache['render_size']}")
    masks = protected_region.build_masks(config, size)
    transferred = {}
    for name, record in masks.items():
        owner_semantic = record["owner_semantic"]
        if owner_semantic not in cache["semantics"]:
            raise RuntimeError(f"FUSION_REGION_OWNER_UNKNOWN:{owner_semantic}")
        slot = cache["semantics"].index(owner_semantic)
        pixel = cache["screen"][slot]
        weight = bilinear(record["weight"][..., None].astype(np.float32),
                          pixel[:, 0], pixel[:, 1])[:, 0]
        transferred[name] = {
            "owner_slot": slot,
            "owner_semantic": owner_semantic,
            "forbidden_slots": [cache["semantics"].index(s)
                                for s in record["forbidden_owner_semantics"]
                                if s in cache["semantics"]],
            "weight": np.where(cache["confidence"][slot] > 0, weight, 0.0).astype(np.float32),
            "priority": record["priority"],
        }
    return transferred


def regularise(ownership: np.ndarray, owned: np.ndarray, decisive: np.ndarray,
               protected: np.ndarray, radius: int) -> tuple[np.ndarray, dict]:
    """Confidence-aware majority filter, confined to each UV island.

    Islands are labelled on the atlas coverage mask, so a majority vote can never pull an
    owner across a chart boundary onto unrelated surface. Texels whose leader already wins
    decisively, and texels inside a protected region, keep their owner untouched.
    """
    from scipy.ndimage import label

    islands, island_count = label(owned)
    changed = 0
    result = ownership.copy()
    candidates = owned & ~decisive & ~protected
    if radius > 0 and candidates.any():
        votes = {}
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                source_y = slice(max(0, -dy), min(ownership.shape[0], ownership.shape[0] - dy))
                target_y = slice(max(0, dy), min(ownership.shape[0], ownership.shape[0] + dy))
                source_x = slice(max(0, -dx), min(ownership.shape[1], ownership.shape[1] - dx))
                target_x = slice(max(0, dx), min(ownership.shape[1], ownership.shape[1] + dx))
                shifted_owner = np.full(ownership.shape, -1, np.int16)
                shifted_island = np.full(islands.shape, -1, np.int32)
                shifted_owner[target_y, target_x] = ownership[source_y, source_x]
                shifted_island[target_y, target_x] = islands[source_y, source_x]
                same = (shifted_island == islands) & (shifted_owner >= 0)
                for value in np.unique(shifted_owner[same]):
                    if value < 0:
                        continue
                    votes.setdefault(int(value), np.zeros(ownership.shape, np.int32))
                    votes[int(value)] += (same & (shifted_owner == value)).astype(np.int32)
        if votes:
            keys = sorted(votes)
            stacked = np.stack([votes[k] for k in keys])
            winner = np.asarray(keys)[np.argmax(stacked, axis=0)]
            replace = candidates & (winner != ownership)
            result[replace] = winner[replace]
            changed = int(replace.sum())
    return result, {"uv_islands": int(island_count), "texels_reassigned": changed,
                    "radius": radius}


def fuse(cache: dict, regions: dict, mode: str, ratio: float, margin: float,
         grazing_cosine: float, detail_ratio: float, protected_min_confidence: float,
         colour_compatibility: float, regularise_radius: int) -> dict:
    confidence = cache["confidence"].copy()
    # Grazing observations are kept only when nothing better exists, so they can supply
    # colour without ever winning ownership of a detailed surface.
    grazing = (cache["facing"] < grazing_cosine) & (confidence > 0)
    confidence[grazing] *= 0.05

    order = np.argsort(-confidence, axis=0)
    leader = order[0]
    runner = order[1]
    columns = np.arange(confidence.shape[1])
    leader_confidence = confidence[leader, columns]
    runner_confidence = confidence[runner, columns]
    observed = leader_confidence > 0

    ratio_value = leader_confidence / np.maximum(runner_confidence, 1e-9)
    leader_detail = cache["detail"][leader, columns]
    runner_detail = cache["detail"][runner, columns]
    detail_value = leader_detail / np.maximum(runner_detail, 1e-9)

    winner_take_all = observed & (
        (ratio_value >= ratio)
        | ((leader_confidence - runner_confidence) >= margin)
        | (detail_value >= detail_ratio)
        | (runner_confidence <= 0)
        | (cache["facing"][runner, columns] < grazing_cosine))

    ownership = np.where(observed, leader, -1)
    protected_flat = np.zeros(confidence.shape[1], bool)
    if mode == "FRONT_PROTECTED_MULTIBAND":
        for record in regions.values():
            slot = record["owner_slot"]
            eligible = (record["weight"] > 0.5) & (
                cache["confidence"][slot] >= protected_min_confidence)
            ownership[eligible] = slot
            winner_take_all |= eligible
            protected_flat |= eligible
            for forbidden in record["forbidden_slots"]:
                confidence[forbidden, eligible] = 0.0

    decisive_flat = winner_take_all

    size = cache["atlas_size"]
    owned = cache["owned"]
    ownership_map = np.full((size, size), -1, np.int16)
    ownership_map[owned] = ownership
    decisive_map = np.zeros((size, size), bool)
    decisive_map[owned] = decisive_flat
    protected_map = np.zeros((size, size), bool)
    protected_map[owned] = protected_flat
    raw_ownership_map = ownership_map.copy()
    ownership_map, regularisation = regularise(
        ownership_map, owned, decisive_map, protected_map, regularise_radius)
    ownership = ownership_map[owned]

    leader_low = np.take_along_axis(cache["low"], np.maximum(ownership, 0)[None, :, None],
                                    axis=0)[0]
    leader_high = np.take_along_axis(cache["high"], np.maximum(ownership, 0)[None, :, None],
                                     axis=0)[0]

    if mode == "WINNER_ONLY":
        colour = leader_low + leader_high
        blend_count = np.ones(confidence.shape[1], np.int32)
    else:
        distance = np.linalg.norm(cache["low"] - leader_low[None], axis=2)
        compatible = (confidence > 0) & (distance <= colour_compatibility)
        compatible[np.maximum(ownership, 0), columns] = observed
        blend = np.where(compatible, confidence ** 2, 0.0)
        total = blend.sum(axis=0)
        blended_low = np.where(
            (total > 0)[:, None],
            np.einsum("vn,vnc->nc", blend, cache["low"]) / np.maximum(total, 1e-12)[:, None],
            leader_low)
        # Detail never averages: it comes from the owner alone, which is the whole point.
        colour = blended_low + leader_high
        blend_count = compatible.sum(axis=0)

    colour = np.where(observed[:, None], colour, 0.0)
    leader_depth_error = cache["depth_delta"][leader, columns]
    return {
        "colour": colour, "observed": observed, "ownership": ownership,
        "ownership_map": ownership_map, "raw_ownership_map": raw_ownership_map,
        "decisive_map": decisive_map, "protected_map": protected_map,
        "leader_confidence": leader_confidence, "ratio_value": ratio_value,
        "leader_depth_error": leader_depth_error,
        "blend_count": blend_count, "regularisation": regularisation,
        "winner_take_all_fraction": float(decisive_flat[observed].mean()) if observed.any() else 0.0,
        "observation_count": (cache["confidence"] > 0).sum(axis=0),
    }
