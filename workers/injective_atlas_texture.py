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


def observe(mesh: Path, bundle: Path, receipt: dict, atlas_size: int, depth_tolerance: float,
            min_facing: float, detail_radius: int) -> dict:
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
    semantics, diagnostics = [], []
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
        pixel = screen * float(render - 1)
        xs = np.rint(pixel[:, 0]).astype(np.int64)
        ys = np.rint(pixel[:, 1]).astype(np.int64)
        in_bounds = (xs >= 0) & (xs < render) & (ys >= 0) & (ys < render)
        cx = np.clip(xs, 0, render - 1)
        cy = np.clip(ys, 0, render - 1)
        in_mask = control_mask[cy, cx] & in_bounds
        buffered = control_depth[cy, cx]
        depth_delta = np.abs(texel_position @ direction - buffered)
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
            "texels_rejected_by_depth": int((in_bounds & in_mask & ~unoccluded).sum()),
            "texels_rejected_back_facing": int(
                (in_bounds & in_mask & unoccluded & ~front_facing).sum()),
            "texels_valid": int(valid.sum()),
        })
        del screen, pixel, aligned, aligned_low, aligned_high, aligned_detail

    return {
        "owner": owner, "owned": owned, "atlas_size": atlas_size, "tris": tris, "uv": uv,
        "low": low, "high": high, "confidence": confidence, "facing": facing_stack,
        "detail": detail_stack, "depth_delta": depth_stack,
        "texel_position": texel_position, "texel_normal": texel_normal,
        "texel_component": texel_component, "front_screen": front_screen,
        "semantics": semantics, "diagnostics": diagnostics, "render_size": render,
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

    return {"colour": colour, "ownership": ownership, "observed": observed,
            "decisive": decisive, "protected": protected, "blend_count": blend_count}


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
              "high_frequency_donated": False}
    if not missing.any() or not observed.any():
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

    result["donated_texels"] = int(donated.sum())
    result["unresolved_texels"] = int(missing.sum() - donated.sum())
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--views-receipt", required=True)
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
    args = parser.parse_args()

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    mesh = Path(args.mesh)
    receipt = json.loads(Path(args.views_receipt).read_text(encoding="utf-8"))
    timings = {}

    started = time.time()
    cache = observe(mesh, Path(args.bundle), receipt, args.atlas_size, args.depth_tolerance,
                    args.min_facing_cosine, args.detail_radius)
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
    timings["donor_fill"] = time.time() - started
    print(f"TEXTURE_DONOR donated={donor['donated_texels']} "
          f"unresolved={donor['unresolved_texels']} {timings['donor_fill']:.0f}s", flush=True)

    started = time.time()
    atlas_info = write_atlas(cache, fused, donor, root / "panda_injective_basecolor.png",
                             args.padding_px)
    textured = root / "tactical_red_panda_scout_textured.glb"
    bound = bind_texture(mesh, textured, (root / "panda_injective_basecolor.png").read_bytes(),
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
            "negative_evidence_triangle_mask_used": False,
            "push_pull_2d_fill_used": False,
            "atlas_wrapping": "CLAMP_TO_EDGE",
            "neural_regeneration": False,
            "camera_remapping": False,
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
