"""Confidence-weighted multiview texture projection onto the repaired panda atlas.

``fast_texture_projection`` owns one view and paints every texel it can reach from it,
which is why the old atlas carried the front face onto the rear. Here each texel is owned
by whichever views can actually see it: an observation only counts once it survives the
foreground-mask, depth, occlusion, bounds and back-facing gates, and surviving observations
are blended by a confidence product rather than averaged flat.

Every camera comes from the bundle's proven contract, and the projection reproduces the
exact transform the controls were rasterised with, so a texel's atlas position and its
pixel in the generated view refer to the same surface point by construction.
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

# Top and bottom see the character at a grazing angle over most of its body, so their
# observations are kept but discounted relative to the four horizontal cameras.
SEMANTIC_RELIABILITY = {"front": 1.0, "rear": 1.0, "left": 1.0, "right": 1.0,
                        "top": 0.85, "bottom": 0.85}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_prefix(view: dict) -> str:
    return str(view.get("control_file_prefix") or view["semantic_name"])


def semantic_of(view: dict) -> str:
    return str(view.get("proven_semantic") or view["semantic_name"])


def foreground_mask(array: np.ndarray) -> np.ndarray:
    """Same border-median rule the inference QA uses, so both agree on what is subject."""
    border = np.concatenate((array[0], array[-1], array[:, 0], array[:, -1]),
                            axis=0).astype(np.float32)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(array.astype(np.float32) - background, axis=2)
    # Saturation alone is not foreground evidence: compression noise in a white source
    # canvas and coloured ground shadows can satisfy that test and later bleed into the
    # atlas.  Require measurable separation from the sampled border matte instead.
    return distance > 12.0


def distance_from_boundary(mask: np.ndarray, rounds: int = 24) -> np.ndarray:
    """Cheap inward chamfer: how many erosions a pixel survives, normalised to [0, 1]."""
    current = mask.copy()
    depth = np.zeros(mask.shape, np.float32)
    for _ in range(rounds):
        eroded = current.copy()
        eroded[1:, :] &= current[:-1, :]
        eroded[:-1, :] &= current[1:, :]
        eroded[:, 1:] &= current[:, :-1]
        eroded[:, :-1] &= current[:, 1:]
        depth += eroded
        current = eroded
        if not current.any():
            break
    return depth / float(rounds)


def register(generated: np.ndarray, target: np.ndarray) -> dict:
    """Recover the small scale/offset between the generated subject and its control mask."""
    def transform(mask, scale, dx, dy):
        image = Image.fromarray(mask.astype(np.uint8) * 255, "L")
        size = max(1, int(round(mask.shape[0] * scale)))
        canvas = Image.new("L", image.size, 0)
        canvas.paste(image.resize((size, size), Image.NEAREST),
                     ((mask.shape[1] - size) // 2 + dx, (mask.shape[0] - size) // 2 + dy))
        return np.asarray(canvas) > 127

    def iou(a, b):
        union = np.logical_or(a, b).sum()
        return float(np.logical_and(a, b).sum() / union) if union else 0.0

    best = {"scale": 1.0, "dx": 0, "dy": 0, "iou": iou(generated, target)}
    span = max(2, generated.shape[0] // 16)
    for scale in (0.90, 0.94, 0.97, 1.0, 1.03, 1.06, 1.10):
        for dx in range(-span, span + 1, max(1, span // 4)):
            for dy in range(-span, span + 1, max(1, span // 4)):
                score = iou(transform(generated, scale, dx, dy), target)
                if score > best["iou"]:
                    best = {"scale": scale, "dx": dx, "dy": dy, "iou": score}
    return best


def apply_registration(image: np.ndarray, fit: dict) -> np.ndarray:
    """Resample the generated view so its subject lands on the control silhouette."""
    size = image.shape[0]
    scaled = max(1, int(round(size * fit["scale"])))
    resized = Image.fromarray(image).resize((scaled, scaled), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), tuple(
        np.median(np.concatenate((image[0], image[-1]), axis=0), axis=0).astype(int)))
    canvas.paste(resized, ((size - scaled) // 2 + fit["dx"], (size - scaled) // 2 + fit["dy"]))
    return np.asarray(canvas)


def bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample a view at fractional pixel coordinates.

    Nearest-neighbour turns a 384px view into visible texel-scale mosaic once it is
    resampled into a 2048 atlas, and the blocks read as noise rather than as detail.
    """
    height, width = image.shape[:2]
    x = np.clip(x, 0, width - 1.001)
    y = np.clip(y, 0, height - 1.001)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = (x - x0)[:, None]
    fy = (y - y0)[:, None]
    top = image[y0, x0] * (1 - fx) + image[y0, x0 + 1] * fx
    bottom = image[y0 + 1, x0] * (1 - fx) + image[y0 + 1, x0 + 1] * fx
    return top * (1 - fy) + bottom * fy


def push_pull_fill(colour: np.ndarray, filled: np.ndarray, rounds: int = 256,
                   domain: np.ndarray | None = None,
                   labels: np.ndarray | None = None):
    """Grow colour by explicit in-bounds neighbours within one surface domain.

    ``domain`` confines completion to geometry-bearing or chart-padding texels.  ``labels``
    prevents a donor crossing between disconnected UV/3D components.  Slicing is used instead
    of ``np.roll`` so atlas edges can never wrap into one another.
    """
    synthesized = np.zeros(filled.shape, bool)
    allowed = np.ones(filled.shape, bool) if domain is None else np.asarray(domain, bool)
    if labels is not None and np.asarray(labels).shape != filled.shape:
        raise ValueError("labels must match filled shape")
    current = filled & allowed
    output = colour.copy()
    for _ in range(rounds):
        if np.all(current[allowed]):
            break
        neighbour_sum = np.zeros(colour.shape, np.float64)
        neighbour_count = np.zeros(filled.shape, np.float64)
        for source, target in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                               ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                               ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                               ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
            source_mask = current[source]
            valid = source_mask & allowed[target]
            if labels is not None:
                valid &= labels[source] == labels[target]
            neighbour_sum[target] += output[source] * valid[..., None]
            neighbour_count[target] += valid
        grow = allowed & (~current) & (neighbour_count > 0)
        if not grow.any():
            break
        output[grow] = (neighbour_sum[grow]
                        / np.maximum(neighbour_count[grow][:, None], 1.0))
        synthesized |= grow
        current |= grow
    return output, synthesized, current


def expand_label_domain(labels: np.ndarray, rounds: int) -> np.ndarray:
    """Build bounded chart-padding labels without crossing atlas edges."""
    result = np.asarray(labels, dtype=np.int32).copy()
    for _ in range(max(0, int(rounds))):
        grown = result.copy()
        for source, target in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                               ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                               ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                               ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
            target_empty = result[target] < 0
            source_valid = result[source] >= 0
            grown[target] = np.where(target_empty & source_valid, result[source], grown[target])
        if np.array_equal(grown, result):
            break
        result = grown
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--views-receipt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-glb", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--atlas-size", type=int, default=2048)
    parser.add_argument("--depth-tolerance", type=float, default=0.010,
                        help="control-space units; a texel further than this behind the "
                             "view's own depth buffer is occluded")
    parser.add_argument("--min-facing-cosine", type=float, default=0.20)
    parser.add_argument("--edge-bleed", type=int, default=12,
                        help="texels of colour pushed past every chart edge; too few and "
                             "bilinear sampling pulls the black gutter into every seam")
    parser.add_argument("--colour-compatibility", type=float, default=60.0,
                        help="max RGB distance from the leading observation before a "
                             "second view is treated as a different surface")
    args = parser.parse_args()

    mesh = Path(args.mesh)
    bundle = Path(args.bundle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(Path(args.views_receipt).read_text(encoding="utf-8"))
    contract = json.loads((bundle / "camera_contract.json").read_text(encoding="utf-8"))
    generated = {item["name"]: Path(item["path"]) for item in receipt["output_images"]}

    positions, normals, uv, tris = read_glb(mesh)
    if uv is None:
        raise RuntimeError("MULTIVIEW_PROJECTION_UV_MISSING")

    # Reproduce the control builder's normalisation exactly. The basis comes from the
    # bundle's own contract, never from this module's default: a bundle built on a
    # different canonical basis would otherwise be sampled in the wrong frame and almost
    # every texel would fail the depth test for a reason that looks like occlusion.
    if "control_space_transform" not in contract:
        raise RuntimeError("MULTIVIEW_PROJECTION_CONTRACT_HAS_NO_BASIS")
    transform = np.asarray(contract["control_space_transform"], dtype=np.float64)
    canonical = positions.astype(np.float64) @ transform.T
    scale = 0.5 / float(np.max(np.abs(canonical)))
    vertices = canonical * scale
    vertex_normals = np.asarray(normals, np.float64) @ transform.T
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-12)

    size = int(args.atlas_size)
    owner, weights = rasterise_atlas(uv, tris, size)
    owned = owner >= 0
    owned_count = int(owned.sum())
    if not owned_count:
        raise RuntimeError("MULTIVIEW_PROJECTION_ATLAS_EMPTY")
    components, _ = triangle_components(positions, tris)
    surface_labels = np.full((size, size), -1, np.int32)
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
    samples, confidences, labels, diagnostics = [], [], [], []
    for view in views:
        index = int(view["index"])
        semantic = semantic_of(view)
        prefix = file_prefix(view)
        image_path = generated.get(f"view_{index}_{semantic}.png")
        if image_path is None or not Path(image_path).is_file():
            raise RuntimeError(f"MULTIVIEW_PROJECTION_VIEW_MISSING:{index}:{semantic}")
        raw = np.asarray(Image.open(image_path).convert("RGB"))
        control_mask = np.asarray(
            Image.open(bundle / f"{prefix}_mask.png").convert("L")) > 127
        control_depth = np.load(bundle / f"{prefix}_depth.npy")
        render = raw.shape[0]
        if control_mask.shape[0] != render:
            raise RuntimeError(f"MULTIVIEW_PROJECTION_SIZE_MISMATCH:{semantic}")

        fit = register(foreground_mask(raw), control_mask)
        aligned = apply_registration(raw, fit)

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
        unoccluded = np.isfinite(buffered) & (depth_delta <= args.depth_tolerance)
        facing = -(texel_normal @ direction)
        front_facing = facing > args.min_facing_cosine
        valid = in_bounds & in_mask & unoccluded & front_facing

        boundary = distance_from_boundary(control_mask)
        # Cubed facing term: where two cameras both see a surface, the one looking at it
        # squarely should dominate outright rather than tie, otherwise the per-texel
        # leader flips between them and the atlas speckles.
        confidence = (
            np.clip(facing, 0.0, 1.0) ** 3.0
            * np.exp(-(depth_delta / max(args.depth_tolerance, 1e-9)) ** 2)
            * (0.10 + 0.90 * boundary[cy, cx])
            * SEMANTIC_RELIABILITY.get(semantic, 1.0)
        )
        confidence = np.where(valid, confidence, 0.0)

        samples.append(bilinear(aligned.astype(np.float64), pixel[:, 0], pixel[:, 1]))
        confidences.append(confidence)
        labels.append(semantic)
        diagnostics.append({
            "raw_index": index,
            "semantic_label": semantic,
            "source_image": str(image_path),
            "registration": fit,
            "texels_in_bounds": int(in_bounds.sum()),
            "texels_rejected_out_of_bounds": int((~in_bounds).sum()),
            "texels_rejected_by_mask": int((in_bounds & ~in_mask).sum()),
            "texels_rejected_by_depth": int((in_bounds & in_mask & ~unoccluded).sum()),
            "texels_rejected_back_facing": int(
                (in_bounds & in_mask & unoccluded & ~front_facing).sum()),
            "texels_valid": int(valid.sum()),
            "mean_confidence_where_valid": float(confidence[valid].mean()) if valid.any() else 0.0,
        })

    sample_stack = np.stack(samples)                     # (V, N, 3)
    confidence_stack = np.stack(confidences)             # (V, N)
    leader = np.argmax(confidence_stack, axis=0)
    leading_colour = np.take_along_axis(
        sample_stack, leader[None, :, None], axis=0)[0]  # (N, 3)

    # Only blend views that agree with the leader: a rear texel that a front camera also
    # reaches through a thin gap would otherwise average a face onto the back of the head.
    distance = np.linalg.norm(sample_stack - leading_colour[None], axis=2)
    compatible = (confidence_stack > 0) & (distance <= args.colour_compatibility)
    compatible[leader, np.arange(len(leader))] = confidence_stack[
        leader, np.arange(len(leader))] > 0
    # Cubing again at blend time keeps a clear winner from being diluted by a
    # grazing second opinion while still letting comparable views average.
    blend_weight = np.where(compatible, confidence_stack ** 3, 0.0)
    total = blend_weight.sum(axis=0)
    observed = total > 0
    blended = np.where(
        observed[:, None],
        np.einsum("vn,vnc->nc", blend_weight, sample_stack)
        / np.maximum(total, 1e-12)[:, None],
        0.0)

    observation_count = (confidence_stack > 0).sum(axis=0)
    blend_count = compatible.sum(axis=0)

    atlas = np.zeros((size, size, 3), np.float64)
    atlas[owned] = blended
    filled = np.zeros((size, size), bool)
    filled[owned] = observed
    ownership = np.full((size, size), -1, np.int16)
    ownership[owned] = np.where(observed, leader, -1)
    confidence_map = np.zeros((size, size), np.float32)
    confidence_map[owned] = np.where(observed, total / np.maximum(blend_count, 1), 0.0)
    observation_map = np.zeros((size, size), np.int16)
    observation_map[owned] = observation_count

    atlas, synthesized_mask, resolved = push_pull_fill(
        atlas, filled, domain=owned, labels=surface_labels)
    # Only geometry-bearing texels count; the empty gutter is not "unresolved surface".
    synthesized_on_surface = synthesized_mask & owned
    unresolved = owned & ~resolved

    # A one-texel bleed past every chart edge stops bilinear sampling from pulling in the
    # gutter and darkening every seam.
    padding_labels = expand_label_domain(surface_labels, args.edge_bleed)
    bled, _bleed_mask, _ = push_pull_fill(
        atlas, resolved, rounds=args.edge_bleed,
        domain=padding_labels >= 0, labels=padding_labels)
    atlas_image = Image.fromarray(np.clip(bled, 0, 255).astype(np.uint8))
    atlas_path = output_dir / "panda_multiview_basecolor.png"
    atlas_image.save(atlas_path)

    seam = np.zeros((size, size), bool)
    for source, target in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                           ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
                           ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                           ((slice(None), slice(None, -1)), (slice(None), slice(1, None)))):
        seam[target] |= owned[target] & owned[source] & (ownership[target] != ownership[source])
    triangle_map = np.zeros((size, size), np.uint32)
    triangle_map[owned] = owner[owned].astype(np.uint32) + 1
    island_map = np.zeros((size, size), np.uint32)
    island_map[owned] = surface_labels[owned].astype(np.uint32) + 1
    for name, array, mode in (
            ("ownership_map.png", ((ownership + 1) * 40).astype(np.uint8), "L"),
            ("triangle_id_map.png", np.clip(triangle_map % 256, 0, 255).astype(np.uint8), "L"),
            ("uv_island_map.png", np.clip(island_map % 256, 0, 255).astype(np.uint8), "L"),
            ("observed_coverage_mask.png", (filled * 255).astype(np.uint8), "L"),
            ("synthesized_coverage_mask.png", (synthesized_on_surface * 255).astype(np.uint8), "L"),
            ("confidence_map.png", (np.clip(confidence_map, 0, 1) * 255).astype(np.uint8), "L"),
            ("seam_map.png", (seam * 255).astype(np.uint8), "L"),
            ("multiview_coverage_mask.png",
             ((observation_map >= 2) * 255).astype(np.uint8), "L")):
        Image.fromarray(array, mode).save(output_dir / name)

    output_glb = Path(args.output_glb)
    before = immutable_buffer_hashes(mesh)
    atlas_bytes = atlas_path.read_bytes()
    textured_triangles = np.zeros(len(tris), dtype=bool)
    textured_triangles[owner[owned][observed]] = True
    bind_texture(mesh, output_glb, atlas_bytes,
                 textured_triangles=textured_triangles)
    after = immutable_buffer_hashes(output_glb)

    def percent(count: int) -> float:
        return round(100.0 * count / owned_count, 4)

    coverage = {
        "atlas_texels_with_geometry": owned_count,
        "directly_observed_percent": percent(int((observation_count >= 1).sum())),
        "multiview_observed_percent": percent(int((observation_count >= 2).sum())),
        "blended_from_multiple_views_percent": percent(int((blend_count >= 2).sum())),
        "synthesized_percent": percent(int(synthesized_on_surface.sum())),
        "unresolved_percent": percent(int(unresolved.sum())),
        "note": ("synthesized and unresolved are reported separately and are NOT folded "
                 "into the observed figure"),
    }
    report = {
        "schema": "multiview_texture_projection_v1",
        "mesh": str(mesh),
        "mesh_sha256": sha256_file(mesh),
        "bundle": str(bundle),
        "views_receipt": str(args.views_receipt),
        "raw_to_semantic": contract.get("raw_to_semantic"),
        "canonical_basis": contract.get("canonical_basis"),
        "control_space_transform": transform.tolist(),
        "atlas": str(atlas_path),
        "atlas_sha256": sha256_bytes(atlas_bytes),
        "atlas_size": size,
        "output_glb": str(output_glb),
        "output_glb_sha256": sha256_file(output_glb),
        "geometry_buffers_before": before,
        "geometry_buffers_after": after,
        "geometry_unchanged": before == after,
        "textured_triangles": int(textured_triangles.sum()),
        "textured_triangle_fraction": round(float(textured_triangles.mean()), 6),
        "coverage": coverage,
        "gates": {
            "depth_tolerance": args.depth_tolerance,
            "min_facing_cosine": args.min_facing_cosine,
            "colour_compatibility": args.colour_compatibility,
            "edge_bleed_texels": args.edge_bleed,
            "semantic_reliability": SEMANTIC_RELIABILITY,
        },
        "per_view": diagnostics,
        "maps": {name: str(output_dir / name) for name in (
            "ownership_map.png", "triangle_id_map.png", "uv_island_map.png",
            "observed_coverage_mask.png", "synthesized_coverage_mask.png",
            "confidence_map.png", "seam_map.png", "multiview_coverage_mask.png")},
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"MULTIVIEW_PROJECTION_DONE observed={coverage['directly_observed_percent']}% "
          f"multiview={coverage['multiview_observed_percent']}% "
          f"synth={coverage['synthesized_percent']}% "
          f"unresolved={coverage['unresolved_percent']}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
