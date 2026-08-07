"""Background key for the shaman anchor that preserves detached subject parts.

The rembg/u2net matte used by the default Mini Turbo path destroys this particular source: it
drops every ornament hanging on a cord from the antler pole, erases the cords themselves, and
collapses the staff's ring head into an opaque black blob. Those failures then show up in the
generated geometry as missing silhouette features plus floating debris.

The source is a studio plate with a smooth, near-uniform background, so a far more faithful matte
is a border flood-fill key: classify pixels close to the background colour, keep only the
background-coloured regions that are *connected to the image border*, and treat everything else as
subject. Interior detail and fully detached ornaments both survive, because neither touches the
border.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


def border_background_colour(rgb: np.ndarray, band: int = 8) -> np.ndarray:
    edges = np.concatenate(
        [
            rgb[:band].reshape(-1, 3),
            rgb[-band:].reshape(-1, 3),
            rgb[:, :band].reshape(-1, 3),
            rgb[:, -band:].reshape(-1, 3),
        ]
    )
    return np.median(edges, axis=0)


def _feather_alpha(
    rgb: np.ndarray,
    subject: np.ndarray,
    background: np.ndarray,
    feather: int,
    excluded: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Recover fractional coverage in a narrow band around the silhouette.

    A binary key answers "is this pixel subject" when the honest answer for an
    edge pixel is "partly". Rigging, railings, a flag on a pole and every
    antialiased outline are made of pixels the camera filled somewhere between
    0 and 1, and rounding them destroys thin structure in one direction and
    builds a stair-stepped outline in the other.

    Each such pixel is a mix of a foreground colour and the plate:

        C = a*F + (1 - a)*B

    B is known -- it is the plate colour already measured from the border. F is
    not, so it is taken from the nearest pixel that is confidently interior,
    which is the standard assumption that a boundary pixel is a mixture of the
    plate and whatever foreground it abuts. Projecting C onto that line gives

        a = clamp(((C - B) . (F - B)) / |F - B|^2, 0, 1)

    Only pixels within `feather` of the boundary are solved. The interior stays
    at 1 and the far plate at 0, so this cannot perforate the subject or revive
    background -- the failure mode of a global unmix.

    Where F is close to B the projection divides by a near-zero length and the
    estimate is meaningless. Those pixels keep their binary value rather than
    taking a fabricated fraction, and they are counted in the receipt.
    """
    if feather <= 0:
        return (np.where(subject, 255, 0).astype(np.uint8),
                {"feather_radius": 0, "partial_alpha_fraction": 0.0,
                 "feather_band_pixels": 0, "feather_degenerate_pixels": 0})

    structure = ndimage.generate_binary_structure(2, 2)
    # Erode by one, not by `feather`. Eroding the full radius would delete any
    # structure thinner than 2*feather+1 from the interior entirely -- exactly
    # the masts, rigging and railings this is meant to rescue -- and then the
    # nearest-interior lookup would hand them a foreground colour borrowed from
    # an unrelated part of the subject. One pixel is enough to step off the
    # antialiased boundary, and it keeps anything at least three pixels wide.
    interior = ndimage.binary_erosion(subject, structure=structure, iterations=1)

    # Structure thinner than three pixels erodes away completely, so it has no
    # interior of its own to sample. The nearest-interior lookup is global -- it
    # does not respect connected components -- so such a structure would silently
    # borrow its foreground colour from whatever unrelated part of the subject
    # happens to be closest. Measured: an isolated one-pixel red mast took its
    # colour from a distant dark body and keyed to alpha 173 instead of 255,
    # i.e. a solid structure rendered two-thirds opaque.
    #
    # Any subject pixel that no interior can reach therefore becomes its own
    # colour source. For a one-pixel mast that is exactly right: the only honest
    # estimate of its foreground colour is the pixel itself.
    reachable = ndimage.binary_dilation(
        interior, structure=structure, iterations=max(feather, 1))
    core = interior | (subject & ~reachable)

    outside = ~ndimage.binary_dilation(subject, structure=structure, iterations=feather)
    band = ~(interior | outside)

    # Nearest core pixel, for the foreground colour. The distance transform runs
    # on the complement, so the indices it returns point at the closest True in
    # `core`.
    if not core.any():
        return (np.where(subject, 255, 0).astype(np.uint8),
                {"feather_radius": int(feather), "partial_alpha_fraction": 0.0,
                 "feather_band_pixels": int(band.sum()),
                 "feather_degenerate_pixels": int(band.sum())})
    _, indices = ndimage.distance_transform_edt(~core, return_indices=True)
    nearest_foreground = rgb[indices[0], indices[1]].astype(np.float32)

    colour = rgb.astype(np.float32)
    plate = background.astype(np.float32)
    direction = nearest_foreground - plate
    denominator = (direction * direction).sum(axis=-1)
    numerator = ((colour - plate) * direction).sum(axis=-1)

    # 8.0 is a squared colour distance, so a foreground within ~2.8 levels of the
    # plate per channel. Below that the projection is noise.
    usable = band & (denominator > 8.0)
    estimate = np.zeros(subject.shape, dtype=np.float32)
    estimate[usable] = np.clip(numerator[usable] / denominator[usable], 0.0, 1.0)

    coverage = np.where(subject, 1.0, 0.0).astype(np.float32)
    coverage[usable] = estimate[usable]

    # Islands the caller already decided to drop sit inside the dilated band, so
    # the solver happily assigns them coverage again -- measured at a full 255,
    # not merely a faint ghost, because a speck's own colour projects cleanly
    # onto its own direction. Feathering would then quietly undo the island drop.
    # The exclusion is applied last so nothing downstream can revive them.
    if excluded is not None:
        coverage[excluded] = 0.0

    alpha = np.round(coverage * 255.0).astype(np.uint8)

    partial = (alpha > 0) & (alpha < 255)
    return alpha, {
        "feather_radius": int(feather),
        "feather_band_pixels": int(band.sum()),
        "feather_degenerate_pixels": int((band & ~usable).sum()),
        "partial_alpha_fraction": round(float(partial.mean()), 6),
    }


def key_alpha(
    rgb: np.ndarray,
    tolerance: float,
    mode: str = "hybrid",
    enclosed_min_area: int = 1500,
    enclosed_tolerance: float | None = None,
    shadow_tolerance: float | None = None,
    shadow_from: float = 0.82,
    shadow_chroma: float | None = None,
    shadow_luma: float = 90.0,
    shadow_smooth: float = 3.0,
    shadow_smooth_window: int = 9,
    close_radius: int = 2,
    min_detached_fraction: float = 0.001,
    feather: int = 2,
) -> tuple[np.ndarray, dict]:
    background = border_background_colour(rgb)
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=-1)

    # The cast shadow on the floor is background that is too dark for the plate tolerance. It only
    # ever occupies the bottom band, where the sole subject content is the dark bird feet, so a
    # looser threshold applied to those rows removes the shadow without touching the character.
    threshold = np.full(distance.shape, float(tolerance), dtype=np.float32)
    if shadow_tolerance is not None and shadow_tolerance > tolerance:
        first_row = int(distance.shape[0] * shadow_from)
        threshold[first_row:] = float(shadow_tolerance)
    candidate = distance < threshold

    # Distance from the plate colour cannot always separate the shadow from what casts it. Measured
    # on the frog, whose feet are dark and stand in their own contact shadow:
    #
    #     shadow      chroma p95   4     distance p50   13 (soft) .. 238 (contact)
    #     feet        chroma p50  15-29  distance p50  340-378
    #
    # The distance ranges overlap -- the contact shadow is darker than parts of the feet -- so any
    # single tolerance either keeps the shadow or punches holes in the toes, and at 220 it does
    # both. Chroma does not overlap: a neutral-grey shadow on a neutral plate stays achromatic
    # however dark it gets, while the subject is pigmented. So key the band on being achromatic AND
    # not fully dark, rather than on being close to white.
    #
    # This is a property of the lighting setup -- one subject, white sweep, soft key -- and not of
    # any one asset. It stays gated behind border connectivity below, so an achromatic pixel inside
    # the subject is only ever removed if it belongs to a genuine enclosed background pocket.
    neutral_shadow = np.zeros(distance.shape, dtype=bool)
    if shadow_chroma is not None:
        first_row = int(distance.shape[0] * shadow_from)
        values = rgb[first_row:].astype(np.float32)
        chroma = values.max(axis=-1) - values.min(axis=-1)
        luminance = values.mean(axis=-1)

        # Chroma alone is not enough at the contact point, where the shadow is both dark and warm
        # from bounce off the subject: measured there it is lum 56 chroma 9, which overlaps the feet
        # instead of separating from them. What does separate is that a cast shadow is smooth and a
        # photographed surface is not. Local standard deviation over a 9px window, measured:
        #
        #     shadow deep / soft     0.9 - 1.5
        #     left / right foot      5.0 - 7.7
        #
        # At std<=3 with lum>=55 this takes 67% of the deep shadow and 80% of the soft shadow while
        # taking 0.4% and 0.0% of the two feet. The luminance floor is what keeps the robe fringe,
        # which is just as smooth as the shadow but much darker (lum 29).
        window = ndimage.uniform_filter(luminance, shadow_smooth_window)
        squares = ndimage.uniform_filter(luminance * luminance, shadow_smooth_window)
        roughness = np.sqrt(np.clip(squares - window * window, 0.0, None))

        neutral_shadow[first_row:] = ((luminance >= float(shadow_luma)) &
                                      (chroma <= float(shadow_chroma)) &
                                      (roughness <= float(shadow_smooth)))
        candidate |= neutral_shadow

    if mode == "hybrid":
        # Border-connected background at a generous tolerance removes the plate and the soft floor
        # shadow without risk to the character, because interior pixels are only reachable through
        # a background-coloured path. Raising tolerance here cannot perforate the body.
        labels, count = ndimage.label(candidate)
        border_labels = set(labels[0].tolist()) | set(labels[-1].tolist())
        border_labels |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
        border_labels.discard(0)
        background_mask = np.isin(labels, list(border_labels)) if border_labels else np.zeros_like(candidate)

        # The pockets enclosed by the antler pole, the hanging cords and the staff ring are real
        # background but never touch the border. Remove them by area so that genuine background
        # panels key out while the small background-coloured speckles inside the robes - which are
        # what perforates the body in a pure colour key - stay part of the subject.
        # Enclosed background needs a tight threshold across the torso, where a loose one punches
        # holes through pale robe fabric, but a loose one in the floor band, where the shadow
        # pocket trapped between the legs is bright and large. Reuse the same row-dependent
        # profile as the plate key.
        enclosed_threshold = np.full(
            distance.shape,
            float(enclosed_tolerance if enclosed_tolerance is not None else tolerance),
            dtype=np.float32,
        )
        if shadow_tolerance is not None:
            enclosed_threshold[int(distance.shape[0] * shadow_from):] = float(shadow_tolerance)
        # The neutral-shadow test has to apply here too, not only to the border flood. The deepest
        # part of a contact shadow is walled off by the feet on both sides, so it is enclosed rather
        # than border-connected, and it sits 100-260 from the plate colour -- far outside any
        # enclosed distance threshold that is safe elsewhere. Keying it on distance is what fails;
        # keying it on being achromatic works regardless of how dark it gets.
        enclosed = ndimage.label(
            ((distance < enclosed_threshold) | neutral_shadow) & ~background_mask)
        enclosed_labels, enclosed_count = enclosed
        if enclosed_count:
            areas = ndimage.sum(
                np.ones_like(enclosed_labels, dtype=bool),
                enclosed_labels,
                range(1, enclosed_count + 1),
            )
            large = [i + 1 for i, area in enumerate(areas) if area >= enclosed_min_area]
            if large:
                background_mask |= np.isin(enclosed_labels, large)
    elif mode == "flood":
        # Border-connected background only. Preserves interior detail but also traps genuine
        # background in the pockets enclosed by the antler pole, the hanging cords and the staff,
        # which then survives as grey slab geometry.
        labels, count = ndimage.label(candidate)
        border_labels = set(labels[0].tolist()) | set(labels[-1].tolist())
        border_labels |= set(labels[:, 0].tolist()) | set(labels[:, -1].tolist())
        border_labels.discard(0)
        background_mask = np.isin(labels, list(border_labels)) if border_labels else np.zeros_like(candidate)
    else:
        # Pure colour key. Every background-coloured pixel is background wherever it sits, so the
        # enclosed pockets and the ring hole in the staff head all key out correctly. The subject
        # is textured and saturated enough that nothing on the character falls inside tolerance.
        count = 0
        background_mask = candidate

    subject = ~background_mask
    if close_radius > 0:
        # Pale, desaturated parts of the character - the bone collar, the bone ornaments - sit
        # close enough to the plate colour that the border flood nibbles speckle holes into them.
        # A closing pass heals those intrusions. It runs on the subject mask, so the genuine large
        # openings (the staff ring, the gaps between cords) are far too big to be closed.
        structure = ndimage.generate_binary_structure(2, 2)
        subject = ndimage.binary_closing(subject, structure=structure, iterations=close_radius)
        # Closing cannot recover a speckle that punched clean through, so also fill any remaining
        # enclosed hole below the area at which an opening is considered genuine.
        holes = ndimage.label(~subject)
        hole_labels, hole_count = holes
        if hole_count:
            hole_areas = ndimage.sum(
                np.ones_like(hole_labels, dtype=bool), hole_labels, range(1, hole_count + 1)
            )
            border_hole = set(hole_labels[0].tolist()) | set(hole_labels[-1].tolist())
            border_hole |= set(hole_labels[:, 0].tolist()) | set(hole_labels[:, -1].tolist())
            small = [
                i + 1
                for i, area in enumerate(hole_areas)
                if area < enclosed_min_area and (i + 1) not in border_hole
            ]
            if small:
                subject |= np.isin(hole_labels, small)

    # Speckle that survived the key as its own island. The docstring above defends
    # keeping detached parts, and that defence is sound for real ornaments -- but
    # measured on both assets this repo uses, the parts it was protecting are not
    # detached at all. On the shaman the ornaments hang from cords that keep them
    # connected: the main component is 547180 px and all 31 others are <= 27 px.
    # On the boat there is exactly one island, 67 px, floating below the hull.
    #
    # That one costs something specific. The bake registers the photograph by
    # fitting its bounding box to the silhouette's, and the island drags the box
    # from y=460 to y=481 -- 21 px on a 473 px subject, a 4.4% vertical stretch
    # applied to the photograph.
    #
    # The floor is a fraction of the largest component rather than a pixel count,
    # so it holds across the 4x resolution range between these two sources. At
    # 1e-3 it clears the boat's island (3.9e-4) and the shaman's worst speck
    # (5e-5) while preserving anything genuinely detached above a thousandth of
    # the body.
    dropped_components = 0
    dropped_pixels = 0
    doomed_mask = None
    if min_detached_fraction > 0:
        island_labels, island_count = ndimage.label(subject)
        if island_count > 1:
            island_sizes = ndimage.sum(
                subject, island_labels, range(1, island_count + 1))
            largest = float(island_sizes.max())
            doomed = [i + 1 for i, size in enumerate(island_sizes)
                      if size < largest * min_detached_fraction]
            if doomed:
                doomed_mask = np.isin(island_labels, doomed)
                dropped_components = len(doomed)
                dropped_pixels = int(doomed_mask.sum())
                subject &= ~doomed_mask

    alpha, feather_stats = _feather_alpha(
        rgb, subject, background, feather, excluded=doomed_mask)

    kept_labels, kept_count = ndimage.label(subject)
    sizes = ndimage.sum(subject, kept_labels, range(1, kept_count + 1))
    stats = {
        "tolerance": tolerance,
        "background_rgb": [float(v) for v in background],
        "candidate_regions": int(count),
        "subject_pixel_fraction": float(subject.mean()),
        "subject_components": int(kept_count),
        "largest_component_fraction": float(sizes.max() / subject.sum()) if kept_count else 0.0,
        "detached_components": int((sizes >= 40).sum() - 1) if kept_count else 0,
        "min_detached_fraction": float(min_detached_fraction),
        "dropped_island_components": dropped_components,
        "dropped_island_pixels": dropped_pixels,
        **feather_stats,
    }
    return alpha, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=float, default=42.0)
    parser.add_argument("--preview", default="")
    parser.add_argument("--stats-json", default="")
    parser.add_argument("--sweep", default="", help="comma separated tolerances to preview instead of writing output")
    parser.add_argument("--enclosed-min-area", type=int, default=1500)
    parser.add_argument("--enclosed-tolerance", type=float, default=None)
    parser.add_argument("--shadow-tolerance", type=float, default=None)
    parser.add_argument("--shadow-from", type=float, default=0.82)
    parser.add_argument(
        "--shadow-chroma", type=float, default=None,
        help="In the shadow band, also key pixels whose max-min RGB spread is at "
             "or below this and whose mean is at least --shadow-luma. Separates a "
             "neutral cast shadow from a pigmented subject standing in it, which "
             "--shadow-tolerance cannot do once the contact shadow is darker than "
             "the subject. Measured: shadow chroma p95 4, feet p50 15-29.")
    parser.add_argument("--shadow-luma", type=float, default=90.0,
                        help="Mean-RGB floor for --shadow-chroma, so that a dark "
                             "achromatic part of the subject is never keyed. This "
                             "is what protects smooth dark cloth: the frog's robe "
                             "fringe is as smooth as the shadow but sits at 29.")
    parser.add_argument("--shadow-smooth", type=float, default=3.0,
                        help="Local standard deviation ceiling for the shadow key. "
                             "A cast shadow is smooth (0.9-1.5) and a photographed "
                             "surface is not (feet 5.0-7.7), which separates them "
                             "at the contact point where chroma and distance both "
                             "overlap.")
    parser.add_argument("--shadow-smooth-window", type=int, default=9,
                        help="Window in pixels for --shadow-smooth.")
    parser.add_argument("--close-radius", type=int, default=2)
    parser.add_argument("--mode", choices=("hybrid", "colour", "flood"), default="hybrid")
    parser.add_argument(
        "--min-detached-fraction", type=float, default=0.001,
        help="Drop subject islands smaller than this fraction of the largest "
             "component. 0 disables. Measured: boat speck 3.9e-4, shaman worst "
             "speckle 5e-5, so 1e-3 clears both and keeps real detached parts.")
    parser.add_argument(
        "--feather", type=int, default=2,
        help="Radius in pixels of the band where fractional coverage is solved "
             "from the compositing equation. 0 restores the binary key.")
    args = parser.parse_args()

    source = Image.open(args.image).convert("RGB")
    rgb = np.asarray(source)

    if args.sweep:
        report = []
        for raw in args.sweep.split(","):
            tolerance = float(raw)
            alpha, stats = key_alpha(rgb, tolerance, args.mode, args.enclosed_min_area, args.enclosed_tolerance, args.shadow_tolerance, args.shadow_from, args.shadow_chroma, args.shadow_luma, args.shadow_smooth, args.shadow_smooth_window, args.close_radius, args.min_detached_fraction, args.feather)
            preview = Image.new("RGB", source.size, (255, 0, 255))
            preview.paste(source, mask=Image.fromarray(alpha, "L"))
            path = Path(args.output).with_name(f"matte_{args.mode}_t{int(tolerance)}.png")
            preview.save(path)
            stats["preview"] = str(path)
            report.append(stats)
            print(
                f"tolerance={tolerance:5.1f} subject={stats['subject_pixel_fraction']*100:5.2f}% "
                f"components={stats['subject_components']:5d} detached>=40px={stats['detached_components']:4d} "
                f"largest={stats['largest_component_fraction']*100:5.2f}% -> {path.name}",
                flush=True,
            )
        if args.stats_json:
            Path(args.stats_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 0

    alpha, stats = key_alpha(rgb, args.tolerance, args.mode, args.enclosed_min_area, args.enclosed_tolerance, args.shadow_tolerance, args.shadow_from, args.shadow_chroma, args.shadow_luma, args.shadow_smooth, args.shadow_smooth_window, args.close_radius, args.min_detached_fraction, args.feather)
    # Neutralise the colour under the transparent region. Leaving the original plate colour there
    # lets any consumer that flattens without honouring alpha reintroduce the background.
    keyed = np.where(alpha[..., None] > 0, rgb, 255).astype(np.uint8)

    # A partially covered pixel holds a mixture, C = a*F + (1-a)*B, but alpha
    # compositing will itself apply the (1-a)*B term against whatever background
    # it draws onto. Storing C unchanged therefore counts the plate twice and
    # leaves a pale halo -- the classic fringe around a feathered cut-out. Solve
    # the same equation the other way for the colour that belongs there:
    #
    #     F = B + (C - B) / a
    #
    # Only for pixels with enough coverage to divide by; below that the estimate
    # amplifies noise faster than it removes fringe, and the mixed colour is the
    # better of two imperfect answers.
    if args.feather > 0:
        coverage = alpha.astype(np.float32) / 255.0
        unmixable = (coverage > 0.15) & (coverage < 1.0)
        if unmixable.any():
            plate = np.asarray(stats["background_rgb"], dtype=np.float32)
            # Divide only where the division is defined. Computing across the
            # whole frame and masking afterwards gives the same answer but walks
            # through a divide-by-zero on every fully transparent pixel, which
            # fills the array with inf and NaN and emits warnings that would
            # train the reader to ignore warnings.
            safe = np.where(unmixable, coverage, 1.0)[..., None]
            recovered = plate + (rgb.astype(np.float32) - plate) / safe
            keyed = np.where(unmixable[..., None],
                             np.clip(recovered, 0, 255), keyed).astype(np.uint8)
        stats["unmixed_pixels"] = int(unmixable.sum())
    rgba = np.dstack([keyed, alpha])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(output)
    if args.preview:
        preview = Image.new("RGB", source.size, (255, 0, 255))
        preview.paste(source, mask=Image.fromarray(alpha, "L"))
        Image.fromarray(np.asarray(preview)).save(args.preview)
    if args.stats_json:
        Path(args.stats_json).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"MATTE_WRITTEN {output} subject={stats['subject_pixel_fraction']*100:.2f}% "
        f"components={stats['subject_components']} detached={stats['detached_components']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




