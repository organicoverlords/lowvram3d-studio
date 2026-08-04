"""What gets generated, and from which pixels.

These cover the decisions the generation stage makes before any GPU is touched,
because those are the ones that quietly produce a wrong-looking scene: a crop
taken from the wrong window, twelve generations where one would do, or a surface
class handed to a generator that can only make objects.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lowvram3d.asset_generation import (  # noqa: E402
    MEASURED_PEAK_VRAM_MB, MIN_CROP_ASPECT, _border_contact, _crop,
    _square_pad, _widen_to_aspect, ladder_for_headroom, plan)
from lowvram3d.region_placement import ground_height_at, place  # noqa: E402


def segmentation(*regions):
    return {"image_dimensions": [1200, 900],
            "camera": {"fov_x_deg": 90.0},
            "regions": list(regions)}


def region(region_id, layer, label, bbox, depth=10.0):
    return {"id": region_id, "layer_type": layer, "semantic_label": label,
            "bbox_norm_xyxy": list(bbox),
            "depth_m": {"near": depth - 1, "median": depth, "far": depth + 1},
            "confidence": 0.8, "observed_fraction": 0.9}


def test_surfaces_are_not_generated():
    """Terrain and water are measured extents, not objects with a silhouette."""
    placement = place(segmentation(
        region("terrain_grass_001", "terrain", "grass", (0.0, 0.7, 1.0, 1.0)),
        region("water_lake_002", "water", "water", (0.0, 0.5, 0.5, 0.7)),
        region("sky_sky_003", "sky", "sky", (0.0, 0.0, 1.0, 0.5)),
    ))
    assert plan(placement) == []


def test_architecture_is_generated_from_its_own_bbox():
    bbox = (0.3, 0.5, 0.8, 0.75)
    placement = place(segmentation(
        region("architecture_house_004", "architecture", "house", bbox)))
    jobs = plan(placement)
    assert len(jobs) == 1
    assert jobs[0]["asset_id"] == "architecture_house_004"
    assert jobs[0]["crop_bbox_norm_xyxy"] == pytest.approx(list(bbox))
    assert jobs[0]["shared_across_instances"] is False


def test_scatter_region_generates_once_for_every_instance():
    """A tree line is one generation reused, not one generation per tree."""
    placement = place(segmentation(
        region("vegetation_tree_005", "vegetation", "tree", (0.0, 0.1, 0.9, 0.7))))
    instances = [a for a in placement["actors"] if a["kind"] == "scatter_instance"]
    assert len(instances) > 1

    jobs = plan(placement)
    assert len(jobs) == 1
    assert jobs[0]["shared_across_instances"] is True
    assert jobs[0]["instance_count"] == len(instances)
    assert len(jobs[0]["actor_indices"]) == len(instances)


def test_scatter_crop_is_a_subject_not_the_whole_line():
    """The crop must be narrower than the region but wide enough to read."""
    region_bbox = (0.0, 0.1, 0.9, 0.7)
    placement = place(segmentation(
        region("vegetation_tree_005", "vegetation", "tree", region_bbox)))
    crop = plan(placement)[0]["crop_bbox_norm_xyxy"]

    assert crop[2] - crop[0] < (region_bbox[2] - region_bbox[0])
    assert crop[0] >= region_bbox[0] and crop[2] <= region_bbox[2]
    aspect = (crop[2] - crop[0]) / (crop[3] - crop[1])
    assert aspect >= MIN_CROP_ASPECT - 1e-9


def test_widening_keeps_the_window_centred_and_inside_bounds():
    widened = _widen_to_aspect([0.46, 0.05, 0.54, 0.72], [0.0, 0.05, 0.92, 0.72])
    assert widened[0] < 0.46 and widened[2] > 0.54
    assert (widened[0] + widened[2]) / 2 == pytest.approx(0.5)
    assert widened[1] == 0.05 and widened[3] == 0.72


def test_widening_never_escapes_the_region():
    """Clamping on one side must not push the window past the other."""
    bounds = [0.0, 0.0, 0.10, 0.9]
    widened = _widen_to_aspect([0.0, 0.0, 0.02, 0.9], bounds)
    assert widened[0] >= bounds[0] and widened[2] <= bounds[2]


def test_widening_leaves_a_healthy_window_alone():
    bbox = [0.3, 0.5, 0.8, 0.75]
    assert _widen_to_aspect(bbox, [0.0, 0.0, 1.0, 1.0]) == bbox


PLANE = {"slope_forward": -0.17564, "slope_right": 0.02765,
         "height_at_camera_m": -1.137}


def clustered(region_id, clusters):
    """A vegetation region already split into clumps by segmentation."""
    base = region("vegetation_tree_001", "vegetation", "tree", (0.0, 0.1, 0.9, 0.7))
    base["id"] = region_id
    base["clusters"] = clusters
    return base


def clump(forward, right, up, size, pixels=5000):
    half = [s / 2 for s in size]
    return {
        "pixel_count": pixels,
        "bbox_norm_xyxy": [0.4, 0.2, 0.5, 0.5],
        "depth_m": {"near": forward - 1, "median": forward, "far": forward + 1},
        "measured_unreal_m": {
            "centroid": [forward, right, up],
            "min": [forward - half[0], right - half[1], up - half[2]],
            "max": [forward + half[0], right + half[1], up + half[2]],
            "size": list(size),
        },
    }


def test_ground_plane_slopes_away_from_the_camera():
    """A constant height would be wrong by metres at one end or the other."""
    near = ground_height_at(PLANE, 2.3, 0.0)
    far = ground_height_at(PLANE, 13.5, 0.0)
    assert near > far
    assert far == pytest.approx(-3.51, abs=0.02)


def test_no_plane_means_no_grounding_guess():
    assert ground_height_at(None, 10.0, 0.0) is None


def test_a_canopy_above_the_ground_gets_a_trunk():
    """Clustering finds canopy; trunks are occluded, so they are inferred."""
    segmentation_with_plane = {
        **segmentation(clustered("vegetation_tree_001",
                                 [clump(10.0, 0.0, 2.0, (2.0, 2.0, 2.0))])),
        "ground_plane_unreal": PLANE,
    }
    actors = place(segmentation_with_plane)["actors"]
    trunks = [a for a in actors if a["kind"] == "trunk_support"]
    assert len(trunks) == 1
    # Thin relative to the crown, and spanning ground to canopy base.
    assert trunks[0]["size_m"][0] < 1.0
    assert trunks[0]["size_m"][2] == pytest.approx(1.0 - ground_height_at(PLANE, 10.0, 0.0), abs=0.01)
    assert "inferred" in trunks[0]


def test_a_canopy_already_on_the_ground_gets_no_trunk():
    segmentation_with_plane = {
        **segmentation(clustered("vegetation_tree_001",
                                 [clump(10.0, 0.0, -2.4, (2.0, 2.0, 2.0))])),
        "ground_plane_unreal": PLANE,
    }
    actors = place(segmentation_with_plane)["actors"]
    assert not [a for a in actors if a["kind"] == "trunk_support"]


def test_a_trunk_that_would_pierce_a_building_is_withdrawn():
    """The canopy centroid is not the trunk's position, and crowns overhang.

    Where the guess collides with something that *was* observed, it is not a
    guess worth making.
    """
    spec = segmentation(
        clustered("vegetation_tree_001", [clump(10.0, 0.0, 4.0, (2.0, 2.0, 2.0))]),
        region("architecture_barn_002", "architecture", "house",
               (0.3, 0.4, 0.7, 0.75), depth=10.0),
    )
    spec["ground_plane_unreal"] = PLANE
    # Put the building exactly where the trunk would descend.
    spec["regions"][1]["measured_unreal_m"] = {
        "centroid": [10.0, 0.0, -1.0], "min": [8.0, -4.0, -3.0],
        "max": [12.0, 4.0, 1.0], "size": [4.0, 8.0, 4.0]}
    result = place(spec)
    assert result["withdrawn_trunk_supports"] == 1
    assert not [a for a in result["actors"] if a["kind"] == "trunk_support"]


LADDER = "384:3000,320:2000,256:1500"


def test_full_ladder_survives_when_the_card_has_room():
    assert ladder_for_headroom(LADDER, 6000) == (LADDER, [])


def test_unaffordable_rungs_are_dropped():
    """Starting a rung without the memory for it costs the whole generation.

    On this card it does not fail as OutOfMemoryError -- it fails as a
    misaligned address minutes in, having spent the run.
    """
    top = max(MEASURED_PEAK_VRAM_MB)
    ladder, dropped = ladder_for_headroom(LADDER, MEASURED_PEAK_VRAM_MB[top] - 1)
    assert not ladder.startswith(f"{top}:")
    assert [d["rung"] for d in dropped] == [f"{top}:3000"]


def test_rungs_with_no_measured_requirement_are_never_dropped():
    """Only drop a rung there is evidence against."""
    assert 256 not in MEASURED_PEAK_VRAM_MB
    ladder, _ = ladder_for_headroom(LADDER, 10)
    assert ladder.startswith("256:")


def test_every_dropped_rung_has_a_recorded_requirement():
    _, dropped = ladder_for_headroom(LADDER, 10)
    for entry in dropped:
        resolution = int(entry["rung"].partition(":")[0])
        assert MEASURED_PEAK_VRAM_MB[resolution] == entry["needs_mb"]


def test_unreadable_vram_leaves_the_ladder_alone():
    assert ladder_for_headroom(LADDER, None) == (LADDER, [])


def test_the_ladder_is_never_emptied():
    """Better to fail honestly at the smallest rung than to skip the asset."""
    ladder, _ = ladder_for_headroom("384:3000", 100)
    assert ladder == "384:3000"


def synthetic_source(tmp_path, subject_box, size=(1200, 900)):
    """A source image plus a mask covering exactly `subject_box` (pixels)."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", size, (120, 140, 90))
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(subject_box, fill=255)
    ImageDraw.Draw(image).rectangle(subject_box, fill=(70, 50, 40))
    image_path, mask_path = tmp_path / "source.png", tmp_path / "mask.png"
    image.save(image_path)
    mask.save(mask_path)
    return image_path, mask_path


def test_a_letterboxed_subject_is_framed_square_before_generation():
    """The generator pads to square regardless; doing it here keeps the subject big.

    A 3:1 barn handed over as-is occupies a third of the conditioning frame, and
    the coarse reconstruction that follows reads as a bad generator.
    """
    from PIL import Image

    framed = _square_pad(Image.new("RGBA", (819, 266), (0, 0, 0, 255)))
    assert framed.size[0] == framed.size[1]
    assert framed.size[0] >= 819
    # The subject stays centred, and the padding is transparent rather than a
    # colour the generator would try to reconstruct.
    assert framed.getpixel((framed.size[0] // 2, framed.size[1] // 2))[3] == 255
    assert framed.getpixel((0, 0))[3] == 0


def test_a_whole_subject_touches_none_of_its_own_edges(tmp_path):
    """Crops are padded so a real object clears its frame -- the barn scores 0.0."""
    image, mask = synthetic_source(tmp_path, (400, 500, 900, 700))
    result = _crop(image, [400 / 1200, 500 / 900, 900 / 1200, 700 / 900],
                   tmp_path / "crop.png", mask)
    assert result["unbounded_crop"] is False
    assert max(result["border_contact"].values()) == 0.0
    assert result["conditioning_size_px"][0] == result["conditioning_size_px"][1]


def test_a_slice_of_a_larger_mass_is_refused(tmp_path):
    """A chunk of hedge has no silhouette; generating from it produces a slab."""
    # Subject extends well past the crop on the left and the bottom.
    image, mask = synthetic_source(tmp_path, (0, 100, 700, 900))
    result = _crop(image, [300 / 1200, 100 / 900, 700 / 1200, 600 / 900],
                   tmp_path / "crop.png", mask)
    assert set(result["unbounded_sides"]) >= {"left", "bottom"}
    assert result["unbounded_crop"] is True


def test_border_contact_is_measured_per_edge():
    from PIL import Image, ImageDraw

    mask = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(mask).rectangle((0, 0, 49, 99), fill=255)
    contact = _border_contact(mask)
    assert contact["left"] == 1.0
    assert contact["right"] == 0.0
    assert contact["top"] == pytest.approx(0.5, abs=0.01)


def test_every_generated_actor_is_accounted_for():
    """Each job's actor indices must address real actors of that region."""
    placement = place(segmentation(
        region("architecture_house_004", "architecture", "house", (0.3, 0.5, 0.8, 0.75)),
        region("vegetation_tree_005", "vegetation", "tree", (0.0, 0.1, 0.9, 0.7)),
        region("terrain_grass_001", "terrain", "grass", (0.0, 0.7, 1.0, 1.0)),
    ))
    actors = placement["actors"]
    for job in plan(placement):
        assert job["actor_indices"]
        for index in job["actor_indices"]:
            assert actors[index]["region_id"] == job["region_id"]
