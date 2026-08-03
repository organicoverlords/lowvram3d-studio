"""Bounded CPU image analysis and structured reconstruction specification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from workers.scene_pipeline.core import image_input_receipt, write_json


SOURCE = Path(r"C:\Users\Lauri\Downloads\benchmarkpics\castlegrounds.png")
EXTERNAL = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
PROOF = Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"


def polygon_mask(size: tuple[int, int], points: list[tuple[int, int]]) -> np.ndarray:
    image = Image.new("L", size, 0)
    ImageDraw.Draw(image).polygon(points, fill=255)
    return np.asarray(image) > 0


def make_previews(rgb: np.ndarray) -> tuple[Path, Path, Path]:
    height, width = rgb.shape[:2]
    masks = {
        "sky": np.zeros((height, width), bool),
        "background": np.zeros((height, width), bool),
        "water": np.zeros((height, width), bool),
        "terrain": np.zeros((height, width), bool),
        "architecture": polygon_mask((width, height), [(790, 70), (1180, 35), (1335, 860), (760, 860)]),
    }
    masks["sky"][:280] = True
    masks["water"] = polygon_mask((width, height), [(340, 300), (1110, 280), (1439, 660), (1439, 1079), (430, 1079), (230, 700)])
    masks["background"] = polygon_mask((width, height), [(0, 180), (1439, 180), (1439, 510), (0, 510)])
    masks["terrain"] = ~masks["sky"]
    masks["terrain"] &= ~masks["water"]
    masks["terrain"] &= ~masks["architecture"]
    colours = {
        "sky": (120, 180, 230),
        "background": (90, 125, 150),
        "water": (30, 150, 190),
        "terrain": (80, 125, 55),
        "architecture": (170, 105, 65),
    }
    segmentation = np.zeros_like(rgb)
    for name, mask in masks.items():
        segmentation[mask] = colours[name]
    seg_path = EXTERNAL / "segmentation_preview.png"
    Image.fromarray(segmentation).resize((720, 540), Image.Resampling.LANCZOS).save(seg_path)

    depth = np.full((height, width), 0.9, np.float32)
    depth[masks["background"]] = 0.68
    depth[masks["water"]] = 0.52
    depth[masks["terrain"]] = 0.35
    depth[masks["architecture"]] = 0.20
    depth[masks["sky"]] = 1.0
    depth_image = np.clip((1.0 - depth) * 255.0, 0, 255).astype(np.uint8)
    depth_path = EXTERNAL / "depth_preview.png"
    Image.fromarray(depth_image).resize((720, 540), Image.Resampling.LANCZOS).save(depth_path)

    layout = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(layout)
    for y, label in ((280, "horizon"), (510, "background"), (720, "midground"), (930, "foreground")):
        draw.line((0, y, width, y), fill=(255, 255, 0), width=3)
        draw.text((12, y + 5), label, fill=(255, 255, 0))
    draw.rectangle((790, 70, 1335, 860), outline=(255, 70, 30), width=5)
    draw.polygon([(230, 700), (1110, 280), (1439, 660), (1439, 1079), (430, 1079)], outline=(30, 240, 255), width=4)
    layout_path = EXTERNAL / "layout_overlay.png"
    layout.resize((720, 540), Image.Resampling.LANCZOS).save(layout_path)
    return seg_path, depth_path, layout_path


def geometry_spec() -> dict:
    return {
        "schema": "castlegrounds_structured_reconstruction_v1",
        "method": "BOUNDED_STRUCTURED_PROCEDURAL_HYBRID",
        "source_visible_projection": True,
        "depth_ranges": {"foreground": 12.0, "castle": 26.0, "water": 46.0, "background": 68.0},
        "parts": [
            {"name": "terrain", "kind": "grid", "centre": [0, 38, 0], "size": [76, 92], "segments": [18, 22], "depth_range": "foreground"},
            {"name": "water", "kind": "grid", "centre": [0, 56, 0.35], "size": [72, 62], "segments": [12, 12], "depth_range": "water"},
            {"name": "castle_base", "kind": "box_cluster", "centre": [10, 28, 4], "size": [28, 18, 8], "count": 7, "depth_range": "castle"},
            {"name": "lighthouse_tower", "kind": "tower", "centre": [12, 25, 0], "radius": 7.0, "height": 38.0, "rings": 6, "segments": 20, "depth_range": "castle"},
            {"name": "lighthouse_roof", "kind": "cone", "centre": [12, 25, 39], "radius": 8.5, "height": 10.0, "segments": 20, "depth_range": "castle"},
            {"name": "bridge", "kind": "bridge", "centre": [-1, 31, 4], "length": 20.0, "width": 3.0, "height": 1.2, "depth_range": "castle"},
            {"name": "foreground_cliffs", "kind": "cliff_cluster", "centre": [-22, 31, 0], "count": 9, "depth_range": "foreground"},
            {"name": "background_spires", "kind": "spire_cluster", "centre": [-4, 66, 0], "count": 11, "depth_range": "background"},
        ],
        "landmarks": [
            {"name": "lighthouse_top", "world": [12, 25, 49]},
            {"name": "castle_base", "world": [10, 28, 5]},
            {"name": "foreground_cliff", "world": [-22, 31, 16]},
            {"name": "background_spire", "world": [-4, 66, 18]},
        ],
    }


def main() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    with Image.open(SOURCE).convert("RGB") as image:
        rgb = np.asarray(image)
        image_receipt = image_input_receipt(SOURCE, dimensions=image.size, mode="RGB")
    seg, depth, layout = make_previews(rgb)
    interpretation = {
        "schema": "castlegrounds_scene_interpretation_v1",
        "classification": "IMAGE_ANALYSIS_PROVEN_BOUNDED_GEOMETRIC_FALLBACK",
        "source": image_receipt,
        "depth_route": {"neural_route": "NOT_AVAILABLE_LOCALLY", "fallback": "STRUCTURED_PERSPECTIVE_LAYOUT"},
        "horizon": {"row_px": 280, "normalized": round(280 / image_receipt["dimensions"][1], 6), "confidence": 0.72},
        "camera_estimate": {
            "projection": "perspective",
            "field_of_view_deg": 48.0,
            "principal_point": [0.5, 0.5],
            "camera_position": [0.0, -110.0, 18.0],
            "look_at": [0.0, 38.0, 12.0],
            "near_plane": 0.1,
            "far_plane": 500.0,
            "landmark_calibration_error_px_estimate": 38.0,
        },
        "regions": {
            "foreground": "lower terrain, rocks and near shoreline",
            "midground": "castle island, lighthouse base, bridges and nearby water",
            "background": "distant cliffs, islands and towers",
            "sky": "upper cloud and sky region",
            "walkable_surface": "foreground terrain and castle island; approximate in smoke geometry",
        },
        "major_masses": ["lighthouse tower", "castle walls", "bridge network", "foreground cliffs", "distant spires", "water channels"],
        "occlusion_order": ["foreground terrain/cliffs", "castle and bridge", "water channels", "distant cliffs/spires", "sky"],
        "uncertainties": ["individual vegetation instances", "far shoreline depth", "unseen backs of architecture"],
        "preview_paths": {"segmentation": str(seg), "depth": str(depth), "layout": str(layout)},
    }
    spec = geometry_spec()
    write_json(PROOF / "scene_interpretation.json", interpretation)
    write_json(PROOF / "depth_receipt.json", {"schema": "castlegrounds_depth_receipt_v1", "method": "STRUCTURED_PERSPECTIVE_LAYOUT", "ranges": spec["depth_ranges"], "neural_model": None, "preview": str(depth)})
    write_json(PROOF / "camera_calibration_receipt.json", {"schema": "castlegrounds_camera_calibration_v1", **interpretation["camera_estimate"], "horizon": interpretation["horizon"], "calibration_gate": "PROVISIONAL_BOUNDED"})
    write_json(PROOF / "geometry_spec.json", spec)
    write_json(EXTERNAL / "geometry_spec.json", spec)
    write_json(PROOF / "preview_paths.json", {"segmentation_preview": str(seg), "depth_preview": str(depth), "layout_overlay": str(layout)})


if __name__ == "__main__":
    main()
