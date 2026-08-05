"""Fresh-import verification of protected Base Color texels."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import bpy
import numpy as np

from common import argv_after_double_dash, import_mesh, reset_scene, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--expected-hash", default="")
    parser.add_argument("--basecolor", default="")
    parser.add_argument("--report", required=True)
    args = parser.parse_args(argv_after_double_dash())
    reset_scene()
    objects = import_mesh(args.glb)
    images = []
    for obj in objects:
        for slot in obj.material_slots:
            material = slot.material
            if not material or not material.use_nodes:
                continue
            bsdf = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
            if not bsdf:
                continue
            base = bsdf.inputs.get("Base Color")
            if base and base.is_linked and base.links[0].from_node.type == "TEX_IMAGE":
                image = base.links[0].from_node.image
                if image:
                    images.append(image)
    unique = {image.name: image for image in images}
    mask_image = bpy.data.images.load(str(Path(args.mask)))
    mask_width, mask_height = mask_image.size
    mask_pixels = np.asarray(mask_image.pixels[:], dtype=np.float32).reshape(mask_height, mask_width, 4)
    mask = mask_pixels[..., 0] > 0.01
    source_hash = args.expected_hash
    if args.basecolor:
        source_image = bpy.data.images.load(str(Path(args.basecolor)))
        sw, sh = source_image.size
        source_pixels = np.asarray(source_image.pixels[:], dtype=np.float32).reshape(sh, sw, 4)
        source_rgb = np.rint(np.clip(source_pixels[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
        if source_rgb.shape[:2] == mask.shape:
            source_hash = hashlib.sha256(source_rgb[mask].tobytes()).hexdigest()
    matches = []
    for image in unique.values():
        width, height = image.size
        pixels = np.asarray(image.pixels[:], dtype=np.float32).reshape(height, width, 4)[..., :3]
        rgb = np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)
        if rgb.shape[:2] != mask.shape:
            continue
        digest = hashlib.sha256(rgb[mask].tobytes()).hexdigest()
        matches.append({"image": image.name, "size": [width, height], "hash": digest,
                        "matches_expected": digest == source_hash})
    passed = bool(source_hash) and any(item["matches_expected"] for item in matches)
    report = {"glb": args.glb, "mask": args.mask, "expected_hash": source_hash,
              "active_basecolor_images": matches, "passed": passed}
    save_json(args.report, report)
    print(f"TEXTURE_PROTECTION_IMPORT passed={passed} images={matches}", flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
