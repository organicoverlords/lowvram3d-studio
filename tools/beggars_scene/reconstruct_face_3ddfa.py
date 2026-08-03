from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct the Antinous reference face with pinned 3DDFA_V2.")
    parser.add_argument("--third-party-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-obj", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--output-meta", required=True)
    return parser.parse_args()


def sample_vertex_colors(image_bgr: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    x = np.clip(np.rint(vertices[0]).astype(np.int32), 0, width - 1)
    y = np.clip(np.rint(vertices[1]).astype(np.int32), 0, height - 1)
    bgr = image_bgr[y, x].astype(np.float32) / 255.0
    return bgr[:, ::-1].copy()


def main() -> int:
    args = parse_args()
    third_party = Path(args.third_party_root).resolve()
    image_path = Path(args.image).resolve()
    output_obj = Path(args.output_obj).resolve()
    output_npz = Path(args.output_npz).resolve()
    output_meta = Path(args.output_meta).resolve()

    if not third_party.is_dir():
        raise SystemExit(f"3DDFA root is missing: {third_party}")
    if not image_path.is_file():
        raise SystemExit(f"Reference image is missing: {image_path}")

    sys.path.insert(0, str(third_party))
    os.chdir(third_party)

    from TDDFA_ONNX import TDDFA_ONNX  # pylint: disable=import-error,import-outside-toplevel

    config_path = third_party / "configs" / "mb1_120x120.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["checkpoint_fp"] = str((third_party / config["checkpoint_fp"]).resolve())
    config["bfm_fp"] = str((third_party / config["bfm_fp"]).resolve())
    config["param_mean_std_fp"] = str(
        (third_party / "configs" / "param_mean_std_62d_120x120.pkl").resolve()
    )

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"OpenCV could not read: {image_path}")
    height, width = image.shape[:2]

    # The public banner is 1000x563. These normalized bounds tightly cover the visible face
    # while excluding the caption and background. Scaling keeps the route stable if the CDN
    # serves a resized copy.
    roi = [
        0.345 * width,
        0.045 * height,
        0.725 * width,
        0.805 * height,
        1.0,
    ]

    tddfa = TDDFA_ONNX(**config)
    param_list, roi_list = tddfa(image, [roi])
    vertex_list = tddfa.recon_vers(param_list, roi_list, dense_flag=True)
    if len(vertex_list) != 1:
        raise SystemExit(f"Expected one reconstructed face, received {len(vertex_list)}")

    vertices_image = vertex_list[0].astype(np.float32)
    triangles = np.asarray(tddfa.tri, dtype=np.int32)
    colors_rgb = sample_vertex_colors(image, vertices_image)

    # Store OBJ in an upright image-aligned coordinate system. Blender performs the final
    # image-space-to-world mapping so the source shot can be matched deterministically.
    vertices_obj = vertices_image.copy()
    vertices_obj[1, :] = height - vertices_obj[1, :]

    output_obj.parent.mkdir(parents=True, exist_ok=True)
    with output_obj.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Pinned 3DDFA_V2 reconstruction with per-vertex RGB\n")
        for index in range(vertices_obj.shape[1]):
            x, y, z = vertices_obj[:, index]
            r, g, b = colors_rgb[index]
            handle.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.6f} {g:.6f} {b:.6f}\n")
        for a, b, c in triangles:
            handle.write(f"f {int(c) + 1} {int(b) + 1} {int(a) + 1}\n")

    np.savez_compressed(
        output_npz,
        vertices_image=vertices_image,
        vertices_obj=vertices_obj,
        triangles=triangles,
        colors_rgb=colors_rgb,
        parameters=np.asarray(param_list[0], dtype=np.float32),
        roi=np.asarray(roi, dtype=np.float32),
        image_size=np.asarray([width, height], dtype=np.int32),
    )

    meta = {
        "classification": "PROVEN",
        "route": "3DDFA_V2_ONNX_PINNED",
        "third_party_commit": "1b6c67601abffc1e9f248b291708aef0e43b55ae",
        "image": str(image_path),
        "image_size": [width, height],
        "roi": [float(value) for value in roi],
        "vertex_count": int(vertices_obj.shape[1]),
        "face_count": int(triangles.shape[0]),
        "obj": str(output_obj),
        "npz": str(output_npz),
    }
    output_meta.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
