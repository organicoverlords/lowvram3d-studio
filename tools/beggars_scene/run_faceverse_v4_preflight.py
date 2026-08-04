from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded FaceVerse v4 full-head reconstruction proof on one private keyframe."
    )
    parser.add_argument("--faceverse-root", required=True)
    parser.add_argument("--model-npy", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--landmarker", required=True)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(vector: np.ndarray) -> np.ndarray:
    length = float(np.sqrt((vector**2).sum()))
    if length <= 1e-8:
        return np.zeros_like(vector)
    return vector / length


def distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(((left - right) ** 2).sum()))


def detect_face_box_and_eyes(image_rgb: np.ndarray, landmarker_path: Path) -> tuple[np.ndarray, np.ndarray]:
    base_options = mp.tasks.BaseOptions
    face_landmarker = mp.tasks.vision.FaceLandmarker
    options_type = mp.tasks.vision.FaceLandmarkerOptions
    running_mode = mp.tasks.vision.RunningMode
    options = options_type(
        base_options=base_options(model_asset_path=str(landmarker_path)),
        running_mode=running_mode.IMAGE,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    with face_landmarker.create_from_options(options) as detector:
        face_image = mp.Image(mp.ImageFormat.SRGB, np.ascontiguousarray(image_rgb.astype(np.uint8)))
        results = detector.detect(face_image)
    if not results.face_landmarks:
        raise RuntimeError("MediaPipe did not detect a face in the selected private keyframe")

    landmarks = np.asarray(
        [(landmark.x * image_rgb.shape[1], landmark.y * image_rgb.shape[0]) for landmark in results.face_landmarks[0]],
        dtype=np.float32,
    )
    if landmarks.shape[0] < 474:
        raise RuntimeError(f"MediaPipe returned only {landmarks.shape[0]} landmarks; 474 are required")

    left_vector = norm(landmarks[362, :2] - landmarks[263, :2])
    left_distance = max(distance(landmarks[362, :2], landmarks[263, :2]), 1e-6)
    left_center = (landmarks[263, :2] + landmarks[362, :2]) * 0.5
    left_eye_x = float(np.dot(landmarks[473] - left_center, left_vector) / left_distance * 3.0)
    left_eye_y = float(
        np.dot(landmarks[473] - left_center, left_vector[[1, 0]]) / left_distance * -1.5
    )

    right_vector = norm(landmarks[33, :2] - landmarks[133, :2])
    right_distance = max(distance(landmarks[33, :2], landmarks[133, :2]), 1e-6)
    right_center = (landmarks[33, :2] + landmarks[133, :2]) * 0.5
    right_eye_x = float(np.dot(landmarks[468] - right_center, right_vector) / right_distance * 3.0)
    right_eye_y = float(
        np.dot(landmarks[468] - right_center, right_vector[[1, 0]]) / right_distance * -1.5
    )

    bbox = np.asarray(
        [
            float(np.min(landmarks[:, 0])),
            float(np.min(landmarks[:, 1])),
            float(np.max(landmarks[:, 0])),
            float(np.max(landmarks[:, 1])),
        ],
        dtype=np.float32,
    )
    eye_coefficients = np.asarray(
        [left_eye_y, left_eye_x, right_eye_y, right_eye_x], dtype=np.float32
    )
    return bbox, eye_coefficients


def write_colored_ply(
    vertices: np.ndarray, colors: np.ndarray, triangles: np.ndarray, output_path: Path
) -> None:
    vertices = np.asarray(vertices, dtype=np.float32)
    colors_u8 = np.clip(np.rint(np.asarray(colors) * 255.0), 0, 255).astype(np.uint8)
    triangles = np.asarray(triangles, dtype=np.int32)
    with output_path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(vertices)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write(f"element face {len(triangles)}\n")
        handle.write("property list uchar int vertex_indices\n")
        handle.write("end_header\n")
        for vertex, color in zip(vertices, colors_u8):
            handle.write(
                f"{vertex[0]:.7f} {vertex[1]:.7f} {vertex[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for triangle in triangles:
            handle.write(f"3 {int(triangle[0])} {int(triangle[1])} {int(triangle[2])}\n")


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main() -> int:
    args = parse_args()
    faceverse_root = Path(args.faceverse_root).resolve()
    model_path = Path(args.model_npy).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    landmarker_path = Path(args.landmarker).resolve()
    input_image_path = Path(args.input_image).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for required in (
        faceverse_root,
        model_path,
        checkpoint_path,
        landmarker_path,
        input_image_path,
    ):
        if not required.exists():
            raise SystemExit(f"Required FaceVerse preflight input is missing: {required}")

    sys.path.insert(0, str(faceverse_root))
    from faceversev4 import FaceVerseRecon  # pylint: disable=import-error,import-outside-toplevel
    from Sim3DR.renderer import render_fvr  # pylint: disable=import-error,import-outside-toplevel

    image_bgr = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"OpenCV could not read the selected keyframe: {input_image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    bbox, eye_coefficients = detect_face_box_and_eyes(image_rgb, landmarker_path)

    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    start = time.perf_counter()
    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)
    load_seconds = time.perf_counter() - start

    inference_start = time.perf_counter()
    coefficients, bbox_list = model.process_imgs(
        image_rgb[np.newaxis, ...], bbox.reshape(1, 1, 4)
    )
    coefficients[:, -4:] = torch.from_numpy(eye_coefficients.reshape(1, 4)).to(
        coefficients.device
    )
    vertices, projected, normals, colors = model.from_coeffs(coefficients, bbox_list)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_start

    triangles = np.asarray(model.fvd["tri"], dtype=np.int32)
    render_rgb, depth = render_fvr(
        image_rgb,
        projected[0],
        triangles,
        normals[0],
        colors[0],
    )
    if render_rgb.shape[:2] != image_rgb.shape[:2]:
        raise RuntimeError(
            f"FaceVerse render shape {render_rgb.shape} does not match input {image_rgb.shape}"
        )

    render_path = output_dir / "faceverse_v4_render.png"
    depth_path = output_dir / "faceverse_v4_depth.png"
    ply_path = output_dir / "faceverse_v4_colored_head.ply"
    coefficients_path = output_dir / "faceverse_v4_coefficients.npz"
    report_path = output_dir / "faceverse_v4_report.json"

    if not cv2.imwrite(str(render_path), cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Could not write FaceVerse render: {render_path}")
    if not cv2.imwrite(str(depth_path), depth):
        raise RuntimeError(f"Could not write FaceVerse depth: {depth_path}")
    write_colored_ply(vertices[0], colors[0], triangles, ply_path)

    coefficient_parts = model.split_coeffs_dict(coefficients)
    np.savez_compressed(
        coefficients_path,
        coefficients=coefficients.detach().cpu().numpy().astype(np.float32),
        bbox=np.asarray(bbox_list, dtype=np.int32),
        eye_coefficients=eye_coefficients.astype(np.float32),
        identity=coefficient_parts["id"].detach().cpu().numpy().astype(np.float32),
        expression=coefficient_parts["exp"].detach().cpu().numpy().astype(np.float32),
        texture=coefficient_parts["tex"].detach().cpu().numpy().astype(np.float32),
        lighting=coefficient_parts["gamma"].detach().cpu().numpy().astype(np.float32),
        rotation=coefficient_parts["angle"].detach().cpu().numpy().astype(np.float32),
        translation=coefficient_parts["trans"].detach().cpu().numpy().astype(np.float32),
        eyes=coefficient_parts["eyes"].detach().cpu().numpy().astype(np.float32),
    )

    render_nonzero_fraction = float(np.mean(np.any(render_rgb > 0, axis=2)))
    depth_nonzero_fraction = float(np.mean(depth > 0))
    vertex_bounds = {
        "min": np.min(vertices[0], axis=0).astype(float).tolist(),
        "max": np.max(vertices[0], axis=0).astype(float).tolist(),
    }
    ver_inds = np.asarray(model.fvd.get("ver_inds", []), dtype=np.int64).reshape(-1)
    component_ranges = ver_inds.astype(int).tolist()

    report = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "route": "FACEVERSE_V4_RESNET50_SINGLE_FRAME_FULL_HEAD",
        "claim": "Inference, mesh export and clean rendering are machine-proven; likeness remains pending visual review.",
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else 0.0
        ),
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "input_dimensions": [int(image_rgb.shape[1]), int(image_rgb.shape[0])],
        "coefficient_dimensions": int(coefficients.shape[1]),
        "identity_dimensions": int(model.id_dims),
        "expression_dimensions": int(model.exp_dims),
        "texture_dimensions": int(model.tex_dims),
        "vertex_count": int(vertices.shape[1]),
        "triangle_count": int(triangles.shape[0]),
        "vertex_bounds": vertex_bounds,
        "component_vertex_boundaries": component_ranges,
        "render_nonzero_fraction": render_nonzero_fraction,
        "depth_nonzero_fraction": depth_nonzero_fraction,
        "bbox": np.asarray(bbox_list, dtype=int).tolist(),
        "eye_coefficients": eye_coefficients.astype(float).tolist(),
        "faceverse_source_commit": os.environ.get(
            "FACEVERSE_SOURCE_COMMIT", "19c67cc4d7234b1ea7d55a185a2cb55fd49bb877"
        ),
        "model_sha256": sha256(model_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "landmarker_sha256": sha256(landmarker_path),
        "outputs": {
            "render": render_path.name,
            "depth": depth_path.name,
            "colored_ply": ply_path.name,
            "coefficients": coefficients_path.name,
        },
        "reference_image_packaged": False,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if coefficients.shape[1] != 621:
        raise RuntimeError(f"Expected 621 FaceVerse coefficients, got {coefficients.shape[1]}")
    if vertices.shape[1] < 10000 or triangles.shape[0] < 10000:
        raise RuntimeError(
            f"FaceVerse mesh is implausibly small: vertices={vertices.shape[1]} triangles={triangles.shape[0]}"
        )
    if render_nonzero_fraction < 0.03 or depth_nonzero_fraction < 0.03:
        raise RuntimeError(
            f"FaceVerse render coverage is too small: rgb={render_nonzero_fraction:.4f} depth={depth_nonzero_fraction:.4f}"
        )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
