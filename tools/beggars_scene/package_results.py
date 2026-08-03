from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate and package the Blender meme recreation outputs.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--final-video", required=True)
    parser.add_argument("--workflow-run-id", required=True)
    parser.add_argument("--head-sha", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_file(path: Path, minimum_bytes: int) -> None:
    if not path.is_file():
        raise SystemExit(f"Required output is missing: {path}")
    size = path.stat().st_size
    if size < minimum_bytes:
        raise SystemExit(f"Required output is implausibly small ({size} bytes): {path}")


def make_contact_sheet(paths: list[Path], output: Path) -> None:
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"OpenCV could not read proof frame: {path}")
        images.append(image)
    target_height = min(image.shape[0] for image in images)
    resized = []
    for image in images:
        width = int(round(image.shape[1] * target_height / image.shape[0]))
        resized.append(cv2.resize(image, (width, target_height), interpolation=cv2.INTER_AREA))
    border = 12
    canvas_height = target_height + border * 2
    canvas_width = sum(image.shape[1] for image in resized) + border * (len(resized) + 1)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:] = (10, 8, 7)
    cursor = border
    for index, image in enumerate(resized):
        canvas[border : border + target_height, cursor : cursor + image.shape[1]] = image
        cv2.putText(
            canvas,
            f"FRAME {index + 1}",
            (cursor + 18, border + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (235, 235, 235),
            2,
            cv2.LINE_AA,
        )
        cursor += image.shape[1] + border
    if not cv2.imwrite(str(output), canvas):
        raise SystemExit(f"Could not write contact sheet: {output}")


def video_metadata(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV could not open final video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise SystemExit(
            f"Invalid final video metadata: fps={fps}, frames={frame_count}, size={width}x{height}"
        )
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": frame_count / fps,
        "resolution": [width, height],
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    final_video = Path(args.final_video).resolve()

    required = {
        "blend": output_dir / "beggars_photoreal_recreation.blend",
        "hero": output_dir / "hero_clean_render.png",
        "wide": output_dir / "wide_scene_proof.png",
        "receipt": output_dir / "scene_receipt.json",
        "video": final_video,
    }
    minimums = {
        "blend": 250_000,
        "hero": 50_000,
        "wide": 50_000,
        "receipt": 500,
        "video": 100_000,
    }
    for name, path in required.items():
        ensure_file(path, minimums[name])

    proof_paths = [output_dir / f"proof_frame_{index:02d}.png" for index in range(1, 4)]
    for path in proof_paths:
        ensure_file(path, 50_000)
    contact_sheet = output_dir / "contact_sheet.png"
    make_contact_sheet(proof_paths, contact_sheet)
    ensure_file(contact_sheet, 80_000)

    receipt = json.loads(required["receipt"].read_text(encoding="utf-8"))
    receipt["workflow"] = {
        "run_id": str(args.workflow_run_id),
        "head_sha": args.head_sha,
    }
    receipt["final_video"] = str(final_video)
    receipt["final_video_metadata"] = video_metadata(final_video)
    receipt["contact_sheet"] = str(contact_sheet)
    receipt["reference_media_uploaded"] = False
    receipt["automated_validation"] = "PROVEN"
    receipt["likeness_match"] = "NOT_PROVEN_PENDING_USER_REVIEW"
    required["receipt"].write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")

    artifact_files = [
        required["blend"],
        required["hero"],
        required["wide"],
        required["receipt"],
        required["video"],
        contact_sheet,
        *proof_paths,
        output_dir / "reference_reconstruction_report.json",
        output_dir / "worker_receipt.json",
    ]
    manifest_files = []
    for path in artifact_files:
        ensure_file(path, 50 if path.suffix.lower() == ".json" else 1000)
        manifest_files.append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    manifest = {
        "classification": "USER_VISUAL_REVIEW_REQUIRED",
        "automated_build_and_validation": "PROVEN",
        "meme_clip_match": "NOT_PROVEN_PENDING_USER_REVIEW",
        "workflow_run_id": str(args.workflow_run_id),
        "head_sha": args.head_sha,
        "reference_media_in_artifact": False,
        "files": manifest_files,
    }
    manifest_path = output_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
