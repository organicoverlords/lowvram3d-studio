from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a numbered contact sheet for the public beggars meme clip."
    )
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--thumb-width", type=int, default=240)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clip_path = Path(args.clip).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(clip_path))
    if not capture.isOpened():
        raise SystemExit(f"OpenCV could not open public reference clip: {clip_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count_reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        raise SystemExit(
            f"Invalid public clip metadata: fps={fps} frames={frame_count_reported} size={width}x{height}"
        )

    thumb_width = int(args.thumb_width)
    thumb_height = max(1, int(round(height * thumb_width / width)))
    label_height = 28
    records: list[dict[str, float | int | str]] = []
    tiles: list[Image.Image] = []
    frame_index = 0

    while True:
        ok, frame_bgr = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_luma = float(gray.mean())
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb).resize(
            (thumb_width, thumb_height), Image.Resampling.LANCZOS
        )
        tile = Image.new("RGB", (thumb_width, thumb_height + label_height), "white")
        tile.paste(frame_image, (0, 0))
        draw = ImageDraw.Draw(tile)
        timestamp = frame_index / fps
        draw.rectangle((0, thumb_height, thumb_width, thumb_height + label_height), fill="white")
        draw.text(
            (6, thumb_height + 5),
            f"frame {frame_index:03d}  {timestamp:0.2f}s  sharp {sharpness:0.0f}",
            fill="black",
        )
        tiles.append(tile)
        records.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": timestamp,
                "sharpness": sharpness,
                "mean_luma": mean_luma,
            }
        )
        frame_index += 1

    capture.release()
    if not tiles:
        raise SystemExit("Public reference clip yielded zero frames")

    columns = max(1, int(args.columns))
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + label_height)),
        (32, 32, 32),
    )
    for index, tile in enumerate(tiles):
        sheet.paste(
            tile,
            ((index % columns) * thumb_width, (index // columns) * (thumb_height + label_height)),
        )

    contact_sheet_path = output_dir / "public_reference_all_frames.jpg"
    report_path = output_dir / "public_reference_frames.json"
    sheet.save(contact_sheet_path, quality=94, subsampling=0)

    sharpest = sorted(records, key=lambda row: float(row["sharpness"]), reverse=True)[:10]
    report = {
        "classification": "PUBLIC_REFERENCE_FRAME_REVIEW_REQUIRED",
        "source": "public Tenor meme clip",
        "source_media_packaged": False,
        "contact_sheet_contains_public_frames": True,
        "fps": fps,
        "reported_frame_count": frame_count_reported,
        "decoded_frame_count": len(records),
        "dimensions": [width, height],
        "contact_sheet": contact_sheet_path.name,
        "sharpest_frames": sharpest,
        "frames": records,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"PUBLIC_REFERENCE_CONTACT_SHEET={contact_sheet_path}")
    print(f"PUBLIC_REFERENCE_DECODED_FRAMES={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
