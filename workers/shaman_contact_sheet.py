"""Build labelled contact sheets from rendered proof images.

Deterministic: images are placed in the order given, each cell is labelled, and
the sheet records its inputs so a reviewer can trace every tile back to a file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def build_contact_sheet(
    images: list[Path],
    destination: Path,
    *,
    labels: list[str] | None = None,
    columns: int = 0,
    cell_width: int = 420,
    title: str = "",
) -> dict:
    if not images:
        raise ValueError("contact sheet requires at least one image")
    labels = labels or [path.stem for path in images]
    if len(labels) != len(images):
        raise ValueError("labels and images must be the same length")

    columns = columns or min(len(images), 5)
    rows = (len(images) + columns - 1) // columns

    loaded = []
    for path in images:
        with Image.open(path) as handle:
            frame = handle.convert("RGB")
            ratio = cell_width / frame.width
            loaded.append(frame.resize((cell_width, max(1, int(frame.height * ratio)))))

    cell_height = max(frame.height for frame in loaded)
    banner = 26
    header = 34 if title else 0
    sheet = Image.new(
        "RGB",
        (columns * cell_width, header + rows * (cell_height + banner)),
        (24, 24, 28),
    )
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((10, 10), title, fill=(235, 235, 240))

    for index, frame in enumerate(loaded):
        column = index % columns
        row = index // columns
        x = column * cell_width
        y = header + row * (cell_height + banner)
        sheet.paste(frame, (x, y + (cell_height - frame.height) // 2))
        draw.text((x + 8, y + cell_height + 6), labels[index][:60], fill=(220, 220, 225))

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination)
    return {
        "contact_sheet": str(destination),
        "tiles": [str(path) for path in images],
        "labels": labels,
        "columns": columns,
        "rows": rows,
        "size": list(sheet.size),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--label", action="append", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--columns", type=int, default=0)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    images = [Path(item) for item in args.image]
    missing = [str(path) for path in images if not path.is_file()]
    if missing:
        raise SystemExit("missing contact-sheet inputs: " + ", ".join(missing))

    result = build_contact_sheet(
        images,
        Path(args.output),
        labels=args.label,
        columns=args.columns,
        title=args.title,
    )
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"CONTACT_SHEET={result['contact_sheet']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
