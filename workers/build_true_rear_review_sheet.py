"""Assemble the sheet a human needs to rule on the true-opposite view.

Four gates in this route have passed on two-faced assets. The last one compared
the correct pair of images and still could not see a face, because it compared
them with whole-image grayscale correlation: the known-bad panda rear carries a
full frontal face and scores 0.162. Silhouette, pose, props and lighting all
differ between a front and a rear view of the same character, so a low
correlation is the *expected* reading whether or not there is a face.

Rather than invent a fifth automatic metric that cannot see the defect either,
this puts the five things a person needs side by side and asks for a verdict.
The verdict is what gates promotion; the correlation is recorded beside it as a
diagnostic so runs stay comparable.

    py workers/build_true_rear_review_sheet.py \
       --run-dir evidence/compare/panda_sixview_512 \
       --source panda2/panda2_2048.png \
       --controls evidence/compare/panda2/controls_512_audited \
       --out review_sheet.png

Then write the verdict beside the run:

    {"verdict": "FACE_PRESENT", "reviewer": "lauri", "note": "second face, clear"}

into <run-dir>/true_rear_verdict.json. Permitted verdicts are FACE_FREE,
FACE_PRESENT and AMBIGUOUS; only FACE_FREE promotes, and AMBIGUOUS fails closed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Panel width. Wide enough that a face is unmistakable at a glance, which is
#: the entire point of the artefact.
PANEL = 420
BAR = 30
BACKGROUND = (24, 24, 28)


def _load(path: Path, size: int):
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), BACKGROUND)
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def build(run_dir: Path, source: Path | None, controls: Path | None,
          out: Path) -> dict:
    import numpy as np
    from PIL import Image, ImageDraw

    receipt_path = run_dir / "inference_receipt.json"
    if not receipt_path.is_file():
        raise SystemExit(f"NO_RECEIPT: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    qa = receipt.get("qa", {})

    # The opposite index comes from the receipt when the run recorded it, and is
    # otherwise re-derived from the camera contract. It is never taken from the
    # label order -- that is the bug this whole document exists because of.
    opposite = qa.get("true_opposite_index")
    contract = None
    if controls is not None and (controls / "camera_contract.json").is_file():
        contract = json.loads((controls / "camera_contract.json").read_text(encoding="utf-8"))
    if opposite is None:
        if contract is None:
            raise SystemExit("NO_TRUE_OPPOSITE_INDEX: pass --controls so it can be derived")
        views = sorted(contract["views"], key=lambda v: int(v["index"]))
        base = np.asarray(views[0]["camera_direction"], np.float64)
        opposite = int(min(range(1, 6), key=lambda i: float(
            np.dot(base, np.asarray(views[i]["camera_direction"], np.float64)))))

    dot = None
    directions = None
    if contract is not None:
        views = sorted(contract["views"], key=lambda v: int(v["index"]))
        base = np.asarray(views[0]["camera_direction"], np.float64)
        other = np.asarray(views[opposite]["camera_direction"], np.float64)
        dot = float(np.dot(base, other))
        directions = (list(base), list(other))

    tiles = sorted(run_dir.glob("view_*.png"))
    by_index = {int(p.name.split("_")[1]): p for p in tiles}
    if 0 not in by_index or opposite not in by_index:
        raise SystemExit(f"MISSING_TILES: need view 0 and view {opposite}")

    panels: list[tuple[str, Path | None]] = [
        ("1. source artwork", source),
        ("2. generated TRUE FRONT (index 0)", by_index[0]),
        (f"3. generated TRUE OPPOSITE (index {opposite})", by_index[opposite]),
    ]
    if controls is not None:
        for candidate in (f"view_{opposite}_normal.png", f"{opposite}_normal.png"):
            if (controls / candidate).is_file():
                panels.append((f"4. TRUE OPPOSITE geometry control", controls / candidate))
                break

    columns = len(panels)
    sheet = Image.new("RGB", (PANEL * columns, PANEL + BAR * 2), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    for column, (label, path) in enumerate(panels):
        x = column * PANEL
        draw.rectangle([x, 0, x + PANEL, BAR], fill=(12, 12, 14))
        draw.text((x + 8, 9), label, fill=(255, 255, 255))
        if path is not None and Path(path).is_file():
            sheet.paste(_load(Path(path), PANEL), (x, BAR))
        else:
            draw.text((x + 12, BAR + 12), "(not supplied)", fill=(150, 150, 150))

    footer = (
        f"true opposite of index 0 is index {opposite}"
        + (f"  |  camera direction dot = {dot:+.3f}" if dot is not None else "")
        + f"  |  correlation direct {qa.get('front_rear_direct_correlation')}"
          f" / mirrored {qa.get('front_rear_mirrored_correlation')} (DIAGNOSTIC ONLY)"
    )
    draw.rectangle([0, PANEL + BAR, PANEL * columns, PANEL + BAR * 2], fill=(12, 12, 14))
    draw.text((8, PANEL + BAR + 9), footer, fill=(230, 230, 120))

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)

    return {
        "schema": "lowvram3d_true_rear_review_sheet_v1",
        "run_dir": str(run_dir),
        "sheet": str(out),
        "true_opposite_index": int(opposite),
        "camera_direction_dot": dot,
        "camera_directions": directions,
        "correlation_diagnostic": {
            "direct": qa.get("front_rear_direct_correlation"),
            "mirrored": qa.get("front_rear_mirrored_correlation"),
        },
        "verdict_file_expected": str(run_dir / "true_rear_verdict.json"),
        "permitted_verdicts": ["FACE_FREE", "FACE_PRESENT", "AMBIGUOUS"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--controls", type=Path, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    report = build(args.run_dir, args.source, args.controls, args.out)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
