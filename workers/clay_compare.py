"""Render orientation-normalised clay sets for several meshes and lay them out for comparison.

Each mesh's up axis, up sign and lateral axis are measured before rendering, so meshes exported by
different tools with different conventions are shown in the same pose. Without that the comparison
silently becomes "which tool used which axis convention".
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from geometry_quality_metrics import canonical_axes, up_axis_sign
from mesh_io import read_glb

REPO_ROOT = Path(__file__).resolve().parents[1]


def render_one(blender: str, glb: Path, out_dir: Path, label: str) -> dict:
    positions, _, _, tris = read_glb(glb)
    positions = positions.astype(np.float64)
    lateral, up, _ = canonical_axes(positions)
    sign = up_axis_sign(positions, tris, up)

    report = out_dir / "clay_report.json"
    command = [
        blender, "--background", "--python-use-system-env",
        "--python", str(REPO_ROOT / "blender" / "clay_render_set.py"), "--",
        "--glb", str(glb), "--output-dir", str(out_dir), "--report", str(report),
        "--up-axis", str(up), "--up-sign", str(sign), "--lateral-axis", str(lateral),
        "--label", label,
    ]
    process = subprocess.run(command, capture_output=True, text=True)
    record = {
        "label": label, "glb": str(glb), "exit": process.returncode,
        "up_axis": "xyz"[up], "up_sign": sign, "lateral_axis": "xyz"[lateral],
        "triangles": int(len(tris)),
    }
    if report.exists():
        record["renders"] = json.loads(report.read_text(encoding="utf-8")).get("renders", {})
    else:
        record["detail"] = (process.stdout + process.stderr)[-800:]
    return record


def contact_sheet(rows: list[dict], destination: Path, cell: int = 384) -> None:
    """One row per mesh, one column per view, labelled down the left."""
    views = ["front", "left_three_quarter", "right_three_quarter", "side", "close_head"]
    label_width = 210
    usable = [r for r in rows if r.get("renders")]
    if not usable:
        return
    height = cell * len(usable)
    sheet = np.full((height, label_width + cell * len(views), 3), 26, np.uint8)
    for row_index, row in enumerate(usable):
        y = row_index * cell
        cv2.putText(sheet, row["label"][:24], (8, y + cell // 2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(sheet, f"{row['triangles']:,} tris", (8, y + cell // 2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1, cv2.LINE_AA)
        for column, view in enumerate(views):
            path = row["renders"].get(view)
            if not path or not Path(path).exists():
                continue
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.resize(image, (cell, cell), interpolation=cv2.INTER_AREA)
            x = label_width + column * cell
            sheet[y:y + cell, x:x + cell] = image
    for column, view in enumerate(views):
        cv2.putText(sheet, view, (label_width + column * cell + 6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), sheet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", action="append", required=True,
                        help="label=path, repeatable; order is preserved in the sheet")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--contact-sheet", default="")
    parser.add_argument("--blender", default=r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
    args = parser.parse_args()

    root = Path(args.output_root)
    rows = []
    for entry in args.mesh:
        label, _, path = entry.partition("=")
        glb = Path(path)
        if not glb.is_file():
            rows.append({"label": label, "glb": str(glb), "exit": -1, "detail": "missing"})
            print(f"CLAY_SKIP {label}: missing {glb}", flush=True)
            continue
        record = render_one(args.blender, glb, root / label, label)
        rows.append(record)
        print(f"CLAY {label}: exit={record['exit']} views={len(record.get('renders', {}))} "
              f"up={record['up_axis']}{int(record['up_sign'])}", flush=True)

    if args.contact_sheet:
        contact_sheet(rows, Path(args.contact_sheet))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({"meshes": rows}, indent=2), encoding="utf-8")
    print(f"CLAY_COMPARE_DONE meshes={len(rows)} sheet={args.contact_sheet}", flush=True)


if __name__ == "__main__":
    main()
