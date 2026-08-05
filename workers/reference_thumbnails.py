"""Render read-only silhouette thumbnails of reference meshes so they can be identified by eye.

Silhouettes are enough to tell an antlered figure with a staff from a tree or a wall, and they cost
a rasterisation rather than a render. Sources are opened read-only; every image is written to a
destination the caller chooses, which is never inside the reference library.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from mesh_io import read_glb
from shaman_texture_views import project, rasterise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--contact-sheet", default="")
    parser.add_argument("--columns", type=int, default=6)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tiles, labels = [], []

    for entry in args.mesh:
        path = Path(entry)
        if not path.is_file() or path.suffix.lower() != ".glb":
            continue
        try:
            positions, _, _, tris = read_glb(path)
        except Exception as error:  # noqa: BLE001
            print(f"THUMB_FAILED {path.name}: {str(error)[:120]}", flush=True)
            continue
        p = positions.astype(np.float64)
        if len(tris) < 2:
            continue
        centre = (p.min(0) + p.max(0)) * 0.5
        verts = p - centre
        ortho = float((verts.max(0) - verts.min(0)).max())
        screen, depth = project(verts, np.array([0.0, 0.0, 1.0]), ortho)
        _, silhouette = rasterise(screen, depth, tris, args.size)
        image = (silhouette.astype(np.uint8) * 255)
        cv2.imwrite(str(out / f"{path.stem}.png"), image)
        tiles.append(image)
        labels.append(path.stem)
        print(f"THUMB {path.stem}", flush=True)

    if args.contact_sheet and tiles:
        columns = max(1, args.columns)
        rows = (len(tiles) + columns - 1) // columns
        cell = args.size
        sheet = np.zeros((rows * (cell + 18), columns * cell), np.uint8)
        for index, (tile, label) in enumerate(zip(tiles, labels)):
            r, c = divmod(index, columns)
            y, x = r * (cell + 18), c * cell
            sheet[y:y + cell, x:x + cell] = tile
            cv2.putText(sheet, label[:28], (x + 2, y + cell + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, 255, 1, cv2.LINE_AA)
        cv2.imwrite(args.contact_sheet, sheet)
        print(f"CONTACT_SHEET {args.contact_sheet} tiles={len(tiles)}", flush=True)


if __name__ == "__main__":
    main()
