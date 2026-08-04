"""Two generators, same photograph, same stages, side by side.

Hunyuan3D-2 ships two checkpoints this machine can run, and they differ in more
than size. `hunyuan3d-dit-v2-mini-turbo` declares `ImageProcessorV2` and takes
exactly one image at 1022 px. `hunyuan3d-dit-v2-mv-turbo` declares
`MVImageProcessorV2`, takes a dict of up to four named views at 518 px each, and
is twice the depth. So the choice is a trade: per-view detail against coverage,
paid for in runtime.

Arguing about that trade from the config files is how this project has
repeatedly got things wrong. This renders both, at every stage that has a
picture, in one image -- so the trade is looked at rather than reasoned about.

    py -3.12 workers/build_generator_comparison.py \\
        --left  name=mini_turbo dir=.../architecture_house_025 asset=architecture_house_025 \\
        --right name=mv         dir=.../compare/mv            asset=barn_mv \\
        --out comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PANEL_HEIGHT = 250
PAD = 14
LABEL_HEIGHT = 32
HEADER_HEIGHT = 30
BACKGROUND = (238, 238, 238)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_column(tokens: list[str]) -> dict:
    return dict(token.split("=", 1) for token in tokens)


def _stages(column: dict) -> list[tuple[str, Path | None, str]]:
    """The stages that have a picture, in pipeline order, with their numbers."""
    directory = Path(column["dir"])
    asset = column["asset"]

    result = (_read_json(directory / f"{asset}_result.json")
              or _read_json(directory / "mini_turbo_result.json"))
    decimation = _read_json(directory / f"{asset}_decimation.json")
    preview = _read_json(directory / "preview.json")
    projection = _read_json(directory / f"{asset}_textured.projection.json")

    attempts = result.get("attempts") or [{}]
    peak = next((a.get("peak_vram_mb") for a in attempts
                 if a.get("peak_vram_mb")), None)
    seconds = result.get("generation_seconds")

    conditioning = next(
        (p for p in (directory / "conditioning.png",
                     directory / "mini_turbo_conditioning.png",
                     directory / "crop.png") if p.is_file()), None)

    return [
        ("conditioning", conditioning,
         f"{result.get('image_dimensions')} · "
         f"{result.get('model_subfolder', '?')}"),
        ("geometry (unlit)", directory / "preview.png",
         f"octree {result.get('octree_resolution')} · "
         f"{result.get('raw_triangles')} tris raw · "
         f"{preview.get('mesh_bodies', decimation.get('bodies', '?'))} bodies"),
        ("appearance", directory / "textured_views.png",
         f"observed {projection.get('observed_face_fraction')} · "
         f"mirrored {projection.get('mirrored_face_fraction')} · "
         f"flat {projection.get('flat_filled_face_fraction')}"),
    ], {
        "generation_seconds": seconds,
        "peak_vram_mb": peak,
        "octree_resolution": result.get("octree_resolution"),
        "num_chunks": result.get("num_chunks"),
        "raw_vertices": result.get("raw_vertices"),
        "raw_triangles": result.get("raw_triangles"),
        "subfolder": result.get("model_subfolder"),
        "weights_bytes": result.get("weights_bytes"),
    }


def _fit(path, height):
    from PIL import Image

    if path is None or not Path(path).is_file():
        return None
    image = Image.open(path).convert("RGB")
    scale = height / image.height
    return image.resize((max(1, int(image.width * scale)), height),
                        Image.LANCZOS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", nargs="+", required=True)
    parser.add_argument("--right", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--panel-height", type=int, default=PANEL_HEIGHT)
    args = parser.parse_args(argv)

    from PIL import Image, ImageDraw

    columns = [_parse_column(args.left), _parse_column(args.right)]
    collected = [_stages(column) for column in columns]

    # A fixed panel width per column, so the same stage lines up across columns
    # even when one generator returns a differently proportioned render.
    panels = [[_fit(path, args.panel_height) for _, path, _ in stages]
              for stages, _ in collected]
    flat = [p for column in panels for p in column if p is not None]
    if not flat:
        raise SystemExit("no stage images found in either column")
    column_width = max(p.width for p in flat) + PAD * 2

    stage_names = [name for name, _, _ in collected[0][0]]
    width = column_width * len(columns)
    height = (HEADER_HEIGHT
              + len(stage_names) * (args.panel_height + LABEL_HEIGHT + PAD)
              + PAD)
    sheet = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(sheet)

    for index, column in enumerate(columns):
        summary = collected[index][1]
        seconds = summary.get("generation_seconds")
        draw.text((index * column_width + PAD, 6),
                  f"{column.get('name', '?')} — "
                  f"{seconds if seconds is not None else '?'} s, "
                  f"peak {summary.get('peak_vram_mb', '?')} MB VRAM",
                  fill=(0, 0, 0))

    for row, name in enumerate(stage_names):
        y = HEADER_HEIGHT + row * (args.panel_height + LABEL_HEIGHT + PAD)
        for index in range(len(columns)):
            x = index * column_width + PAD
            note = collected[index][0][row][2]
            draw.text((x, y), f"{row + 1}. {name}", fill=(0, 0, 0))
            draw.text((x, y + 15), note, fill=(90, 90, 90))
            panel = panels[index][row]
            if panel is not None:
                sheet.paste(panel, (x, y + LABEL_HEIGHT))
            else:
                draw.text((x, y + LABEL_HEIGHT + 20), "(not produced)",
                          fill=(150, 60, 60))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)

    receipt = {
        "schema_version": "generator_comparison_v1",
        "classification": "PROVEN",
        "out": str(out),
        "columns": [
            {"name": columns[i].get("name"), **collected[i][1],
             "stages_present": [n for (n, _, _), p
                                in zip(collected[i][0], panels[i])
                                if p is not None],
             "stages_missing": [n for (n, _, _), p
                                in zip(collected[i][0], panels[i])
                                if p is None]}
            for i in range(len(columns))],
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
