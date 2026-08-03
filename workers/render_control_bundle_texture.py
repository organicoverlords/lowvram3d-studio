"""Textured re-render of a CPU control bundle, plus raw-index and semantic contact sheets.

Camera semantics cannot be read off a mask: an orthographic silhouette from +d and -d is
the same shape mirrored, so front and rear have identical pixel areas.  Landmarks - muzzle,
tail, backpack, rifle - only become visible once the base-colour texture is sampled, which
is what this does, reusing the bundle's own triangle-ID and barycentric buffers so the
result is in exactly the camera frame the controls were built in.

Filenames come from the bundle's camera contract, never from a positional VIEW_NAMES tuple.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mesh_io import read_glb


def glb_json_and_bin(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    if raw[:4] != b"glTF":
        raise RuntimeError("NOT_GLB")
    json_length = struct.unpack_from("<I", raw, 12)[0]
    meta = json.loads(raw[20:20 + json_length])
    bin_start = 20 + ((json_length + 3) // 4) * 4 + 8
    bin_length = struct.unpack_from("<I", raw, bin_start - 8)[0]
    return meta, raw[bin_start:bin_start + bin_length]


def base_colour_image(path: Path) -> Image.Image:
    meta, binary = glb_json_and_bin(path)
    images = meta.get("images")
    if not images:
        raise RuntimeError("GLB_HAS_NO_IMAGE")
    image = images[0]
    if "bufferView" in image:
        view = meta["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        payload = binary[start:start + view["byteLength"]]
    elif str(image.get("uri", "")).startswith("data:"):
        payload = base64.b64decode(str(image["uri"]).split(",", 1)[1])
    else:
        raise RuntimeError("GLB_IMAGE_NOT_SELF_CONTAINED")
    return Image.open(io.BytesIO(payload)).convert("RGB")


def sample(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width = texture.shape[:2]
    # glTF texture space puts v=0 at the top of the image.
    xs = np.clip((uv[:, 0] % 1.0) * (width - 1), 0, width - 1).astype(np.int64)
    ys = np.clip((uv[:, 1] % 1.0) * (height - 1), 0, height - 1).astype(np.int64)
    return texture[ys, xs]


def label(image: Image.Image, text: str) -> Image.Image:
    framed = Image.new("RGB", (image.width, image.height + 18), (18, 18, 20))
    framed.paste(image, (0, 18))
    ImageDraw.Draw(framed).text((4, 4), text, fill=(240, 240, 240))
    return framed


def contact_sheet(tiles: list[Image.Image], columns: int = 3) -> Image.Image:
    rows = (len(tiles) + columns - 1) // columns
    width, height = tiles[0].size
    sheet = Image.new("RGB", (width * columns, height * rows), (18, 18, 20))
    for position, tile in enumerate(tiles):
        sheet.paste(tile, ((position % columns) * width, (position // columns) * height))
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    mesh = Path(args.mesh)
    bundle = Path(args.bundle)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _positions, _normals, uv, tris = read_glb(mesh)
    if uv is None:
        raise RuntimeError("CONTROL_TEXTURE_UV_MISSING")
    texture = np.asarray(base_colour_image(mesh))
    contract = json.loads((bundle / "camera_contract.json").read_text(encoding="utf-8"))

    raw_tiles, semantic_tiles, records = [], [], []
    for view in sorted(contract["views"], key=lambda item: int(item["index"])):
        index = int(view["index"])
        prefix = str(view["semantic_name"])
        ids = np.load(bundle / f"{prefix}_triangle_ids.npy")
        bary = np.load(bundle / f"{prefix}_barycentric.npy")
        visible = ids >= 0
        pixel_uv = np.einsum("nc,ncd->nd", bary[visible], uv[tris[ids[visible]]])
        canvas = np.full(ids.shape + (3,), 24, dtype=np.uint8)
        canvas[visible] = sample(texture, pixel_uv)
        image = Image.fromarray(canvas)
        image.save(output_dir / f"raw{index}_{prefix}_textured.png")
        raw_tiles.append(label(image, f"raw{index}"))
        semantic_tiles.append(label(image, str(view.get("proven_semantic") or prefix)))
        records.append({
            "raw_index": index,
            "bundle_prefix": prefix,
            "declared_semantic": view.get("proven_semantic"),
            "azimuth_deg": view["azimuth_deg"],
            "elevation_deg": view["elevation_deg"],
            "camera_position_control_space": view["camera_position"],
            "camera_direction_control_space": view["camera_direction"],
            "camera_up_control_space": view["camera_up"],
            "textured_render": str(output_dir / f"raw{index}_{prefix}_textured.png"),
            "foreground_pixels": int(visible.sum()),
        })

    contact_sheet(raw_tiles).save(output_dir / "contact_sheet_a_raw_index.png")
    contact_sheet(semantic_tiles).save(output_dir / "contact_sheet_b_semantic.png")
    report = {
        "schema": "control_bundle_texture_render_v1",
        "mesh": str(mesh),
        "bundle": str(bundle),
        "contact_sheet_raw_index": str(output_dir / "contact_sheet_a_raw_index.png"),
        "contact_sheet_semantic": str(output_dir / "contact_sheet_b_semantic.png"),
        "filenames_from_camera_contract": True,
        "views": records,
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CONTROL_TEXTURE_RENDER_DONE views={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
