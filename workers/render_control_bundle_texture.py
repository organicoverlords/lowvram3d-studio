"""Textured control renders and the raw / semantic / mapping contact sheets.

Camera semantics cannot be read off a mask: an orthographic silhouette from +d and -d is
the same shape mirrored, so front and rear have identical pixel areas.  Landmarks - muzzle,
tail, backpack, rifle - only become visible once the base-colour texture is sampled, which
is what this does, reusing the bundle's own triangle-ID and barycentric buffers so the
result is in exactly the camera frame the controls were built in.

Every filename and every label comes from the camera contract: the on-disk arrays are found
through ``control_file_prefix`` and the label through ``proven_semantic``.  There is no
positional VIEW_NAMES tuple anywhere in this file, so a relabelled bundle cannot silently
pick up the wrong semantic from its ordering.
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
    """Resolve the image a primitive's material actually samples.

    Taking images[0] silently reads a stale atlas whenever a GLB carries more than one -
    which is exactly what happens after a re-texture appends the new one, and it made a
    replaced texture look like it had kept detail it never had.
    """
    meta, binary = glb_json_and_bin(path)
    images = meta.get("images")
    if not images:
        raise RuntimeError("GLB_HAS_NO_IMAGE")
    index = 0
    bound = []
    for mesh in meta.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            material_index = primitive.get("material")
            if material_index is None:
                continue
            pbr = meta["materials"][material_index].get("pbrMetallicRoughness", {})
            texture = pbr.get("baseColorTexture")
            if texture is None:
                continue
            source = meta["textures"][int(texture["index"])].get("source")
            if source is not None:
                bound.append(int(source))
    if bound:
        if len(set(bound)) > 1:
            raise RuntimeError(f"GLB_MULTIPLE_BASE_COLOUR_IMAGES:{sorted(set(bound))}")
        index = bound[0]
    image = images[index]
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


def file_prefix(view: dict) -> str:
    """The on-disk array prefix, which is not the semantic label once a bundle is relabelled."""
    return str(view.get("control_file_prefix") or view["semantic_name"])


def caption(image: Image.Image, lines: list[str]) -> Image.Image:
    height = 12 * len(lines) + 8
    framed = Image.new("RGB", (image.width, image.height + height), (18, 18, 20))
    framed.paste(image, (0, height))
    draw = ImageDraw.Draw(framed)
    for position, text in enumerate(lines):
        draw.text((4, 3 + position * 12), text, fill=(240, 240, 240))
    return framed


def contact_sheet(tiles: list[Image.Image], columns: int = 3) -> Image.Image:
    rows = (len(tiles) + columns - 1) // columns
    width = max(tile.width for tile in tiles)
    height = max(tile.height for tile in tiles)
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
    views = sorted(contract["views"], key=lambda item: int(item["index"]))

    modalities: dict[str, list[Image.Image]] = {"textured": [], "position": [], "normal": []}
    records = []
    for view in views:
        index = int(view["index"])
        prefix = file_prefix(view)
        semantic = str(view.get("proven_semantic") or view["semantic_name"])
        ids = np.load(bundle / f"{prefix}_triangle_ids.npy")
        bary = np.load(bundle / f"{prefix}_barycentric.npy")
        visible = ids >= 0
        pixel_uv = np.einsum("nc,ncd->nd", bary[visible], uv[tris[ids[visible]]])
        canvas = np.full(ids.shape + (3,), 24, dtype=np.uint8)
        canvas[visible] = sample(texture, pixel_uv)
        textured_path = output_dir / f"raw{index}_{semantic}_textured.png"
        Image.fromarray(canvas).save(textured_path)

        modalities["textured"].append(Image.fromarray(canvas))
        modalities["position"].append(Image.open(bundle / f"{prefix}_position.png").convert("RGB"))
        modalities["normal"].append(Image.open(bundle / f"{prefix}_normal.png").convert("RGB"))
        records.append({
            "raw_index": index,
            "semantic_label": semantic,
            "control_file_prefix": prefix,
            "azimuth_deg": view["azimuth_deg"],
            "elevation_deg": view["elevation_deg"],
            "camera_direction_control_space": view["camera_direction"],
            "camera_direction_mesh_local": view.get("camera_direction_mesh_local"),
            "textured_render": str(textured_path),
            "foreground_pixels": int(visible.sum()),
        })

    sheets = {}
    for modality, images in modalities.items():
        variants = {
            "a_raw_index": [f"raw{item['raw_index']}" for item in records],
            "b_semantic_corrected": [item["semantic_label"] for item in records],
            "c_raw_and_semantic_mapping": None,
        }
        for name, labels in variants.items():
            if labels is None:
                tiles = [caption(image, [
                    f"raw{item['raw_index']} = {item['semantic_label']}",
                    f"az {item['azimuth_deg']:.2f}  el {item['elevation_deg']:.2f}",
                ]) for image, item in zip(images, records)]
            else:
                tiles = [caption(image, [text]) for image, text in zip(images, labels)]
            path = output_dir / f"contact_sheet_{name}_{modality}.png"
            contact_sheet(tiles).save(path)
            sheets[f"{name}_{modality}"] = str(path)

    report = {
        "schema": "control_bundle_texture_render_v2",
        "mesh": str(mesh),
        "bundle": str(bundle),
        "camera_contract_classification": contract.get("classification"),
        "raw_to_semantic": contract.get("raw_to_semantic"),
        "filenames_from_camera_contract": True,
        "hardcoded_view_name_tuple_used": False,
        "contact_sheets": sheets,
        "views": records,
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CONTROL_TEXTURE_RENDER_DONE views={len(records)} sheets={len(sheets)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
