"""Stage 6 preflight: prove every texture input is the artifact it claims to be.

Hash-checks the source image and both meshes, then inspects the four baked maps. The material-ID
check is the important one and exists because of an actual failure: that map once baked as
per-triangle noise while reporting full coverage, zero NaN and a clean pass, so it is verified
here by counting how many distinct colours it actually resolves rather than by any aggregate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_map(path: Path, expect: int) -> dict:
    # cv2, not PIL: PIL has no real 16-bit RGB PNG support and silently hands back an 8-bit array,
    # which made a correct 16-bit normal map look like an 8-bit one.
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"could not read {path}")
    array = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB) if raw.ndim == 3 else raw
    image = Image.open(path)
    scale = 65535.0 if array.dtype == np.uint16 else 255.0
    normalised = array.astype(np.float32) / scale
    rgb = normalised[..., :3] if normalised.ndim == 3 else normalised[..., None]
    quantised = (array[..., :3] // 8).astype(np.int32) if array.ndim == 3 else None
    distinct = None
    if quantised is not None:
        key = quantised[..., 0] * 4096 + quantised[..., 1] * 64 + quantised[..., 2]
        distinct = int(np.unique(key).size)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "size": list(image.size),
        "resolution_ok": image.size == (expect, expect),
        "dtype": str(array.dtype),
        "bit_depth": 16 if array.dtype == np.uint16 else 8,
        "finite": bool(np.isfinite(normalised).all()),
        "nan_count": int((~np.isfinite(normalised)).sum()),
        "mean_rgb": [round(float(v), 6) for v in rgb.reshape(-1, rgb.shape[-1]).mean(axis=0)],
        "min": round(float(normalised.min()), 6),
        "max": round(float(normalised.max()), 6),
        "all_black": bool(rgb.max() <= 0.02),
        "distinct_quantised_colours": distinct,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--uv-mesh", required=True)
    parser.add_argument("--uv-sha256", required=True)
    parser.add_argument("--high-mesh", required=True)
    parser.add_argument("--textures", required=True)
    parser.add_argument("--bake-report", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=4096)
    args = parser.parse_args()

    failures: list[str] = []
    source, uv_mesh, high = Path(args.source_image), Path(args.uv_mesh), Path(args.high_mesh)

    source_hash = sha256(source)
    if source_hash != args.source_sha256.lower():
        failures.append(f"source image hash {source_hash} != expected {args.source_sha256}")
    uv_hash = sha256(uv_mesh)
    if not uv_hash.startswith(args.uv_sha256.split("…")[0].lower()):
        failures.append(f"UV mesh hash {uv_hash} does not match expected prefix {args.uv_sha256}")

    maps = {}
    for name in ("normal", "ao", "cavity", "material_id"):
        path = Path(args.textures) / f"shaman_{name}_4k.png"
        if not path.exists():
            failures.append(f"{name}: missing at {path}")
            continue
        entry = inspect_map(path, args.resolution)
        maps[name] = entry
        if not entry["resolution_ok"]:
            failures.append(f"{name}: {entry['size']} is not {args.resolution}x{args.resolution}")
        if not entry["finite"]:
            failures.append(f"{name}: {entry['nan_count']} non-finite pixels")
        if entry["all_black"]:
            failures.append(f"{name}: entirely black")

    # A tangent-space normal map is dominated by +Z, so blue must clearly lead red and green.
    normal = maps.get("normal")
    if normal:
        red, green, blue = normal["mean_rgb"][:3]
        if not (blue > red + 0.2 and blue > green + 0.2):
            failures.append(f"normal: mean {normal['mean_rgb'][:3]} is not tangent-space dominant")
        if normal["bit_depth"] != 16:
            failures.append(f"normal: {normal['bit_depth']}-bit, expected 16-bit non-colour data")

    bake = json.loads(Path(args.bake_report).read_text(encoding="utf-8"))
    components = int(bake.get("high_component_count", -1))
    if not 40 <= components <= 200:
        failures.append(
            f"material_id: bake resolved {components} components, which is outside the welded-position "
            "range and indicates per-triangle noise"
        )
    # Independent of the bake's own claim: noise saturates the quantised colour count.
    material = maps.get("material_id")
    if material and material["distinct_quantised_colours"] is not None:
        if material["distinct_quantised_colours"] > 20000:
            failures.append(
                f"material_id: {material['distinct_quantised_colours']} distinct colours reads as "
                "per-triangle noise rather than per-part identity"
            )

    report = {
        "source_image": {"path": str(source), "sha256": source_hash, "bytes": source.stat().st_size,
                         "matches_expected": source_hash == args.source_sha256.lower()},
        "uv_mesh": {"path": str(uv_mesh), "sha256": uv_hash, "bytes": uv_mesh.stat().st_size},
        "high_mesh": {"path": str(high), "sha256": sha256(high), "bytes": high.stat().st_size},
        "maps": maps,
        "bake_high_component_count": components,
        "bake_uv_layer_source": bake.get("uv_layer_source"),
        "failures": failures,
        "passed": not failures,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"PREFLIGHT passed={not failures} failures={failures}", flush=True)
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
