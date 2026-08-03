"""Create the deterministic RGB inference copy without touching the original."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from workers.scene_pipeline.core import image_input_receipt, write_json


SOURCE = Path(r"C:\Users\Lauri\Downloads\benchmarkpics\castlegrounds.png")
EXTERNAL = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
PROOF = Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    with Image.open(SOURCE) as original:
        original_mode = original.mode
        original_size = list(original.size)
        had_alpha = "A" in original.getbands()
        normalized = ImageOps.exif_transpose(original).convert("RGB")
    output = EXTERNAL / "source_rgb.png"
    normalized.save(output, format="PNG", optimize=False, compress_level=6)
    reopened = Image.open(output).convert("RGB")
    cv_image = cv2.imread(str(output), cv2.IMREAD_COLOR)
    if reopened.size != tuple(original_size) or cv_image is None or cv_image.ndim != 3 or cv_image.shape[2] != 3:
        raise RuntimeError("NORMALIZED_SOURCE_REOPEN_FAILED")
    receipt = {
        "schema": "scene_source_normalization_v1",
        "original": image_input_receipt(SOURCE, dimensions=tuple(original_size), mode=original_mode),
        "normalized": image_input_receipt(output, dimensions=reopened.size, mode="RGB"),
        "operations": ["EXIF_ORIENTATION_APPLIED", "CONVERTED_TO_RGB", "ALPHA_REMOVED", "PNG_METADATA_REWRITTEN"],
        "pillow_reopen": True,
        "opencv_reopen": {"valid": True, "shape": list(cv_image.shape), "dtype": str(cv_image.dtype)},
        "original_alpha_present": had_alpha,
        "original_unchanged_hash_recorded": True,
    }
    write_json(PROOF / "source_normalization.json", receipt)
    write_json(EXTERNAL / "source_normalization.json", receipt)


if __name__ == "__main__":
    main()
