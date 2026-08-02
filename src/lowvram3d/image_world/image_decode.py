"""Deterministic image decoding for image-to-world inputs.

OpenCV remains the fast path. Pillow is used only as an explicit recovery path
for malformed or truncated images that OpenCV refuses to decode. The selected
decoder is returned as provenance rather than hidden from proof receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np


class OpenCvLike(Protocol):
    IMREAD_UNCHANGED: int

    def imread(self, filename: str, flags: int) -> np.ndarray | None: ...


@dataclass(frozen=True)
class DecodedImage:
    bgr_or_bgra: np.ndarray
    decoder: str
    recovered: bool

    def validate(self) -> None:
        image = np.asarray(self.bgr_or_bgra)
        if image.dtype != np.uint8:
            raise ValueError("decoded image must use uint8 pixels")
        if image.ndim != 3 or image.shape[2] not in {3, 4}:
            raise ValueError("decoded image must have BGR or BGRA channels")
        if image.shape[0] < 1 or image.shape[1] < 1:
            raise ValueError("decoded image dimensions must be positive")
        if not image.flags.c_contiguous:
            raise ValueError("decoded image must be C-contiguous")
        if not self.decoder:
            raise ValueError("decoder provenance cannot be empty")


def decode_image_bgr(source: Path, cv2_module: OpenCvLike) -> DecodedImage:
    """Decode ``source`` as contiguous BGR/BGRA pixels.

    The function never changes the source file and never silently rewrites a
    damaged input. OpenCV is attempted first. Pillow's truncated-image mode is
    limited to the fallback call and restored afterward.
    """

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)

    image = cv2_module.imread(str(path), cv2_module.IMREAD_UNCHANGED)
    if image is not None:
        decoded = DecodedImage(
            bgr_or_bgra=np.ascontiguousarray(image, dtype=np.uint8),
            decoder="opencv",
            recovered=False,
        )
        decoded.validate()
        return decoded

    from PIL import Image, ImageFile

    previous_truncated_setting = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(path) as pil_image:
            pil_image.load()
            if "A" in pil_image.getbands():
                rgba = np.asarray(pil_image.convert("RGBA"), dtype=np.uint8)
                pixels = np.ascontiguousarray(rgba[:, :, [2, 1, 0, 3]])
                decoder = "pillow_truncated_rgba"
            else:
                rgb = np.asarray(pil_image.convert("RGB"), dtype=np.uint8)
                pixels = np.ascontiguousarray(rgb[:, :, ::-1])
                decoder = "pillow_truncated_rgb"
    except Exception as exc:
        raise RuntimeError(
            f"OpenCV and tolerant Pillow could not decode {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_truncated_setting

    decoded = DecodedImage(
        bgr_or_bgra=pixels,
        decoder=decoder,
        recovered=True,
    )
    decoded.validate()
    return decoded
