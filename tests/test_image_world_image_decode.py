from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFile

from lowvram3d.image_world.image_decode import DecodedImage, decode_image_bgr


class FakeCv2:
    IMREAD_UNCHANGED = -1

    def __init__(self, result):
        self.result = result
        self.calls = []

    def imread(self, filename: str, flags: int):
        self.calls.append((filename, flags))
        return self.result


def write_rgb(path: Path) -> None:
    image = np.zeros((3, 4, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 20
    image[..., 2] = 30
    Image.fromarray(image, mode="RGB").save(path)


def test_opencv_fast_path_preserves_bgr_and_provenance(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"present")
    bgr = np.zeros((2, 3, 3), dtype=np.uint8)
    bgr[..., 0] = 9
    cv2 = FakeCv2(bgr)

    decoded = decode_image_bgr(source, cv2)

    decoded.validate()
    assert decoded.decoder == "opencv"
    assert decoded.recovered is False
    assert decoded.bgr_or_bgra.flags.c_contiguous
    assert np.array_equal(decoded.bgr_or_bgra, bgr)
    assert cv2.calls == [(str(source), cv2.IMREAD_UNCHANGED)]


def test_pillow_fallback_converts_rgb_to_bgr(tmp_path):
    source = tmp_path / "source.png"
    write_rgb(source)
    cv2 = FakeCv2(None)

    decoded = decode_image_bgr(source, cv2)

    assert decoded.decoder == "pillow_truncated_rgb"
    assert decoded.recovered is True
    assert decoded.bgr_or_bgra.shape == (3, 4, 3)
    assert decoded.bgr_or_bgra[0, 0].tolist() == [30, 20, 10]


def test_pillow_fallback_converts_rgba_to_bgra(tmp_path):
    source = tmp_path / "source.png"
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[...] = [1, 2, 3, 4]
    Image.fromarray(rgba, mode="RGBA").save(source)

    decoded = decode_image_bgr(source, FakeCv2(None))

    assert decoded.decoder == "pillow_truncated_rgba"
    assert decoded.bgr_or_bgra[0, 0].tolist() == [3, 2, 1, 4]


def test_fallback_restores_global_pillow_truncated_setting(tmp_path):
    source = tmp_path / "source.png"
    write_rgb(source)
    previous = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        decode_image_bgr(source, FakeCv2(None))
        assert ImageFile.LOAD_TRUNCATED_IMAGES is False
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous


def test_both_decoders_failing_reports_source(tmp_path):
    source = tmp_path / "broken.jpg"
    source.write_bytes(b"not an image")

    with pytest.raises(RuntimeError, match="OpenCV and tolerant Pillow") as exc_info:
        decode_image_bgr(source, FakeCv2(None))

    assert str(source) in str(exc_info.value)


def test_missing_source_fails_before_decoder_call(tmp_path):
    cv2 = FakeCv2(None)
    source = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError):
        decode_image_bgr(source, cv2)

    assert cv2.calls == []


def test_decoded_image_rejects_non_contiguous_pixels():
    source = np.zeros((4, 4, 3), dtype=np.uint8)[:, ::2]
    decoded = DecodedImage(source, decoder="test", recovered=False)

    with pytest.raises(ValueError, match="C-contiguous"):
        decoded.validate()
