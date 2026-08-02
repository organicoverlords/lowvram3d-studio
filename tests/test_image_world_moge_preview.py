from pathlib import Path

import numpy as np

from lowvram3d.image_world.moge_preview import depth_preview, normal_preview, save_moge_previews


class FakeCv2:
    COLOR_RGB2BGR = 1

    @staticmethod
    def cvtColor(image, _code):
        return image[..., ::-1]

    @staticmethod
    def imwrite(path, image):
        Path(path).write_bytes(np.asarray(image).tobytes())
        return True


def test_depth_preview_preserves_invalid_as_black_and_near_is_brighter():
    depth = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask = np.array([[True, True], [False, True]])
    preview, low, high = depth_preview(depth, mask)
    assert preview[1, 0] == 0
    assert preview[0, 0] > preview[1, 1]
    assert high > low


def test_constant_depth_is_valid():
    preview, low, high = depth_preview(np.ones((3, 3)), np.ones((3, 3), bool))
    assert preview.shape == (3, 3)
    assert high > low


def test_normal_preview_maps_axes_and_masks_invalid_pixels():
    normal = np.array([[[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]])
    mask = np.array([[True, False]])
    preview = normal_preview(normal, mask)
    assert preview[0, 0].tolist() == [255, 128, 0]
    assert preview[0, 1].tolist() == [0, 0, 0]


def test_save_moge_previews_writes_compact_proofs(tmp_path):
    geometry = tmp_path / "geometry"
    geometry.mkdir()
    np.save(geometry / "depth.npy", np.arange(12, dtype=np.float32).reshape(3, 4), allow_pickle=False)
    np.save(geometry / "mask.npy", np.ones((3, 4), dtype=np.uint8), allow_pickle=False)
    np.save(geometry / "normal.npy", np.zeros((3, 4, 3), dtype=np.float32), allow_pickle=False)
    output = tmp_path / "previews"
    summary = save_moge_previews(geometry, output, cv2_module=FakeCv2)
    assert summary.previews == ("depth.png", "mask.png", "normal.png")
    assert (output / "preview-summary.json").is_file()
    assert all((output / name).stat().st_size > 0 for name in summary.previews)
