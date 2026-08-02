import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from workers.normalize_conditioning import normalize_conditioning


class ConditioningNormalizationTests(unittest.TestCase):
    def _paths(self, root):
        return [root / name for name in ("out.png", "audit.json", "overlay.png", "compare.png")]

    def test_useful_original_alpha_is_preserved_without_matte_call(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            rgba = np.zeros((40, 20, 4), dtype=np.uint8)
            rgba[8:32, 5:15, :3] = (20, 80, 120)
            rgba[8:32, 5:15, 3] = 255
            Image.fromarray(rgba, "RGBA").save(source)
            out, audit, overlay, compare = self._paths(root)
            with mock.patch("workers.normalize_conditioning.key_alpha", side_effect=AssertionError("matte must not run")):
                result = normalize_conditioning(source, out, audit, overlay, compare, size=64)
            self.assertEqual(result["route"], "original_alpha_preserved")
            with Image.open(out) as normalized:
                self.assertEqual(normalized.size, (64, 64))
            self.assertTrue(result["normalization"]["clipping_prevented"])

    def test_opaque_source_uses_existing_matte_route_and_normalizes_square(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            rgb = np.full((32, 48, 3), 255, dtype=np.uint8)
            rgb[5:27, 12:36] = (80, 30, 20)
            Image.fromarray(rgb, "RGB").save(source)
            out, audit, overlay, compare = self._paths(root)
            result = normalize_conditioning(source, out, audit, overlay, compare, size=64)
            self.assertEqual(result["route"], "existing_pipeline_matte")
            with Image.open(out) as normalized:
                self.assertEqual(normalized.size, (64, 64))
            self.assertTrue(json.loads(audit.read_text())["normalized_alpha"]["alpha_valid"])

    def test_empty_foreground_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.png"
            Image.new("RGBA", (20, 20), (0, 0, 0, 0)).save(source)
            out, audit, overlay, compare = self._paths(root)
            with self.assertRaises(ValueError):
                normalize_conditioning(source, out, audit, overlay, compare, size=64)


if __name__ == "__main__":
    unittest.main()
