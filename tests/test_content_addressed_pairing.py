from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

from workers.content_addressed_pairing import (
    build_fixture_report,
    inventory_snapshot,
    pair_images_to_models,
    sha256_file,
    stability_report,
)


def png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    height, width = len(pixels), len(pixels[0])
    raw = b"".join(b"\x00" + bytes(value for rgb in row for value in rgb) for row in pixels)
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)
    body = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    path.write_bytes(body)


class ContentAddressedPairingTests(unittest.TestCase):
    def test_same_content_at_new_path_retains_identity_and_history_input(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "first.glb"
            second = root / "moved.glb"
            first.write_bytes(b"same model bytes")
            before = inventory_snapshot([root])
            first.rename(second)
            after = inventory_snapshot([root])
            self.assertEqual(before["identities"], after["identities"])
            self.assertEqual(before["files"][0]["asset_id"], after["files"][0]["asset_id"])
            self.assertNotEqual(before["files"][0]["path"], after["files"][0]["path"])

    def test_duplicate_formats_are_linked_by_hash(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "asset.glb").write_bytes(b"duplicate export")
            (root / "asset.fbx").write_bytes(b"duplicate export")
            report = build_fixture_report(root)
            self.assertEqual(len(report["duplicate_exports"]), 1)
            self.assertEqual(len(report["duplicate_exports"][0]["paths"]), 2)

    def test_near_duplicate_pngs_group(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            png(root / "crop-a.png", [[(0, 0, 0)] * 8 for _ in range(8)])
            changed = [[(0, 0, 0)] * 8 for _ in range(8)]
            changed[-1][-1] = (1, 1, 1)
            png(root / "crop-b.png", changed)
            report = build_fixture_report(root)
            self.assertEqual(len(report["image_groups"]), 1)

    def test_one_image_can_rank_multiple_provenance_models(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "concept.png"
            png(image, [[(20, 20, 20)] * 4 for _ in range(4)])
            for name in ("candidate-a.glb", "candidate-b.glb"):
                model = root / name
                model.write_bytes(name.encode())
                (root / f"{name}.json").write_text(json.dumps({"source_image": image.name}), encoding="utf-8")
            report = build_fixture_report(root)
            proven = [item for item in report["pairing"]["proposals"] if item["classification"] == "PROVEN_HIGH_CONFIDENCE_PAIR"]
            self.assertEqual(len(proven), 2)

    def test_filename_only_match_cannot_be_high_confidence_and_ambiguous_is_reviewed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "boat-concept.png"
            png(image, [[(20, 20, 20)] * 4 for _ in range(4)])
            (root / "boat-candidate-a.glb").write_bytes(b"a")
            (root / "boat-candidate-b.glb").write_bytes(b"b")
            report = build_fixture_report(root)
            proposals = report["pairing"]["proposals"]
            self.assertTrue(all(item["classification"] != "PROVEN_HIGH_CONFIDENCE_PAIR" for item in proposals))
            self.assertTrue(all(item["review_required"] for item in proposals))

    def test_cache_reuses_unchanged_file_and_snapshot_changes_are_provisional(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / "model.glb"
            path.write_bytes(b"stable")
            cache = {}
            first = inventory_snapshot([root], cache)
            second = inventory_snapshot([root], cache)
            self.assertFalse(first["files"][0]["cache_hit"])
            self.assertTrue(second["files"][0]["cache_hit"])
            path.write_bytes(b"changed")
            third = inventory_snapshot([root], cache)
            self.assertFalse(stability_report(first, third)["stable"])
            self.assertEqual(stability_report(first, third)["classification"], "PROVISIONAL_SNAPSHOT_MOVING_DATASET")

    def test_source_file_is_not_modified(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "source.png"
            png(image, [[(1, 2, 3)] * 2 for _ in range(2)])
            before = sha256_file(image)
            build_fixture_report(root)
            self.assertEqual(before, sha256_file(image))


if __name__ == "__main__":
    unittest.main()
