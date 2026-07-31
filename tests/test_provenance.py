from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from lowvram3d.provenance import PROVENANCE_SCHEMA, provenance_path, sha256_file
from lowvram3d.runner import StageFailure, artifact_is_valid, run_stage


_COPY_COMMAND = (
    "from pathlib import Path; import sys; "
    "Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes() + b'-output')"
)


class StageProvenanceTests(unittest.TestCase):
    def run_copy_stage(self, root: Path, source: Path, output: Path):
        return run_stage(
            "copy",
            [sys.executable, "-c", _COPY_COMMAND, str(source), str(output)],
            root,
            root / "logs",
            {"mesh": str(output)},
            999999,
            timeout_seconds=30,
        )

    def test_success_seals_output_and_records_receipt_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.bin"
            output = root / "output.bin"
            source.write_bytes(b"source")

            receipt = self.run_copy_stage(root, source, output)

            self.assertEqual(receipt.status, "passed")
            self.assertEqual(receipt.provenance_schema, PROVENANCE_SCHEMA)
            self.assertTrue(receipt.command_fingerprint)
            self.assertEqual(receipt.input_fingerprints[str(source.resolve())], sha256_file(source))
            self.assertEqual(receipt.artifact_fingerprints["mesh"], sha256_file(output))
            self.assertEqual(receipt.provenance_files["mesh"], str(provenance_path(output)))
            self.assertTrue(provenance_path(output).is_file())
            self.assertTrue(artifact_is_valid(output))

    def test_mutating_sealed_output_invalidates_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.bin"
            output = root / "output.bin"
            source.write_bytes(b"source")
            self.run_copy_stage(root, source, output)

            output.write_bytes(b"silently replaced by a stale artifact")

            self.assertFalse(artifact_is_valid(output))

    def test_mutating_stage_input_invalidates_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.bin"
            output = root / "output.bin"
            source.write_bytes(b"source-v1")
            self.run_copy_stage(root, source, output)

            source.write_bytes(b"source-v2")

            self.assertFalse(artifact_is_valid(output))

    def test_failed_rerun_invalidates_old_output_even_when_file_survives(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.bin"
            output = root / "output.bin"
            source.write_bytes(b"source")
            self.run_copy_stage(root, source, output)
            self.assertTrue(artifact_is_valid(output))

            with self.assertRaises(StageFailure):
                run_stage(
                    "copy",
                    [sys.executable, "-c", "raise SystemExit(9)", str(source), str(output)],
                    root,
                    root / "logs",
                    {"mesh": str(output)},
                    999999,
                    timeout_seconds=30,
                )

            self.assertTrue(output.is_file())
            self.assertFalse(provenance_path(output).exists())
            self.assertFalse(artifact_is_valid(output))

    def test_legacy_unsealed_artifact_remains_readable(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "legacy.bin"
            output.write_bytes(b"legacy")
            self.assertTrue(artifact_is_valid(output))


if __name__ == "__main__":
    unittest.main()
