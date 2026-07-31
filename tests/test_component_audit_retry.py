from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lowvram3d.component_audit import AuditConfig
from workers.component_audit_cleanup import (
    _ambiguity_only,
    _topology_safe,
    run_cleanup,
)


def audit_result(*, errors: list[str], boundary_after: int = 0) -> dict:
    return {
        "success": not errors,
        "errors": errors,
        "warnings": [],
        "output": "unused.glb",
        "topology_before": {
            "faces": 100,
            "boundary_edges": 0,
            "non_manifold_edges": 0,
        },
        "topology_after": {
            "faces": 96,
            "boundary_edges": boundary_after,
            "non_manifold_edges": 0,
        },
        "faces_removed": 4,
        "faces_removed_percent": 4.0,
        "main_component_faces_before": 80,
        "main_component_faces_after": 80,
        "passes": [{}],
        "final_audit": {"audit_required_count": 2, "decisions": []},
    }


class AmbiguityPolicyTests(unittest.TestCase):
    def test_expected_audit_ambiguity_is_classified(self):
        result = audit_result(
            errors=[
                "cleanup did not converge within max_passes",
                "2 visible components remain audit-required",
            ]
        )
        self.assertTrue(_ambiguity_only(result))
        self.assertTrue(_topology_safe(result))

    def test_unrelated_process_error_is_not_ambiguity(self):
        result = audit_result(errors=["worker process crashed"])
        self.assertFalse(_ambiguity_only(result))

    def test_topology_regression_is_not_safe(self):
        result = audit_result(
            errors=["2 visible components remain audit-required"],
            boundary_after=3,
        )
        self.assertFalse(_topology_safe(result))


class PreserveOriginalTests(unittest.TestCase):
    def test_safe_ambiguity_preserves_original_bytes_and_continues(self):
        baseline = audit_result(
            errors=[
                "cleanup did not converge within max_passes",
                "2 visible components remain audit-required",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.glb"
            output = root / "clean.glb"
            payload = b"turbo-bird-class-valid-glb"
            source.write_bytes(payload)
            with patch(
                "workers.component_audit_cleanup.audit_and_cleanup",
                return_value=baseline,
            ):
                result = run_cleanup(
                    str(source),
                    str(output),
                    asset_type="character",
                    source_image=None,
                    config=AuditConfig(),
                    seed=0,
                )

        self.assertTrue(result["success"])
        self.assertEqual(output.read_bytes() if output.exists() else payload, payload)
        self.assertEqual(result["faces_removed"], 0)
        self.assertTrue(result["manual_review_required"])
        self.assertEqual(
            result["audit_policy"]["selected"],
            "preserve_original_on_audit_ambiguity",
        )

    def test_real_topology_regression_still_fails(self):
        baseline = audit_result(
            errors=["2 visible components remain audit-required"],
            boundary_after=4,
        )
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "source.glb"
            output = root / "clean.glb"
            source.write_bytes(b"mesh")
            with patch(
                "workers.component_audit_cleanup.audit_and_cleanup",
                return_value=baseline,
            ):
                result = run_cleanup(
                    str(source),
                    str(output),
                    asset_type="character",
                    source_image=None,
                    config=AuditConfig(),
                    seed=0,
                )
        self.assertFalse(result["success"])
        self.assertFalse(output.exists())
        self.assertEqual(result["audit_policy"]["selected"], "hard_failure")


if __name__ == "__main__":
    unittest.main()
