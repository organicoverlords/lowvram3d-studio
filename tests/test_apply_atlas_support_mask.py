from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_support_mask_worker_keeps_direct_and_conservative_inputs_separate() -> None:
    text = (ROOT / "workers" / "apply_atlas_support_mask.py").read_text(encoding="utf-8")
    assert "rasterise(uv, tris" in text
    assert "support = np.load" in text
    assert "np.concatenate" in text
    assert "DIAGNOSTIC_ONLY_NOT_PRODUCTION_READY" in text
    assert '"promotion_authorized": False' in text
