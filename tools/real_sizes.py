"""Shared scale lookup for the deliverable builders.

Kept in one place because the FBX export and the inspection scene must agree.
If they disagree, an asset is one size in Unreal and another in the lineup, and
nothing in either file would show it.
"""

import json
from pathlib import Path

REPO = Path(r"C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803")
SIZES = REPO / "evidence" / "deliverables" / "REAL_SIZES.json"


def load():
    return json.loads(SIZES.read_text(encoding="utf-8"))


def subject_of(stem):
    """Subject name from a deliverable filename: 'whale_trellis1024_tex2048'."""
    return stem.split("_", 1)[0]


def scale_for(stem, dims, table=None):
    """Factor that takes a normalised mesh to its real size.

    dims is the mesh's (x, y, z) extent as imported. Which of those the target
    refers to depends on the subject's axis: 'height' uses z, 'longest' uses
    whichever is largest. Returns (factor, metres, axis) so the caller can
    record what it applied rather than re-deriving it.
    """
    table = table or load()
    entry = table["subjects"].get(subject_of(stem), table["default"])
    metres = float(entry["metres"])
    axis = entry.get("axis", "height")
    current = float(dims[2]) if axis == "height" else float(max(dims))
    if current <= 0:
        return 1.0, metres, axis
    return metres / current, metres, axis
