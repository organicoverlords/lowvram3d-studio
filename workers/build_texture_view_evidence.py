"""Backend-neutral, exact CPU view evidence builder.

This is the production name for the existing deterministic CPU rasterizer.  It deliberately
does not inspect image semantics: a view is evidence only for the mesh triangles its registered
camera actually exposes.
"""
from __future__ import annotations

from build_view_evidence import main, rasterise  # noqa: F401


if __name__ == "__main__":
    raise SystemExit(main())
