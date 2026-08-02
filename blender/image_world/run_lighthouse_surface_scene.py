"""Blender entry point that forwards only arguments after ``--``."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main() -> None:
    try:
        separator = sys.argv.index("--")
    except ValueError as exc:
        raise SystemExit("Blender invocation must separate script arguments with --") from exc
    script = Path(__file__).with_name("build_lighthouse_surface_scene.py")
    forwarded = sys.argv[separator + 1 :]
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
