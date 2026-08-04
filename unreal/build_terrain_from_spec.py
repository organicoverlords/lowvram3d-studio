"""Reusable terrain-stage entrypoint.

The complete scene adapter owns idempotent layer ordering. This thin entrypoint
keeps the terrain stage addressable without introducing a second mutation path.
"""
from pathlib import Path

_target = Path(__file__).with_name("build_complete_scene_layers.py")
exec(compile(_target.read_text(encoding="utf-8"), str(_target), "exec"))
