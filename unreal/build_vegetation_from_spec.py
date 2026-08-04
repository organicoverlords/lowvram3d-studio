"""Reusable vegetation-stage entrypoint delegated to the unified builder."""
from pathlib import Path

_target = Path(__file__).with_name("build_complete_scene_layers.py")
exec(compile(_target.read_text(encoding="utf-8"), str(_target), "exec"))
