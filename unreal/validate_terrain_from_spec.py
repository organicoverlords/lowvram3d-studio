"""Reusable terrain validation entrypoint delegated to the unified validator."""
from pathlib import Path

_target = Path(__file__).with_name("validate_complete_scene_layers.py")
exec(compile(_target.read_text(encoding="utf-8"), str(_target), "exec"))
