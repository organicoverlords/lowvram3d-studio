"""Static guard that owner-aware projection routes bind only covered triangles."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_fast_projection_owner_route_passes_triangle_coverage_mask() -> None:
    source = (ROOT / "workers" / "fast_texture_projection.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = [node for node in ast.walk(tree)
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name)
                           and target.id == "textured_triangles"
                           for target in node.targets)]
    assert any(
        isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "triangle_coverage_mask"
        and len(node.value.args) == 2
        and isinstance(node.value.args[0], ast.Name)
        and node.value.args[0].id == "owner"
        and isinstance(node.value.args[1], ast.Call)
        and isinstance(node.value.args[1].func, ast.Name)
        and node.value.args[1].func.id == "len"
        and len(node.value.args[1].args) == 1
        and isinstance(node.value.args[1].args[0], ast.Name)
        and node.value.args[1].args[0].id == "triangles"
        for node in assignments
    )
    bind_calls = [node for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name)
                  and node.func.id == "bind_texture"]
    assert any(any(keyword.arg == "textured_triangles"
                   and isinstance(keyword.value, ast.Name)
                   and keyword.value.id == "textured_triangles"
                   for keyword in call.keywords)
               for call in bind_calls)
