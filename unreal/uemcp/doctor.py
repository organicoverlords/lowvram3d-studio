"""One command that answers "why can't the agent talk to Unreal right now?".

Checks every surface an agent might reach for and names the specific repair for
each failure, so no step is left to guesswork.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

from .bridge import Bridge, BridgeError, discover_port
from .editor_mcp import EditorMCP

DEFAULT_PROJECT = Path(r"C:\Users\Lauri\Desktop\UnrealAITest58")


def _editor_processes() -> list[int]:
    import subprocess

    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-Process UnrealEditor -ErrorAction SilentlyContinue | "
             "Select-Object -ExpandProperty Id"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except Exception:
        return []
    return [int(line) for line in out.split() if line.strip().isdigit()]


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=2):
            return True
    except OSError:
        return False


def diagnose(project_root: Path | str | None = None) -> dict[str, Any]:
    root = Path(project_root or os.environ.get("UE_PROJECT_ROOT") or DEFAULT_PROJECT)
    report: dict[str, Any] = {"project_root": str(root), "checks": [], "ok": True}

    def check(name: str, ok: bool, detail: str, repair: str = "",
              advisory: bool = False) -> None:
        report["checks"].append(
            {"name": name, "ok": ok, "advisory": advisory, "detail": detail,
             "repair": repair if not ok else ""})
        if not ok and not advisory:
            report["ok"] = False

    pids = _editor_processes()
    check("editor_running", bool(pids), f"UnrealEditor pids: {pids or 'none'}",
          f'Launch the editor on {root / (root.name + ".uproject")}')

    lockfile = root / "Saved" / "UE_MCP_Bridge" / "port.json"
    bridge_port = None
    try:
        bridge_port = discover_port(root)
        check("bridge_lockfile", True, f"{lockfile} -> port {bridge_port}")
    except BridgeError as exc:
        check("bridge_lockfile", False, str(exc),
              "Enable the UE_MCP_Bridge plugin for this project and restart the editor")

    if bridge_port:
        bridge = Bridge(root, bridge_port)
        try:
            health = bridge.health()
            ready = bool(health.get("editorReady"))
            check("bridge_health", ready, json.dumps(health),
                  "The editor is still loading, or a modal dialog is blocking the game thread")
        except Exception as exc:
            check("bridge_health", False, f"{type(exc).__name__}: {exc}",
                  f"Port {bridge_port} is not accepting length-prefixed JSON")

        try:
            value = bridge.python_json("import json\nresult = json.dumps({'python': True})")
            check("bridge_python", bool(value.get("python")), json.dumps(value),
                  "PythonScriptPlugin is disabled for this project")
        except Exception as exc:
            check("bridge_python", False, f"{type(exc).__name__}: {exc}",
                  "Enable PythonScriptPlugin and restart the editor")

    check("editor_mcp_port", _port_open(8000), "127.0.0.1:8000",
          "Enable the ModelContextProtocol plugin and restart the editor")
    if _port_open(8000):
        try:
            with EditorMCP() as client:
                toolsets = client.list_toolsets()
            check("editor_mcp_toolsets", bool(toolsets),
                  f"{len(toolsets)} toolsets registered")
        except Exception as exc:
            check("editor_mcp_toolsets", False, f"{type(exc).__name__}: {exc}",
                  "The MCP server is listening but not answering; restart the editor")

    # Advisory: this library falls back to the pinned project root, but the npm
    # MCP server has no such fallback and silently targets the wrong project.
    env_root = os.environ.get("UE_PROJECT_ROOT")
    check("env_ue_project_root", env_root is not None,
          f"UE_PROJECT_ROOT={env_root!r}",
          "Harmless for this CLI, but the unreal-engine MCP server needs it in its "
          "env block; without it the server defaults to the process cwd and looks "
          "for the wrong project's bridge",
          advisory=True)

    return report


def main() -> int:
    report = diagnose()
    for item in report["checks"]:
        mark = "ok  " if item["ok"] else ("warn" if item["advisory"] else "FAIL")
        print(f"[{mark}] {item['name']}: {item['detail']}")
        if item["repair"]:
            print(f"        repair: {item['repair']}")
    print("\nOVERALL:", "HEALTHY" if report["ok"] else "DEGRADED")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
