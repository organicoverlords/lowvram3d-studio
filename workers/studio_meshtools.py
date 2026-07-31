from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import requests


def parse_sse(response: requests.Response) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    for raw in response.iter_lines(decode_unicode=True):
        if not raw or raw.startswith(":"):
            continue
        if not raw.startswith("data:"):
            continue
        payload = json.loads(raw[5:].strip())
        events.append(payload)
        if payload.get("type") == "error":
            raise RuntimeError(str(payload.get("detail") or "3D Gen Studio mesh tool failed"))
        if payload.get("type") == "done":
            terminal = payload
    if terminal is None:
        raise RuntimeError("3D Gen Studio mesh tool stream ended without a done event")
    return terminal, events


def run_tool(
    base_url: str,
    operation: str,
    input_path: Path,
    output_path: Path,
    report_path: Path,
    options: dict[str, Any],
    timeout: int,
) -> None:
    endpoint = {"auto-retopo": "/meshes/auto-retopo", "auto-uv": "/meshes/auto-uv"}[operation]
    health = requests.get(base_url.rstrip("/") + "/health", timeout=10)
    health.raise_for_status()
    with input_path.open("rb") as handle:
        response = requests.post(
            base_url.rstrip("/") + endpoint,
            files={"meshFile": (input_path.name, handle, "model/gltf-binary")},
            data={"options": json.dumps(options), "format": "glb"},
            stream=True,
            timeout=(30, timeout),
        )
        response.raise_for_status()
        terminal, events = parse_sse(response)
    encoded = terminal.get("mesh_b64")
    if not encoded:
        raise RuntimeError("3D Gen Studio mesh tool returned no mesh payload")
    mesh_bytes = base64.b64decode(encoded)
    if not mesh_bytes:
        raise RuntimeError("3D Gen Studio mesh tool returned an empty mesh")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(mesh_bytes)
    report = {
        "success": True,
        "backend": "3d_gen_studio_mesh_tools",
        "operation": operation,
        "endpoint": endpoint,
        "options": options,
        "stats": terminal.get("stats"),
        "progress_events": events,
        "output": str(output_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8200")
    parser.add_argument("--operation", choices=("auto-retopo", "auto-uv"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--options-json", default="{}")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()
    run_tool(
        args.url,
        args.operation,
        Path(args.input),
        Path(args.output),
        Path(args.report),
        json.loads(args.options_json),
        args.timeout,
    )


if __name__ == "__main__":
    main()
