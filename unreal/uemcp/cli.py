"""Command line front end: ``python -m uemcp <verb>``.

Every verb prints JSON on stdout so an agent can parse the result instead of
scraping prose.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .bridge import Bridge, BridgeError
from .doctor import diagnose
from .editor_mcp import EditorMCP
from .viewport import Viewport


def _emit(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _vec(text: str | None) -> tuple[float, float, float] | None:
    if not text:
        return None
    parts = [float(p) for p in text.replace(",", " ").split()]
    if len(parts) != 3:
        raise SystemExit(f"expected three numbers, got {text!r}")
    return parts[0], parts[1], parts[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uemcp", description=__doc__)
    parser.add_argument("--project", default=None, help="Unreal project root")
    sub = parser.add_subparsers(dest="verb", required=True)

    sub.add_parser("doctor", help="diagnose every agent-to-editor surface")
    sub.add_parser("health", help="bridge health check")
    sub.add_parser("viewport", help="print viewport location, rotation and FOV")
    sub.add_parser("toolsets", help="list toolsets on the in-editor MCP server")

    p = sub.add_parser("python", help="run Python inside the editor")
    p.add_argument("source", help="inline code, or @path/to/script.py")
    p.add_argument("--result-variable", default="result")
    p.add_argument("--json", action="store_true", help="decode the result as JSON")

    p = sub.add_parser("purge", help="drop cached editor-side Python modules")
    p.add_argument("prefix")

    p = sub.add_parser("console", help="run an editor console command")
    p.add_argument("command")

    p = sub.add_parser("look-at", help="move the viewport camera and aim it")
    p.add_argument("--eye", required=True, help='"x y z"')
    p.add_argument("--target", required=True, help='"x y z"')

    p = sub.add_parser("focus", help="frame a named actor in the viewport")
    p.add_argument("actor_label")

    p = sub.add_parser("orbit", help="orbit the viewport camera around a point")
    p.add_argument("--target", required=True, help='"x y z"')
    p.add_argument("--distance", type=float, required=True)
    p.add_argument("--yaw", type=float, required=True)
    p.add_argument("--pitch", type=float, default=-20.0)

    p = sub.add_parser("shot", help="render an off-screen PNG of the editor world")
    p.add_argument("output")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fov", type=float, default=90.0)
    p.add_argument("--location", default=None, help='"x y z"')
    p.add_argument("--focus-actor", default=None)

    p = sub.add_parser("camera-shot", help="render exactly what a named CameraActor sees")
    p.add_argument("camera_label")
    p.add_argument("output")
    p.add_argument("--width", type=int, required=True)
    p.add_argument("--height", type=int, required=True)

    p = sub.add_parser("camera", help="print a named CameraActor's projection contract")
    p.add_argument("camera_label")

    p = sub.add_parser("call", help="call a tool on the in-editor MCP server")
    p.add_argument("tool_name")
    p.add_argument("--toolset", default=None)
    p.add_argument("--arguments", default="{}", help="JSON object")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.verb == "doctor":
        report = diagnose(args.project)
        _emit(report)
        return 0 if report["ok"] else 1

    if args.verb == "toolsets":
        with EditorMCP() as client:
            _emit(client.list_toolsets())
        return 0

    if args.verb == "call":
        with EditorMCP() as client:
            _emit(client.call(args.tool_name, json.loads(args.arguments), args.toolset))
        return 0

    bridge = Bridge(args.project)
    view = Viewport(bridge)

    try:
        if args.verb == "health":
            _emit(bridge.health())
        elif args.verb == "viewport":
            _emit(view.info())
        elif args.verb == "python":
            source = args.source
            if source.startswith("@"):
                source = Path(source[1:]).read_text(encoding="utf-8")
            if args.json:
                _emit(bridge.python_json(source, args.result_variable))
            else:
                _emit(bridge.python(source, args.result_variable))
        elif args.verb == "purge":
            _emit(bridge.purge_modules(args.prefix))
        elif args.verb == "console":
            _emit(bridge.console(args.command))
        elif args.verb == "look-at":
            _emit(view.look_at(_vec(args.eye), _vec(args.target)))
        elif args.verb == "focus":
            _emit(view.focus(args.actor_label))
        elif args.verb == "orbit":
            _emit(view.orbit(_vec(args.target), args.distance, args.yaw, args.pitch))
        elif args.verb == "shot":
            _emit(view.capture(args.output, args.width, args.height, args.fov,
                               _vec(args.location), focus_actor_label=args.focus_actor))
        elif args.verb == "camera-shot":
            _emit(view.capture_from_camera(args.camera_label, args.output,
                                           args.width, args.height))
        elif args.verb == "camera":
            _emit(view.camera_contract(args.camera_label))
        else:
            raise SystemExit(f"unhandled verb {args.verb}")
    except BridgeError as exc:
        _emit({"error": str(exc)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
