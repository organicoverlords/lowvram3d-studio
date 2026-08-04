"""Canonical client layer for driving a running Unreal Editor from an agent.

Two independent transports reach the same editor process:

``bridge``
    The ``UE_MCP_Bridge`` plugin's length-prefixed JSON-RPC TCP server. This is
    the only surface that exposes ``execute_python``, so it is the workhorse.

``editor_mcp``
    Epic's ``ModelContextProtocol`` plugin on ``http://127.0.0.1:8000/mcp``.
    Streamable HTTP, ~840 registered toolset tools, no arbitrary Python.

Both are loopback-only and require the editor to be running.
"""

from .bridge import Bridge, BridgeError, discover_port
from .editor_mcp import EditorMCP, EditorMCPError
from .viewport import Viewport, look_at_rotation

__all__ = [
    "Bridge",
    "BridgeError",
    "discover_port",
    "EditorMCP",
    "EditorMCPError",
    "Viewport",
    "look_at_rotation",
]
