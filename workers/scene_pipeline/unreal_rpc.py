"""Canonical local RPC client for the Unreal Editor bridge.

The editor-side adapter uses MCP JSON-RPC semantics over a localhost,
length-prefixed byte stream.  The framing is deliberately separate from the
MCP/JSON-RPC message model so the stdio or Streamable HTTP MCP sidecar can be
changed without changing editor requests.
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from typing import Any, Callable


MAX_FRAME_BYTES = 100 * 1024 * 1024
PROTOCOL_VERSION = "2025-11-25"


class UnrealRPCError(RuntimeError):
    """A protocol, transport, or JSON-RPC error."""


def encode_frame(message: dict[str, Any], *, max_bytes: int = MAX_FRAME_BYTES) -> bytes:
    """Encode one UTF-8 JSON-RPC message with a 4-byte big-endian length."""

    payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if not payload or len(payload) > max_bytes:
        raise UnrealRPCError(f"frame size {len(payload)} is outside 1..{max_bytes}")
    return struct.pack(">I", len(payload)) + payload


def recv_exact(sock: Any, size: int) -> bytes:
    """Read exactly *size* bytes, handling short TCP reads and EOF."""

    if size < 0:
        raise UnrealRPCError("negative receive size")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise UnrealRPCError(f"connection closed after {size - remaining}/{size} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_frame(sock: Any, *, max_bytes: int = MAX_FRAME_BYTES) -> dict[str, Any]:
    """Read and validate one length-prefixed JSON object."""

    header = recv_exact(sock, 4)
    size = struct.unpack(">I", header)[0]
    if size == 0 or size > max_bytes:
        raise UnrealRPCError(f"invalid frame size {size}")
    try:
        message = json.loads(recv_exact(sock, size).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise UnrealRPCError(f"invalid UTF-8 JSON frame: {exc}") from exc
    if not isinstance(message, dict):
        raise UnrealRPCError("JSON-RPC frame must contain an object")
    return message


@dataclass
class UnrealRPCClient:
    """Small synchronous client for the canonical editor-side RPC contract."""

    host: str = "127.0.0.1"
    port: int = 55557
    timeout: float = 10.0
    socket_factory: Callable[..., Any] = socket.create_connection

    def __post_init__(self) -> None:
        self._sock: Any | None = None
        self._next_id = 1

    def connect(self) -> "UnrealRPCClient":
        if self._sock is None:
            self._sock = self.socket_factory((self.host, self.port), self.timeout)
            self._sock.settimeout(self.timeout)
        return self

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "UnrealRPCClient":
        return self.connect()

    def __exit__(self, *_: object) -> None:
        self.close()

    def call(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if not method or not isinstance(method, str):
            raise UnrealRPCError("method must be a non-empty string")
        self.connect()
        request_id = self._next_id
        self._next_id += 1
        request = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}
        self._sock.sendall(encode_frame(request))
        response = decode_frame(self._sock)
        if response.get("jsonrpc") != "2.0" or response.get("id") != request_id:
            raise UnrealRPCError(f"invalid response correlation for {method}: {response}")
        if "error" in response:
            raise UnrealRPCError(f"{method} failed: {response['error']}")
        if "result" not in response:
            raise UnrealRPCError(f"response omitted result for {method}")
        return response["result"]

    def initialize(self, client_name: str = "lowvram3d-scene-pipeline") -> dict[str, Any]:
        result = self.call(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {"name": client_name, "version": "1.0.0"},
            },
        )
        if not isinstance(result, dict) or result.get("protocolVersion") != PROTOCOL_VERSION:
            raise UnrealRPCError(f"unsupported editor protocol response: {result}")
        return result

    def tools_list(self) -> list[dict[str, Any]]:
        result = self.call("tools/list")
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list):
            raise UnrealRPCError(f"tools/list returned invalid result: {result}")
        return tools

    def tools_call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        result = self.call("tools/call", {"name": name, "arguments": arguments or {}})
        if not isinstance(result, dict) or not isinstance(result.get("content"), list):
            raise UnrealRPCError(f"tools/call returned invalid result: {result}")
        return result
