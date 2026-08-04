"""Client for Epic's in-editor ModelContextProtocol server (streamable HTTP).

``urllib`` and ``requests`` both fail against this server: it answers
``tools/call`` with a ``text/event-stream`` body that is never terminated and
carries no ``Content-Length``, which those libraries report as an immediate EOF
(you get ``200 OK`` with a zero-byte body and no error). This client speaks
HTTP/1.1 over a raw socket so it can stop reading at the first complete SSE
frame instead.

Tool naming: ``tools/list`` advertises only ``list_toolsets``,
``describe_toolset`` and ``call_tool``. Everything else is reached through
``call_tool`` with the toolset name and the bare tool name passed separately --
sending the fully-qualified ``Toolset.tool`` string as ``tool_name`` returns
"Tool not found".
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any

HOST = "127.0.0.1"
PORT = 8000
PATH = "/mcp"


class EditorMCPError(RuntimeError):
    pass


class EditorMCP:
    def __init__(self, host: str = HOST, port: int = PORT, timeout: float = 300.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.session_id: str | None = None
        self._sock: socket.socket | None = None
        self._buf = b""
        self._id = 0

    # -- transport ---------------------------------------------------------
    def _send(self, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        head = (
            f"POST {PATH} HTTP/1.1\r\nHost: {self.host}:{self.port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Accept: application/json, text/event-stream\r\n"
            f"Content-Length: {len(payload)}\r\n"
        )
        if self.session_id:
            head += f"Mcp-Session-Id: {self.session_id}\r\n"
        head += "\r\n"
        assert self._sock is not None
        self._sock.sendall(head.encode("utf-8") + payload)

    def _recv_chunk(self, deadline: float) -> None:
        if time.time() > deadline:
            raise EditorMCPError("timed out reading from the editor MCP server")
        assert self._sock is not None
        chunk = self._sock.recv(65536)
        if not chunk:
            raise EditorMCPError("editor closed the MCP connection")
        self._buf += chunk

    def _read_response(self, timeout: float) -> dict[str, Any] | None:
        deadline = time.time() + timeout
        while b"\r\n\r\n" not in self._buf:
            self._recv_chunk(deadline)
        raw_head, self._buf = self._buf.split(b"\r\n\r\n", 1)
        headers: dict[str, str] = {}
        for line in raw_head.decode("utf-8", "replace").split("\r\n")[1:]:
            if ":" in line:
                key, value = line.split(":", 1)
                headers[key.strip().lower()] = value.strip()
        if headers.get("mcp-session-id"):
            self.session_id = headers["mcp-session-id"]

        if "content-length" in headers:
            need = int(headers["content-length"])
            while len(self._buf) < need:
                self._recv_chunk(deadline)
            body, self._buf = self._buf[:need], self._buf[need:]
            text = body.decode("utf-8", "replace").strip()
            return json.loads(text) if text.startswith("{") else None

        # Event stream: stop at the first complete data frame.
        while True:
            for line in self._buf.decode("utf-8", "replace").splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    frame = line[5:].strip()
                    if frame.startswith("{"):
                        self._buf = b""
                        return json.loads(frame)
            self._recv_chunk(deadline)

    # -- protocol ----------------------------------------------------------
    def connect(self) -> dict[str, Any]:
        self._sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self._sock.settimeout(self.timeout)
        self._buf = b""
        self._id += 1
        self._send({"jsonrpc": "2.0", "id": self._id, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "uemcp", "version": "1"}}})
        info = self._read_response(60) or {}
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        self._read_response(60)
        return info

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "EditorMCP":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def rpc(self, method: str, params: dict[str, Any] | None = None,
            timeout: float | None = None) -> dict[str, Any]:
        if self._sock is None:
            self.connect()
        self._id += 1
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": self._id, "method": method}
        if params is not None:
            body["params"] = params
        self._send(body)
        response = self._read_response(timeout or self.timeout) or {}
        if "error" in response:
            raise EditorMCPError(f"{method}: {response['error']}")
        return response

    # -- toolsets ----------------------------------------------------------
    @staticmethod
    def _text(response: dict[str, Any]) -> str:
        result = response.get("result")
        if not isinstance(result, dict):
            return ""
        return "\n".join(item.get("text", "") for item in (result.get("content") or [])
                         if item.get("type") == "text")

    def list_toolsets(self) -> list[str]:
        text = self._text(self.rpc("tools/call", {"name": "list_toolsets", "arguments": {}}))
        names = []
        for line in text.splitlines():
            if line.startswith("- ") and ":" in line:
                name = line[2:].split(":", 1)[0].strip()
                # Descriptions wrap onto bulleted lines; real names are dotted.
                if "." in name and " " not in name:
                    names.append(name)
        return names

    def describe_toolset(self, toolset: str) -> dict[str, Any]:
        text = self._text(self.rpc("tools/call", {
            "name": "describe_toolset", "arguments": {"toolset_name": toolset}}))
        return json.loads(text)

    def call(self, tool_name: str, arguments: dict[str, Any] | None = None,
             toolset: str | None = None, timeout: float | None = None) -> Any:
        """Invoke a toolset tool.

        ``tool_name`` must be the bare name; pass the owning toolset separately.
        """
        args: dict[str, Any] = {"tool_name": tool_name, "arguments": arguments or {}}
        if toolset:
            args["toolset_name"] = toolset
        text = self._text(self.rpc("tools/call", {"name": "call_tool", "arguments": args}, timeout))
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text

    def is_ready(self) -> bool:
        try:
            self.connect()
            return bool(self.list_toolsets())
        except Exception:
            return False
        finally:
            self.close()
