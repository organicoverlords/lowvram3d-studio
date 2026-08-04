"""Length-prefixed JSON-RPC client for the ``UE_MCP_Bridge`` editor plugin.

Wire format (``FMCPBridgeServer::ProcessLengthPrefixedMessages``):

    [4-byte big-endian payload length][UTF-8 JSON payload]

The plugin peeks at the first byte of a new connection to choose a framing:
``{``/``[`` selects newline-delimited JSON, ``G`` selects a WebSocket upgrade,
and anything else selects length-prefixed framing. A 4-byte length header for a
realistic payload always starts with ``0x00``, so length-prefixed framing is
selected simply by writing the header first.

That probe uses a single five-second ``select()``. A client that connects and
then stays silent is handed to the WebSocket handshake path and dropped. This
client therefore opens a connection per call and writes immediately, which
sidesteps the defect entirely. Do not "optimise" it into a persistent idle
socket without fixing the plugin first -- see docs/unreal-mcp/README.md.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import time
from pathlib import Path
from typing import Any

DEFAULT_PROJECT = Path(r"C:\Users\Lauri\Desktop\UnrealAITest58")
MAX_PAYLOAD = 100 * 1024 * 1024


class BridgeError(RuntimeError):
    """Raised when the bridge is unreachable or returns an error envelope."""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    except Exception:
        return True


def discover_port(project_root: Path | str | None = None) -> int:
    """Resolve the bridge port from the project's lockfile.

    The plugin derives a port from the project path and republishes it to
    ``Saved/UE_MCP_Bridge/port.json`` on every boot, so the lockfile -- not a
    hardcoded 55557 -- is the only trustworthy source.
    """
    root = Path(project_root or os.environ.get("UE_PROJECT_ROOT") or DEFAULT_PROJECT)
    lockfile = root / "Saved" / "UE_MCP_Bridge" / "port.json"
    if not lockfile.is_file():
        raise BridgeError(
            f"no bridge lockfile at {lockfile}; the editor is not running, or "
            f"UE_MCP_Bridge is disabled for this project"
        )
    data = json.loads(lockfile.read_text(encoding="utf-8"))
    port = int(data["port"])
    pid = int(data.get("pid", 0))
    if pid and not _pid_alive(pid):
        raise BridgeError(
            f"lockfile names editor pid {pid}, which is gone; the file is stale. "
            f"Restart the editor or delete {lockfile}"
        )
    return port


class Bridge:
    """One connection per call against the editor bridge."""

    def __init__(
        self,
        project_root: Path | str | None = None,
        port: int | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.project_root = Path(project_root or os.environ.get("UE_PROJECT_ROOT") or DEFAULT_PROJECT)
        self.timeout = timeout
        self._port = port

    @property
    def port(self) -> int:
        if self._port is None:
            self._port = discover_port(self.project_root)
        return self._port

    # -- transport ---------------------------------------------------------
    def call(self, method: str, params: dict[str, Any] | None = None,
             timeout: float | None = None) -> dict[str, Any]:
        timeout = timeout or self.timeout
        payload = json.dumps({"method": method, "params": params or {}}).encode("utf-8")
        deadline = time.time() + timeout
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=timeout)
        try:
            sock.settimeout(timeout)
            sock.sendall(struct.pack(">I", len(payload)) + payload)

            def exact(count: int) -> bytes:
                buf = b""
                while len(buf) < count:
                    if time.time() > deadline:
                        raise BridgeError(f"timed out reading response to {method!r}")
                    chunk = sock.recv(min(65536, count - len(buf)))
                    if not chunk:
                        raise BridgeError(f"editor closed the connection during {method!r}")
                    buf += chunk
                return buf

            length = struct.unpack(">I", exact(4))[0]
            if not 0 < length <= MAX_PAYLOAD:
                raise BridgeError(f"implausible response length {length} for {method!r}")
            body = exact(length)
        finally:
            sock.close()

        envelope = json.loads(body.decode("utf-8"))
        if "error" in envelope:
            raise BridgeError(f"{method}: {envelope['error']}")
        return envelope.get("result", envelope)

    # -- convenience -------------------------------------------------------
    def health(self) -> dict[str, Any]:
        return self.call("health_check", timeout=15.0)

    def is_ready(self) -> bool:
        try:
            return bool(self.health().get("editorReady"))
        except Exception:
            return False

    def python(self, code: str, result_variable: str = "result",
               timeout: float | None = None) -> dict[str, Any]:
        """Run Python inside the editor and read one variable back.

        ``code`` executes in the ``__main__`` scope with ``ExecuteFile``
        semantics, so a top-level ``return`` is a syntax error. Assign the value
        you want instead; it is evaluated afterwards and returned as ``result``.
        """
        response = self.call(
            "execute_python",
            {"code": code, "resultVariable": result_variable},
            timeout=timeout,
        )
        if not response.get("success"):
            log = "\n".join(entry.get("output", "") for entry in response.get("log_output", []))
            raise BridgeError(f"python failed in editor:\n{log or response}")
        return response

    def python_json(self, code: str, result_variable: str = "result",
                    timeout: float | None = None) -> Any:
        """Run Python whose result variable holds a JSON string, and decode it.

        The bridge evaluates the variable with ``repr`` semantics, so a ``str``
        arrives wrapped in quotes. Strip one layer before decoding.
        """
        response = self.python(code, result_variable, timeout)
        raw = response.get("result", "")
        if not response.get("resultVariableResolved"):
            raise BridgeError(f"result variable {result_variable!r} was never assigned")
        text = raw.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
            text = text[1:-1].encode().decode("unicode_escape")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise BridgeError(f"result variable was not JSON ({exc}): {raw[:400]}") from exc

    def run_file(self, path: Path | str, args: list[str] | None = None,
                 result_variable: str = "result", timeout: float | None = None) -> dict[str, Any]:
        return self.call(
            "run_python_file",
            {"path": str(path), "args": args or [], "resultVariable": result_variable},
            timeout=timeout,
        )

    def purge_modules(self, prefix: str) -> dict[str, Any]:
        """Drop cached editor-side Python modules so edited tools reload.

        The embedded interpreter caches imports for the whole editor session;
        without this an edited module keeps running its old code.
        """
        return self.call("purge_python_modules", {"prefix": prefix})

    def console(self, command: str) -> dict[str, Any]:
        return self.call("execute_command", {"command": command})
