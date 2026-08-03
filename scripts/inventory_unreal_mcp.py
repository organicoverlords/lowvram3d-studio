"""Read-only inventory of the existing Unreal MCP client/server stack."""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import concurrent.futures
import time
import urllib.error
import urllib.request
from pathlib import Path


PROJECT = Path(r"C:\Users\Lauri\Desktop\UnrealAITest58")
UPROJECT = PROJECT / "UnrealAITest58.uproject"
SERVER = Path(r"C:\Users\Lauri\AppData\Roaming\npm\node_modules\ultimate-unreal-engine-mcp\dist\cli.js")
NODE = Path(r"C:\Program Files\nodejs\node.exe")
OUTPUT = Path(__file__).resolve().parents[1] / "evidence" / "latest-unreal-mcp" / "mcp_tool_inventory.json"


class Rpc:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [str(NODE), str(SERVER)],
            cwd=str(SERVER.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.next_id = 1

    def request(self, method: str, params: dict) -> dict:
        request_id = self.next_id
        self.next_id += 1
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            assert self.proc.stdout is not None
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP server exited: {self.proc.poll()}")
            message = json.loads(line)
            if message.get("id") == request_id:
                if "error" in message:
                    raise RuntimeError(json.dumps(message["error"]))
                return message.get("result", {})
        raise TimeoutError(method)

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def port_health(port: int) -> dict:
    result = {"port": port, "loopback": "127.0.0.1", "tcp_listening": False}
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            result["tcp_listening"] = True
    except OSError as exc:
        result["error"] = str(exc)
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{port}/mcp", method="GET")
        with urllib.request.urlopen(request, timeout=2) as response:
            result["http_status"] = response.status
            result["http_content_type"] = response.headers.get("Content-Type")
    except Exception as exc:
        result["http_error"] = str(exc)
    return result


def classify_tool(tool: dict) -> list[str]:
    name = tool.get("name", "").lower()
    description = tool.get("description", "").lower()
    categories: list[str] = []
    if any(token in name for token in ("level", "map", "world")):
        categories.append("map_or_world")
    if any(token in name for token in ("actor", "spawn", "transform", "property")):
        categories.append("actor_mutation_or_query")
    if any(token in name for token in ("save", "reload", "load")) or "save" in description:
        categories.append("save_or_load")
    if any(token in name for token in ("async", "job", "queue", "tick", "latent")) or any(token in description for token in ("async", "multi-tick", "latent")):
        categories.append("async_or_multitick_candidate")
    if not categories:
        categories.append("other")
    return categories


def http_request(session_id: str | None, request_id: int, method: str, params: dict) -> tuple[dict, str | None]:
    payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
    body_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {**({"Mcp-Session-Id": session_id, "Mcp-Protocol-Version": "2025-11-25"} if session_id else {}), "Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    request = urllib.request.Request(
        "http://127.0.0.1:8000/mcp",
        data=body_bytes,
        headers=headers,
        method="POST",
    )
    # Epic's dynamic tool-search responses are SSE. curl reliably consumes
    # these short, finite responses on Windows; urllib can wait for the event
    # stream terminator even after the JSON event has arrived.
    command = ["curl.exe", "--silent", "--show-error", "--max-time", "60", "--dump-header", "-", "--request", "POST", "http://127.0.0.1:8000/mcp"]
    for key, value in headers.items():
        command.extend(["--header", f"{key}: {value}"])
    command.extend(["--data-binary", body_bytes.decode("utf-8")])
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    stream_lines: list[str] = []
    deadline = time.monotonic() + 60
    assert process.stdout is not None
    while time.monotonic() < deadline:
        line = process.stdout.readline()
        if not line:
            break
        stream_lines.append(line)
        # MCP's response is a single JSON body or one complete SSE data line.
        if line.startswith("data:") or ("\r\n\r\n" in "".join(stream_lines) and line.lstrip().startswith("{")):
            break
    process.kill()
    process.wait(timeout=5)
    output = "".join(stream_lines)
    header_text, _, body = output.partition("\r\n\r\n")
    if not body:
        header_text, _, body = output.partition("\n\n")
    new_session = session_id
    for line in header_text.splitlines():
        if line.lower().startswith("mcp-session-id:"):
            new_session = line.split(":", 1)[1].strip() or session_id
            break
    if body.lstrip().startswith("event:"):
        data_lines = [line[6:].strip() for line in body.splitlines() if line.startswith("data:")]
        body = data_lines[-1] if data_lines else "{}"
    return (json.loads(body) if body.strip() else {}), new_session


def http_session() -> tuple[str, dict, list[dict]]:
    initialized, session = http_request(None, 1, "initialize", {"protocolVersion": "2025-11-25", "capabilities": {}, "clientInfo": {"name": "scene-pipeline-inventory", "version": "1.0"}})
    http_request(session, 2, "notifications/initialized", {})
    tools, session = http_request(session, 3, "tools/list", {})
    return session or "", initialized.get("result", {}), tools.get("result", {}).get("tools", [])


def describe_one(toolset_name: str) -> tuple[str, dict]:
    session, _, _ = http_session()
    response, _ = http_request(session, 10, "tools/call", {"name": "describe_toolset", "arguments": {"toolset_name": toolset_name}})
    result = response.get("result", {})
    text = ""
    for item in result.get("content", []):
        if item.get("type") == "text":
            text += item.get("text", "")
    return toolset_name, {"description_text": text, "raw_result": result}


def main() -> int:
    project = json.loads(UPROJECT.read_text(encoding="utf-8"))
    enabled_plugins = [item.get("Name") for item in project.get("Plugins", []) if item.get("Enabled")]
    rpc = Rpc()
    try:
        initialize = rpc.request("initialize", {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "scene-pipeline-inventory", "version": "1.0"}})
        assert rpc.proc.stdin is not None
        rpc.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}) + "\n")
        rpc.proc.stdin.flush()
        tool_result = rpc.request("tools/list", {})
        tools = tool_result.get("tools", [])
        inventory = {
            "schema_version": "unreal_mcp_tool_inventory_v1",
            "classification": "PROVEN" if tools else "BLOCKED_NO_TOOLS",
            "repository": str(Path(__file__).resolve().parents[1]),
            "project": str(UPROJECT),
            "mcp_endpoint_requested": "http://127.0.0.1:8000/mcp",
            "mcp_endpoint_health": port_health(8000),
            "existing_bridge_health": port_health(3000),
            "server": {"node": str(NODE), "server": str(SERVER), "server_info": initialize.get("serverInfo"), "protocol_version": initialize.get("protocolVersion")},
            "enabled_project_plugins": enabled_plugins,
            "required_plugin_presence": {name: name in enabled_plugins for name in ("ModelContextProtocol", "ToolsetRegistry", "AllToolsets", "PythonScriptPlugin")},
            "tool_count": len(tools),
            "toolsets": sorted({item.split("_", 2)[1] if item.startswith("ue_") and "_" in item else "unscoped" for item in (tool.get("name", "") for tool in tools)}),
            "tools": [{**tool, "capability_categories": classify_tool(tool)} for tool in tools],
            "mutation_capability_summary": {
                "map_or_world_tools": [tool.get("name") for tool in tools if "map_or_world" in classify_tool(tool)],
                "actor_tools": [tool.get("name") for tool in tools if "actor_mutation_or_query" in classify_tool(tool)],
                "save_or_load_tools": [tool.get("name") for tool in tools if "save_or_load" in classify_tool(tool)],
                "async_or_multitick_candidates": [tool.get("name") for tool in tools if "async_or_multitick_candidate" in classify_tool(tool)],
            },
            "inventory_is_read_only": True,
            "castlegrounds_source_map_mutated": False,
        }
    finally:
        rpc.close()
    http_session_id, http_initialize, http_meta_tools = http_session()
    list_response, _ = http_request(http_session_id, 4, "tools/call", {"name": "list_toolsets", "arguments": {}})
    list_text = "\n".join(item.get("text", "") for item in list_response.get("result", {}).get("content", []) if item.get("type") == "text")
    toolset_names = []
    toolset_descriptions = {}
    for line in list_text.splitlines():
        if line.startswith("- ") and ":" in line:
            name, description = line[2:].split(":", 1)
            toolset_names.append(name.strip())
            toolset_descriptions[name.strip()] = description.strip()
    # The stdio catalog above already contains the complete 134-tool schema.
    # Describe only the toolsets relevant to the requested map/actor/save
    # workflow; asking Epic's server for every dynamic toolset in parallel can
    # starve the editor's HTTP worker and is unnecessary for this read-only
    # capability decision.
    describe_names = [name for name in sorted(set(toolset_names)) if any(token in name for token in (
        "EditorAppToolset", "ActorTools", "AssetTools", "SceneTools", "ScenePipelineTools"))]
    described = dict(describe_one(name) for name in describe_names)
    inventory["official_http_mcp"] = {
        "endpoint": "http://127.0.0.1:8000/mcp",
        "initialize": http_initialize,
        "meta_tools": http_meta_tools,
        "toolset_count": len(set(toolset_names)),
        "toolset_names": sorted(set(toolset_names)),
        "all_toolset_names": sorted(set(toolset_names)),
        "all_toolset_descriptions": toolset_descriptions,
        "described_toolset_count": len(described),
        "toolsets": described,
        "read_only_inventory": True,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "tool_count": inventory["tool_count"], "endpoint_8000": inventory["mcp_endpoint_health"]}, indent=2))
    return 0 if inventory["tool_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
