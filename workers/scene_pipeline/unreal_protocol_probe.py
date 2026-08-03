"""Run a read-only canonical MCP/Unreal bridge probe and write a receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unreal_rpc import UnrealRPCClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lockfile = args.project / "Saved" / "UE_MCP_Bridge" / "port.json"
    receipt: dict[str, object] = {"schema": "unreal_canonical_mcp_probe_v1", "lockfile": str(lockfile)}
    try:
        lock = json.loads(lockfile.read_text(encoding="utf-8"))
        port = int(lock["port"])
        with UnrealRPCClient(port=port) as client:
            init = client.initialize()
            tools = client.tools_list()
            health = client.call("health_check")
            read_only_call = client.tools_call(
                "execute_python",
                {"code": 'result={"protocol":"canonical","read_only":True}'},
            )
            receipt.update(
                {
                    "classification": "PROVEN",
                    "port": port,
                    "initialize": init,
                    "tool_count": len(tools),
                    "health": health,
                    "read_only_tools_call": read_only_call,
                    "framing": "uint32_be_length_prefixed_utf8_json",
                    "mcp_message_model": "jsonrpc_2.0",
                }
            )
    except Exception as exc:  # receipt must preserve the precise blocker
        receipt.update({"classification": "BLOCKED", "error": str(exc)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0 if receipt["classification"] == "PROVEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
