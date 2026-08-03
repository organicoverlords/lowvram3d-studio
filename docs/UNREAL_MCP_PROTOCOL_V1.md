# Unreal editor protocol v1

The previous bridge mixed three incompatible contracts: WebSocket JSON-RPC,
length-prefixed `{type, params}` commands, and MCP-style JSON-RPC. That made a
healthy listener look broken to one client and made failures appear as generic
EOFs. The new contract has one canonical editor-side protocol and explicit
compatibility adapters.

## Contract

The canonical editor connection is loopback TCP only:

```text
uint32 big-endian byte length
UTF-8 JSON-RPC 2.0 object
```

The payload is standard MCP JSON-RPC. The session begins with `initialize`,
then uses `tools/list`, `tools/call`, and ordinary request/response correlation
by JSON-RPC `id`. Maximum frame size is 100 MiB; short reads, invalid UTF-8,
invalid JSON, oversized frames, and mismatched response IDs fail closed.

The MCP-facing sidecar remains the process that speaks stdio or Streamable
HTTP to an agent. The Unreal plugin is an editor adapter, not a second,
slightly different MCP server. A compatibility adapter may accept the older
`{type, params}` envelope during migration, but new clients must not depend on
it.

## Capability and lifecycle rules

- Bind to `127.0.0.1`; never expose the editor listener on `0.0.0.0`.
- Publish the selected port and process identity in a lockfile.
- `initialize` returns the negotiated protocol version, server identity, and
  capabilities.
- Every request has one response with the same JSON-RPC ID, or a structured
  JSON-RPC error.
- Long-running captures use a job receipt and explicit cancellation rather
  than an unbounded socket timeout.
- Read-only health and capability probes are separate from mutating editor
  tools.
- Screenshot commands return an artifact receipt containing camera ID,
  resolution, render route, map, actor reference, file hash, and completion
  status. A PNG without that receipt is not visual proof.

## Why this boundary

MCP standardizes JSON-RPC messages and currently standardizes stdio and
Streamable HTTP transports. A custom TCP transport is allowed, but it must
preserve JSON-RPC and document framing. This adapter therefore keeps the
custom part to a small, deterministic byte-stream frame while leaving MCP
semantics intact. Unreal's own Remote Control API remains an optional engine
side route for property/function access; it is not mixed into the MCP framing
layer.

## Migration

1. Use `workers/scene_pipeline/unreal_rpc.py` for new local editor calls.
2. Keep the WebSocket adapter only for existing CLI clients.
3. Remove the legacy `{type, params}` adapter after all sidecars have moved to
   canonical JSON-RPC.
4. Use the official Unreal Remote Control HTTP/WebSocket API for engine-native
   property/function operations where it is sufficient; use the local bridge
   only for pipeline-specific commands such as deterministic screenshot jobs.
