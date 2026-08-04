# Agent instructions

Applies to Claude Code, Codex and OpenCode working in this repository.

## Talking to Unreal Engine

Read `docs/unreal-mcp/README.md` before the first editor interaction of a
session. It is short, and it is the difference between working and guessing.

The rules that matter most:

1. **Start with the doctor.** `python -m uemcp doctor` from `unreal/`. It
   checks every surface and names the repair for each failure. Do not
   improvise a diagnosis before running it.

2. **Prefer `uemcp` over the `mcp__unreal-engine__*` tools.** Those tools
   depend on a plugin probe defect that is patched in source but not yet
   rebuilt. `uemcp` opens a connection per call and writes immediately, so it
   is unaffected.

3. **Never hardcode the bridge port.** It is derived from the project path and
   republished to `<project>/Saved/UE_MCP_Bridge/port.json` on every boot.
   `uemcp` reads the lockfile and verifies the owning pid is alive.

4. **Do not use an HTTP library against port 8000.** It answers with an
   unterminated event stream that `urllib` and `requests` report as an empty
   body with no error. Use `uemcp.EditorMCP`.

5. **`execute_python` needs a result variable.** Code runs with `ExecuteFile`
   semantics, so a top-level `return` is a syntax error. Assign a variable and
   read it back — `Bridge.python_json` handles the quoting.

6. **Purge after editing editor-side Python.** The embedded interpreter caches
   imports for the session: `python -m uemcp purge <prefix>`.

7. **Capture with `shot` / `camera-shot`, not screenshots.** They are
   off-screen scene renders: no editor UI, no window focus needed, no PIE, no
   player pawn, and they return a byte size you can assert on.

## Evidence

Claims about the editor need a receipt. A capture is proven by a PNG whose
dimensions, size and content were checked — not by a manifest asserting that a
capture happened. When a step fails, say so with the actual error rather than
narrowing the scope and reporting success.

## Housekeeping

If MCP servers appear duplicated or the editor log fills with
"Connected / Lost connection", sweep orphans:

```bash
pwsh -File scripts/windows/ue-mcp-cleanup.ps1
```

Never launch an Unreal MCP server with `npx -y`. Point at the pinned
`node <path>/dist/cli.js` entrypoint instead; `npx` re-resolves from the
network every launch and leaks process chains.
