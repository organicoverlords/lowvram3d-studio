# Driving Unreal Engine from an agent

Everything here was verified against the live editor on 2026-08-04:
UE **5.8.0-55116800**, embedded Python **3.11.8**, project
`C:\Users\Lauri\Desktop\UnrealAITest58`.

If something does not work, run the doctor first. It names the specific repair
for every failure it finds:

```bash
python -m uemcp doctor
```

Run it from `C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803\unreal`.

---

## The three surfaces

A running editor exposes three independent channels. They are not
interchangeable, and picking the wrong one is the single biggest source of
wasted time.

| Surface | Endpoint | Reach | Use it for |
|---|---|---|---|
| **UE_MCP_Bridge** | TCP `127.0.0.1:49538` | 713 handlers **+ arbitrary `execute_python`** | Everything. This is the workhorse. |
| **ModelContextProtocol** (Epic) | HTTP `127.0.0.1:8000/mcp` | ~54 toolsets, ~842 tools, **no** arbitrary Python | Structured asset/Sequencer/Niagara/PCG work |
| **UnrealOpenCode** | TCP `127.0.0.1:3000` | project-specific | Legacy; prefer the two above |

The bridge port is **derived from the project path**, not fixed. It is
republished on every editor boot to:

```
<project>/Saved/UE_MCP_Bridge/port.json
```

Always read that lockfile. Never hardcode `55557` — that default belongs to an
older plugin generation and matches nothing on this machine.

---

## The fast path

```bash
cd C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803\unreal

python -m uemcp doctor                       # is anything reachable?
python -m uemcp viewport                     # where is the camera?
python -m uemcp focus Castlegrounds_ReconstructedMesh
python -m uemcp orbit --target "0 0 200" --distance 1200 --yaw 35
python -m uemcp shot C:/tmp/look.png --width 1280 --height 720
```

### Viewport control

| Verb | Effect |
|---|---|
| `viewport` | Print current location, rotation, FOV |
| `look-at --eye "x y z" --target "x y z"` | Place the camera and aim it |
| `focus <ActorLabel>` | Frame an actor, like pressing **F** |
| `orbit --target "x y z" --distance D --yaw Y [--pitch P]` | Orbit a point and aim inward |
| `shot <out.png> [--width --height --fov --location --focus-actor]` | Off-screen render |
| `camera <CameraLabel>` | Read a CameraActor's exact projection contract |
| `camera-shot <CameraLabel> <out.png> --width W --height H` | Render exactly what that camera sees |

`shot` and `camera-shot` are **scene-capture renders**, not screengrabs. They
carry no editor UI, do not need the window focused, never touch PIE or the
player pawn, and return the byte size so the caller can assert the file is
real. Prefer them over any screenshot verb for visual evidence.

### Running Python in the editor

```bash
python -m uemcp python "import unreal; result = str(unreal.SystemLibrary.get_engine_version())"
python -m uemcp python @scripts/probe.py --json
```

From Python:

```python
from uemcp import Bridge

bridge = Bridge()
data = bridge.python_json("""
import json, unreal
subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
result = json.dumps([str(a.get_actor_label()) for a in subsystem.get_all_level_actors()])
""")
```

---

## Traps that cost real time

Each of these produced a confident-looking wrong answer before it was pinned
down. They are listed with the symptom you will actually see.

### The MCP server returns `{}` for every live tool

`ultimate-unreal-engine-mcp` reads `UE_PROJECT_ROOT` and falls back to
`process.cwd()`. Launched by a desktop app, the cwd is not your project, so it
hunts for a bridge that does not exist and every tool fails identically.

**Fix:** set `UE_PROJECT_ROOT` (and `UE_PLUGIN_PORT`) in the server's `env`
block. Already applied to Claude, Codex and OpenCode — see *Configuration*.

### `urllib`/`requests` see an empty body from port 8000

The editor answers `tools/call` with a `text/event-stream` body carrying no
`Content-Length` that is never terminated. `urllib` reports this as `200 OK`
with a **zero-byte body and no exception** — it looks like the editor returned
nothing.

**Fix:** `uemcp.EditorMCP` speaks HTTP/1.1 over a raw socket and stops at the
first complete SSE frame. Do not reach for an HTTP library here.

### "Tool not found" on port 8000 for a tool you can see listed

`tools/list` advertises only `list_toolsets`, `describe_toolset` and
`call_tool`. Everything else goes through `call_tool`, with the toolset name
and the bare tool name passed **separately**. Sending the fully-qualified
`Toolset.tool` string as `tool_name` fails.

```python
client.call("get_execution_environment",
            toolset="editor_toolset.toolsets.programmatic.ProgrammaticToolset")
```

### `execute_python` returns `"None"`

The code runs with `ExecuteFile` semantics in the `__main__` scope, so a
top-level `return` is a syntax error and nothing is captured by default.
**Assign a variable and name it** via `resultVariable`; it is evaluated
afterwards. `uemcp` defaults that name to `result`.

The value comes back with `repr` semantics, so a `str` arrives quoted.
`Bridge.python_json` strips one layer before decoding — use it rather than
parsing the raw field.

### An edited editor-side Python module keeps running old code

The embedded interpreter caches imports for the whole editor session.

```bash
python -m uemcp purge lowvram3d
```

### The editor log fills with "Connected / Lost connection"

The plugin probed a new connection with a **single five-second `select()`** and
handed a still-silent socket to the WebSocket handshake, which closed it. The
npm client opens its socket at startup and stays quiet until a tool call
arrives, so it was dropped, reconnected, and dropped again — with every tool
call timing out at 30 s.

**Fix:** patched in
`Plugins/UE_MCP_Bridge/Source/UE_MCP_Bridge/Private/BridgeServer.cpp` to poll
in one-second slices and wait indefinitely. Each connection already runs on its
own thread, so waiting is free.

> **This patch is source-only until the plugin is rebuilt.** The rebuild needs
> the editor closed. Until then, use `uemcp` (which opens a connection per call
> and writes immediately, sidestepping the probe entirely) rather than the
> `mcp__unreal-engine__*` tools.

### Port 8000 is never listening, even with `bAutoStartServer=True`

`UModelContextProtocolSettings` is declared
`UCLASS(config=EditorPerProjectUserSettings)` in the
**`ModelContextProtocolEngine`** module. A block placed in `DefaultEditor.ini`,
or under a `[/Script/ModelContextProtocol....]` section, parses without error
and is **silently ignored** — the server then never binds and the port looks
dead for no visible reason.

The only file and section the engine reads:

```ini
; Config/DefaultEditorPerProjectUserSettings.ini
[/Script/ModelContextProtocolEngine.ModelContextProtocolSettings]
bAutoStartServer=True
ServerPortNumber=8000
ServerUrlPath=/mcp
```

Command-line `-ModelContextProtocolStartServer` forces it on regardless.

### `mcp__unreal-engine__*` tools time out after exactly 30 s

Distinct from the reconnect storm above, and it survives the probe fix. The npm
client stamps every command with a `correlationId` and resolves its promise
only when a response carries the same id back. This plugin build has **no
`correlationId` handling at all**, so the reply is never matched and the client
times out on every call.

The npm package (0.1.25, the latest published) is older than the installed
plugin. Until they are reconciled, use `uemcp`, which does not depend on
correlation.

### A modal dialog freezes every tool call

Editor modals block the game thread, and MCP tool calls execute there — so a
dialog waiting for a click stalls every surface at once. `UE_MCP_Bridge`
installs a `ModalMessageDialog` hook to mitigate this, but if calls hang with
the editor otherwise healthy, look at the editor window for a dialog before
investigating anything else.

### Duplicate MCP servers pile up

`npx -y <package>` re-resolves from the network on every launch and leaks a
three-process chain (`npx` → `cmd` → `node`). Fifteen orphaned chains were
found in one session, all reconnecting to the editor.

**Fix:** configs now point at a pinned `node <path>/dist/cli.js`. To sweep up:

```bash
pwsh -File scripts/windows/ue-mcp-cleanup.ps1
```

It keeps chains owned by a running client and reports the editor's live ports.

---

## Configuration

All three clients are wired to the same verified surfaces. Backups from before
the change sit beside each file with a `.bak-20260804` suffix.

| Client | File | Server name |
|---|---|---|
| Claude | `%APPDATA%\Claude\claude_desktop_config.json` | `unreal-engine` |
| Codex desktop | `%USERPROFILE%\.codex\config.toml` | `unreal_engine` |
| OpenCode | `%USERPROFILE%\.config\opencode\opencode.json` | `unreal_engine`, `unreal-official` |

Each sets:

```
UE_PROJECT_ROOT = C:\Users\Lauri\Desktop\UnrealAITest58
UE_PLUGIN_PORT  = 49538
```

Removed as dead: Claude's `unreal` server (pointed at a `mcp_bridge.py` that
does not exist) and OpenCode's `unreal` / `monolith` servers.

`UE_PLUGIN_PORT` is pinned because the installed npm client (0.1.25, the latest
published) cannot read the port lockfile. **If the project is ever moved or
renamed, the derived port changes** — re-read `port.json` and update the three
configs.

---

## Reference

- `docs/unreal-mcp/bridge-handlers.txt` — all 713 bridge handler names.
- `unreal/uemcp/` — the client library: `bridge`, `editor_mcp`, `viewport`,
  `doctor`, `cli`.

Handlers worth knowing:

| Handler | Use |
|---|---|
| `execute_python`, `run_python_file` | Arbitrary editor Python |
| `purge_python_modules` | Drop cached editor-side modules |
| `capture_scene_png` | Deterministic off-screen PNG render |
| `get_viewport_info`, `set_viewport_camera`, `focus_viewport_on_actor` | Viewport |
| `hit_test_viewport_pixel` | Which actor is under a pixel |
| `place_actor`, `place_actors_batch`, `set_actor_property` | Scene edits |
| `execute_command` | Editor console command |
| `check_for_crashes`, `list_crashes`, `get_crash_info` | Post-mortem |
