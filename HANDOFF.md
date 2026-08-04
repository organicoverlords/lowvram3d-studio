# Handoff — 2026-08-04

Repo: `C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803`
Branch: `agent/scene-pipeline-smoke-20260803`
Head: `7fbbce8` (session started at `2e228c1`)
Unreal project: `C:\Users\Lauri\Desktop\UnrealAITest58\UnrealAITest58.uproject` (UE 5.8.0)

**Start here:** `docs/unreal-mcp/README.md`, then `docs/pipelines/README.md`.
Run `python -m uemcp doctor` from `unreal/` before diagnosing anything.

---

## 1. What actually works now

One command takes an image to a built, rendered Unreal scene:

```bash
PYTHONPATH=src python -m lowvram3d.image_to_scene_pipeline \
  --image "C:/Users/Lauri/Downloads/benchmarkpics/treesandbarn.png" \
  --project "C:/Users/Lauri/Desktop/UnrealAITest58/UnrealAITest58.uproject" \
  --scene-id barn_auto --input-kind scene --quality-tier preview \
  --output-root "/Game/AgentProof/BarnAuto" --evidence-root evidence/barn-auto
```

Ends with `PIPELINE_CLASSIFICATION=SCENE_BUILT` and 16 actors in
`/Game/AgentProof/BarnAuto/Maps/L_barn_auto`.

Chain: image → MoGe-2 depth → SegFormer regions → unprojected placement →
spawned in Unreal → rendered. Every stage writes a receipt under
`--evidence-root`.

---

## 2. The two pipelines

Full rationale in `docs/pipelines/README.md`. The short version:

| | **A — Photometric** | **B — Structural** |
|---|---|---|
| Bet | Reproduce *this photograph* | Build a *real scene* it describes |
| Fails when | Camera moves off-axis | Compared to the source image |

**They must be graded on different tests.** Photometric matches the source view
*by construction*, so scoring it that way proves nothing — that is how a flat
textured shell passed as a scene for several sessions. `pipeline_result_v1`
carries `graded_on` / `not_applicable` and omits inapplicable metrics rather
than reporting zero, so they cannot be silently averaged.

`offaxis_stability` is the one metric both must satisfy, and the only one that
would have caught the flat shell early.

Dispatcher: `python -m lowvram3d.pipelines --list`.

---

## 3. Environments (this matters — nothing shares one)

| Purpose | Interpreter |
|---|---|
| MoGe, segmentation, splats | `%LOCALAPPDATA%\LowVRAM3DStudio\envs\image-world-moge\Scripts\python.exe` (torch 2.8.0+cu128, CUDA) |
| Image comparison, pipeline driver | `%LOCALAPPDATA%\Programs\Python\Python312\python.exe` (numpy, PIL, cv2, trimesh) |
| Perceptual critic | `...\envs\visualqa\Scripts\python.exe` (transformers, CPU torch) |
| **Do not use** | Python311 on PATH — no numpy/PIL/torch |

`torch` in Python312 is an **empty namespace directory**, not an install:
`torch.__file__` is `None`. It imports and then fails on first attribute access.

`depth_stage.py` shells out to the MoGe interpreter and degrades visibly when
it is missing — it never silently produces a flat result.

---

## 4. Talking to Unreal

`unreal/uemcp/` is the canonical client. `python -m uemcp doctor` checks every
surface and names the repair.

Three independent channels reach the editor:

- **UE_MCP_Bridge** — TCP, port **derived from the project path** and published
  to `<project>/Saved/UE_MCP_Bridge/port.json` each boot (49538 here). 713
  handlers, the only one with `execute_python`. **This is the workhorse.**
- **ModelContextProtocol** — `http://127.0.0.1:8000/mcp`, ~54 toolsets, no
  arbitrary Python.
- **UnrealOpenCode** — port 3000, legacy.

### Traps that each cost hours

- **Never hardcode port 55557.** Read the lockfile.
- **Never use `urllib`/`requests` against port 8000** — it answers with an
  unterminated event stream that reports as `200 OK` with a zero-byte body and
  no exception. Use `uemcp.EditorMCP` (raw socket).
- **`execute_python` needs a named result variable.** `ExecuteFile` semantics,
  so top-level `return` is a syntax error.
- **`connect_material_property` fails silently on a wrong pin name.**
  `VertexColor` accepts only the default output; `TextureSample` accepts
  `"RGB"`. Always connect *and then verify*.
- **Long editor work can exceed the handler timeout while still completing.**
  A 14 MB import "timed out" and had succeeded. Check state before retrying.
- **MSYS rewrites leading-slash arguments.** `--output-root /Game/X` arrives as
  `C:/Program Files/Git/Game/X`. `unreal_stage.normalise_package_root` recovers
  it; do the same anywhere else a package path crosses a shell.
- **A modal dialog blocks the game thread** and stalls every channel at once.
  Look at the editor window before investigating anything else.

### `mcp__unreal-engine__*` tools are unusable

The npm client (`ultimate-unreal-engine-mcp` 0.1.25) correlates responses by a
`correlationId` this plugin build **never echoes**, so every call times out at
30 s. The reconnect storm was a separate defect and is fixed (patched
`BridgeServer.cpp`, rebuilt). Configs for Claude / Codex / OpenCode are
corrected and pinned (backups `.bak-20260804`). **Use `uemcp` instead.**

---

## 5. Bugs found and fixed (do not reintroduce)

Each of these silently discarded the previous stage's work while every receipt
still read `PROVEN`:

1. **Material emissive never connected.** `M_CastlegroundsSourceProjection` is
   `MSM_UNLIT`; three `TextureSample` nodes pointed at the right texture with
   none wired to an output. Every capture in project history rendered black.
2. **Nanite ate the mesh.** glTF import enables it; the mesh then reports and
   renders a 1,770-triangle fallback proxy for a 502,846-triangle
   reconstruction. Disabled on import.
3. **Double metre→centimetre conversion.** The importer already applies ×100.
   Scaling the actor again put a 544 m scene at 25 km across.
4. **Non-idempotent builders.** `new_level()` on an existing path *loads* it, so
   reruns stacked copies — three meshes, three cameras, and in the hybrid map
   four competing directional lights. Builders now remove what they own first.
5. **180° import roll.** Diagnosed by scoring the render against the source
   under all four flip candidates (`rot180` correlation 0.50, everything else
   negative). Corrected on the mesh, not the camera.
6. **MCP HTTP server never started.** `bAutoStartServer` was in
   `DefaultEditor.ini` under the wrong module; the class is
   `config=EditorPerProjectUserSettings` in `ModelContextProtocolEngine`. Moved
   to `Config/DefaultEditorPerProjectUserSettings.ini`.

---

## 6. Open items, highest leverage first

1. **Per-object models via Hunyuan3D Mini Turbo.** The user asked for this
   explicitly. Placement specs are already the right input: each region carries
   a source bbox to crop, a measured world size to scale to, and a position.
   Stage is `region bbox → crop → Mini Turbo → GLB → import → swap for the
   primitive`. Runtime notes: standalone Python, `hy3dgen` on `PYTHONPATH`, DiT
   weights under the `hub\` HF tree. **Structural output is still engine
   primitives until this lands.**
2. **A/mesh glTF axis mapping.** Yaw 0 (what the standard convention predicts)
   renders an almost-empty frame; geometry sits near yaw −50°. Centroid framing
   is a workaround. **Settle it by measuring** the mapping from known vertex
   positions — do not guess a fourth time.
3. **A/splats needs a UE plugin.** PLY is standard INRIA 3DGS and validated
   (`scripts/render_splat_ply.py`, needs no GPU). Luma AI ship a free UE plugin;
   installing one is a project change, so ask first.
4. **Occlusion inpainting.** Single-view leaves holes by definition.
   [SplatFill](https://arxiv.org/abs/2509.07809) operates on the splat
   representation already exported.
5. **`mcp__unreal-engine__*` correlationId mismatch** — the npm package is older
   than the installed plugin. Reconcile or retire.

---

## 7. Test suite caveat

`python -m pytest` fails collection on 33 files for missing
`PIL`/`numpy`/`cv2`/`psutil`. **This is pre-existing**, confirmed by stashing
all session changes and reproducing identically on a clean tree. It is an
interpreter-on-PATH problem, not a regression — but it does mean the "52 passing
tests" baseline is not reproducible from a default shell. Use Python312.

---

## 8. Working habits that paid off

- **Measure, don't eyeball.** The 180° roll was settled by scoring four
  candidate orientations, not by looking. The result contradicted the standard
  convention.
- **Verify writes, don't trust return values.** Unreal's Python API returns
  success for several operations that silently did nothing.
- **Distrust green receipts.** Every major defect this session sat behind a
  `PROVEN` classification. A PNG existing is not evidence it contains anything.
- **Render off-screen, not screenshots.** `capture_scene_png` / `uemcp shot`
  need no window focus, no PIE, no player pawn, and return a byte size to
  assert on.
