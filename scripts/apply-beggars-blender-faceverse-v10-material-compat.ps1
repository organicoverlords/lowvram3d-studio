[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'blender\build_beggars_meme_scene_faceverse_v10.py'
if (-not (Test-Path -LiteralPath $target)) {
    throw "FaceVerse v10 Blender source is missing: $target"
}

$text = Get-Content -LiteralPath $target -Raw
$old = @'
def render_projection_variants(output_dir: Path) -> list[dict[str, Any]]:
    material = bpy.data.materials.get(PROJECTION_MATERIAL_NAME)
    if material is None or not material.use_nodes:
        raise RuntimeError("V10 projected material is missing after scene build")
    strength_node = material.node_tree.nodes.get(PROJECTION_STRENGTH_NODE)
'@
$new = @'
def render_projection_variants(output_dir: Path) -> list[dict[str, Any]]:
    material = bpy.data.materials.get(PROJECTION_MATERIAL_NAME)
    if material is None or not material.use_nodes or material.node_tree.nodes.get(PROJECTION_STRENGTH_NODE) is None:
        face = bpy.data.objects.get("CHAR_Antinous")
        if face is None or not hasattr(face.data, "materials"):
            raise RuntimeError("V10 face object is missing after save/reload")
        candidates = [
            slot
            for slot in face.data.materials
            if slot is not None
            and slot.use_nodes
            and slot.node_tree is not None
            and slot.node_tree.nodes.get(PROJECTION_STRENGTH_NODE) is not None
        ]
        if len(candidates) != 1:
            names = [slot.name for slot in face.data.materials if slot is not None]
            raise RuntimeError(
                f"V10 projected material could not be resolved from face slots: candidates={len(candidates)} slots={names}"
            )
        material = candidates[0]
        print(f"BLENDER_FACEVERSE_V10_RELOADED_MATERIAL=PROVEN NAME={material.name}")
    strength_node = material.node_tree.nodes.get(PROJECTION_STRENGTH_NODE)
'@

if ($text.Contains($old)) {
    $text = $text.Replace($old, $new)
}
elseif (-not $text.Contains('BLENDER_FACEVERSE_V10_RELOADED_MATERIAL=PROVEN')) {
    throw 'Could not locate the FaceVerse v10 material-resolution anchor.'
}

[System.IO.File]::WriteAllText(
    (Resolve-Path -LiteralPath $target),
    $text,
    [System.Text.UTF8Encoding]::new($false)
)

$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "FaceVerse Python is missing: $python"
}
& $python -m py_compile $target
if ($LASTEXITCODE -ne 0) {
    throw 'FaceVerse v10 Blender source failed compilation after material repair.'
}

$patched = Get-Content -LiteralPath $target -Raw
if (-not $patched.Contains('BLENDER_FACEVERSE_V10_RELOADED_MATERIAL=PROVEN')) {
    throw 'FaceVerse v10 reloaded-material gate is absent after patch.'
}
Write-Host 'BLENDER_FACEVERSE_V10_MATERIAL_COMPAT=PROVEN'
