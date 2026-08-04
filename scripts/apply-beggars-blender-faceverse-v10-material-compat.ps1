[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'blender\build_beggars_meme_scene_faceverse_v10.py'
if (-not (Test-Path -LiteralPath $target)) {
    throw "FaceVerse v10 Blender source is missing: $target"
}

$text = Get-Content -LiteralPath $target -Raw

$oldFace = @'
def create_face_v10(*args, **kwargs):
    face, follow, targets = _original_create_corrected_face(*args, **kwargs)
    apply_projection_attributes(face)
    return face, follow, targets
'@
$newFace = @'
def create_face_v10(*args, **kwargs):
    face, follow, targets = _original_create_corrected_face(*args, **kwargs)
    apply_projection_attributes(face)
    projected_material = projected_skin_material_v10()
    if len(face.data.materials) == 0:
        face.data.materials.append(projected_material)
    else:
        face.data.materials[0] = projected_material
    for polygon in face.data.polygons:
        polygon.material_index = 0
    face["v10_projected_material_slot"] = 0
    print(
        f"BLENDER_FACEVERSE_V10_PROJECTED_MATERIAL_ASSIGNED=PROVEN "
        f"SLOT=0 NAME={projected_material.name}"
    )
    return face, follow, targets
'@
if ($text.Contains($oldFace)) {
    $text = $text.Replace($oldFace, $newFace)
}
elseif (-not $text.Contains('BLENDER_FACEVERSE_V10_PROJECTED_MATERIAL_ASSIGNED=PROVEN')) {
    throw 'Could not locate the FaceVerse v10 face-material assignment anchor.'
}

if (-not $text.Contains('BLENDER_FACEVERSE_V10_RELOADED_SLOT_ZERO=PROVEN')) {
    $renderPattern = '(?s)def render_projection_variants\(output_dir: Path\) -> list\[dict\[str, Any\]\]:\r?\n.*?    strength_node = material\.node_tree\.nodes\.get\(PROJECTION_STRENGTH_NODE\)\r?\n'
    if (-not [regex]::IsMatch($text, $renderPattern)) {
        throw 'Could not locate the FaceVerse v10 projection-variant material block.'
    }
    $newRender = @'
def render_projection_variants(output_dir: Path) -> list[dict[str, Any]]:
    face = bpy.data.objects.get("CHAR_Antinous")
    if face is None or not hasattr(face.data, "materials"):
        raise RuntimeError("V10 face object is missing after save/reload")
    if len(face.data.materials) < 1 or face.data.materials[0] is None:
        raise RuntimeError("V10 projected material slot zero is empty after save/reload")
    material = face.data.materials[0]
    if (
        not material.use_nodes
        or material.node_tree is None
        or material.node_tree.nodes.get(PROJECTION_STRENGTH_NODE) is None
    ):
        names = [slot.name for slot in face.data.materials if slot is not None]
        raise RuntimeError(
            f"V10 face slot zero is not the projected material: slot0={material.name} slots={names}"
        )
    print(f"BLENDER_FACEVERSE_V10_RELOADED_SLOT_ZERO=PROVEN NAME={material.name}")
    strength_node = material.node_tree.nodes.get(PROJECTION_STRENGTH_NODE)
'@
    $text = [regex]::Replace($text, $renderPattern, $newRender.TrimEnd() + "`n", 1)
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
foreach ($marker in @(
    'BLENDER_FACEVERSE_V10_PROJECTED_MATERIAL_ASSIGNED=PROVEN',
    'face.data.materials[0] = projected_material',
    'BLENDER_FACEVERSE_V10_RELOADED_SLOT_ZERO=PROVEN',
    'material = face.data.materials[0]'
)) {
    if (-not $patched.Contains($marker)) {
        throw "FaceVerse v10 direct material-slot gate is absent: $marker"
    }
}
Write-Host 'BLENDER_FACEVERSE_V10_DIRECT_MATERIAL_SLOT=PROVEN'
Write-Host 'BLENDER_FACEVERSE_V10_MATERIAL_COMPAT=PROVEN'
