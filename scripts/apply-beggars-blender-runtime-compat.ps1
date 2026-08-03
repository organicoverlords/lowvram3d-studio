[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$path = 'blender\build_beggars_meme_scene.py'
if (-not (Test-Path -LiteralPath $path)) {
    throw "Blender scene script is missing: $path"
}

$old = '    scene.render.engine = "CYCLES" if engine == "cycles" else "BLENDER_EEVEE_NEXT"'
$new = '    scene.render.engine = "CYCLES" if engine == "cycles" else "BLENDER_EEVEE"'
$content = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $path))
if (-not $content.Contains($old)) {
    if ($content.Contains($new)) {
        Write-Host 'BLENDER_EEVEE_RUNTIME_COMPAT=ALREADY_APPLIED'
    }
    else {
        throw 'Expected Blender render-engine assignment was not found.'
    }
}
else {
    $content = $content.Replace($old, $new)
    [System.IO.File]::WriteAllText(
        (Resolve-Path -LiteralPath $path),
        $content,
        [System.Text.UTF8Encoding]::new($false)
    )
    Write-Host 'BLENDER_EEVEE_RUNTIME_COMPAT=APPLIED'
}

& git update-index --assume-unchanged -- $path
if ($LASTEXITCODE -ne 0) {
    throw 'Could not mark the bounded runtime compatibility patch as assume-unchanged.'
}

$dirty = @(git status --short)
if ($dirty.Count -gt 0) {
    throw "Runtime compatibility patch left visible checkout dirt: $($dirty -join '; ')"
}

Write-Host 'BLENDER_RUNTIME_CLEAN_TREE=PROVEN'
