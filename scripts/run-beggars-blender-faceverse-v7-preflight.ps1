[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$sourcePath = 'scripts\run-beggars-blender-faceverse-v6-preflight.ps1'
if (-not (Test-Path -LiteralPath $sourcePath)) {
    throw "Proven Blender v6 runner is missing: $sourcePath"
}

$text = Get-Content -LiteralPath $sourcePath -Raw
$replacements = [ordered]@{
    'blender-faceverse-v6-preflight-' = 'blender-faceverse-v7-preflight-'
    'build_beggars_meme_scene_faceverse_v6.py' = 'build_beggars_meme_scene_faceverse_v7.py'
    'BLENDER_V6_FACEVERSE_RUNTIME' = 'BLENDER_V7_FACEVERSE_RUNTIME'
    'BLENDER_FACEVERSE_V6_REFINED_SAVE_RELOAD_STILLS' = 'BLENDER_FACEVERSE_V7_CONFORMING_HAIR_SAVE_RELOAD_STILLS'
    'BLENDER_FACEVERSE_V6_OUTPUT' = 'BLENDER_FACEVERSE_V7_OUTPUT'
    'Blender v6' = 'Blender v7'
    'model-space Blender v6' = 'model-space Blender v7'
    'B6_OUTPUT' = 'B7_OUTPUT'
}
foreach ($entry in $replacements.GetEnumerator()) {
    if (-not $text.Contains([string]$entry.Key)) {
        throw "Could not locate required v7 runner replacement anchor: $($entry.Key)"
    }
    $text = $text.Replace([string]$entry.Key, [string]$entry.Value)
}

$tempScript = Join-Path $env:TEMP "beggars-blender-faceverse-v7-$PID.ps1"
try {
    [System.IO.File]::WriteAllText(
        $tempScript,
        $text,
        [System.Text.UTF8Encoding]::new($false)
    )
    $tokens = $null
    $errors = $null
    [void][System.Management.Automation.Language.Parser]::ParseFile(
        $tempScript,
        [ref]$tokens,
        [ref]$errors
    )
    if (@($errors).Count -gt 0) {
        throw "Generated Blender v7 runner failed parsing: $($errors[0].Message)"
    }
    & powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File $tempScript
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "Generated Blender v7 runner failed with exit code $exitCode"
    }
}
finally {
    Remove-Item -LiteralPath $tempScript -Force -ErrorAction SilentlyContinue
}

Write-Host 'BLENDER_FACEVERSE_V7_RUNNER=PROVEN'
exit 0
