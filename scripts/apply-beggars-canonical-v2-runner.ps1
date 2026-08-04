[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$runner = 'scripts\run-beggars-canonical-diagnostic.ps1'
$oldRenderer = 'blender\diagnose_beggars_canonical_face.py'
$newRenderer = 'blender\diagnose_beggars_canonical_face_v2.py'
$marker = "    foreach (`$name in @(`r`n"

if (-not (Test-Path -LiteralPath $runner)) {
    throw "Canonical diagnostic runner is missing: $runner"
}
if (-not (Test-Path -LiteralPath $newRenderer)) {
    throw "Landmark v2 renderer is missing: $newRenderer"
}

$text = Get-Content -LiteralPath $runner -Raw
if ($text.Contains($oldRenderer)) {
    $text = $text.Replace($oldRenderer, $newRenderer)
    Write-Host 'CANONICAL_V2_RENDERER_ROUTE=APPLIED'
}
elseif ($text.Contains($newRenderer)) {
    Write-Host 'CANONICAL_V2_RENDERER_ROUTE=ALREADY_APPLIED'
}
else {
    throw 'Canonical runner did not contain the expected renderer path.'
}

if (-not $text.Contains('CANONICAL_V2_OUTPUT_ALIASES=PROVEN')) {
    $aliasBlock = @'
    $v2ExactAliases = [ordered]@{
        'v2_01_canonical_face_only.png' = 'canonical_01_face_only.png'
        'v2_02_landmark_features.png' = 'canonical_02_features_no_hair.png'
        'v2_03_hair_and_features.png' = 'canonical_03_complete_neutral.png'
        'diagnostic_v2.json' = 'diagnostic.json'
    }
    foreach ($entry in $v2ExactAliases.GetEnumerator()) {
        $source = Join-Path $ArtifactRoot $entry.Key
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Landmark v2 output is missing: $($entry.Key)"
        }
        Copy-Item -LiteralPath $source -Destination (Join-Path $ArtifactRoot $entry.Value) -Force
    }
    $openPose = @(Get-ChildItem -LiteralPath $ArtifactRoot -Filter 'v2_04_open_pose_*.png' -File)
    $grinPose = @(Get-ChildItem -LiteralPath $ArtifactRoot -Filter 'v2_05_grin_pose_*.png' -File)
    if ($openPose.Count -ne 1 -or $grinPose.Count -ne 1) {
        throw "Expected one open-pose and one grin-pose render; found open=$($openPose.Count) grin=$($grinPose.Count)"
    }
    Copy-Item -LiteralPath $openPose[0].FullName -Destination (Join-Path $ArtifactRoot 'canonical_04_complete_pose.png') -Force
    Copy-Item -LiteralPath $grinPose[0].FullName -Destination (Join-Path $ArtifactRoot 'canonical_05_complete_grin.png') -Force
    Write-Host 'CANONICAL_V2_OUTPUT_ALIASES=PROVEN'

'@
    if (-not $text.Contains($marker)) {
        $marker = "    foreach (`$name in @(`n"
    }
    if (-not $text.Contains($marker)) {
        throw 'Canonical runner validation marker was not found.'
    }
    $text = $text.Replace($marker, $aliasBlock + $marker)
}

[System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $runner), $text, [System.Text.UTF8Encoding]::new($false))

$python = 'C:\AI\LowVRAM3D-cache\beggars-scene-v2\venv\Scripts\python.exe'
& $python -m py_compile $newRenderer
if ($LASTEXITCODE -ne 0) {
    throw 'Landmark v2 renderer failed Python compilation.'
}

$tokens = $null
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path -LiteralPath $runner), [ref]$tokens, [ref]$errors)
if (@($errors).Count -gt 0) {
    throw "Patched canonical runner failed PowerShell parsing: $($errors[0].Message)"
}

Write-Host 'CANONICAL_V2_RUNNER_PATCH=PROVEN'
