[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$runner = 'scripts\run-beggars-canonical-diagnostic.ps1'
$oldRenderer = 'blender\diagnose_beggars_canonical_face.py'
$newRenderer = 'blender\diagnose_beggars_canonical_face_v3.py'
$validationMarker = "    foreach (`$name in @(`r`n"

if (-not (Test-Path -LiteralPath $runner)) {
    throw "Canonical diagnostic runner is missing: $runner"
}
if (-not (Test-Path -LiteralPath $newRenderer)) {
    throw "Projective v3 renderer is missing: $newRenderer"
}

$text = Get-Content -LiteralPath $runner -Raw
if ($text.Contains($oldRenderer)) {
    $text = $text.Replace($oldRenderer, $newRenderer)
    Write-Host 'CANONICAL_V3_RENDERER_ROUTE=APPLIED'
}
elseif ($text.Contains($newRenderer)) {
    Write-Host 'CANONICAL_V3_RENDERER_ROUTE=ALREADY_APPLIED'
}
else {
    throw 'Canonical runner did not contain the expected renderer path.'
}

$sequenceArgument = "        '--sequence',`$canonical,`r`n        '--output-dir',`$ArtifactRoot"
if (-not $text.Contains($sequenceArgument)) {
    $sequenceArgument = "        '--sequence',`$canonical,`n        '--output-dir',`$ArtifactRoot"
}
if ($text.Contains($sequenceArgument) -and -not $text.Contains("'--tracked-sequence',`$tracked")) {
    $replacement = "        '--sequence',`$canonical,`r`n        '--tracked-sequence',`$tracked,`r`n        '--keyframe-image',`$keyframe,`r`n        '--bfm-pkl',(Join-Path `$ThreeDdfaRoot 'configs\bfm_noneck_v3.pkl'),`r`n        '--output-dir',`$ArtifactRoot"
    if ($sequenceArgument.Contains("`n") -and -not $sequenceArgument.Contains("`r`n")) {
        $replacement = $replacement.Replace("`r`n", "`n")
    }
    $text = $text.Replace($sequenceArgument, $replacement)
    Write-Host 'CANONICAL_V3_PRIVATE_INPUTS=APPLIED'
}
elseif ($text.Contains("'--tracked-sequence',`$tracked")) {
    Write-Host 'CANONICAL_V3_PRIVATE_INPUTS=ALREADY_APPLIED'
}
else {
    throw 'Canonical runner Blender argument block did not match the expected source form.'
}

if (-not $text.Contains('CANONICAL_V3_OUTPUT_ALIASES=PROVEN')) {
    $aliasBlock = @'
    $v3ExactAliases = [ordered]@{
        'v3_01_projective_face.png' = 'canonical_01_face_only.png'
        'v3_02_face_teeth_no_hair.png' = 'canonical_02_features_no_hair.png'
        'v3_03_face_teeth_hair.png' = 'canonical_03_complete_neutral.png'
        'diagnostic_v3.json' = 'diagnostic.json'
    }
    foreach ($entry in $v3ExactAliases.GetEnumerator()) {
        $source = Join-Path $ArtifactRoot $entry.Key
        if (-not (Test-Path -LiteralPath $source)) {
            throw "Projective v3 output is missing: $($entry.Key)"
        }
        Copy-Item -LiteralPath $source -Destination (Join-Path $ArtifactRoot $entry.Value) -Force
    }
    $mildPose = @(Get-ChildItem -LiteralPath $ArtifactRoot -Filter 'v3_04_mild_pose_*.png' -File)
    $grinPose = @(Get-ChildItem -LiteralPath $ArtifactRoot -Filter 'v3_05_grin_pose_*.png' -File)
    if ($mildPose.Count -ne 1 -or $grinPose.Count -ne 1) {
        throw "Expected one mild-pose and one grin-pose render; found mild=$($mildPose.Count) grin=$($grinPose.Count)"
    }
    Copy-Item -LiteralPath $mildPose[0].FullName -Destination (Join-Path $ArtifactRoot 'canonical_04_complete_pose.png') -Force
    Copy-Item -LiteralPath $grinPose[0].FullName -Destination (Join-Path $ArtifactRoot 'canonical_05_complete_grin.png') -Force
    Write-Host 'CANONICAL_V3_OUTPUT_ALIASES=PROVEN'

'@
    if (-not $text.Contains($validationMarker)) {
        $validationMarker = "    foreach (`$name in @(`n"
    }
    if (-not $text.Contains($validationMarker)) {
        throw 'Canonical runner validation marker was not found.'
    }
    $text = $text.Replace($validationMarker, $aliasBlock + $validationMarker)
}

[System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $runner), $text, [System.Text.UTF8Encoding]::new($false))

$python = 'C:\AI\LowVRAM3D-cache\beggars-scene-v2\venv\Scripts\python.exe'
& $python -m py_compile $newRenderer
if ($LASTEXITCODE -ne 0) {
    throw 'Projective v3 renderer failed Python compilation.'
}

$tokens = $null
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path -LiteralPath $runner), [ref]$tokens, [ref]$errors)
if (@($errors).Count -gt 0) {
    throw "Patched canonical runner failed PowerShell parsing: $($errors[0].Message)"
}

Write-Host 'CANONICAL_V3_RUNNER_PATCH=PROVEN'
