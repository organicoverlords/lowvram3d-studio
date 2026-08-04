[CmdletBinding()]
param(
    [string]$ExpectedBranch = 'feature/procedural-jungle-playable-20260804'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $RepoRoot) { throw 'Not inside the expected Git repository' }
Set-Location -LiteralPath $RepoRoot
$Remote = (git config --get remote.origin.url).Trim()
$Branch = (git branch --show-current).Trim()
$Head = (git rev-parse HEAD).Trim()
if ($Remote -notmatch 'organicoverlords/lowvram3d-studio(\.git)?$') { throw "Repository mismatch: $Remote" }
if ($Branch -ne $ExpectedBranch) { throw "Branch mismatch: $Branch" }

$BaselineCommit = 'bdf71585456e0c97aa77234f1bc01a7cf67c7bf4'
$BaselineBlob = '39fed93175360450c99891fbdf61ba0ba1d7ed71'
$BootstrapPath = 'scripts/procedural_jungle/run-procedural-jungle-local-worker.ps1'
& git merge-base --is-ancestor $BaselineCommit HEAD
if ($LASTEXITCODE -ne 0) { throw "Baseline commit is not an ancestor of HEAD: $BaselineCommit" }
$ActualBaselineBlob = (& git rev-parse "$BaselineCommit`:$BootstrapPath").Trim()
if ($LASTEXITCODE -ne 0 -or $ActualBaselineBlob -ne $BaselineBlob) {
    throw "Baseline bootstrap identity mismatch: expected=$BaselineBlob actual=$ActualBaselineBlob"
}

$WrapperTemp = Join-Path $env:RUNNER_TEMP "procedural-jungle-diagnostic-$Head"
if (Test-Path -LiteralPath $WrapperTemp) { Remove-Item -LiteralPath $WrapperTemp -Recurse -Force }
New-Item -ItemType Directory -Path $WrapperTemp -Force | Out-Null
$PatchedBootstrap = Join-Path $WrapperTemp 'run-procedural-jungle-local-worker-diagnostic.ps1'
$BaselineLines = @(& git show "$BaselineCommit`:$BootstrapPath")
if ($LASTEXITCODE -ne 0 -or $BaselineLines.Count -lt 10) { throw 'Could not read the proven baseline bootstrap' }
$BaselineText = $BaselineLines -join "`n"

$InsertionMarker = '# Windows PowerShell 5 turns native stderr into terminating error records under'
$MarkerMatches = [regex]::Matches($BaselineText, [regex]::Escape($InsertionMarker))
if ($MarkerMatches.Count -ne 1) {
    throw "Could not prove unique bootstrap insertion marker; matches=$($MarkerMatches.Count)"
}

$Diagnostic = @'
$BuildScript = Join-Path $SourceRoot 'scripts\procedural_jungle\build-procedural-jungle.ps1'
if (-not (Test-Path -LiteralPath $BuildScript -PathType Leaf)) {
    throw "Decoded jungle build script missing: $BuildScript"
}
$BuildLines = @(Get-Content -LiteralPath $BuildScript)
Write-Host "JUNGLE_BUILD_SCRIPT_LINE_COUNT=$($BuildLines.Count)"
$Needles = @(
    'No valid tactical red panda GLB was found',
    'rig_animate_panda.py',
    'tactical_red_panda_walk.fbx',
    'panda_source.json'
)
$MatchedIndexes = New-Object System.Collections.Generic.List[int]
for ($Index = 0; $Index -lt $BuildLines.Count; $Index++) {
    foreach ($Needle in $Needles) {
        if ($BuildLines[$Index] -like "*$Needle*") {
            $MatchedIndexes.Add($Index)
            break
        }
    }
}
if ($MatchedIndexes.Count -lt 1) {
    throw 'Could not locate panda-selection markers in decoded build script'
}
$Start = [Math]::Max(0, (($MatchedIndexes | Measure-Object -Minimum).Minimum - 35))
$End = [Math]::Min($BuildLines.Count - 1, (($MatchedIndexes | Measure-Object -Maximum).Maximum + 55))
Write-Host "JUNGLE_BUILD_DIAGNOSTIC_RANGE=$($Start + 1)-$($End + 1)"
for ($Index = $Start; $Index -le $End; $Index++) {
    Write-Host ('JUNGLE_BUILD_LINE_{0:D4}={1}' -f ($Index + 1), $BuildLines[$Index])
}
throw 'JUNGLE_BUILD_DIAGNOSTIC_CAPTURED'
'@

$PatchedText = $BaselineText.Replace($InsertionMarker, $Diagnostic + "`n`n" + $InsertionMarker)
Set-Content -LiteralPath $PatchedBootstrap -Value $PatchedText -Encoding utf8
Write-Host "BOOTSTRAP_BASELINE_COMMIT=$BaselineCommit"
Write-Host "BOOTSTRAP_BASELINE_BLOB=$BaselineBlob"
Write-Host 'JUNGLE_BUILD_DIAGNOSTIC_INJECTION=PROVEN'

& $PatchedBootstrap -ExpectedBranch $ExpectedBranch
$WorkerExit = $LASTEXITCODE
if ($WorkerExit -ne 0) { throw "Diagnostic direct worker exited with code $WorkerExit" }
