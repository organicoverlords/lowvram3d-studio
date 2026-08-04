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

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-runtime-diagnostic-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedBootstrap = Join-Path $TempRoot 'run-procedural-jungle-runtime-diagnostic.ps1'
$BaselineLines = @(& git show "$BaselineCommit`:$BootstrapPath")
if ($LASTEXITCODE -ne 0 -or $BaselineLines.Count -lt 10) { throw 'Could not read baseline bootstrap' }
$BaselineText = $BaselineLines -join "`n"
$InsertionMarker = '# Windows PowerShell 5 turns native stderr into terminating error records under'
if ([regex]::Matches($BaselineText, [regex]::Escape($InsertionMarker)).Count -ne 1) {
    throw 'Could not prove unique bootstrap insertion marker'
}

$Diagnostic = @'
$Patterns = 'JUNGLE_PROOF_CAPTURE|JUNGLE_RUNTIME_PROOF|RequestScreenshot|SetViewTarget|ViewTarget|PandaRoute|panda_travel|TravelDistance|ProofCamera|camera_count|capture_count'
$RuntimeFiles = @(
    Get-ChildItem -LiteralPath $SourceRoot -Recurse -File -ErrorAction Stop |
        Where-Object { $_.Extension -in @('.cpp', '.h', '.ps1', '.py') } |
        Where-Object { (Get-Content -LiteralPath $_.FullName -Raw) -match $Patterns }
)
if ($RuntimeFiles.Count -lt 1 -or $RuntimeFiles.Count -gt 12) {
    throw "Unexpected runtime proof source file count: $($RuntimeFiles.Count)"
}
Write-Host "RUNTIME_DIAGNOSTIC_FILE_COUNT=$($RuntimeFiles.Count)"
foreach ($File in ($RuntimeFiles | Sort-Object FullName)) {
    $Relative = $File.FullName.Substring($SourceRoot.Length).TrimStart('\')
    $Lines = @(Get-Content -LiteralPath $File.FullName)
    Write-Host "RUNTIME_SOURCE_BEGIN=$Relative lines=$($Lines.Count)"
    for ($Index = 0; $Index -lt $Lines.Count; $Index++) {
        Write-Host ('RUNTIME_SOURCE_{0:D4}={1}' -f ($Index + 1), $Lines[$Index])
    }
    Write-Host "RUNTIME_SOURCE_END=$Relative"
}
throw 'JUNGLE_RUNTIME_SOURCE_DIAGNOSTIC_CAPTURED'
'@

$PatchedText = $BaselineText.Replace($InsertionMarker, $Diagnostic + "`n`n" + $InsertionMarker)
Set-Content -LiteralPath $PatchedBootstrap -Value $PatchedText -Encoding utf8
Write-Host 'JUNGLE_RUNTIME_SOURCE_DIAGNOSTIC_INJECTION=PROVEN'
& $PatchedBootstrap -ExpectedBranch $ExpectedBranch
$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 0) { throw "Runtime diagnostic exited with code $ExitCode" }
