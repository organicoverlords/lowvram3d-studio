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

$BaseCommit = '84c2259677c3211c5dbe55da814677c40d635d10'
$BaseBlob = '5301f2769e455318b6ae396c37692a5d8778b3aa'
$WrapperPath = 'scripts/procedural_jungle/run-procedural-jungle-local-worker.ps1'
& git merge-base --is-ancestor $BaseCommit HEAD
if ($LASTEXITCODE -ne 0) { throw "Runtime repair base is not an ancestor of HEAD: $BaseCommit" }
$ActualBlob = (& git rev-parse "$BaseCommit`:$WrapperPath").Trim()
if ($LASTEXITCODE -ne 0 -or $ActualBlob -ne $BaseBlob) {
    throw "Runtime repair base identity mismatch: expected=$BaseBlob actual=$ActualBlob"
}

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-ps51-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedWrapper = Join-Path $TempRoot 'run-procedural-jungle-local-worker-ps51.ps1'
$BaseLines = @(& git show "$BaseCommit`:$WrapperPath")
if ($LASTEXITCODE -ne 0 -or $BaseLines.Count -lt 10) { throw 'Could not read the proven runtime repair wrapper' }
$WrapperText = $BaseLines -join "`n"

$Old = '$x=$sr.ReadToEnd();$sr.Dispose()'
$New = '$x=$sr.ReadToEnd();$sr.Dispose();$Utf8NoBomTargets=[regex]::Matches($x,''utf8NoBOM'').Count;if($Utf8NoBomTargets -lt 1 -or $Utf8NoBomTargets -gt 8){throw "Unexpected utf8NoBOM compatibility target count: $Utf8NoBomTargets"};$x=$x.Replace(''utf8NoBOM'',''utf8'');Write-Host "POWERSHELL51_UTF8_COMPAT_PATCH=PROVEN targets=$Utf8NoBomTargets"'
$MatchCount = [regex]::Matches($WrapperText, [regex]::Escape($Old)).Count
if ($MatchCount -ne 1) { throw "Could not prove unique decompression patch point; matches=$MatchCount" }
$WrapperText = $WrapperText.Replace($Old, $New)
[IO.File]::WriteAllText($PatchedWrapper, $WrapperText, (New-Object Text.UTF8Encoding($false)))

# Never upload stale captures or acceptance from an earlier failed run.
$OutputRoot = 'C:\AI\ProceduralJungle\20260804'
$ProofRoot = Join-Path $OutputRoot 'proof'
if (Test-Path -LiteralPath $ProofRoot) {
    Get-ChildItem -LiteralPath $ProofRoot -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'capture_*.png' -or $_.Name -in @('gameplay_runtime_proof.json', 'contact_sheet.png', 'visual_capture_audit.json') } |
        Remove-Item -Force
}
foreach ($StalePath in @(
    (Join-Path $OutputRoot 'acceptance.json'),
    (Join-Path $RepoRoot 'evidence\latest-procedural-jungle\acceptance.json'),
    (Join-Path $RepoRoot 'evidence\latest-procedural-jungle\workflow_receipt.json')
)) {
    if (Test-Path -LiteralPath $StalePath) { Remove-Item -LiteralPath $StalePath -Force }
}

Write-Host "RUNTIME_REPAIR_BASE_COMMIT=$BaseCommit"
Write-Host "RUNTIME_REPAIR_BASE_BLOB=$BaseBlob"
Write-Host 'POWERSHELL51_WRAPPER_PATCH=PROVEN'
Write-Host 'STALE_JUNGLE_PROOF_CLEARED=PROVEN'

& $PatchedWrapper -ExpectedBranch $ExpectedBranch
$WorkerExit = $LASTEXITCODE
if ($WorkerExit -ne 0) { throw "PowerShell-5-compatible jungle worker failed with exit code $WorkerExit" }
