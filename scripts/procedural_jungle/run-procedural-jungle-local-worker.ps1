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

$RecoveryCommit = '0c975ad69eb5390bcc5d6d6cb17319b7af18b5e8'
$RecoveryBlob = '374ea42f870ad8fd4b278f9f138dd1415165e464'
$WrapperPath = 'scripts/procedural_jungle/run-procedural-jungle-local-worker.ps1'
& git merge-base --is-ancestor $RecoveryCommit HEAD
if ($LASTEXITCODE -ne 0) { throw "Recovery commit is not an ancestor of HEAD: $RecoveryCommit" }
$ActualRecoveryBlob = (& git rev-parse "$RecoveryCommit`:$WrapperPath").Trim()
if ($LASTEXITCODE -ne 0 -or $ActualRecoveryBlob -ne $RecoveryBlob) {
    throw "Recovery wrapper identity mismatch: expected=$RecoveryBlob actual=$ActualRecoveryBlob"
}

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-entrytext-fix-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedWrapper = Join-Path $TempRoot 'run-procedural-jungle-local-worker-entrytext-fixed.ps1'
$RecoveryLines = @(& git show "$RecoveryCommit`:$WrapperPath")
if ($LASTEXITCODE -ne 0 -or $RecoveryLines.Count -lt 10) { throw 'Could not read the proven recovery wrapper' }
$RecoveryText = $RecoveryLines -join "`n"

$OldRead = '$BuildText = (Get-Content -LiteralPath $BuildScript -Raw) -replace "`r`n", "`n"'
$NewRead = '$BuildText = $EntryText -replace "`r`n", "`n"'
$OldWrite = 'Set-Content -LiteralPath $BuildScript -Value $BuildText -Encoding utf8'
$NewWrite = '$EntryText = $BuildText'
$ReadMatches = [regex]::Matches($RecoveryText, [regex]::Escape($OldRead)).Count
$WriteMatches = [regex]::Matches($RecoveryText, [regex]::Escape($OldWrite)).Count
if ($ReadMatches -ne 1 -or $WriteMatches -ne 1) {
    throw "Could not prove unique cached-entry transformations: read=$ReadMatches write=$WriteMatches"
}
$RecoveryText = $RecoveryText.Replace($OldRead, $NewRead).Replace($OldWrite, $NewWrite)
Set-Content -LiteralPath $PatchedWrapper -Value $RecoveryText -Encoding utf8
Write-Host "RECOVERY_WRAPPER_COMMIT=$RecoveryCommit"
Write-Host "RECOVERY_WRAPPER_BLOB=$RecoveryBlob"
Write-Host 'PANDA_PATCH_CACHED_ENTRYTEXT_FIX=PROVEN'

& $PatchedWrapper -ExpectedBranch $ExpectedBranch
$WorkerExit = $LASTEXITCODE
if ($WorkerExit -ne 0) { throw "EntryText-fixed direct worker failed with exit code $WorkerExit" }
