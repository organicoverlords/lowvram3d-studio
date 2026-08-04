[CmdletBinding()]
param(
    [string]$ExpectedBranch = 'feature/procedural-jungle-playable-20260804'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$RepoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $RepoRoot) { throw 'Not inside a Git repository' }
Set-Location -LiteralPath $RepoRoot

$Remote = (git config --get remote.origin.url).Trim()
$Branch = (git branch --show-current).Trim()
$Head = (git rev-parse HEAD).Trim()
$Status = @(git status --short)
if ($Remote -notmatch 'organicoverlords/lowvram3d-studio(\.git)?$') { throw "Repository mismatch: $Remote" }
if ($Branch -ne $ExpectedBranch) { throw "Branch mismatch: $Branch" }
if ($Status.Count -ne 0) { throw "Repository is dirty before V3 byte-length wrapper: $($Status -join '; ')" }

git fetch origin $Branch --quiet
$RemoteHead = (git rev-parse "origin/$Branch").Trim()
if ($Head -ne $RemoteHead) { throw "Checkout head differs from remote: $Head vs $RemoteHead" }

$SourceCommit = '0b30584a12ffc5b961a0890a851aca82d6b381fb'
$RelativePath = 'scripts/procedural_jungle/run-procedural-jungle-local-worker.ps1'
$SourceLines = @(git show "${SourceCommit}:$RelativePath")
if ($LASTEXITCODE -ne 0 -or $SourceLines.Count -lt 100) {
    throw "Unable to recover pinned V3 worker from $SourceCommit"
}
$SourceText = ($SourceLines -join "`n") + "`n"

$CharacterGuardLine = 'if ($InstallerText.Length -ne 46901) { throw "V3 installer byte-safe text length mismatch: expected=46901 actual=$($InstallerText.Length)" }'
$WriteLine = '[IO.File]::WriteAllText($Installer, $InstallerText, (New-Object Text.UTF8Encoding($false)))'
$GuardCount = [regex]::Matches($SourceText, [regex]::Escape($CharacterGuardLine)).Count
$WriteCount = [regex]::Matches($SourceText, [regex]::Escape($WriteLine)).Count
if ($GuardCount -ne 1) { throw "Pinned exact character-guard line count is not one: $GuardCount" }
if ($WriteCount -ne 1) { throw "Pinned exact WriteAllText line count is not one: $WriteCount" }

$PatchedText = $SourceText.Replace($CharacterGuardLine + "`n", '')
$ByteCheckedWrite = @'
[IO.File]::WriteAllText($Installer, $InstallerText, (New-Object Text.UTF8Encoding($false)))
$InstallerByteLength = ([IO.File]::ReadAllBytes($Installer)).Length
if ($InstallerByteLength -ne 46901) {
    throw "V3 installer UTF-8 byte length mismatch: expected=46901 actual=$InstallerByteLength"
}
'@
$PatchedText = $PatchedText.Replace($WriteLine, $ByteCheckedWrite.TrimEnd())
if ($PatchedText.Contains($CharacterGuardLine)) { throw 'Pinned character-length guard remains after exact patch' }
if ($PatchedText -notmatch 'InstallerByteLength -ne 46901') { throw 'UTF-8 byte-length guard was not installed' }

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-v3-bytefix-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedWorker = Join-Path $TempRoot 'run-procedural-jungle-v3-bytefixed.ps1'
[IO.File]::WriteAllText($PatchedWorker, $PatchedText, (New-Object Text.UTF8Encoding($false)))

Write-Host "JUNGLE_V3_PINNED_WORKER_SOURCE=$SourceCommit"
Write-Host 'JUNGLE_V3_EXACT_PINNED_LINES_PATCH=PROVEN'
Write-Host 'JUNGLE_V3_UTF8_BYTE_LENGTH_GUARD_PATCH=PROVEN'
& powershell -NoProfile -ExecutionPolicy Bypass -File $PatchedWorker -ExpectedBranch $ExpectedBranch
if ($LASTEXITCODE -ne 0) { throw "Byte-fixed V3 worker failed with exit code $LASTEXITCODE" }
