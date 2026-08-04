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

$AppendPattern = '(?m)^if \(\$InstallerText\.Length -eq 46900\) \{ \$InstallerText \+= "`n" \}\r?\n'
$LengthPattern = '(?m)^if \(\$InstallerText\.Length -ne 46901\) \{ throw "V3 installer byte-safe text length mismatch: expected=46901 actual=\$\(\$InstallerText\.Length\)" \}\r?\n'
$WriteLine = '[IO.File]::WriteAllText($Installer, $InstallerText, $Utf8NoBom)'
$ByteCheck = @'
[IO.File]::WriteAllText($Installer, $InstallerText, $Utf8NoBom)
$InstallerByteLength = ([IO.File]::ReadAllBytes($Installer)).Length
if ($InstallerByteLength -ne 46901) {
    throw "V3 installer UTF-8 byte length mismatch: expected=46901 actual=$InstallerByteLength"
}
'@

if ([regex]::Matches($SourceText, $AppendPattern).Count -ne 1) {
    throw 'Pinned worker append-line match count is not exactly one'
}
if ([regex]::Matches($SourceText, $LengthPattern).Count -ne 1) {
    throw 'Pinned worker character-length guard match count is not exactly one'
}
if (($SourceText.Split($WriteLine).Count - 1) -ne 1) {
    throw 'Pinned worker WriteAllText match count is not exactly one'
}

$PatchedText = [regex]::Replace($SourceText, $AppendPattern, '', 1)
$PatchedText = [regex]::Replace($PatchedText, $LengthPattern, '', 1)
$PatchedText = $PatchedText.Replace($WriteLine, $ByteCheck.TrimEnd())
if ($PatchedText -match 'InstallerText\.Length -ne 46901' -or $PatchedText -match 'InstallerText\.Length -eq 46900') {
    throw 'Character-count installer guard remains after patching'
}
if ($PatchedText -notmatch 'InstallerByteLength -ne 46901') {
    throw 'UTF-8 byte-length installer guard was not installed'
}

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-v3-bytefix-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedWorker = Join-Path $TempRoot 'run-procedural-jungle-v3-bytefixed.ps1'
[IO.File]::WriteAllText($PatchedWorker, $PatchedText, (New-Object Text.UTF8Encoding($false)))

Write-Host "JUNGLE_V3_PINNED_WORKER_SOURCE=$SourceCommit"
Write-Host 'JUNGLE_V3_UTF8_BYTE_LENGTH_GUARD_PATCH=PROVEN'
& powershell -NoProfile -ExecutionPolicy Bypass -File $PatchedWorker -ExpectedBranch $ExpectedBranch
if ($LASTEXITCODE -ne 0) { throw "Byte-fixed V3 worker failed with exit code $LASTEXITCODE" }
