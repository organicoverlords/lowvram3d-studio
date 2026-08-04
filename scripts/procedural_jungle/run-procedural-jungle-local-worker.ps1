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
if ($Status.Count -ne 0) { throw "Repository is dirty before V3 checked-in guard wrapper: $($Status -join '; ')" }
git fetch origin $Branch --quiet
$RemoteHead = (git rev-parse "origin/$Branch").Trim()
if ($Head -ne $RemoteHead) { throw "Checkout head differs from remote: $Head vs $RemoteHead" }

$SourceCommit = '498e600c5873377fd0f55b8179b86ec62780b32c'
$RelativePath = 'scripts/procedural_jungle/run-procedural-jungle-local-worker.ps1'
$SourceLines = @(git show "${SourceCommit}:${RelativePath}")
if ($LASTEXITCODE -ne 0 -or $SourceLines.Count -lt 150) {
    throw "Unable to recover protected-coexistence V3 wrapper from $SourceCommit"
}
$SourceText = ($SourceLines -join "`n") + "`n"
$Anchor = @'
$OwnedProjectRoot = 'C:\Users\Lauri\Desktop\ProceduralJungle58'
$ProtectedProjectPath = 'C:\Users\Lauri\Desktop\UnrealAITest58\UnrealAITest58.uproject'
'@.Trim()
$AnchorIndex = $SourceText.LastIndexOf($Anchor, [StringComparison]::Ordinal)
if ($AnchorIndex -lt 0) { throw 'Protected-coexistence preflight anchor is missing' }
$ExpectedPrefix = '$CoexistencePreflight = @''' + "`n"
if ($AnchorIndex -lt $ExpectedPrefix.Length) { throw 'Protected-coexistence anchor has no room for its declaration prefix' }
$ActualPrefix = $SourceText.Substring($AnchorIndex - $ExpectedPrefix.Length, $ExpectedPrefix.Length)
if ($ActualPrefix -ne $ExpectedPrefix) {
    throw "Protected-coexistence anchor is not inside the expected here-string declaration: $ActualPrefix"
}

$Insertion = @'
$DecodedBuildPath = Join-Path $ExtractRoot 'scripts\procedural_jungle\build-procedural-jungle.ps1'
$GuardPatchScript = Join-Path $RepoRoot 'scripts\procedural_jungle\patch_decoded_build_unreal_guard.py'
if (-not (Test-Path -LiteralPath $GuardPatchScript -PathType Leaf)) {
    throw "Decoded build guard patch script is missing: $GuardPatchScript"
}
& $Python $GuardPatchScript --path $DecodedBuildPath
if ($LASTEXITCODE -ne 0) {
    throw "Decoded build-script Unreal guard patch failed with exit code $LASTEXITCODE"
}
$DecodedBuildText = Get-Content -LiteralPath $DecodedBuildPath -Raw
if ($DecodedBuildText -notmatch 'JUNGLE_BUILD_INTERNAL_UNREAL_GUARD_DELEGATED=PROVEN') {
    throw 'Decoded build-script delegation marker is missing after checked-in Python patch'
}
Write-Host 'JUNGLE_DECODED_BUILD_GUARD_CHECKED_IN_PATCH=PROVEN'

'@
$PatchedSource = $SourceText.Substring(0, $AnchorIndex) + $Insertion.TrimEnd() + "`n" + $SourceText.Substring($AnchorIndex)
if ($PatchedSource -notmatch 'JUNGLE_DECODED_BUILD_GUARD_CHECKED_IN_PATCH=PROVEN') {
    throw 'Checked-in decoded-guard patch insertion marker is missing'
}

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-v3-checked-in-guard-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedWrapper = Join-Path $TempRoot 'run-procedural-jungle-v3-checked-in-guard.ps1'
[IO.File]::WriteAllText($PatchedWrapper, $PatchedSource, (New-Object Text.UTF8Encoding($false)))

Write-Host "JUNGLE_V3_COEXISTENCE_WRAPPER_SOURCE=$SourceCommit"
Write-Host 'JUNGLE_V3_CHECKED_IN_GUARD_PATCH_INJECTION=PROVEN'
& powershell -NoProfile -ExecutionPolicy Bypass -File $PatchedWrapper -ExpectedBranch $ExpectedBranch
if ($LASTEXITCODE -ne 0) { throw "Checked-in-guard-patched V3 wrapper failed with exit code $LASTEXITCODE" }
