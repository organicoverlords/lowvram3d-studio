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
if ($Status.Count -ne 0) { throw "Repository is dirty before V3 assembly wrapper: $($Status -join '; ')" }

git fetch origin $Branch --quiet
$RemoteHead = (git rev-parse "origin/$Branch").Trim()
if ($Head -ne $RemoteHead) { throw "Checkout head differs from remote: $Head vs $RemoteHead" }

$SourceCommit = '5ee2f112714d451f22fee8e7ab337ef766804f9c'
$RelativePath = 'scripts/procedural_jungle/run-procedural-jungle-local-worker.ps1'
$SourceLines = @(git show "${SourceCommit}:$RelativePath")
if ($LASTEXITCODE -ne 0 -or $SourceLines.Count -lt 150) {
    throw "Unable to recover safe-preflight V3 worker from $SourceCommit"
}
$SourceText = ($SourceLines -join "`n") + "`n"

$StartMarker = '$InstallerBuilder = New-Object Text.StringBuilder'
$EndMarker = '& $Python $Installer --target $ExtractRoot --report $OverlayReport'
$StartIndex = $SourceText.IndexOf($StartMarker, [StringComparison]::Ordinal)
$EndIndex = $SourceText.IndexOf($EndMarker, [StringComparison]::Ordinal)
if ($StartIndex -lt 0) { throw 'V3 installer assembly start marker was not found' }
if ($EndIndex -le $StartIndex) { throw 'V3 installer invocation end marker was not found after the start marker' }
if ($SourceText.IndexOf($StartMarker, $StartIndex + 1, [StringComparison]::Ordinal) -ge 0) { throw 'V3 installer assembly start marker is not unique' }
if ($SourceText.IndexOf($EndMarker, $EndIndex + 1, [StringComparison]::Ordinal) -ge 0) { throw 'V3 installer invocation marker is not unique' }

$Replacement = @'
$InstallerBuilder = New-Object Text.StringBuilder
foreach ($Fragment in $InstallerFragments) {
    [void]$InstallerBuilder.Append((Get-Content -LiteralPath $Fragment.FullName -Raw))
}
$InstallerText = $InstallerBuilder.ToString().Replace("`r`n", "`n")
$InstallerText = $InstallerText.TrimEnd([char[]]@([char]13, [char]10)) + "`n"
$Installer = Join-Path $TempRoot 'apply-procedural-jungle-v3-overlay.py'
$Utf8NoBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText($Installer, $InstallerText, $Utf8NoBom)
$InstallerByteLength = ([IO.File]::ReadAllBytes($Installer)).Length
if ($InstallerByteLength -ne 46901) {
    throw "V3 installer UTF-8 byte length mismatch: expected=46901 actual=$InstallerByteLength"
}
$InstallerSha = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
$DiagnosticLogRoot = 'C:\AI\ProceduralJungle\20260804\logs'
New-Item -ItemType Directory -Path $DiagnosticLogRoot -Force | Out-Null
Get-ChildItem -LiteralPath $DiagnosticLogRoot -Filter 'jungle-v3-installer-*' -File -ErrorAction SilentlyContinue | Remove-Item -Force
[IO.File]::WriteAllText((Join-Path $DiagnosticLogRoot 'jungle-v3-installer-reconstructed.py.log'), $InstallerText, $Utf8NoBom)
$FragmentMetadata = New-Object Collections.Generic.List[string]
for ($Index = 0; $Index -lt $InstallerFragments.Count; $Index++) {
    $FragmentText = (Get-Content -LiteralPath $InstallerFragments[$Index].FullName -Raw).Replace("`r`n", "`n")
    $FragmentPath = Join-Path $DiagnosticLogRoot ("jungle-v3-installer-fragment-{0:D2}.py.log" -f $Index)
    [IO.File]::WriteAllText($FragmentPath, $FragmentText, $Utf8NoBom)
    $FragmentSha = (Get-FileHash -LiteralPath $FragmentPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $FragmentMetadata.Add(("INDEX={0};NAME={1};LENGTH={2};SHA256={3}" -f $Index, $InstallerFragments[$Index].Name, $FragmentText.Length, $FragmentSha))
}
[IO.File]::WriteAllLines((Join-Path $DiagnosticLogRoot 'jungle-v3-installer-fragments.log'), $FragmentMetadata, $Utf8NoBom)
$ExpectedInstallerSha = '8caff1745534e6c73ae21b86a6cb5ee035bad5512741a1c5561bc3b649a348d5'
if ($InstallerSha -ne $ExpectedInstallerSha) {
    throw "V3 installer hash mismatch: expected=$ExpectedInstallerSha actual=$InstallerSha"
}
Write-Host "JUNGLE_V3_INSTALLER_UTF8_BYTES=$InstallerByteLength"
Write-Host "JUNGLE_V3_INSTALLER_CANONICAL_SHA256=$InstallerSha"
'@

$PatchedText = $SourceText.Substring(0, $StartIndex) + $Replacement.TrimEnd() + "`n" + $SourceText.Substring($EndIndex)
if ($PatchedText -match 'InstallerText\.Length -ne 46901' -or $PatchedText -match 'InstallerText\.Length -eq 46900') {
    throw 'Rejected character-count installer logic remains after block replacement'
}
if ($PatchedText -notmatch 'InstallerByteLength -ne 46901') { throw 'UTF-8 byte-length gate is missing after block replacement' }
if ($PatchedText -notmatch 'JUNGLE_OWNED_UNREAL_PROCESS_PREFLIGHT_V1') { throw 'Safe Unreal process preflight was lost during block replacement' }

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-v3-assemblyfix-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedWorker = Join-Path $TempRoot 'run-procedural-jungle-v3-assemblyfixed.ps1'
[IO.File]::WriteAllText($PatchedWorker, $PatchedText, (New-Object Text.UTF8Encoding($false)))

Write-Host "JUNGLE_V3_SAFE_PREFLIGHT_SOURCE=$SourceCommit"
Write-Host 'JUNGLE_V3_INSTALLER_ASSEMBLY_BLOCK_REPLACEMENT=PROVEN'
& powershell -NoProfile -ExecutionPolicy Bypass -File $PatchedWorker -ExpectedBranch $ExpectedBranch
if ($LASTEXITCODE -ne 0) { throw "Assembly-fixed V3 worker failed with exit code $LASTEXITCODE" }
