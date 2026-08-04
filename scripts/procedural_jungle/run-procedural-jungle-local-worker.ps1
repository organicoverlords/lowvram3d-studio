[CmdletBinding()]
param(
    [string]$ExpectedBranch = 'feature/procedural-jungle-playable-20260804'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $RepoRoot) { throw 'Not inside a Git repository' }
Set-Location -LiteralPath $RepoRoot

$Remote = (git config --get remote.origin.url).Trim()
$Branch = (git branch --show-current).Trim()
$Head = (git rev-parse HEAD).Trim()
$Status = @(git status --short)
if ($Remote -notmatch 'organicoverlords/lowvram3d-studio(\.git)?$') { throw "Repository mismatch: $Remote" }
if ($Branch -ne $ExpectedBranch) { throw "Branch mismatch: $Branch" }
if ($Status.Count -ne 0) { throw "Repository is dirty before bundle export: $($Status -join '; ')" }

$BundleRoot = Join-Path $RepoRoot 'worker-bundles\procedural-jungle-visual-overhaul'
$Chunks = @(Get-ChildItem -LiteralPath $BundleRoot -Filter 'chunk-*.b64' -File | Sort-Object Name)
if ($Chunks.Count -ne 4) { throw "Expected exactly four visual-overhaul chunks; found $($Chunks.Count)" }

$Builder = New-Object Text.StringBuilder
foreach ($Chunk in $Chunks) {
    $Text = Get-Content -LiteralPath $Chunk.FullName -Raw
    [void]$Builder.Append(($Text -replace '\s', ''))
}
$ArchiveBytes = [Convert]::FromBase64String($Builder.ToString())
$ActualSha = [BitConverter]::ToString(([Security.Cryptography.SHA256]::Create()).ComputeHash($ArchiveBytes)).Replace('-', '').ToLowerInvariant()
$ExpectedSha = 'd78dfd8b68e1a47573446c038bfaaf03d996751b931160266279ee5814613129'
if ($ActualSha -ne $ExpectedSha) { throw "Reconstructed archive hash mismatch: expected=$ExpectedSha actual=$ActualSha" }

$LogRoot = 'C:\AI\ProceduralJungle\20260804\logs'
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
Get-ChildItem -LiteralPath $LogRoot -Filter 'visual-overhaul-diagnostic-*' -File -ErrorAction SilentlyContinue | Remove-Item -Force

foreach ($Chunk in $Chunks) {
    $Destination = Join-Path $LogRoot ("visual-overhaul-diagnostic-{0}.b64.log" -f $Chunk.BaseName)
    Copy-Item -LiteralPath $Chunk.FullName -Destination $Destination -Force
}
$ArchiveLog = Join-Path $LogRoot 'visual-overhaul-diagnostic-archive.tar.xz.log'
[IO.File]::WriteAllBytes($ArchiveLog, $ArchiveBytes)
$Metadata = @(
    "HEAD=$Head",
    "ARCHIVE_SHA256=$ActualSha",
    "ARCHIVE_BYTES=$($ArchiveBytes.Length)",
    "CHUNK_COUNT=$($Chunks.Count)",
    'BLENDER_INVOKED=FALSE',
    'UNREAL_INVOKED=FALSE',
    'GPU_INVOKED=FALSE',
    'CODEX_INVOKED=FALSE',
    'CLAUDE_INVOKED=FALSE',
    'MAGICMUSIC_INVOKED=FALSE'
) -join "`r`n"
[IO.File]::WriteAllText((Join-Path $LogRoot 'visual-overhaul-diagnostic-metadata.log'), $Metadata, (New-Object Text.UTF8Encoding($false)))

Write-Host 'JUNGLE_VISUAL_OVERHAUL_BUNDLE_EXPORT=PROVEN'
Write-Host $Metadata
throw 'BUNDLE_DIAGNOSTIC_COMPLETE: exact archive and chunks exported for offline repair'
