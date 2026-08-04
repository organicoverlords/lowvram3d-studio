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
if ($Remote -notmatch 'organicoverlords/lowvram3d-studio(\.git)?$') { throw "Repository mismatch: $Remote" }
if ($Branch -ne $ExpectedBranch) { throw "Branch mismatch: $Branch" }

$BundleRoot = Join-Path $RepoRoot 'worker-bundles\procedural-jungle-direct-worker'
$Chunks = @(Get-ChildItem -LiteralPath $BundleRoot -Filter 'chunk-*.b64' -File | Sort-Object Name)
if ($Chunks.Count -ne 9) { throw "Unexpected worker bundle chunk count: $($Chunks.Count)" }

$EncodedBuilder = New-Object Text.StringBuilder
foreach ($Chunk in $Chunks) {
    $ChunkText = Get-Content -LiteralPath $Chunk.FullName -Raw
    [void]$EncodedBuilder.Append(($ChunkText -replace '\s', ''))
}
$BundleBytes = [Convert]::FromBase64String($EncodedBuilder.ToString())

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-source-export-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$ZipPath = Join-Path $TempRoot 'worker.zip'
$ExtractRoot = Join-Path $TempRoot 'extracted'
[IO.File]::WriteAllBytes($ZipPath, $BundleBytes)
Expand-Archive -LiteralPath $ZipPath -DestinationPath $ExtractRoot -Force

$LogRoot = 'C:\AI\ProceduralJungle\20260804\logs'
New-Item -ItemType Directory -Path $LogRoot -Force | Out-Null
Get-ChildItem -LiteralPath $LogRoot -Filter 'source_snapshot_*.log' -File -ErrorAction SilentlyContinue | Remove-Item -Force

$Selected = @(
    'blender\procedural_jungle\generate_jungle_assets.py',
    'project_template\Source\ProceduralJungle58\JunglePopulationActor.cpp',
    'project_template\Source\ProceduralJungle58\JunglePopulationActor.h',
    'project_template\Source\ProceduralJungle58\JungleProofDirector.cpp',
    'project_template\Source\ProceduralJungle58\JungleProofDirector.h',
    'project_template\Source\ProceduralJungle58\PandaWalkerCharacter.cpp',
    'project_template\Source\ProceduralJungle58\PandaWalkerCharacter.h',
    'scripts\procedural_jungle\build-procedural-jungle.ps1',
    'scripts\procedural_jungle\make_contact_sheet.py',
    'scripts\procedural_jungle\validate_generated.py',
    'unreal\procedural_jungle\build_unreal_scene.py',
    'unreal\procedural_jungle\audit_unreal_scene.py'
)

$Manifest = New-Object System.Collections.Generic.List[object]
foreach ($RelativePath in $Selected) {
    $SourcePath = Join-Path $ExtractRoot $RelativePath
    if (-not (Test-Path -LiteralPath $SourcePath -PathType Leaf)) { throw "Selected bundle source missing: $RelativePath" }
    $SafeName = ($RelativePath -replace '[\\/:*?"<>|]', '__')
    $Destination = Join-Path $LogRoot ("source_snapshot_{0}.log" -f $SafeName)
    [IO.File]::WriteAllBytes($Destination, [IO.File]::ReadAllBytes($SourcePath))
    $Hash = (Get-FileHash -LiteralPath $SourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
    $Manifest.Add([ordered]@{
        relative_path = $RelativePath
        snapshot_path = $Destination
        bytes = (Get-Item -LiteralPath $SourcePath).Length
        sha256 = $Hash
    })
    Write-Host "SOURCE_SNAPSHOT=$RelativePath|BYTES=$((Get-Item -LiteralPath $SourcePath).Length)|SHA256=$Hash"
}

$ManifestPath = Join-Path $LogRoot 'source_snapshot_manifest.log'
$ManifestText = [ordered]@{
    classification = 'PROVEN'
    branch = $Branch
    head = $Head
    bundle_chunk_count = $Chunks.Count
    bundle_zip_bytes = $BundleBytes.Length
    files = $Manifest
    blender_invoked = $false
    unreal_invoked = $false
    codex_invoked = $false
    claude_invoked = $false
    magicmusic_invoked = $false
} | ConvertTo-Json -Depth 10
[IO.File]::WriteAllText($ManifestPath, $ManifestText, (New-Object Text.UTF8Encoding($false)))

Write-Host "JUNGLE_FULL_SOURCE_EXPORT=PROVEN"
Write-Host "SOURCE_SNAPSHOT_COUNT=$($Manifest.Count)"
Write-Host 'DIAGNOSTIC_ONLY=TRUE'
Write-Host 'BLENDER_INVOKED=FALSE'
Write-Host 'UNREAL_INVOKED=FALSE'
throw 'DIAGNOSTIC_COMPLETE_NO_BUILD'
