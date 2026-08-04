[CmdletBinding()]
param(
    [string]$ExpectedBranch = 'feature/procedural-jungle-playable-20260804'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Get-CleanBase64Bytes {
    param(
        [Parameter(Mandatory=$true)][string]$Directory,
        [Parameter(Mandatory=$true)][int]$ExpectedChunkCount
    )
    $Chunks = @(Get-ChildItem -LiteralPath $Directory -Filter 'chunk-*.b64' -File | Sort-Object Name)
    if ($Chunks.Count -ne $ExpectedChunkCount) {
        throw "Unexpected chunk count at ${Directory}: expected=$ExpectedChunkCount actual=$($Chunks.Count)"
    }
    $Builder = New-Object Text.StringBuilder
    foreach ($Chunk in $Chunks) {
        $Text = Get-Content -LiteralPath $Chunk.FullName -Raw
        [void]$Builder.Append(($Text -replace '\s', ''))
    }
    if ($Builder.Length -lt 1) { throw "Empty encoded bundle: $Directory" }
    return [Convert]::FromBase64String($Builder.ToString())
}

$RepoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $RepoRoot) { throw 'Not inside a Git repository' }
Set-Location -LiteralPath $RepoRoot
$Remote = (git config --get remote.origin.url).Trim()
$Branch = (git branch --show-current).Trim()
$Head = (git rev-parse HEAD).Trim()
$Status = @(git status --short)
if ($Remote -notmatch 'organicoverlords/lowvram3d-studio(\.git)?$') { throw "Repository mismatch: $Remote" }
if ($Branch -ne $ExpectedBranch) { throw "Branch mismatch: $Branch" }
if ($Status.Count -ne 0) { throw "Repository is dirty before V3 worker: $($Status -join '; ')" }
git fetch origin $Branch --quiet
$RemoteHead = (git rev-parse "origin/$Branch").Trim()
if ($Head -ne $RemoteHead) { throw "Checkout head differs from remote: $Head vs $RemoteHead" }

$BaseBundleRoot = Join-Path $RepoRoot 'worker-bundles\procedural-jungle-direct-worker'
$BaseBytes = Get-CleanBase64Bytes -Directory $BaseBundleRoot -ExpectedChunkCount 9
$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-v3-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$BaseZip = Join-Path $TempRoot 'base-worker.zip'
$ExtractRoot = Join-Path $TempRoot 'worker'
$OverlayReport = Join-Path $TempRoot 'overlay_installation.json'
[IO.File]::WriteAllBytes($BaseZip, $BaseBytes)
$BaseSha = (Get-FileHash -LiteralPath $BaseZip -Algorithm SHA256).Hash.ToLowerInvariant()
Expand-Archive -LiteralPath $BaseZip -DestinationPath $ExtractRoot -Force

$RequiredBaseFiles = @(
    'blender\procedural_jungle\generate_jungle_assets.py',
    'blender\procedural_jungle\rig_animate_panda.py',
    'scripts\procedural_jungle\build-procedural-jungle.ps1',
    'scripts\procedural_jungle\validate_generated.py',
    'scripts\procedural_jungle\make_contact_sheet.py',
    'unreal\procedural_jungle\build_unreal_scene.py',
    'unreal\procedural_jungle\audit_unreal_scene.py',
    'project_template\Source\ProceduralJungle58\JungleProofDirector.cpp',
    'project_template\Source\ProceduralJungle58\PandaWalkerCharacter.cpp',
    'project_template\Source\ProceduralJungle58\JunglePopulationActor.cpp'
)
foreach ($Relative in $RequiredBaseFiles) {
    $Path = Join-Path $ExtractRoot $Relative
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Base worker file missing after ZIP expansion: $Relative" }
}

$PythonCandidates = @(
    "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\control\Scripts\python.exe",
    $(if (Get-Command python.exe -ErrorAction SilentlyContinue) { (Get-Command python.exe).Source } else { $null })
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
$Python = $PythonCandidates | Select-Object -First 1
if (-not $Python) { throw 'No Python interpreter is available for the V3 overlay installer' }
$InstallerBundleRoot = Join-Path $RepoRoot 'worker-bundles\procedural-jungle-v3-installer'
$InstallerFragments = @(Get-ChildItem -LiteralPath $InstallerBundleRoot -Filter 'chunk-*.pyfrag' -File | Sort-Object Name)
if ($InstallerFragments.Count -ne 6) {
    throw "Unexpected V3 installer fragment count: expected=6 actual=$($InstallerFragments.Count)"
}
$InstallerBuilder = New-Object Text.StringBuilder
foreach ($Fragment in $InstallerFragments) {
    [void]$InstallerBuilder.Append((Get-Content -LiteralPath $Fragment.FullName -Raw))
}
$InstallerText = $InstallerBuilder.ToString()
if ($InstallerText.Length -ne 46901) { throw "V3 installer byte-safe text length mismatch: expected=46901 actual=$($InstallerText.Length)" }
$Installer = Join-Path $TempRoot 'apply-procedural-jungle-v3-overlay.py'
[IO.File]::WriteAllText($Installer, $InstallerText, (New-Object Text.UTF8Encoding($false)))
$InstallerSha = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
$ExpectedInstallerSha = '8caff1745534e6c73ae21b86a6cb5ee035bad5512741a1c5561bc3b649a348d5'
if ($InstallerSha -ne $ExpectedInstallerSha) {
    throw "V3 installer hash mismatch: expected=$ExpectedInstallerSha actual=$InstallerSha"
}
& $Python $Installer --target $ExtractRoot --report $OverlayReport
if ($LASTEXITCODE -ne 0) { throw "V3 overlay installer failed with exit code $LASTEXITCODE" }
$Overlay = Get-Content -LiteralPath $OverlayReport -Raw | ConvertFrom-Json
if ($Overlay.classification -ne 'PROVEN' -or $Overlay.marker -ne 'JUNGLE_VISUAL_OVERHAUL_V3_INSTALLER' -or [int]$Overlay.file_count -ne 11) {
    throw "V3 overlay installation rejected: $($Overlay | ConvertTo-Json -Compress)"
}

$FinalGenerator = Get-Content -LiteralPath (Join-Path $ExtractRoot 'blender\procedural_jungle\generate_jungle_assets.py') -Raw
$FinalBuild = Get-Content -LiteralPath (Join-Path $ExtractRoot 'scripts\procedural_jungle\build-procedural-jungle.ps1') -Raw
$FinalProof = Get-Content -LiteralPath (Join-Path $ExtractRoot 'project_template\Source\ProceduralJungle58\JungleProofDirector.cpp') -Raw
if ($FinalGenerator -notmatch 'procedural_jungle_manifest_v3' -or $FinalGenerator -notmatch 'DENSE_JUNGLE_VISUAL_ASSETS_GENERATED') { throw 'V3 generator markers missing after installation' }
if ($FinalBuild -match 'RenderOffScreen' -or $FinalBuild -match 'utf8NoBOM') { throw 'Rejected runtime flag or unsupported encoding remains after V3 installation' }
if ($FinalProof -notmatch 'PrepareNextCapture' -or $FinalProof -notmatch 'panda_framed_capture_count') { throw 'V3 proof-camera implementation missing after installation' }

Write-Host "JUNGLE_BASE_WORKER_ZIP_SHA256=$BaseSha"
Write-Host 'JUNGLE_BASE_WORKER_DECODE=PROVEN'
Write-Host "JUNGLE_VISUAL_OVERHAUL_V3_INSTALLER_SHA256=$InstallerSha"
Write-Host 'JUNGLE_VISUAL_OVERHAUL_V3_INSTALLATION=PROVEN'
Write-Host "JUNGLE_VISUAL_OVERHAUL_V3_FILE_COUNT=$($Overlay.file_count)"
Write-Host "JUNGLE_WORKER_HEAD=$Head"
Write-Host 'CODEX_INVOKED=FALSE'
Write-Host 'CLAUDE_INVOKED=FALSE'
Write-Host 'MAGICMUSIC_INVOKED=FALSE'

$BuildPath = Join-Path $ExtractRoot 'scripts\procedural_jungle\build-procedural-jungle.ps1'
& powershell -NoProfile -ExecutionPolicy Bypass -File $BuildPath -SourceRoot $ExtractRoot
if ($LASTEXITCODE -ne 0) { throw "Procedural-jungle V3 pipeline failed with exit code $LASTEXITCODE" }

$AcceptancePath = 'C:\AI\ProceduralJungle\20260804\acceptance.json'
if (-not (Test-Path -LiteralPath $AcceptancePath -PathType Leaf)) { throw "Acceptance output missing: $AcceptancePath" }
$Acceptance = Get-Content -LiteralPath $AcceptancePath -Raw | ConvertFrom-Json
if ($Acceptance.classification -ne 'JUNGLE_PANDA_PLAYABLE_PROVEN') { throw "Playable acceptance rejected: $($Acceptance.classification)" }
if ($Acceptance.visual_quality_classification -ne 'DENSE_JUNGLE_VISUAL_QUALITY_PROVEN') { throw "Visual acceptance rejected: $($Acceptance.visual_quality_classification)" }
if ([int]$Acceptance.generated_variant_count -lt 50) { throw 'V3 variant count below 50' }
if ([int]$Acceptance.generated_instance_count -lt 4000) { throw 'V3 instance count below 4000' }
if ([int]$Acceptance.canopy_instance_count -lt 650) { throw 'V3 canopy count below 650' }
if ([int]$Acceptance.understory_instance_count -lt 3150) { throw 'V3 understory count below 3150' }
if ([int]$Acceptance.panda_framed_capture_count -lt 4) { throw 'V3 panda-framed captures below 4' }
if ([int]$Acceptance.vegetation_rich_frame_count -lt 6) { throw 'V3 vegetation-rich frames below 6' }
Write-Host 'JUNGLE_VISUAL_OVERHAUL_V3=PROVEN'
Write-Host ($Acceptance | ConvertTo-Json -Depth 30)
