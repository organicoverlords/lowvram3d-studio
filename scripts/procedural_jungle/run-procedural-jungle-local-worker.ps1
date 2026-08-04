[CmdletBinding()]
param(
    [string]$ExpectedBranch = 'feature/procedural-jungle-playable-20260804'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

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
if ($Status.Count -ne 0) { throw 'Repository is dirty before jungle build' }

$BaseBundleRoot = Join-Path $RepoRoot 'worker-bundles\procedural-jungle-direct-worker'
$OverhaulBundleRoot = Join-Path $RepoRoot 'worker-bundles\procedural-jungle-visual-overhaul'
$BaseBytes = Get-CleanBase64Bytes -Directory $BaseBundleRoot -ExpectedChunkCount 9
$OverhaulBytes = Get-CleanBase64Bytes -Directory $OverhaulBundleRoot -ExpectedChunkCount 4

$ExpectedOverhaulSha = 'd78dfd8b68e1a47573446c038bfaaf03d996751b931160266279ee5814613129'
$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-visual-overhaul-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$BaseZip = Join-Path $TempRoot 'base-worker.zip'
$OverhaulArchive = Join-Path $TempRoot 'visual-overhaul.tar.xz'
$ExtractRoot = Join-Path $TempRoot 'worker'
[IO.File]::WriteAllBytes($BaseZip, $BaseBytes)
[IO.File]::WriteAllBytes($OverhaulArchive, $OverhaulBytes)

$ActualOverhaulSha = (Get-FileHash -LiteralPath $OverhaulArchive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualOverhaulSha -ne $ExpectedOverhaulSha) {
    throw "Visual-overhaul archive hash mismatch: expected=$ExpectedOverhaulSha actual=$ActualOverhaulSha"
}

Expand-Archive -LiteralPath $BaseZip -DestinationPath $ExtractRoot -Force
$Tar = Get-Command tar.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -First 1
if (-not $Tar) { throw 'Windows tar.exe is required to extract the audited visual-overhaul archive' }
& $Tar -xJf $OverhaulArchive -C $ExtractRoot
if ($LASTEXITCODE -ne 0) { throw "Visual-overhaul extraction failed with exit code $LASTEXITCODE" }

$ManifestPath = Join-Path $ExtractRoot 'MANIFEST.sha256'
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) { throw 'Visual-overhaul manifest is missing' }
$VerifiedFiles = 0
foreach ($Line in @(Get-Content -LiteralPath $ManifestPath)) {
    if ($Line -notmatch '^([0-9a-f]{64})\s+\./(.+)$') { throw "Malformed overhaul manifest line: $Line" }
    $ExpectedHash = $Matches[1]
    $RelativePath = $Matches[2]
    if ($RelativePath -eq 'MANIFEST.sha256') { continue }
    $TargetPath = Join-Path $ExtractRoot ($RelativePath -replace '/', '\')
    if (-not (Test-Path -LiteralPath $TargetPath -PathType Leaf)) { throw "Overhaul file missing: $RelativePath" }
    $ActualHash = (Get-FileHash -LiteralPath $TargetPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne $ExpectedHash) {
        throw "Overhaul file hash mismatch: path=$RelativePath expected=$ExpectedHash actual=$ActualHash"
    }
    $VerifiedFiles++
}
if ($VerifiedFiles -ne 10) { throw "Unexpected verified-overhaul file count: $VerifiedFiles" }

$GeneratorPath = Join-Path $ExtractRoot 'blender\procedural_jungle\generate_jungle_assets.py'
$BuildPath = Join-Path $ExtractRoot 'scripts\procedural_jungle\build-procedural-jungle.ps1'
$ProofCppPath = Join-Path $ExtractRoot 'project_template\Source\ProceduralJungle58\JungleProofDirector.cpp'
$PandaCppPath = Join-Path $ExtractRoot 'project_template\Source\ProceduralJungle58\PandaWalkerCharacter.cpp'
$VisualAuditPath = Join-Path $ExtractRoot 'scripts\procedural_jungle\audit_visual_captures.py'

$GeneratorText = Get-Content -LiteralPath $GeneratorPath -Raw
$BuildText = Get-Content -LiteralPath $BuildPath -Raw
$ProofText = Get-Content -LiteralPath $ProofCppPath -Raw
$PandaText = Get-Content -LiteralPath $PandaCppPath -Raw
$VisualAuditText = Get-Content -LiteralPath $VisualAuditPath -Raw
if ($GeneratorText -notmatch 'procedural_jungle_manifest_v2') { throw 'Dense-jungle generator marker missing' }
if ($GeneratorText -notmatch 'DENSE_JUNGLE_VISUAL_ASSETS_GENERATED') { throw 'Dense-jungle classification marker missing' }
if ($BuildText -match 'RenderOffScreen') { throw 'Rejected off-screen rendering flag remains in build script' }
if ($BuildText -match 'utf8NoBOM') { throw 'Unsupported Windows PowerShell encoding remains in build script' }
if ($ProofText -notmatch 'PrepareNextCapture' -or $ProofText -notmatch 'panda_framed_capture_count') { throw 'Dynamic panda capture implementation missing' }
if ($PandaText -notmatch 'bRunPhysicsWithNoController') { throw 'Controller-free panda movement fix missing' }
if ($VisualAuditText -notmatch 'DENSE_JUNGLE_VISUAL_QUALITY_PROVEN') { throw 'Dense-jungle visual audit marker missing' }

Write-Host 'JUNGLE_BASE_BUNDLE_DECODE=PROVEN'
Write-Host "JUNGLE_VISUAL_OVERHAUL_ARCHIVE_SHA256=$ActualOverhaulSha"
Write-Host "JUNGLE_VISUAL_OVERHAUL_FILE_COUNT=$VerifiedFiles"
Write-Host 'JUNGLE_VISUAL_OVERHAUL_OVERLAY=PROVEN'
Write-Host "JUNGLE_WORKER_HEAD=$Head"
Write-Host 'CODEX_INVOKED=FALSE'
Write-Host 'CLAUDE_INVOKED=FALSE'
Write-Host 'MAGICMUSIC_INVOKED=FALSE'

& powershell -NoProfile -ExecutionPolicy Bypass -File $BuildPath -SourceRoot $ExtractRoot
if ($LASTEXITCODE -ne 0) { throw "Dense-jungle pipeline failed with exit code $LASTEXITCODE" }

$AcceptancePath = 'C:\AI\ProceduralJungle\20260804\acceptance.json'
if (-not (Test-Path -LiteralPath $AcceptancePath -PathType Leaf)) { throw "Acceptance output missing: $AcceptancePath" }
$Acceptance = Get-Content -LiteralPath $AcceptancePath -Raw | ConvertFrom-Json
if ($Acceptance.classification -ne 'JUNGLE_PANDA_PLAYABLE_PROVEN') { throw "Playable acceptance rejected: $($Acceptance.classification)" }
if ($Acceptance.visual_quality -ne 'PROVEN') { throw "Visual quality rejected: $($Acceptance.visual_quality)" }
if ($Acceptance.visual_quality_classification -ne 'DENSE_JUNGLE_VISUAL_QUALITY_PROVEN') {
    throw "Visual-quality classification rejected: $($Acceptance.visual_quality_classification)"
}
if ([int]$Acceptance.generated_variant_count -lt 40) { throw 'Dense-jungle variant count below 40' }
if ([int]$Acceptance.generated_instance_count -lt 3300) { throw 'Dense-jungle instance count below 3300' }
if ([int]$Acceptance.canopy_instance_count -lt 380) { throw 'Canopy instance count below 380' }
if ([int]$Acceptance.understory_instance_count -lt 2500) { throw 'Understory instance count below 2500' }
if ([int]$Acceptance.panda_framed_capture_count -lt 4) { throw 'Too few panda-framed captures' }
if ([int]$Acceptance.vegetation_rich_frame_count -lt 6) { throw 'Too few vegetation-rich captures' }
if ([double]$Acceptance.average_saturation -lt 0.19) { throw 'Capture saturation remains too low' }
if ([double]$Acceptance.average_near_white_fraction -gt 0.16) { throw 'Capture set remains overexposed' }

Write-Host 'DENSE_JUNGLE_VISUAL_OVERHAUL=PROVEN'
Write-Host ($Acceptance | ConvertTo-Json -Depth 30)
