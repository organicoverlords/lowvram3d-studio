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
if ($Chunks.Count -lt 1) { throw "No worker bundle chunks found: $BundleRoot" }

$EncodedBuilder = New-Object Text.StringBuilder
foreach ($Chunk in $Chunks) {
    $ChunkText = Get-Content -LiteralPath $Chunk.FullName -Raw
    [void]$EncodedBuilder.Append(($ChunkText -replace '\s', ''))
}
$Encoded = $EncodedBuilder.ToString()
if ([string]::IsNullOrWhiteSpace($Encoded)) { throw 'Worker bundle is empty' }

$BundleBytes = [Convert]::FromBase64String($Encoded)
$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-bundle-inspect-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$ZipPath = Join-Path $TempRoot 'worker.zip'
$ExtractRoot = Join-Path $TempRoot 'extracted'
[IO.File]::WriteAllBytes($ZipPath, $BundleBytes)
Expand-Archive -LiteralPath $ZipPath -DestinationPath $ExtractRoot -Force

$Files = @(Get-ChildItem -LiteralPath $ExtractRoot -Recurse -File | Sort-Object FullName)
if ($Files.Count -lt 1) { throw 'Decoded worker bundle contains no files' }

Write-Host "JUNGLE_BUNDLE_INSPECTION=PROVEN"
Write-Host "BUNDLE_HEAD=$Head"
Write-Host "BUNDLE_CHUNK_COUNT=$($Chunks.Count)"
Write-Host "BUNDLE_ZIP_BYTES=$($BundleBytes.Length)"
Write-Host "BUNDLE_FILE_COUNT=$($Files.Count)"

foreach ($File in $Files) {
    $Relative = $File.FullName.Substring($ExtractRoot.Length).TrimStart('\')
    Write-Host "BUNDLE_FILE=$Relative|BYTES=$($File.Length)"
}

$SourceExtensions = @('.py', '.ps1', '.cpp', '.h', '.hpp', '.cs', '.ini', '.json', '.md')
$KeywordPattern = '(?i)(^\s*(def|class|function)\s+|tree|canopy|understory|grass|fern|vine|root|rock|terrain|river|waterfall|pool|water|foam|mist|spray|material|roughness|color|sun|light|sky|fog|exposure|camera|capture|screenshot|instance|population|acceptance|visual)'
$TotalPrinted = 0
$MaxPrinted = 2600

foreach ($File in $Files) {
    if ($SourceExtensions -notcontains $File.Extension.ToLowerInvariant()) { continue }
    $Relative = $File.FullName.Substring($ExtractRoot.Length).TrimStart('\')
    $Lines = @(Get-Content -LiteralPath $File.FullName)
    $Indexes = New-Object 'System.Collections.Generic.SortedSet[int]'
    for ($Index = 0; $Index -lt $Lines.Count; $Index++) {
        if ($Lines[$Index] -match $KeywordPattern) {
            foreach ($ContextIndex in @(($Index - 2), ($Index - 1), $Index, ($Index + 1), ($Index + 2))) {
                if ($ContextIndex -ge 0 -and $ContextIndex -lt $Lines.Count) {
                    [void]$Indexes.Add($ContextIndex)
                }
            }
        }
    }
    if ($Indexes.Count -eq 0) { continue }
    Write-Host "SOURCE_EXCERPT_BEGIN=$Relative"
    foreach ($Index in $Indexes) {
        if ($TotalPrinted -ge $MaxPrinted) { break }
        $Number = $Index + 1
        Write-Host ('{0:D5}: {1}' -f $Number, $Lines[$Index])
        $TotalPrinted++
    }
    Write-Host "SOURCE_EXCERPT_END=$Relative"
    if ($TotalPrinted -ge $MaxPrinted) { break }
}

Write-Host "SOURCE_EXCERPT_LINES=$TotalPrinted"
Write-Host 'DIAGNOSTIC_ONLY=TRUE'
Write-Host 'BLENDER_INVOKED=FALSE'
Write-Host 'UNREAL_INVOKED=FALSE'
Write-Host 'CODEX_INVOKED=FALSE'
Write-Host 'CLAUDE_INVOKED=FALSE'
Write-Host 'MAGICMUSIC_INVOKED=FALSE'
throw 'DIAGNOSTIC_COMPLETE_NO_BUILD'
