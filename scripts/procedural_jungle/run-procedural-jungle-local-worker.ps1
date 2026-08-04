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

# Preserve the previously proven bootstrap byte-for-byte as the source of truth,
# then inject only the two fixes evidenced by workflow run 30861450720.
$BaselineCommit = 'bdf71585456e0c97aa77234f1bc01a7cf67c7bf4'
$BaselineBlob = '39fed93175360450c99891fbdf61ba0ba1d7ed71'
$BootstrapPath = 'scripts/procedural_jungle/run-procedural-jungle-local-worker.ps1'
& git merge-base --is-ancestor $BaselineCommit HEAD
if ($LASTEXITCODE -ne 0) { throw "Baseline commit is not an ancestor of HEAD: $BaselineCommit" }
$ActualBaselineBlob = (& git rev-parse "$BaselineCommit`:$BootstrapPath").Trim()
if ($LASTEXITCODE -ne 0 -or $ActualBaselineBlob -ne $BaselineBlob) {
    throw "Baseline bootstrap identity mismatch: expected=$BaselineBlob actual=$ActualBaselineBlob"
}

$WrapperTemp = Join-Path $env:RUNNER_TEMP "procedural-jungle-bootstrap-$Head"
if (Test-Path -LiteralPath $WrapperTemp) { Remove-Item -LiteralPath $WrapperTemp -Recurse -Force }
New-Item -ItemType Directory -Path $WrapperTemp -Force | Out-Null
$PatchedBootstrap = Join-Path $WrapperTemp 'run-procedural-jungle-local-worker-patched.ps1'
$BaselineLines = @(& git show "$BaselineCommit`:$BootstrapPath")
if ($LASTEXITCODE -ne 0 -or $BaselineLines.Count -lt 10) { throw 'Could not read the proven baseline bootstrap' }
$BaselineText = $BaselineLines -join "`n"

$InsertionMarker = '# Windows PowerShell 5 turns native stderr into terminating error records under'
$MarkerMatches = [regex]::Matches($BaselineText, [regex]::Escape($InsertionMarker))
if ($MarkerMatches.Count -ne 1) {
    throw "Could not prove unique bootstrap insertion marker; matches=$($MarkerMatches.Count)"
}

$EvidenceFixes = @'
# Workflow run 30861450720 proved that generated materials are created correctly,
# but load_asset logs a hard error when probing a path that does not exist yet.
# Guard every assignment-form loader call with the same receiver's existence test.
$SceneScript = Join-Path $SourceRoot 'unreal\procedural_jungle\build_unreal_scene.py'
if (-not (Test-Path -LiteralPath $SceneScript)) { throw "Decoded Unreal scene script missing: $SceneScript" }
$SceneText = Get-Content -LiteralPath $SceneScript -Raw
$AssetLoadCandidatePattern = '(?m)^[ \t]*[A-Za-z_][A-Za-z0-9_]*[ \t]*=[^\r\n]*\.load_asset\([^\r\n]*$'
$AssetLoadCandidates = @([regex]::Matches($SceneText, $AssetLoadCandidatePattern))
$AssetLoadPattern = '(?m)^(?<indent>[ \t]*)(?<lhs>[A-Za-z_][A-Za-z0-9_]*)[ \t]*=[ \t]*(?<receiver>[A-Za-z_][A-Za-z0-9_.]*)\.load_asset\((?<arg>.+)\)[ \t]*$'
$AssetLoadMatches = @([regex]::Matches($SceneText, $AssetLoadPattern))
if ($AssetLoadCandidates.Count -lt 1 -or $AssetLoadCandidates.Count -gt 16) {
    throw "Unexpected assignment-form load_asset candidate count: $($AssetLoadCandidates.Count)"
}
if ($AssetLoadMatches.Count -ne $AssetLoadCandidates.Count) {
    $CandidateText = ($AssetLoadCandidates | ForEach-Object { $_.Value.Trim() }) -join ' | '
    throw "Not every assignment-form load_asset call has a simple dotted receiver: matches=$($AssetLoadMatches.Count) candidates=$($AssetLoadCandidates.Count) lines=$CandidateText"
}
$AssetLoadEvaluator = [Text.RegularExpressions.MatchEvaluator]{
    param([Text.RegularExpressions.Match]$Match)
    $Indent = $Match.Groups['indent'].Value
    $Left = $Match.Groups['lhs'].Value
    $Receiver = $Match.Groups['receiver'].Value
    $Argument = $Match.Groups['arg'].Value.Trim()
    return "$Indent$Left = $Receiver.load_asset($Argument) if $Receiver.does_asset_exist($Argument) else None"
}
$SceneText = [regex]::Replace($SceneText, $AssetLoadPattern, $AssetLoadEvaluator)
$RemainingUnguardedLoads = @([regex]::Matches($SceneText, $AssetLoadPattern))
if ($RemainingUnguardedLoads.Count -ne 0) {
    throw "Unguarded assignment-form load_asset calls remain after patch: $($RemainingUnguardedLoads.Count)"
}
Set-Content -LiteralPath $SceneScript -Value $SceneText -Encoding utf8
Write-Host 'UNREAL_ASSET_LOAD_GUARD_PATCH=PROVEN'
Write-Host "UNREAL_ASSET_LOADS_GUARDED=$($AssetLoadMatches.Count)"

# The same run proved every HISM was Static while JunglePopulationActor.SceneRoot
# remained Movable, so PIE refused all attachments. Keep vegetation static for
# performance and make the one population root Static before project compilation.
$PopulationCppFiles = @(
    Get-ChildItem -LiteralPath $SourceRoot -Recurse -File -Filter '*.cpp' -ErrorAction Stop |
        Where-Object { (Get-Content -LiteralPath $_.FullName -Raw) -match 'JunglePopulationActor' }
)
if ($PopulationCppFiles.Count -lt 1 -or $PopulationCppFiles.Count -gt 3) {
    throw "Unexpected JunglePopulationActor source count: $($PopulationCppFiles.Count)"
}
$RootPattern = '(?m)^(?<indent>[ \t]*)(?<root>[A-Za-z_][A-Za-z0-9_]*)[ \t]*=[ \t]*CreateDefaultSubobject<USceneComponent>\(TEXT\("SceneRoot"\)\);[ \t]*$'
$RootMatchTotal = 0
foreach ($PopulationCpp in $PopulationCppFiles) {
    $CppText = Get-Content -LiteralPath $PopulationCpp.FullName -Raw
    $RootMatches = [regex]::Matches($CppText, $RootPattern)
    if ($RootMatches.Count -eq 0) { continue }
    if ($RootMatches.Count -ne 1) {
        throw "Ambiguous SceneRoot constructor in $($PopulationCpp.FullName); matches=$($RootMatches.Count)"
    }
    if ($CppText -match 'SceneRoot[^\r\n]*SetMobility\(EComponentMobility::Static\)') {
        throw "Population SceneRoot already has a static mobility assignment: $($PopulationCpp.FullName)"
    }
    $RootEvaluator = [Text.RegularExpressions.MatchEvaluator]{
        param([Text.RegularExpressions.Match]$Match)
        $Indent = $Match.Groups['indent'].Value
        $Root = $Match.Groups['root'].Value
        return $Match.Value + "`r`n" + $Indent + $Root + '->SetMobility(EComponentMobility::Static);'
    }
    $CppText = [regex]::Replace($CppText, $RootPattern, $RootEvaluator, 1)
    Set-Content -LiteralPath $PopulationCpp.FullName -Value $CppText -Encoding utf8
    $RootMatchTotal++
}
if ($RootMatchTotal -ne 1) { throw "Could not prove one JunglePopulationActor SceneRoot patch; matches=$RootMatchTotal" }
Write-Host 'JUNGLE_POPULATION_STATIC_ROOT_PATCH=PROVEN'
'@

$PatchedText = $BaselineText.Replace($InsertionMarker, $EvidenceFixes + "`n`n" + $InsertionMarker)
Set-Content -LiteralPath $PatchedBootstrap -Value $PatchedText -Encoding utf8
Write-Host "BOOTSTRAP_BASELINE_COMMIT=$BaselineCommit"
Write-Host "BOOTSTRAP_BASELINE_BLOB=$BaselineBlob"
Write-Host 'RUN11_EVIDENCE_FIX_INJECTION=PROVEN'

& $PatchedBootstrap -ExpectedBranch $ExpectedBranch
$WorkerExit = $LASTEXITCODE
if ($WorkerExit -ne 0) { throw "Patched direct worker failed with exit code $WorkerExit" }
