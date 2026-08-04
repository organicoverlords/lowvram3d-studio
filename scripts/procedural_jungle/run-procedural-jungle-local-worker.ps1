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
# Guard first-create material probes that previously emitted hard Unreal errors.
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

# Keep all HISM children and their population root Static.
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

# The approved source GLB disappeared after the rig was generated. Reuse is allowed only when
# the preserved generated FBX, Blend, report, and log prove the exact original source identity.
$BuildScript = Join-Path $SourceRoot 'scripts\procedural_jungle\build-procedural-jungle.ps1'
if (-not (Test-Path -LiteralPath $BuildScript -PathType Leaf)) { throw "Decoded jungle build script missing: $BuildScript" }
$BuildText = (Get-Content -LiteralPath $BuildScript -Raw) -replace "`r`n", "`n"
$RigOld = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('JFBhbmRhR2VuZXJhdGVkUm9vdCA9IEpvaW4tUGF0aCAkR2VuZXJhdGVkUm9vdCAncGFuZGEnCk5ldy1JdGVtIC1JdGVtVHlwZSBEaXJlY3RvcnkgLVBhdGggJFBhbmRhR2VuZXJhdGVkUm9vdCAtRm9yY2UgfCBPdXQtTnVsbAokUGFuZGFSaWdSZXBvcnQgPSBKb2luLVBhdGggJFBhbmRhR2VuZXJhdGVkUm9vdCAncGFuZGFfcmlnX3JlcG9ydC5qc29uJwpJbnZva2UtQ2hlY2tlZCAtTmFtZSAnQkxFTkRFUl9SSUdfQU5JTUFURV9QQU5EQScgLUxvZ1BhdGggKEpvaW4tUGF0aCAkTG9nc1Jvb3QgJ2JsZW5kZXJfcmlnX3BhbmRhLmxvZycpIC1Db21tYW5kIHsKICAgICYgJEJsZW5kZXIgLS1iYWNrZ3JvdW5kIC0tZmFjdG9yeS1zdGFydHVwIC0tcHl0aG9uIChKb2luLVBhdGggJFNvdXJjZVJvb3QgJ2JsZW5kZXJccHJvY2VkdXJhbF9qdW5nbGVccmlnX2FuaW1hdGVfcGFuZGEucHknKSAtLSBgCiAgICAgICAgLS1pbnB1dCAkUGFuZGFTb3VyY2UgLS1vdXRwdXQtcm9vdCAkUGFuZGFHZW5lcmF0ZWRSb290IC0tcmVwb3J0ICRQYW5kYVJpZ1JlcG9ydAp9'))
$RigNew = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('JFBhbmRhR2VuZXJhdGVkUm9vdCA9IEpvaW4tUGF0aCAkR2VuZXJhdGVkUm9vdCAncGFuZGEnCk5ldy1JdGVtIC1JdGVtVHlwZSBEaXJlY3RvcnkgLVBhdGggJFBhbmRhR2VuZXJhdGVkUm9vdCAtRm9yY2UgfCBPdXQtTnVsbAokUGFuZGFSaWdSZXBvcnQgPSBKb2luLVBhdGggJFBhbmRhR2VuZXJhdGVkUm9vdCAncGFuZGFfcmlnX3JlcG9ydC5qc29uJwppZiAoJFJldXNlRXhpc3RpbmdQYW5kYVJpZykgewogICAgV3JpdGUtSG9zdCAnQkxFTkRFUl9SSUdfQU5JTUFURV9QQU5EQT1SRVVTRURfUFJPVkVOX09VVFBVVCcKfSBlbHNlIHsKICAgIEludm9rZS1DaGVja2VkIC1OYW1lICdCTEVOREVSX1JJR19BTklNQVRFX1BBTkRBJyAtTG9nUGF0aCAoSm9pbi1QYXRoICRMb2dzUm9vdCAnYmxlbmRlcl9yaWdfcGFuZGEubG9nJykgLUNvbW1hbmQgewogICAgICAgICYgJEJsZW5kZXIgLS1iYWNrZ3JvdW5kIC0tZmFjdG9yeS1zdGFydHVwIC0tcHl0aG9uIChKb2luLVBhdGggJFNvdXJjZVJvb3QgJ2JsZW5kZXJccHJvY2VkdXJhbF9qdW5nbGVccmlnX2FuaW1hdGVfcGFuZGEucHknKSAtLSBgCiAgICAgICAgICAgIC0taW5wdXQgJFBhbmRhU291cmNlIC0tb3V0cHV0LXJvb3QgJFBhbmRhR2VuZXJhdGVkUm9vdCAtLXJlcG9ydCAkUGFuZGFSaWdSZXBvcnQKICAgIH0KfQ=='))
$RigMatchCount = [regex]::Matches($BuildText, [regex]::Escape($RigOld)).Count
if ($RigMatchCount -ne 1) { throw "Could not prove unique panda rig invocation block; matches=$RigMatchCount" }
$BuildText = $BuildText.Replace($RigOld, $RigNew)

$SelectionStartMarker = '# Select the panda without trusting a filename alone.'
$SelectionEndMarker = "Write-JsonFile `$PandaIdentity (Join-Path `$OutputRoot 'panda_source.json')"
$SelectionStart = $BuildText.IndexOf($SelectionStartMarker, [StringComparison]::Ordinal)
$SelectionEnd = $BuildText.IndexOf($SelectionEndMarker, [StringComparison]::Ordinal)
if ($SelectionStart -lt 0 -or $SelectionEnd -lt $SelectionStart) {
    throw "Could not locate panda source-selection block: start=$SelectionStart end=$SelectionEnd"
}
if ($BuildText.IndexOf($SelectionStartMarker, $SelectionStart + 1, [StringComparison]::Ordinal) -ge 0) {
    throw 'Panda source-selection start marker is not unique'
}
if ($BuildText.IndexOf($SelectionEndMarker, $SelectionEnd + 1, [StringComparison]::Ordinal) -ge 0) {
    throw 'Panda source-selection end marker is not unique'
}
$SelectionLineEnd = $BuildText.IndexOf("`n", $SelectionEnd + $SelectionEndMarker.Length)
if ($SelectionLineEnd -lt 0) { $SelectionLineEnd = $BuildText.Length } else { $SelectionLineEnd++ }
$SelectionNew = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('IyBTZWxlY3QgdGhlIHBhbmRhIHdpdGhvdXQgdHJ1c3RpbmcgYSBmaWxlbmFtZSBhbG9uZS4gQSByZXBhaXJlZCBmaW5hbCBjYW5kaWRhdGUgaXMgYWNjZXB0ZWQgb25seQojIHdoZW4gaXRzIGFkamFjZW50IGFjY2VwdGFuY2UgcmVjb3JkIHByb3ZlcyB0aGUgY29tcGxldGUgdGV4dHVyZSByZXBhaXIuIFRoZSBmYWxsYmFjayByZW1haW5zIHRoZQojIGtub3duIHN0cnVjdHVyYWwgcmVwYWlyIGFuZCB0aGUgZG93bmdyYWRlIGlzIGV4cGxpY2l0IGluIHRoZSByZWNlaXB0LgokUGFuZGFSb290ID0gJ0M6XEFJXExvd1ZSQU0zRC1iZW5jaG1hcmtzXG1pbml0dXJiby0zc3RlcC1leHBlcmltZW50LTIwMjYwODAzXHRhY3RpY2FsX3JlZF9wYW5kYV9zY291dCcKJEZpbmFsQWNjZXB0YW5jZSA9IEpvaW4tUGF0aCAkUGFuZGFSb290ICdwYW5kYV9mdWxsX3BpcGVsaW5lX3JlcGFpcl8yMDI2MDgwNFxhY2NlcHRhbmNlLmpzb24nCiRGaW5hbENhbmRpZGF0ZSA9IEpvaW4tUGF0aCAkUGFuZGFSb290ICdwYW5kYV9mdWxsX3BpcGVsaW5lX3JlcGFpcl8yMDI2MDgwNFxmaW5hbF9jYW5kaWRhdGVcdGFjdGljYWxfcmVkX3BhbmRhX3Njb3V0X3RleHR1cmVkLmdsYicKJEZhbGxiYWNrQ2FuZGlkYXRlID0gSm9pbi1QYXRoICRQYW5kYVJvb3QgJ2Jhcl9sb2NhbF9jbG9zdXJlX3YxXHRhY3RpY2FsX3JlZF9wYW5kYV9zY291dF9iYXJfcmVwYWlyZWQuZ2xiJwokUGFuZGFTb3VyY2UgPSAkbnVsbAokUGFuZGFTb3VyY2VDbGFzc2lmaWNhdGlvbiA9ICRudWxsCiRSZXVzZUV4aXN0aW5nUGFuZGFSaWcgPSAkZmFsc2UKJEV4cGVjdGVkRmFsbGJhY2tTaGEgPSAnNzhjNTUxMzMxNjVlOTMxYmM4ZDY3NjU2MTBhNjc5ZDFkMThiYWRjZGMxNzg4MjBhNjllMzFiN2IzMmJjYmZiOCcKJEV4cGVjdGVkRmFsbGJhY2tCeXRlcyA9IDUwMjQ0NDAwCmlmICgoVGVzdC1QYXRoIC1MaXRlcmFsUGF0aCAkRmluYWxBY2NlcHRhbmNlKSAtYW5kIChUZXN0LVBhdGggLUxpdGVyYWxQYXRoICRGaW5hbENhbmRpZGF0ZSkpIHsKICAgIHRyeSB7CiAgICAgICAgJGEgPSBHZXQtQ29udGVudCAtTGl0ZXJhbFBhdGggJEZpbmFsQWNjZXB0YW5jZSAtUmF3IHwgQ29udmVydEZyb20tSnNvbgogICAgICAgIGlmICgkYS5jbGFzc2lmaWNhdGlvbiAtZXEgJ0ZST05UX1JFQVJfVEVYVFVSRV9SRVBBSVJfUFJPVkVOJykgewogICAgICAgICAgICAkUGFuZGFTb3VyY2UgPSAkRmluYWxDYW5kaWRhdGUKICAgICAgICAgICAgJFBhbmRhU291cmNlQ2xhc3NpZmljYXRpb24gPSAnQUNDRVBURURfVEVYVFVSRURfRklOQUwnCiAgICAgICAgfQogICAgfSBjYXRjaCB7CiAgICAgICAgV3JpdGUtV2FybmluZyAiSWdub3JpbmcgaW52YWxpZCBwYW5kYSBhY2NlcHRhbmNlIEpTT046ICRfIgogICAgfQp9CmlmICgtbm90ICRQYW5kYVNvdXJjZSAtYW5kIChUZXN0LVBhdGggLUxpdGVyYWxQYXRoICRGYWxsYmFja0NhbmRpZGF0ZSkpIHsKICAgICRQYW5kYVNvdXJjZSA9ICRGYWxsYmFja0NhbmRpZGF0ZQogICAgJFBhbmRhU291cmNlQ2xhc3NpZmljYXRpb24gPSAnU1RSVUNUVVJBTF9SRVBBSVJfVEVYVFVSRV9ET1dOR1JBREUnCn0KaWYgKC1ub3QgJFBhbmRhU291cmNlKSB7CiAgICAkUGFuZGFHZW5lcmF0ZWRSb290ID0gSm9pbi1QYXRoICRHZW5lcmF0ZWRSb290ICdwYW5kYScKICAgICRQYW5kYVJpZ1JlcG9ydCA9IEpvaW4tUGF0aCAkUGFuZGFHZW5lcmF0ZWRSb290ICdwYW5kYV9yaWdfcmVwb3J0Lmpzb24nCiAgICAkUGFuZGFGYnggPSBKb2luLVBhdGggJFBhbmRhR2VuZXJhdGVkUm9vdCAndGFjdGljYWxfcmVkX3BhbmRhX3dhbGsuZmJ4JwogICAgJFBhbmRhQmxlbmQgPSBKb2luLVBhdGggJFBhbmRhR2VuZXJhdGVkUm9vdCAndGFjdGljYWxfcmVkX3BhbmRhX3dhbGsuYmxlbmQnCiAgICAkUmlnTG9nID0gSm9pbi1QYXRoICRMb2dzUm9vdCAnYmxlbmRlcl9yaWdfcGFuZGEubG9nJwogICAgJFJlcXVpcmVkUmV1c2VGaWxlcyA9IEAoJFBhbmRhUmlnUmVwb3J0LCAkUGFuZGFGYngsICRQYW5kYUJsZW5kLCAkUmlnTG9nKQogICAgJE1pc3NpbmdSZXVzZUZpbGVzID0gQCgkUmVxdWlyZWRSZXVzZUZpbGVzIHwgV2hlcmUtT2JqZWN0IHsgLW5vdCAoVGVzdC1QYXRoIC1MaXRlcmFsUGF0aCAkXyAtUGF0aFR5cGUgTGVhZikgfSkKICAgIGlmICgkTWlzc2luZ1JldXNlRmlsZXMuQ291bnQgLW5lIDApIHsKICAgICAgICB0aHJvdyAiT3JpZ2luYWwgcGFuZGEgR0xCIGlzIGFic2VudCBhbmQgcHJvdmVuIGdlbmVyYXRlZCByaWcgb3V0cHV0cyBhcmUgaW5jb21wbGV0ZTogJCgkTWlzc2luZ1JldXNlRmlsZXMgLWpvaW4gJyB8ICcpIgogICAgfQogICAgdHJ5IHsKICAgICAgICAkRXhpc3RpbmdSaWcgPSBHZXQtQ29udGVudCAtTGl0ZXJhbFBhdGggJFBhbmRhUmlnUmVwb3J0IC1SYXcgfCBDb252ZXJ0RnJvbS1Kc29uCiAgICB9IGNhdGNoIHsKICAgICAgICB0aHJvdyAiRXhpc3RpbmcgcGFuZGEgcmlnIHJlcG9ydCBpcyBpbnZhbGlkIEpTT046ICRQYW5kYVJpZ1JlcG9ydCA6OiAkXyIKICAgIH0KICAgICRSaWdTb3VyY2VTaGEgPSAoW3N0cmluZ10kRXhpc3RpbmdSaWcuc291cmNlX3NoYTI1NikuVG9Mb3dlckludmFyaWFudCgpCiAgICBpZiAoJEV4aXN0aW5nUmlnLmNsYXNzaWZpY2F0aW9uIC1uZSAnUEFOREFfV0FMS19SSUdfR0VORVJBVEVEJykgeyB0aHJvdyAiVW5leHBlY3RlZCBleGlzdGluZyBwYW5kYSByaWcgY2xhc3NpZmljYXRpb246ICQoJEV4aXN0aW5nUmlnLmNsYXNzaWZpY2F0aW9uKSIgfQogICAgaWYgKCRSaWdTb3VyY2VTaGEgLW5lICRFeHBlY3RlZEZhbGxiYWNrU2hhKSB7IHRocm93ICJFeGlzdGluZyBwYW5kYSByaWcgc291cmNlIGhhc2ggbWlzbWF0Y2g6ICRSaWdTb3VyY2VTaGEiIH0KICAgIGlmIChbaW50NjRdJEV4aXN0aW5nUmlnLnNvdXJjZV9zaXplIC1uZSBbaW50NjRdJEV4cGVjdGVkRmFsbGJhY2tCeXRlcykgeyB0aHJvdyAiRXhpc3RpbmcgcGFuZGEgcmlnIHNvdXJjZSBzaXplIG1pc21hdGNoOiAkKCRFeGlzdGluZ1JpZy5zb3VyY2Vfc2l6ZSkiIH0KICAgIGlmIChbaW50XSRFeGlzdGluZ1JpZy5ib25lX2NvdW50IC1sdCAyNSkgeyB0aHJvdyAiRXhpc3RpbmcgcGFuZGEgcmlnIGhhcyB0b28gZmV3IGJvbmVzOiAkKCRFeGlzdGluZ1JpZy5ib25lX2NvdW50KSIgfQogICAgaWYgKFtpbnRdJEV4aXN0aW5nUmlnLm1lc2hfdmVydGljZXMgLW5lIDQ1NjA5MiAtb3IgW2ludF0kRXhpc3RpbmdSaWcubWVzaF9mYWNlcyAtbmUgNjQ0MzQ4KSB7CiAgICAgICAgdGhyb3cgIkV4aXN0aW5nIHBhbmRhIHJpZyBnZW9tZXRyeSBpZGVudGl0eSBtaXNtYXRjaDogdmVydGljZXM9JCgkRXhpc3RpbmdSaWcubWVzaF92ZXJ0aWNlcykgZmFjZXM9JCgkRXhpc3RpbmdSaWcubWVzaF9mYWNlcykiCiAgICB9CiAgICBpZiAoLW5vdCAkRXhpc3RpbmdSaWcuYW5pbWF0aW9uLmxvb3BpbmcgLW9yIC1ub3QgJEV4aXN0aW5nUmlnLmFuaW1hdGlvbi5pbl9wbGFjZSkgeyB0aHJvdyAnRXhpc3RpbmcgcGFuZGEgYW5pbWF0aW9uIGlzIG5vdCBsb29waW5nIGFuZCBpbi1wbGFjZScgfQogICAgaWYgKFtpbnRdJEV4aXN0aW5nUmlnLmFuaW1hdGlvbi5mcmFtZV9zdGFydCAtbmUgMSAtb3IgW2ludF0kRXhpc3RpbmdSaWcuYW5pbWF0aW9uLmZyYW1lX2VuZCAtbmUgMzMgLW9yIFtpbnRdJEV4aXN0aW5nUmlnLmFuaW1hdGlvbi5mcHMgLW5lIDI0KSB7CiAgICAgICAgdGhyb3cgIkV4aXN0aW5nIHBhbmRhIGFuaW1hdGlvbiB0aW1pbmcgbWlzbWF0Y2g6ICQoJEV4aXN0aW5nUmlnLmFuaW1hdGlvbiB8IENvbnZlcnRUby1Kc29uIC1Db21wcmVzcykiCiAgICB9CiAgICAkUmlnTWF0ZXJpYWxzID0gQCgkRXhpc3RpbmdSaWcubWF0ZXJpYWxzIHwgRm9yRWFjaC1PYmplY3QgeyBbc3RyaW5nXSRfIH0pCiAgICBmb3JlYWNoICgkUmVxdWlyZWRNYXRlcmlhbCBpbiBAKCdSYXN0ZXJBdGxhcycsICdSYXN0ZXJOZXV0cmFsU3ludGhlc2lzJykpIHsKICAgICAgICBpZiAoJFJpZ01hdGVyaWFscyAtbm90Y29udGFpbnMgJFJlcXVpcmVkTWF0ZXJpYWwpIHsgdGhyb3cgIkV4aXN0aW5nIHBhbmRhIHJpZyBpcyBtaXNzaW5nIG1hdGVyaWFsOiAkUmVxdWlyZWRNYXRlcmlhbCIgfQogICAgfQogICAgJFJlcG9ydGVkRmJ4ID0gW0lPLlBhdGhdOjpHZXRGdWxsUGF0aChbc3RyaW5nXSRFeGlzdGluZ1JpZy5vdXRwdXRzLmZieCkKICAgICRSZXBvcnRlZEJsZW5kID0gW0lPLlBhdGhdOjpHZXRGdWxsUGF0aChbc3RyaW5nXSRFeGlzdGluZ1JpZy5vdXRwdXRzLmJsZW5kKQogICAgaWYgKCRSZXBvcnRlZEZieCAtbmUgW0lPLlBhdGhdOjpHZXRGdWxsUGF0aCgkUGFuZGFGYngpKSB7IHRocm93ICJFeGlzdGluZyByaWcgRkJYIHBhdGggbWlzbWF0Y2g6ICRSZXBvcnRlZEZieCIgfQogICAgaWYgKCRSZXBvcnRlZEJsZW5kIC1uZSBbSU8uUGF0aF06OkdldEZ1bGxQYXRoKCRQYW5kYUJsZW5kKSkgeyB0aHJvdyAiRXhpc3RpbmcgcmlnIEJsZW5kIHBhdGggbWlzbWF0Y2g6ICRSZXBvcnRlZEJsZW5kIiB9CiAgICAkRmJ4SXRlbSA9IEdldC1JdGVtIC1MaXRlcmFsUGF0aCAkUGFuZGFGYngKICAgICRCbGVuZEl0ZW0gPSBHZXQtSXRlbSAtTGl0ZXJhbFBhdGggJFBhbmRhQmxlbmQKICAgIGlmIChbaW50NjRdJEZieEl0ZW0uTGVuZ3RoIC1sdCAxMDAwMDAwIC1vciBbaW50NjRdJEJsZW5kSXRlbS5MZW5ndGggLWx0IDEwMDAwMDApIHsKICAgICAgICB0aHJvdyAiRXhpc3RpbmcgcGFuZGEgcmlnIG91dHB1dHMgYXJlIGltcGxhdXNpYmx5IHNtYWxsOiBmYng9JCgkRmJ4SXRlbS5MZW5ndGgpIGJsZW5kPSQoJEJsZW5kSXRlbS5MZW5ndGgpIgogICAgfQogICAgJFJpZ0xvZ1RleHQgPSBHZXQtQ29udGVudCAtTGl0ZXJhbFBhdGggJFJpZ0xvZyAtUmF3CiAgICBpZiAoJFJpZ0xvZ1RleHQgLW5vdG1hdGNoIChbcmVnZXhdOjpFc2NhcGUoJEV4cGVjdGVkRmFsbGJhY2tTaGEpKSAtb3IgJFJpZ0xvZ1RleHQgLW5vdG1hdGNoICdQQU5EQV9XQUxLX1JJR19HRU5FUkFURUQnKSB7CiAgICAgICAgdGhyb3cgJ0V4aXN0aW5nIHBhbmRhIHJpZyBsb2cgZG9lcyBub3QgcHJvdmUgdGhlIGV4cGVjdGVkIHNvdXJjZSBoYXNoIGFuZCBnZW5lcmF0ZWQgY2xhc3NpZmljYXRpb24nCiAgICB9CiAgICAkUGFuZGFTb3VyY2VDbGFzc2lmaWNhdGlvbiA9ICdTVFJVQ1RVUkFMX1JFUEFJUl9URVhUVVJFX0RPV05HUkFERScKICAgICRSZXVzZUV4aXN0aW5nUGFuZGFSaWcgPSAkdHJ1ZQogICAgJFBhbmRhSWRlbnRpdHkgPSBbb3JkZXJlZF1AewogICAgICAgIHNvdXJjZSA9ICRGYWxsYmFja0NhbmRpZGF0ZQogICAgICAgIGNsYXNzaWZpY2F0aW9uID0gJFBhbmRhU291cmNlQ2xhc3NpZmljYXRpb24KICAgICAgICBzaGEyNTYgPSAkRXhwZWN0ZWRGYWxsYmFja1NoYQogICAgICAgIGJ5dGVzID0gJEV4cGVjdGVkRmFsbGJhY2tCeXRlcwogICAgICAgIHNlbGVjdGVkX2F0ID0gKEdldC1EYXRlKS5Ub1VuaXZlcnNhbFRpbWUoKS5Ub1N0cmluZygnbycpCiAgICAgICAgcmV1c2VfbW9kZSA9ICdQUk9WRU5fR0VORVJBVEVEX1JJR19SRVVTRV9BRlRFUl9TT1VSQ0VfTE9TUycKICAgICAgICByaWdfcmVwb3J0ID0gJFBhbmRhUmlnUmVwb3J0CiAgICAgICAgcmlnX2ZieCA9ICRQYW5kYUZieAogICAgICAgIHJpZ19mYnhfc2hhMjU2ID0gR2V0LVNoYTI1NiAkUGFuZGFGYngKICAgICAgICByaWdfZmJ4X2J5dGVzID0gJEZieEl0ZW0uTGVuZ3RoCiAgICAgICAgcmlnX2JsZW5kID0gJFBhbmRhQmxlbmQKICAgICAgICByaWdfYmxlbmRfc2hhMjU2ID0gR2V0LVNoYTI1NiAkUGFuZGFCbGVuZAogICAgICAgIHJpZ19ibGVuZF9ieXRlcyA9ICRCbGVuZEl0ZW0uTGVuZ3RoCiAgICB9Cn0gZWxzZSB7CiAgICAkUGFuZGFJZGVudGl0eSA9IFtvcmRlcmVkXUB7CiAgICAgICAgc291cmNlID0gJFBhbmRhU291cmNlCiAgICAgICAgY2xhc3NpZmljYXRpb24gPSAkUGFuZGFTb3VyY2VDbGFzc2lmaWNhdGlvbgogICAgICAgIHNoYTI1NiA9IEdldC1TaGEyNTYgJFBhbmRhU291cmNlCiAgICAgICAgYnl0ZXMgPSAoR2V0LUl0ZW0gLUxpdGVyYWxQYXRoICRQYW5kYVNvdXJjZSkuTGVuZ3RoCiAgICAgICAgc2VsZWN0ZWRfYXQgPSAoR2V0LURhdGUpLlRvVW5pdmVyc2FsVGltZSgpLlRvU3RyaW5nKCdvJykKICAgICAgICByZXVzZV9tb2RlID0gJ0ZSRVNIX1NPVVJDRScKICAgIH0KfQpXcml0ZS1Kc29uRmlsZSAkUGFuZGFJZGVudGl0eSAoSm9pbi1QYXRoICRPdXRwdXRSb290ICdwYW5kYV9zb3VyY2UuanNvbicpCmlmICgkUmV1c2VFeGlzdGluZ1BhbmRhUmlnKSB7CiAgICBXcml0ZS1Ib3N0ICdQQU5EQV9HRU5FUkFURURfUklHX1JFVVNFPVBST1ZFTicKICAgIFdyaXRlLUhvc3QgIlBBTkRBX1JFVVNFRF9GQlhfU0hBMjU2PSQoJFBhbmRhSWRlbnRpdHkucmlnX2ZieF9zaGEyNTYpIgogICAgV3JpdGUtSG9zdCAiUEFOREFfUkVVU0VEX0JMRU5EX1NIQTI1Nj0kKCRQYW5kYUlkZW50aXR5LnJpZ19ibGVuZF9zaGEyNTYpIgp9'))
$BuildText = $BuildText.Substring(0, $SelectionStart) + $SelectionNew + "`n" + $BuildText.Substring($SelectionLineEnd)
Set-Content -LiteralPath $BuildScript -Value $BuildText -Encoding utf8
Write-Host 'PANDA_PROVEN_RIG_REUSE_PATCH=PROVEN'
Write-Host 'PANDA_SOURCE_SELECTION_BLOCK_PATCHED=1'
Write-Host 'PANDA_RIG_INVOCATION_BLOCK_PATCHED=1'
'@

$PatchedText = $BaselineText.Replace($InsertionMarker, $EvidenceFixes + "`n`n" + $InsertionMarker)
Set-Content -LiteralPath $PatchedBootstrap -Value $PatchedText -Encoding utf8
Write-Host "BOOTSTRAP_BASELINE_COMMIT=$BaselineCommit"
Write-Host "BOOTSTRAP_BASELINE_BLOB=$BaselineBlob"
Write-Host 'JUNGLE_RUN11_RECOVERY_INJECTION=PROVEN'

& $PatchedBootstrap -ExpectedBranch $ExpectedBranch
$WorkerExit = $LASTEXITCODE
if ($WorkerExit -ne 0) { throw "Patched direct worker failed with exit code $WorkerExit" }
