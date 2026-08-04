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

$BaseCommit = '84c2259677c3211c5dbe55da814677c40d635d10'
$BaseBlob = '5301f2769e455318b6ae396c37692a5d8778b3aa'
$WrapperPath = 'scripts/procedural_jungle/run-procedural-jungle-local-worker.ps1'
& git merge-base --is-ancestor $BaseCommit HEAD
if ($LASTEXITCODE -ne 0) { throw "Runtime repair base is not an ancestor of HEAD: $BaseCommit" }
$ActualBlob = (& git rev-parse "$BaseCommit`:$WrapperPath").Trim()
if ($LASTEXITCODE -ne 0 -or $ActualBlob -ne $BaseBlob) {
    throw "Runtime repair base identity mismatch: expected=$BaseBlob actual=$ActualBlob"
}

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-capture-path-fix-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedWrapper = Join-Path $TempRoot 'run-procedural-jungle-capture-path-fix.ps1'
$BaseLines = @(& git show "$BaseCommit`:$WrapperPath")
if ($LASTEXITCODE -ne 0 -or $BaseLines.Count -lt 10) { throw 'Could not read the proven runtime repair wrapper' }
$WrapperText = $BaseLines -join "`n"

$AssemblyOld = '$t=$t.Replace($m,$m+"`n"+$x);Set-Content $o $t -Encoding utf8'
$AssemblyMatches = [regex]::Matches($WrapperText, [regex]::Escape($AssemblyOld)).Count
if ($AssemblyMatches -ne 1) { throw "Could not prove unique runtime-payload assembly line; matches=$AssemblyMatches" }
$FixCode = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('JENhcHR1cmVCbG9ja1N0YXJ0TWFya2VyID0gJyRQPUpvaW4tUGF0aCAkU291cmNlUm9vdCAnJ3Byb2plY3RfdGVtcGxhdGVcU291cmNlXFByb2NlZHVyYWxKdW5nbGU1OFxKdW5nbGVQcm9vZkRpcmVjdG9yLmNwcCcnJwokQ2FwdHVyZUJsb2NrRW5kTWFya2VyID0gJ1dyaXRlLUhvc3QgJydKVU5HTEVfREVMQVlFRF9WSUVXVEFSR0VUX0NBUFRVUkVfUEFUQ0g9UFJPVkVOJycnCiRDYXB0dXJlQmxvY2tTdGFydCA9ICRSZWNvdmVyeVRleHQuSW5kZXhPZigkQ2FwdHVyZUJsb2NrU3RhcnRNYXJrZXIsIFtTdHJpbmdDb21wYXJpc29uXTo6T3JkaW5hbCkKJENhcHR1cmVCbG9ja0VuZCA9ICRSZWNvdmVyeVRleHQuSW5kZXhPZigkQ2FwdHVyZUJsb2NrRW5kTWFya2VyLCBbU3RyaW5nQ29tcGFyaXNvbl06Ok9yZGluYWwpCmlmICgkQ2FwdHVyZUJsb2NrU3RhcnQgLWx0IDAgLW9yICRDYXB0dXJlQmxvY2tFbmQgLWxlICRDYXB0dXJlQmxvY2tTdGFydCkgewogICAgdGhyb3cgIkNvdWxkIG5vdCBsb2NhdGUgSnVuZ2xlUHJvb2ZEaXJlY3RvciBjYXB0dXJlIHBhdGNoIGJsb2NrOiBzdGFydD0kQ2FwdHVyZUJsb2NrU3RhcnQgZW5kPSRDYXB0dXJlQmxvY2tFbmQiCn0KJENhcHR1cmVCbG9jayA9ICRSZWNvdmVyeVRleHQuU3Vic3RyaW5nKCRDYXB0dXJlQmxvY2tTdGFydCwgJENhcHR1cmVCbG9ja0VuZCAtICRDYXB0dXJlQmxvY2tTdGFydCkKJExvd2VyUE1hdGNoZXMgPSBbcmVnZXhdOjpNYXRjaGVzKCRDYXB0dXJlQmxvY2ssICdcJHBeYicpLkNvdW50CmlmICgkTG93ZXJQTWF0Y2hlcyAtbmUgMykgewogICAgdGhyb3cgIlVuZXhwZWN0ZWQgbG93ZXJjYXNlIGNhcHR1cmUtcGF0dGVybiB2YXJpYWJsZSBjb3VudDogJExvd2VyUE1hdGNoZXMiCn0KJEZpeGVkQ2FwdHVyZUJsb2NrID0gJENhcHR1cmVCbG9jay5SZXBsYWNlKCckcCcsICckQ2FwdHVyZVBhdHRlcm4nKQokUmVjb3ZlcnlUZXh0ID0gJFJlY292ZXJ5VGV4dC5TdWJzdHJpbmcoMCwgJENhcHR1cmVCbG9ja1N0YXJ0KSArICRGaXhlZENhcHR1cmVCbG9jayArICRSZWNvdmVyeVRleHQuU3Vic3RyaW5nKCRDYXB0dXJlQmxvY2tFbmQpCldyaXRlLUhvc3QgJ0pVTkdMRV9DQVB UVVJFX1BBVFRFUk5fUEFUSF9DT0xMSVNJT05fUEFUQ0g9UFJPVkVOJwpXcml0ZS1Ib3N0ICJKVU5HTEVfQ0FQVFVSRV9QQVRURVJOX1JFRkVSRU5DRVNfUkVOQU1FRD0kTG93ZXJQTWF0Y2hlcyIKCiRQczUxV3JpdGVQYXR0ZXJuID0gJyg/bSleKD88aW5kZW50PlsgXHRdKilTZXQtQ29udGVudCAtTGl0ZXJhbFBhdGggKD88cGF0aD5cJFtBLVphLXpfXVtBLVphLXowLTlfLl0qKSAtVmFsdWUgKD88dmFsdWU+XCRbQS1aYS16X11bQS1aYS16MC05Xy5dKikgLUVuY29kaW5nIHV0ZjhbIFx0XSokJwokUHM1MVdyaXRlTWF0Y2hlcyA9IEAoW3JlZ2V4XTo6TWF0Y2hlcygkUmVjb3ZlcnlUZXh0LCAkUHM1MVdyaXRlUGF0dGVybikpCmlmICgkUHM1MVdyaXRlTWF0Y2hlcy5Db3VudCAtbmUgNykgewogICAgJE1hdGNoZWRMaW5lcyA9ICgkUHM1MVdyaXRlTWF0Y2hlcyB8IEZvckVhY2gtT2JqZWN0IHsgJF8uVmFsdWUuVHJpbSgpIH0pIC1qb2luICcgfCAnCiAgICB0aHJvdyAiVW5leHBlY3RlZCBQb3dlclNoZWxsIDUuMSBTZXQtQ29udGVudCBjb21wYXRpYmlsaXR5IHRhcmdldCBjb3VudDogJCgkUHM1MVdyaXRlTWF0Y2hlcy5Db3VudCk7IGxpbmVzPSRNYXRjaGVkTGluZXMiCn0KJFBzNTFXcml0ZUV2YWx1YXRvciA9IFtUZXh0LlJlZ3VsYXJFeHByZXNzaW9ucy5NYXRjaEV2YWx1YXRvcl17CiAgICBwYXJhbShbVGV4dC5SZWd1bGFyRXhwcmVzc2lvbnMuTWF0Y2hdJE1hdGNoKQogICAgJEluZGVudCA9ICRNYXRjaC5Hcm91cHNbJ2luZGVudCddLlZhbHVlCiAgICAkUGF0aEV4cHJlc3Npb24gPSAkTWF0Y2guR3JvdXBzWydwYXRoJ10uVmFsdWUKICAgICRWYWx1ZUV4cHJlc3Npb24gPSAkTWF0Y2guR3JvdXBzWyd2YWx1ZSddLlZhbHVlCiAgICByZXR1cm4gJEluZGVudCArICdbSU8uRmlsZV06OldyaXRlQWxsVGV4dChbc3RyaW5nXSgnICsgJFBhdGhFeHByZXNzaW9uICsgJyksIFtzdHJpbmddKCcgKyAkVmFsdWVFeHByZXNzaW9uICsgJyksIChOZXctT2JqZWN0IFRleHQuVVRGOEVuY29kaW5nKCRmYWxzZSkpKScKfQokUmVjb3ZlcnlUZXh0ID0gW3JlZ2V4XTo6UmVwbGFjZSgkUmVjb3ZlcnlUZXh0LCAkUHM1MVdyaXRlUGF0dGVybiwgJFBzNTFXcml0ZUV2YWx1YXRvcikKJFBzNTFSZW1haW5pbmcgPSBAKFtyZWdleF06Ok1hdGNoZXMoJFJlY292ZXJ5VGV4dCwgJFBzNTFXcml0ZVBhdHRlcm4pKQppZiAoJFBzNTFSZW1haW5pbmcuQ291bnQgLW5lIDApIHsKICAgIHRocm93ICJQb3dlclNoZWxsIDUuMSBTZXQtQ29udGVudCBjb21wYXRpYmlsaXR5IGNhbGxzIHJlbWFpbjogJCgkUHM1MVJlbWFpbmluZy5Db3VudCkiCn0KV3JpdGUtSG9zdCAnUE9XRVJTSEVMTDUxX0dFTkVSQVRFRF9XUklURVNfUEFUQ0g9UFJPVkVOJwpXcml0ZS1Ib3N0ICJQT1dFUlNIRUxMNTEfR0VORVJBVEVEX1dSSVRFU19QQVRDSEVEPSQoJFBzNTFXcml0ZU1hdGNoZXMuQ291bnQpIgo='))
$AssemblyNew = '$t=$t.Replace($m,$m+"`n"+$x+"`n"+$FixCode);[IO.File]::WriteAllText($o,$t,(New-Object Text.UTF8Encoding($false)))'
$WrapperText = $WrapperText.Replace($AssemblyOld, $AssemblyNew)
[IO.File]::WriteAllText($PatchedWrapper, $WrapperText, (New-Object Text.UTF8Encoding($false)))

$OutputRoot = 'C:\AI\ProceduralJungle\20260804'
$ProofRoot = Join-Path $OutputRoot 'proof'
if (Test-Path -LiteralPath $ProofRoot) {
    Get-ChildItem -LiteralPath $ProofRoot -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'capture_*.png' -or $_.Name -in @('gameplay_runtime_proof.json', 'contact_sheet.png', 'visual_capture_audit.json') } |
        Remove-Item -Force
}
foreach ($StalePath in @(
    (Join-Path $OutputRoot 'acceptance.json'),
    (Join-Path $RepoRoot 'evidence\latest-procedural-jungle\acceptance.json'),
    (Join-Path $RepoRoot 'evidence\latest-procedural-jungle\workflow_receipt.json')
)) {
    if (Test-Path -LiteralPath $StalePath) { Remove-Item -LiteralPath $StalePath -Force }
}

Write-Host "RUNTIME_REPAIR_BASE_COMMIT=$BaseCommit"
Write-Host "RUNTIME_REPAIR_BASE_BLOB=$BaseBlob"
Write-Host 'JUNGLE_CAPTURE_PATH_COLLISION_FIX_INJECTION=PROVEN'
Write-Host 'STALE_JUNGLE_PROOF_CLEARED=PROVEN'

& $PatchedWrapper -ExpectedBranch $ExpectedBranch
$WorkerExit = $LASTEXITCODE
if ($WorkerExit -ne 0) { throw "Capture-path-fixed jungle worker failed with exit code $WorkerExit" }
