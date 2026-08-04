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

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-ps51-generated-writes-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedWrapper = Join-Path $TempRoot 'run-procedural-jungle-ps51-generated-writes.ps1'
$BaseLines = @(& git show "$BaseCommit`:$WrapperPath")
if ($LASTEXITCODE -ne 0 -or $BaseLines.Count -lt 10) { throw 'Could not read the proven runtime repair wrapper' }
$WrapperText = $BaseLines -join "`n"

$AssemblyOld = '$t=$t.Replace($m,$m+"`n"+$x);Set-Content $o $t -Encoding utf8'
$AssemblyMatches = [regex]::Matches($WrapperText, [regex]::Escape($AssemblyOld)).Count
if ($AssemblyMatches -ne 1) { throw "Could not prove unique runtime-payload assembly line; matches=$AssemblyMatches" }
$CompatibilityCode = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('JFBzNTFXcml0ZVBhdHRlcm4gPSAnKD9tKV4oPzxpbmRlbnQ+WyBcdF0qKVNldC1Db250ZW50IC1MaXRlcmFsUGF0aCAoPzxwYXRoPlwkW0EtWmEtel9dW0EtWmEtejAtOV8uXSopIC1WYWx1ZSAoPzx2YWx1ZT5cJFtBLVphLXpfXVtBLVphLXowLTlfLl0qKSAtRW5jb2RpbmcgdXRmOFsgXHRdKiQnCiRQczUxV3JpdGVNYXRjaGVzID0gQChbcmVnZXhdOjpNYXRjaGVzKCRSZWNvdmVyeVRleHQsICRQczUxV3JpdGVQYXR0ZXJuKSkKaWYgKCRQczUxV3JpdGVNYXRjaGVzLkNvdW50IC1uZSA3KSB7CiAgICAkTWF0Y2hlZExpbmVzID0gKCRQczUxV3JpdGVNYXRjaGVzIHwgRm9yRWFjaC1PYmplY3QgeyAkXy5WYWx1ZS5UcmltKCkgfSkgLWpvaW4gJyB8ICcKICAgIHRocm93ICJVbmV4cGVjdGVkIFBvd2VyU2hlbGwgNS4xIFNldC1Db250ZW50IGNvbXBhdGliaWxpdHkgdGFyZ2V0IGNvdW50OiAkKCRQczUxV3JpdGVNYXRjaGVzLkNvdW50KTsgbGluZXM9JE1hdGNoZWRMaW5lcyIKfQokUHM1MVdyaXRlRXZhbHVhdG9yID0gW1RleHQuUmVndWxhckV4cHJlc3Npb25zLk1hdGNoRXZhbHVhdG9yXXsKICAgIHBhcmFtKFtUZXh0LlJlZ3VsYXJFeHByZXNzaW9ucy5NYXRjaF0kTWF0Y2gpCiAgICAkSW5kZW50ID0gJE1hdGNoLkdyb3Vwc1snaW5kZW50J10uVmFsdWUKICAgICRQYXRoRXhwcmVzc2lvbiA9ICRNYXRjaC5Hcm91cHNbJ3BhdGgnXS5WYWx1ZQogICAgJFZhbHVlRXhwcmVzc2lvbiA9ICRNYXRjaC5Hcm91cHNbJ3ZhbHVlJ10uVmFsdWUKICAgIHJldHVybiAkSW5kZW50ICsgJ1tJTy5GaWxlXTo6V3JpdGVBbGxUZXh0KFtzdHJpbmddKCcgKyAkUGF0aEV4cHJlc3Npb24gKyAnKSwgW3N0cmluZ10oJyArICRWYWx1ZUV4cHJlc3Npb24gKyAnKSwgKE5ldy1PYmplY3QgVGV4dC5VVEY4RW5jb2RpbmcoJGZhbHNlKSkpJwp9CiRSZWNvdmVyeVRleHQgPSBbcmVnZXhdOjpSZXBsYWNlKCRSZWNvdmVyeVRleHQsICRQczUxV3JpdGVQYXR0ZXJuLCAkUHM1MVdyaXRlRXZhbHVhdG9yKQokUHM1MVJlbWFpbmluZyA9IEAoW3JlZ2V4XTo6TWF0Y2hlcygkUmVjb3ZlcnlUZXh0LCAkUHM1MVdyaXRlUGF0dGVybikpCmlmICgkUHM1MVJlbWFpbmluZy5Db3VudCAtbmUgMCkgewogICAgdGhyb3cgIlBvd2VyU2hlbGwgNS4xIFNldC1Db250ZW50IGNvbXBhdGliaWxpdHkgY2FsbHMgcmVtYWluOiAkKCRQczUxUmVtYWluaW5nLkNvdW50KSIKfQpXcml0ZS1Ib3N0ICdQT1dFUlNIRUxMNTEfR0VORVJBVEVEX1dSSVRFU19QQVRDSD1QUk9WRU4nCldyaXRlLUhvc3QgIlBPV0VSU0hFTEw1MV9HRU5FUkFURURfV1JJVEVTX1BBVENIRUQ9JCgkUHM1MVdyaXRlTWF0Y2hlcy5Db3VudCkiCg=='))
$AssemblyNew = '$t=$t.Replace($m,$m+"`n"+$x+"`n"+$CompatibilityCode);[IO.File]::WriteAllText($o,$t,(New-Object Text.UTF8Encoding($false)))'
$WrapperText = $WrapperText.Replace($AssemblyOld, $AssemblyNew)
[IO.File]::WriteAllText($PatchedWrapper, $WrapperText, (New-Object Text.UTF8Encoding($false)))

# Never accept or upload stale visual/runtime proof from an earlier attempt.
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
Write-Host 'POWERSHELL51_GENERATED_WRITE_INJECTION=PROVEN'
Write-Host 'STALE_JUNGLE_PROOF_CLEARED=PROVEN'

& $PatchedWrapper -ExpectedBranch $ExpectedBranch
$WorkerExit = $LASTEXITCODE
if ($WorkerExit -ne 0) { throw "PowerShell-5-compatible jungle worker failed with exit code $WorkerExit" }
