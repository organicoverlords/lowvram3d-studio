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

$BaseCommit = '2932d9b6465dbd0c0532a832bcf359367911a34e'
$BaseBlob = 'f0d035d23a0b3542d51afe9ca58c58f3ba811c0d'
$WrapperPath = 'scripts/procedural_jungle/run-procedural-jungle-local-worker.ps1'
& git merge-base --is-ancestor $BaseCommit HEAD
if ($LASTEXITCODE -ne 0) { throw "Literal-marker repair base is not an ancestor of HEAD: $BaseCommit" }
$ActualBlob = (& git rev-parse "$BaseCommit`:$WrapperPath").Trim()
if ($LASTEXITCODE -ne 0 -or $ActualBlob -ne $BaseBlob) {
    throw "Literal-marker repair base identity mismatch: expected=$BaseBlob actual=$ActualBlob"
}

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-literal-counter-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedWrapper = Join-Path $TempRoot 'run-procedural-jungle-literal-counter.ps1'
$BaseLines = @(& git show "$BaseCommit`:$WrapperPath")
if ($LASTEXITCODE -ne 0 -or $BaseLines.Count -lt 10) { throw 'Could not read the literal-marker runtime wrapper' }
$WrapperText = $BaseLines -join "`n"

$InsertionMarker = '$AssemblyNew ='
$InsertionMatches = [regex]::Matches($WrapperText, [regex]::Escape($InsertionMarker)).Count
if ($InsertionMatches -ne 1) { throw "Could not prove unique FixCode insertion point; matches=$InsertionMatches" }
$Injection = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('JE9sZENvdW50TGluZSA9ICckTG93ZXJQTWF0Y2hlcyA9IFtyZWdleF06Ok1hdGNoZXMoJENhcHR1cmVCbG9jaywgJydcJHBcYicnKS5Db3VudCcKJE5ld0NvdW50TGluZXMgPSBbVGV4dC5FbmNvZGluZ106OlVURjguR2V0U3RyaW5nKFtDb252ZXJ0XTo6RnJvbUJhc2U2NFN0cmluZygnSkV4dmQyVnlVRlJ2YTJWdUlEMGdKeVJ3Sndva1RHOTNaWEpRVFdGMFkyaGxjeUE5SUNna1EyRndkSFZ5WlVKc2IyTnJMa3hsYm1kMGFDQXRJQ1JEWVhCMGRYSmxRbXh2WTJzdVVtVndiR0ZqWlNna1RHOTNaWEpRVkc5clpXNHNJQ2NuS1M1TVpXNW5kR2dwSUM4Z0pFeHZkMlZ5VUZSdmEyVnVMa3hsYm1kMGFBPT0nKSkKJE9sZENvdW50TWF0Y2hlcyA9IFtyZWdleF06Ok1hdGNoZXMoJEZpeENvZGUsIFtyZWdleF06OkVzY2FwZSgkT2xkQ291bnRMaW5lKSkuQ291bnQKaWYgKCRPbGRDb3VudE1hdGNoZXMgLW5lIDEpIHsKICAgIHRocm93ICJDb3VsZCBub3QgcHJvdmUgdW5pcXVlIGxvd2VyY2FzZSBjYXB0dXJlLXBhdHRlcm4gY291bnRlcjsgbWF0Y2hlcz0kT2xkQ291bnRNYXRjaGVzIgp9CiRGaXhDb2RlID0gJEZpeENvZGUuUmVwbGFjZSgkT2xkQ291bnRMaW5lLCAkTmV3Q291bnRMaW5lcykKV3JpdGUtSG9zdCAnSlVOR0xFX0xJVEVSQUxfQ0FQVFVSRV9UT0tFTl9DT1VOVEVSX1BBVENIPVBST1ZFTicK'))
$WrapperText = $WrapperText.Replace($InsertionMarker, $Injection + "`n" + $InsertionMarker)
[IO.File]::WriteAllText($PatchedWrapper, $WrapperText, (New-Object Text.UTF8Encoding($false)))

Write-Host "LITERAL_COUNTER_BASE_COMMIT=$BaseCommit"
Write-Host "LITERAL_COUNTER_BASE_BLOB=$BaseBlob"
Write-Host 'JUNGLE_LITERAL_COUNTER_INJECTION=PROVEN'

& $PatchedWrapper -ExpectedBranch $ExpectedBranch
$WorkerExit = $LASTEXITCODE
if ($WorkerExit -ne 0) { throw "Literal-counter jungle worker failed with exit code $WorkerExit" }
