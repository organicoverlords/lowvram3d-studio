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

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-counter-assignment-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedWrapper = Join-Path $TempRoot 'run-procedural-jungle-counter-assignment.ps1'
$BaseLines = @(& git show "$BaseCommit`:$WrapperPath")
if ($LASTEXITCODE -ne 0 -or $BaseLines.Count -lt 10) { throw 'Could not read the literal-marker runtime wrapper' }
$WrapperText = $BaseLines -join "`n"

$InsertionMarker = '$AssemblyNew ='
$InsertionMatches = [regex]::Matches($WrapperText, [regex]::Escape($InsertionMarker)).Count
if ($InsertionMatches -ne 1) { throw "Could not prove unique FixCode insertion point; matches=$InsertionMatches" }
$Injection = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('JENvdW50ZXJQYXR0ZXJuID0gJyg/bSleXCRMb3dlclBNYXRjaGVzWyBcdF0qPVteXHJcbl0qJCcKJENvdW50ZXJNYXRjaGVzID0gW3JlZ2V4XTo6TWF0Y2hlcygkRml4Q29kZSwgJENvdW50ZXJQYXR0ZXJuKS5Db3VudAppZiAoJENvdW50ZXJNYXRjaGVzIC1uZSAxKSB7CiAgICB0aHJvdyAiQ291bGQgbm90IHByb3ZlIHVuaXF1ZSBsb3dlcmNhc2UgY2FwdHVyZS1wYXR0ZXJuIGNvdW50ZXIgYXNzaWdubWVudDsgbWF0Y2hlcz0kQ291bnRlck1hdGNoZXMiCn0KJE5ld0NvdW50ZXJMaW5lcyA9IFtUZXh0LkVuY29kaW5nXTo6VVRGOC5HZXRTdHJpbmcoW0NvbnZlcnRdOjpGcm9tQmFzZTY0U3RyaW5nKCdKRXh2ZDJWeVVGUnZhMlZ1SUQwZ0p5UndKd29rVEc5M1pYSlFUV0YwWTJobGN5QTlJQ2drUTJGd2RIVnlaVUpzYjJOckxreGxibWQwYUNBdElDUkRZWEIwZFhKbFFteHZZMnN1VW1Wd2JHRmpaU2drVEc5M1pYSlFWRzlyWlc0c0lDY25LUzVNWlc1bmRHZ3BJQzhnSkV4dmQyVnlVRlJ2YTJWdUxreGxibWQwYUE9PScpKQokRml4Q29kZSA9IFtyZWdleF06OlJlcGxhY2UoJEZpeENvZGUsICRDb3VudGVyUGF0dGVybiwgW1RleHQuUmVndWxhckV4cHJlc3Npb25zLk1hdGNoRXZhbHVhdG9yXXsgcGFyYW0oJE1hdGNoKSAkTmV3Q291bnRlckxpbmVzIH0sIDEpCldyaXRlLUhvc3QgJ0pVTkdMRV9MSVRFUkFMX0NBUFRVUkVfQ09VTlRFUl9BU1NJR05NRU5UX1BBVENIPVBST1ZFTicK'))
$WrapperText = $WrapperText.Replace($InsertionMarker, $Injection + "`n" + $InsertionMarker)
[IO.File]::WriteAllText($PatchedWrapper, $WrapperText, (New-Object Text.UTF8Encoding($false)))

Write-Host "COUNTER_ASSIGNMENT_BASE_COMMIT=$BaseCommit"
Write-Host "COUNTER_ASSIGNMENT_BASE_BLOB=$BaseBlob"
Write-Host 'JUNGLE_COUNTER_ASSIGNMENT_INJECTION=PROVEN'

& $PatchedWrapper -ExpectedBranch $ExpectedBranch
$WorkerExit = $LASTEXITCODE
if ($WorkerExit -ne 0) { throw "Counter-assignment jungle worker failed with exit code $WorkerExit" }
