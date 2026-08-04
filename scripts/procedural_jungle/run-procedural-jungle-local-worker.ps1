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

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-encoding-diagnostic-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$DiagnosticWrapper = Join-Path $TempRoot 'run-procedural-jungle-encoding-diagnostic.ps1'
$BaseLines = @(& git show "$BaseCommit`:$WrapperPath")
if ($LASTEXITCODE -ne 0 -or $BaseLines.Count -lt 10) { throw 'Could not read the proven runtime repair wrapper' }
$WrapperText = $BaseLines -join "`n"

$AssemblyOld = '$t=$t.Replace($m,$m+"`n"+$x);Set-Content $o $t -Encoding utf8'
$AssemblyMatches = [regex]::Matches($WrapperText, [regex]::Escape($AssemblyOld)).Count
if ($AssemblyMatches -ne 1) { throw "Could not prove unique runtime-payload assembly line; matches=$AssemblyMatches" }
$DiagnosticCode = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('JEVuY29kaW5nRGlhZ25vc3RpYyA9IE5ldy1PYmplY3QgU3lzdGVtLkNvbGxlY3Rpb25zLkdlbmVyaWMuTGlzdFtzdHJpbmddCiRFbmNvZGluZ1ZhcmlhYmxlc0ZvdW5kID0gMApmb3JlYWNoICgkVmFyaWFibGVOYW1lIGluIEAoJ1JlY292ZXJ5VGV4dCcsICdFbnRyeVRleHQnKSkgewogICAgJFZhcmlhYmxlID0gR2V0LVZhcmlhYmxlIC1OYW1lICRWYXJpYWJsZU5hbWUgLUVycm9yQWN0aW9uIFNpbGVudGx5Q29udGludWUKICAgIGlmICgkbnVsbCAtZXEgJFZhcmlhYmxlIC1vciAtbm90ICgkVmFyaWFibGUuVmFsdWUgLWlzIFtzdHJpbmddKSkgeyBjb250aW51ZSB9CiAgICAkRW5jb2RpbmdWYXJpYWJsZXNGb3VuZCsrCiAgICAkTGluZU51bWJlciA9IDAKICAgIGZvcmVhY2ggKCRMaW5lIGluICgkVmFyaWFibGUuVmFsdWUgLXNwbGl0ICJgcj9gbiIpKSB7CiAgICAgICAgJExpbmVOdW1iZXIrKwogICAgICAgIGlmICgkTGluZSAtbWF0Y2ggJyg/aSkoLUVuY29kaW5nfHV0ZjhOb0JPTXxXcml0ZUFsbFRleHR8U2V0LUNvbnRlbnR8T3V0LUZpbGV8QWRkLUNvbnRlbnR8VGVlLU9iamVjdHxFeHBvcnQtQ3N2KScpIHsKICAgICAgICAgICAgJEVuY29kaW5nRGlhZ25vc3RpYy5BZGQoIiRWYXJpYWJsZU5hbWVgOiRMaW5lTnVtYmVyYDokTGluZSIpCiAgICAgICAgfQogICAgfQp9CmlmICgkRW5jb2RpbmdWYXJpYWJsZXNGb3VuZCAtbHQgMSkgewogICAgdGhyb3cgJ1J1bnRpbWUgZW5jb2RpbmcgZGlhZ25vc3RpYyBmb3VuZCBuZWl0aGVyIFJlY292ZXJ5VGV4dCBub3IgRW50cnlUZXh0Jwp9CldyaXRlLUhvc3QgIlJVTlRJTUVfRU5DT0RJTkdfVkFSSUFCTEVTPSRFbmNvZGluZ1ZhcmlhYmxlc0ZvdW5kIgpXcml0ZS1Ib3N0ICJSVU5USU1FX0VOQ09ESU5HX0xJTkVfQ09VTlQ9JCgkRW5jb2RpbmdEaWFnbm9zdGljLkNvdW50KSIKZm9yZWFjaCAoJERpYWdub3N0aWNMaW5lIGluICRFbmNvZGluZ0RpYWdub3N0aWMpIHsKICAgIFdyaXRlLUhvc3QgIlJVTlRJTUVfRU5DT0RJTkdfTElORT0kRGlhZ25vc3RpY0xpbmUiCn0KdGhyb3cgJ1JVTlRJTUVfRU5DT0RJTkdfRElBR05PU1RJQ19DT01QTEVURScK'))
$AssemblyNew = '$t=$t.Replace($m,$m+"`n"+$x+"`n"+$DiagnosticCode);[IO.File]::WriteAllText($o,$t,(New-Object Text.UTF8Encoding($false)))'
$WrapperText = $WrapperText.Replace($AssemblyOld, $AssemblyNew)
[IO.File]::WriteAllText($DiagnosticWrapper, $WrapperText, (New-Object Text.UTF8Encoding($false)))

Write-Host "RUNTIME_DIAGNOSTIC_BASE_COMMIT=$BaseCommit"
Write-Host "RUNTIME_DIAGNOSTIC_BASE_BLOB=$BaseBlob"
Write-Host 'RUNTIME_ENCODING_DIAGNOSTIC_INJECTION=PROVEN'

& $DiagnosticWrapper -ExpectedBranch $ExpectedBranch
$WorkerExit = $LASTEXITCODE
if ($WorkerExit -ne 0) { throw "Runtime encoding diagnostic exited with code $WorkerExit" }
throw 'Runtime encoding diagnostic unexpectedly completed without the fail-fast marker'
