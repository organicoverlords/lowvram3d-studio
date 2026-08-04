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

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-write-context-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$DiagnosticWrapper = Join-Path $TempRoot 'run-procedural-jungle-write-context.ps1'
$BaseLines = @(& git show "$BaseCommit`:$WrapperPath")
if ($LASTEXITCODE -ne 0 -or $BaseLines.Count -lt 10) { throw 'Could not read the proven runtime repair wrapper' }
$WrapperText = $BaseLines -join "`n"

$AssemblyOld = '$t=$t.Replace($m,$m+"`n"+$x);Set-Content $o $t -Encoding utf8'
$AssemblyMatches = [regex]::Matches($WrapperText, [regex]::Escape($AssemblyOld)).Count
if ($AssemblyMatches -ne 1) { throw "Could not prove unique runtime-payload assembly line; matches=$AssemblyMatches" }
$DiagnosticCode = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('JFJlY292ZXJ5TGluZXMgPSBAKCRSZWNvdmVyeVRleHQgLXNwbGl0ICJgcj9gbiIpCiRDb250ZXh0U3RhcnQgPSA5MAokQ29udGV4dEVuZCA9IFtNYXRoXTo6TWluKDE1NSwgJFJlY292ZXJ5TGluZXMuQ291bnQpCldyaXRlLUhvc3QgIlJVTlRJTUVfV1JJVEVfQ09OVEVYVF9UT1RBTF9MSU5FUz0kKCRSZWNvdmVyeUxpbmVzLkNvdW50KSIKZm9yICgkSW5kZXggPSAkQ29udGV4dFN0YXJ0OyAkSW5kZXggLWxlICRDb250ZXh0RW5kOyAkSW5kZXgrKykgewogICAgV3JpdGUtSG9zdCAiUlVOVElNRV9XUklURV9DT05URVhUPSRJbmRleGA6JCgkUmVjb3ZlcnlMaW5lc1skSW5kZXggLSAxXSkiCn0KdGhyb3cgJ1JVTlRJTUVfV1JJVEVfQ09OVEVYVF9ESUFHTk9TVElDX0NPTVBMRVRFJwo='))
$AssemblyNew = '$t=$t.Replace($m,$m+"`n"+$x+"`n"+$DiagnosticCode);[IO.File]::WriteAllText($o,$t,(New-Object Text.UTF8Encoding($false)))'
$WrapperText = $WrapperText.Replace($AssemblyOld, $AssemblyNew)
[IO.File]::WriteAllText($DiagnosticWrapper, $WrapperText, (New-Object Text.UTF8Encoding($false)))

Write-Host "RUNTIME_DIAGNOSTIC_BASE_COMMIT=$BaseCommit"
Write-Host "RUNTIME_DIAGNOSTIC_BASE_BLOB=$BaseBlob"
Write-Host 'RUNTIME_WRITE_CONTEXT_DIAGNOSTIC_INJECTION=PROVEN'

& $DiagnosticWrapper -ExpectedBranch $ExpectedBranch
$WorkerExit = $LASTEXITCODE
if ($WorkerExit -ne 0) { throw "Runtime write-context diagnostic exited with code $WorkerExit" }
throw 'Runtime write-context diagnostic unexpectedly completed without the fail-fast marker'
