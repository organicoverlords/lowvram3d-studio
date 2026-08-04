[CmdletBinding()]
param(
    [string]$ExpectedBranch = 'feature/procedural-jungle-playable-20260804'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$RepoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $RepoRoot) { throw 'Not inside a Git repository' }
Set-Location -LiteralPath $RepoRoot
$Remote = (git config --get remote.origin.url).Trim()
$Branch = (git branch --show-current).Trim()
$Head = (git rev-parse HEAD).Trim()
$Status = @(git status --short)
if ($Remote -notmatch 'organicoverlords/lowvram3d-studio(\.git)?$') { throw "Repository mismatch: $Remote" }
if ($Branch -ne $ExpectedBranch) { throw "Branch mismatch: $Branch" }
if ($Status.Count -ne 0) { throw "Repository is dirty before V3 Python-guard wrapper: $($Status -join '; ')" }
git fetch origin $Branch --quiet
$RemoteHead = (git rev-parse "origin/$Branch").Trim()
if ($Head -ne $RemoteHead) { throw "Checkout head differs from remote: $Head vs $RemoteHead" }

$SourceCommit = '498e600c5873377fd0f55b8179b86ec62780b32c'
$RelativePath = 'scripts/procedural_jungle/run-procedural-jungle-local-worker.ps1'
$SourceLines = @(git show "${SourceCommit}:${RelativePath}")
if ($LASTEXITCODE -ne 0 -or $SourceLines.Count -lt 150) {
    throw "Unable to recover protected-coexistence V3 wrapper from $SourceCommit"
}
$SourceText = ($SourceLines -join "`n") + "`n"
$Anchor = '$OwnedProjectRoot = ''C:\Users\Lauri\Desktop\ProceduralJungle58'''
$AnchorCount = [regex]::Matches($SourceText, [regex]::Escape($Anchor)).Count
if ($AnchorCount -ne 1) { throw "Protected-coexistence preflight anchor count is not exactly one: $AnchorCount" }

$Insertion = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('JERlY29kZWRCdWlsZFBhdGggPSBKb2luLVBhdGggJEV4dHJhY3RSb290ICdzY3JpcHRzXHByb2NlZHVyYWxfanVuZ2xlXGJ1aWxkLXByb2NlZHVyYWwtanVuZ2xlLnBzMScKJERlY29kZWRHdWFyZFBhdGNoID0gSm9pbi1QYXRoICRUZW1wUm9vdCAncGF0Y2gtZGVjb2RlZC1idWlsZC11bnJlYWwtZ3VhcmQucHknCiREZWNvZGVkR3VhcmRQYXRjaENvZGUgPSBbVGV4dC5FbmNvZGluZ106OlVURjguR2V0U3RyaW5nKFtDb252ZXJ0XTo6RnJvbUJhc2U2NFN0cmluZygnWm5KdmJTQmZYMloxZEhWeVpWOWZJR2x0Y0c5eWRDQmhibTVoZEdWdWRITUtDbWx0Y0c5eWRDQmhjbWR3WVhKelpRcG1jbTl0SUhCaGRHaHNhV0lnYVcxd2IzSjBJQXBLUmZWSFVVRlNSQ0E5SUNJaUpFVjRhWE4wYVc1blZXNXlaV0ZzSUQwZ1FDaEhaWFF0VUhKdlkyVnpjeUJWYm5KbFlXeEZaxEdhRzlyTENCVmJuSmxZV3hGWkdsMGIzSXRRMjFrSUMxRmNuSnZja0ZqZEdsdmJpQlRhV3hsYm5Sc2VVTnZiblJwYm5WbEtRcHBaaUFvSkVWNGRqTnBhR2x1WjFWdWNtVmhiQzVEYjNWdWRDQXRaM1FnTUNrZ2V3b2dJQ0FnZEdoeWIzY2dJbFZ1Y21WaGJDQkZaR2wwYjNJZ2FYTWdZV3h5WldGa2VTQnlkVzV1YVc1bk95QnlaV1oxYzJsdVp5QjBieUJwYm5SbGNtWmxjbVVnZDJsMGFDQlFTVVE2Y3lrNklDUW9KRWs0YVhOMGFXNW5WVzV5WldGc0xrbGtJQzFxYjJsdUlDY3NKeWt1TEZ4dUlpd0tJQ0FnSUNJaUtRcDlDaUlpSWdwT1JVVkVURVVnUFNBaVZXNXlaV0ZzSUVWa2FYUnZjaUJwY3lCaGJISmxZV1I1SUhKMWJtNXBibWNnT3lCeVpXWjFjMmx1WnlCMGJ5QnBiblJsY21abGNtVWdkMmwwYUNCUVNVUW9jeWs2SWdwU1JVVk1RVU5GVFVsT1VDQTlJQ0pYY21sMFpTMUliM04wSUNkS1ZVNUhURVZmUWxWSlRGUmZTVTVVUlZKT1FVeGZWVTVTUlVGTVgwZFZRVkpFWDBSRlRFVkhRVlJGUkQxUVVrOVdSVTRuWDF4dUlpd0tDbVJsWmlCdFlXbHVLQ2tnTFQ0Z1RtOXVaVG9LSUNBZ2NHRnljMlZ5SUQwZ1lYSm5jR0Z5YzJWeUxsaHdZWEp6WlhKa1p5Z3BDaUFnSUNCd1lYUm9JRDBnVUdGMGFDaGhjbWR6TG5CaGRHZ3BDaUFnSUNCelpXeG1JRDBnY0dGMGFDNXlaV04xYkdWbEtDaGhjbWR6TG5CaGRHZ3BDaUFnSUNCelpXeG1JRDBnY0dGMGFDaGhjbWR6TG5CaGRHZ3BDaUFnSUNCemRXMXRZWEo1SUQwZ2NHRjBhQzV5WldGa1gzUmxlSFFvWlc1amIyUnBibWc5SW5WMFppMDRJaWt1Y21Wd2JHRmpaU2dpWEhKdVhHMXVJaXdnSWx4dUlpaz0nKSkKW0lPLkZpbGVdOjpXcml0ZUFsbFRleHQoJERlY29kZWRHdWFyZFBhdGNoLCAkRGVjb2RlZEd1YXJkUGF0Y2hDb2RlLCAoTmV3LU9iamVjdCBUZXh0LlVURjhFbmNvZGluZygkZmFsc2UpKSkKJiAkUHl0aG9uICREZWNvZGVkR3VhcmRQYXRjaCAtLXBhdGggJERlY29kZWRCdWlsZFBhdGgKaWYgKCRMQVNU RVhJVENPREUgLW5lIDApIHsgdGhyb3cgIkRlY29kZWQgYnVpbGQtc2NyaXB0IFVucmVhbCBndWFyZCBwYXRjaCBmYWlsZWQgd2l0aCBleGl0IGNvZGUgJExBU1RFWElUQ09ERSIgfQokRGVjb2RlZEJ1aWxkVGV4dCA9IEdldC1Db250ZW50IC1MaXRlcmFsUGF0aCAkRGVjb2RlZEJ1aWxkUGF0aCAtUmF3CmlmICgkRGVjb2RlZEJ1aWxkVGV4dCAtbm90bWF0Y2ggJ0pVTkdMRV9CVUlMRF9JTlRFUk5BTF9VTlJFQUxfR1VBUkRfREVMRUdBVEVEPVBST1ZFTicpIHsKICAgIHRocm93ICdEZWNvZGVkIGJ1aWxkLXNjcmlwdCBkZWxlZ2F0aW9uIG1hcmtlciBpcyBtaXNzaW5nIGFmdGVyIFB5dGhvbiBwYXRjaCcKfQpXcml0ZS1Ib3N0ICdKVU5HTEVfREVDT0RFRF9CVUlMRF9HVUFSRF9QWVRIT05fUEFUQ0g9UFJPVkVOJwoK'))
$PatchedSource = $SourceText.Replace($Anchor, $Insertion.TrimEnd() + "`n" + $Anchor)
if ($PatchedSource -notmatch 'JUNGLE_DECODED_BUILD_GUARD_PYTHON_PATCH=PROVEN') {
    throw 'Python decoded-guard patch insertion marker is missing'
}

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-v3-python-guard-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedWrapper = Join-Path $TempRoot 'run-procedural-jungle-v3-python-guard.ps1'
[IO.File]::WriteAllText($PatchedWrapper, $PatchedSource, (New-Object Text.UTF8Encoding($false)))

Write-Host "JUNGLE_V3_COEXISTENCE_WRAPPER_SOURCE=$SourceCommit"
Write-Host 'JUNGLE_V3_DECODED_BUILD_PYTHON_PATCH_INJECTION=PROVEN'
& powershell -NoProfile -ExecutionPolicy Bypass -File $PatchedWrapper -ExpectedBranch $ExpectedBranch
if ($LASTEXITCODE -ne 0) { throw "Python-guard-patched V3 wrapper failed with exit code $LASTEXITCODE" }
