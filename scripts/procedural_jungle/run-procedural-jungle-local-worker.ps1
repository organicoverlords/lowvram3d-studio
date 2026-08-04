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
if ($Status.Count -ne 0) { throw "Repository is dirty before V3 internal-guard wrapper: $($Status -join '; ')" }
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
$Anchor = '$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-v3-coexist-$Head"'
$AnchorCount = [regex]::Matches($SourceText, [regex]::Escape($Anchor)).Count
if ($AnchorCount -ne 1) { throw "Protected-coexistence wrapper anchor count is not exactly one: $AnchorCount" }

$Insertion = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('JEJ1aWxkTWFya2VyRm9ySW5qZWN0aW9uID0gJyRCdWlsZFBhdGggPSBKb2luLVBhdGggJEV4dHJhY3RSb290ICcnc2NyaXB0c1xwcm9jZWR1cmFsX2p1bmdsZVxidWlsZC1wcm9jZWR1cmFsLWp1bmdsZS5wczEnJycKJEJ1aWxkTWFya2VyQ291bnQgPSBbcmVnZXhdOjpNYXRjaGVzKCRQYXRjaGVkVGV4dCwgW3JlZ2V4XTo6RXNjYXBlKCRCdWlsZE1hcmtlckZvckluamVjdGlvbikpLkNvdW50CmlmICgkQnVpbGRNYXJrZXJDb3VudCAtbmUgMSkgewogICAgdGhyb3cgIkRlY29kZWQgYnVpbGQtc2NyaXB0IG1hcmtlciBjb3VudCBpcyBub3QgZXhhY3RseSBvbmU6ICRCdWlsZE1hcmtlckNvdW50Igp9CiRSdW50aW1lQnVpbGRHdWFyZFBhdGNoID0gQCcKJEJ1aWxkUGF0aCA9IEpvaW4tUGF0aCAkRXh0cmFjdFJvb3QgJ3NjcmlwdHNccHJvY2VkdXJhbF9qdW5nbGVcYnVpbGQtcHJvY2VkdXJhbC1qdW5nbGUucHMxJwokQnVpbGRUZXh0ID0gKEdldC1Db250ZW50IC1MaXRlcmFsUGF0aCAkQnVpbGRQYXRoIC1SYXcpLlJlcGxhY2UoImByYG4iLCAiYG4iKQokR3VhcmRMaW5lcyA9IEAoCiAgICAnJEV4aXN0aW5nVW5yZWFsID0gQChHZXQtUHJvY2VzcyBVbnJlYWxFZGl0b3IsIFVucmVhbEVkaXRvci1DbWQgLUVycm9yQWN0aW9uIFNpbGVudGx5Q29udGludWUpJywKICAgICdpZiAoJEV4aXN0aW5nVW5yZWFsLkNvdW50IC1ndCAwKSB7JywKICAgICcgICAgdGhyb3cgIlVucmVhbCBFZGl0b3IgaXMgYWxyZWFkeSBydW5uaW5nOyByZWZ1c2luZyB0byBpbnRlcmZlcmUgd2l0aCBQSUQocyk6ICQoJEV4aXN0aW5nVW5yZWFsLklkIC1qb2luICcnLCcnKSIgJywKICAgICd9Jwp9CiRHdWFyZEJsb2NrID0gJEd1YXJkTGluZXMgLWpvaW4gImBuIgokR3VhcmRDb3VudCA9IFtyZWdleF06Ok1hdGNoZXMoJEJ1aWxkVGV4dCwgW3JlZ2V4XTo6RXNjYXBlKCRHdWFyZEJsb2NrKSkuQ291bnQKaWYgKCRHdWFyZENvdW50IC1uZSAxKSB7CiAgICB0aHJvdyAiRGVjb2RlZCBidWlsZC1zY3JpcHQgVW5yZWFsIGd1YXJkIGNvdW50IGlzIG5vdCBleGFjdGx5IG9uZTogJEd1YXJkQ291bnQiCn0KJEJ1aWxkVGV4dCA9ICRCdWlsZFRleHQuUmVwbGFjZSgKICAgICRHdWFyZEJsb2NrLAogICAgIldyaXRlLUhvc3QgJ0pVTkdMRV9CVUlMRF9JTlRFUk5BTF9VTlJFQUxfR1VBUkRfREVMRUdBVEVEPVBST1ZFTiciCikKaWYgKCRCdWlsZFRleHQuQ29udGFpbnMoJ1VucmVhbCBFZGl0b3IgaXMgYWxyZWFkeSBydW5uaW5nOyByZWZ1c2luZyB0byBpbnRlcmZlcmUgd2l0aCBQSUQocyk6JykpIHsKICAgIHRocm93ICdEZWNvZGVkIGJ1aWxkLXNjcmlwdCBibGFua2V0IFVucmVhbCBndWFyZCByZW1haW5zIGFmdGVyIGRlbGVnYXRpb24gcGF0Y2gnCn0KW0lPLkZpbGVdOjpXcml0ZUFsbFRleHQoJEJ1aWxkUGF0aCwgJEJ1aWxkVGV4dCwgKE5ldy1PYmplY3QgVGV4dC5VVEY4RW5jb2RpbmcoJGZhbHNlKSkpCldyaXRlLUhvc3QgJ0pVTkdMRV9CVUlMRF9JTlRFUk5BTF9VTlJFQUxfR1VBUkRfUEFUQ0g9UFJPVkVOJwonQAoKJFBhdGNoZWRUZXh0ID0gJFBhdGNoZWRUZXh0LlJlcGxhY2UoJEJ1aWxkTWFya2VyRm9ySW5qZWN0aW9uLCAkUnVudGltZUJ1aWxkR3VhcmRQYXRjaC5UcmltRW5kKCkpCmlmICgkUGF0Y2hlZFRleHQgLW5vdG1hdGNoICdKVU5HTEVfQlVJTERfSU5URVJOQUxfVU5SRUFMX0dVQVJEX1BBVENIPVBST1ZFTicpIHsKICAgIHRocm93ICdEZWNvZGVkIGJ1aWxkLXNjcmlwdCBndWFyZCBwYXRjaCB3YXMgbm90IGluamVjdGVkIGludG8gdGhlIFYzIHdvcmtlcicKfQoK'))
$PatchedSource = $SourceText.Replace($Anchor, $Insertion.TrimEnd() + "`n" + $Anchor)
if ($PatchedSource -notmatch 'JUNGLE_BUILD_INTERNAL_UNREAL_GUARD_PATCH=PROVEN') {
    throw 'Internal build-script guard patch insertion marker is missing'
}

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-v3-internalguard-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedWrapper = Join-Path $TempRoot 'run-procedural-jungle-v3-internalguard.ps1'
[IO.File]::WriteAllText($PatchedWrapper, $PatchedSource, (New-Object Text.UTF8Encoding($false)))

Write-Host "JUNGLE_V3_COEXISTENCE_WRAPPER_SOURCE=$SourceCommit"
Write-Host 'JUNGLE_V3_INTERNAL_BUILD_GUARD_INJECTION=PROVEN'
& powershell -NoProfile -ExecutionPolicy Bypass -File $PatchedWrapper -ExpectedBranch $ExpectedBranch
if ($LASTEXITCODE -ne 0) { throw "Internal-guard-patched V3 wrapper failed with exit code $LASTEXITCODE" }
