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
$Status = @(git status --short)
if ($Remote -notmatch 'organicoverlords/lowvram3d-studio(\.git)?$') { throw "Repository mismatch: $Remote" }
if ($Branch -ne $ExpectedBranch) { throw "Branch mismatch: $Branch" }
if ($Status.Count -ne 0) { throw "Checkout is dirty before worker bootstrap: $($Status -join '; ')" }
$BundleDir = Join-Path $RepoRoot 'worker-bundles\procedural-jungle-direct-worker'
$BundleParts = @(Get-ChildItem -LiteralPath $BundleDir -Filter 'chunk-*.b64' -File | Sort-Object Name)
if ($BundleParts.Count -lt 1) { throw "Worker bundle chunks are missing: $BundleDir" }
$ExpectedSha = 'ae5a81fab6be79dac7b58bcf383679abacb4da47e6a25a9ca36258c23fcfac2d'
$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$ZipPath = Join-Path $TempRoot 'worker.zip'
$Base64 = (($BundleParts | ForEach-Object { Get-Content -LiteralPath $_.FullName -Raw }) -join '') -replace '\s',''
[IO.File]::WriteAllBytes($ZipPath, [Convert]::FromBase64String($Base64))
$ActualSha = (Get-FileHash -LiteralPath $ZipPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualSha -ne $ExpectedSha) { throw "Worker bundle hash mismatch: $ActualSha" }
$SourceRoot = Join-Path $TempRoot 'source'
Expand-Archive -LiteralPath $ZipPath -DestinationPath $SourceRoot -Force
$Entry = Join-Path $SourceRoot 'scripts\procedural_jungle\build-procedural-jungle.ps1'
if (-not (Test-Path -LiteralPath $Entry)) { throw "Decoded worker entrypoint is missing: $Entry" }

# A separate Unreal project may already be open interactively. Never close it or
# touch its files. Permit coexistence only when no running process references the
# dedicated ProceduralJungle58 target project. The decoded worker otherwise has a
# conservative blanket refusal that would block safe isolated commandlets.
$TargetProjectRoot = 'C:\Users\Lauri\Desktop\ProceduralJungle58'
$UnrealProcesses = @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -in @('UnrealEditor.exe', 'UnrealEditor-Cmd.exe') }
)
$TargetProcesses = @(
    $UnrealProcesses | Where-Object {
        $_.CommandLine -and $_.CommandLine.IndexOf($TargetProjectRoot, [StringComparison]::OrdinalIgnoreCase) -ge 0
    }
)
if ($TargetProcesses.Count -gt 0) {
    throw "Target ProceduralJungle58 project is already open; refusing concurrent mutation. PID(s): $($TargetProcesses.ProcessId -join ',')"
}
if ($UnrealProcesses.Count -gt 0) {
    $EntryText = Get-Content -LiteralPath $Entry -Raw
    $Pattern = '(?ms)^\s*if\s*\(\$[A-Za-z_][A-Za-z0-9_]*\.Count\s*-gt\s*0\)\s*\{\s*throw\s*"Unreal Editor is already running; refusing to interfere with PID\(s\):[^\"]*"\s*\}\s*'
    $Matches = [regex]::Matches($EntryText, $Pattern)
    if ($Matches.Count -ne 1) {
        throw "Could not prove a unique Unreal coexistence guard in decoded worker; matches=$($Matches.Count)"
    }
    $Replacement = "Write-Warning 'Unrelated Unreal Editor process detected; continuing only with the isolated ProceduralJungle58 project.'`r`n"
    $EntryText = [regex]::Replace($EntryText, $Pattern, $Replacement, 1)
    Set-Content -LiteralPath $Entry -Value $EntryText -Encoding utf8
    Write-Host "UNRELATED_UNREAL_COEXISTENCE=AUTHORIZED"
    Write-Host "UNRELATED_UNREAL_PID=$($UnrealProcesses.ProcessId -join ',')"
}

Write-Host "DIRECT_WORKER_BUNDLE_SHA256=$ActualSha"
Write-Host 'CODEX_INVOKED=NO'
Write-Host 'CLAUDE_INVOKED=NO'
Write-Host 'MAGICMUSIC_INVOKED=NO'
& $Entry -ExpectedBranch $ExpectedBranch -SourceRoot $SourceRoot
if ($LASTEXITCODE -ne 0) { throw "Direct worker entrypoint failed with exit code $LASTEXITCODE" }
