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
if ($Status.Count -ne 0) { throw "Repository is dirty before V3 assembly wrapper: $($Status -join '; ')" }

git fetch origin $Branch --quiet
$RemoteHead = (git rev-parse "origin/$Branch").Trim()
if ($Head -ne $RemoteHead) { throw "Checkout head differs from remote: $Head vs $RemoteHead" }

$SourceCommit = '5ee2f112714d451f22fee8e7ab337ef766804f9c'
$RelativePath = 'scripts/procedural_jungle/run-procedural-jungle-local-worker.ps1'
$SourceLines = @(git show "${SourceCommit}:$RelativePath")
if ($LASTEXITCODE -ne 0 -or $SourceLines.Count -lt 150) {
    throw "Unable to recover safe-preflight V3 worker from $SourceCommit"
}
$SourceText = ($SourceLines -join "`n") + "`n"

$StartMarker = '$InstallerBuilder = New-Object Text.StringBuilder'
$EndMarker = '& $Python $Installer --target $ExtractRoot --report $OverlayReport'
$StartIndex = $SourceText.IndexOf($StartMarker, [StringComparison]::Ordinal)
$EndIndex = $SourceText.IndexOf($EndMarker, [StringComparison]::Ordinal)
if ($StartIndex -lt 0 -or $EndIndex -le $StartIndex) { throw 'V3 installer assembly markers are invalid' }
if ($SourceText.IndexOf($StartMarker, $StartIndex + 1, [StringComparison]::Ordinal) -ge 0) { throw 'V3 installer start marker is not unique' }
if ($SourceText.IndexOf($EndMarker, $EndIndex + 1, [StringComparison]::Ordinal) -ge 0) { throw 'V3 installer invocation marker is not unique' }

$Replacement = @'
$InstallerBuilder = New-Object Text.StringBuilder
foreach ($Fragment in $InstallerFragments) {
    [void]$InstallerBuilder.Append((Get-Content -LiteralPath $Fragment.FullName -Raw))
}
$InstallerText = $InstallerBuilder.ToString().Replace("`r`n", "`n")
$InstallerText = $InstallerText.TrimEnd([char[]]@([char]13, [char]10)) + "`n"
$Installer = Join-Path $TempRoot 'apply-procedural-jungle-v3-overlay.py'
$Utf8NoBom = New-Object Text.UTF8Encoding($false)
[IO.File]::WriteAllText($Installer, $InstallerText, $Utf8NoBom)
$InstallerByteLength = ([IO.File]::ReadAllBytes($Installer)).Length
if ($InstallerByteLength -ne 46901) {
    throw "V3 installer UTF-8 byte length mismatch: expected=46901 actual=$InstallerByteLength"
}
$InstallerSha = (Get-FileHash -LiteralPath $Installer -Algorithm SHA256).Hash.ToLowerInvariant()
$DiagnosticLogRoot = 'C:\AI\ProceduralJungle\20260804\logs'
New-Item -ItemType Directory -Path $DiagnosticLogRoot -Force | Out-Null
Get-ChildItem -LiteralPath $DiagnosticLogRoot -Filter 'jungle-v3-installer-*' -File -ErrorAction SilentlyContinue | Remove-Item -Force
[IO.File]::WriteAllText((Join-Path $DiagnosticLogRoot 'jungle-v3-installer-reconstructed.py.log'), $InstallerText, $Utf8NoBom)
$FragmentMetadata = New-Object Collections.Generic.List[string]
for ($Index = 0; $Index -lt $InstallerFragments.Count; $Index++) {
    $FragmentText = (Get-Content -LiteralPath $InstallerFragments[$Index].FullName -Raw).Replace("`r`n", "`n")
    $FragmentPath = Join-Path $DiagnosticLogRoot ("jungle-v3-installer-fragment-{0:D2}.py.log" -f $Index)
    [IO.File]::WriteAllText($FragmentPath, $FragmentText, $Utf8NoBom)
    $FragmentSha = (Get-FileHash -LiteralPath $FragmentPath -Algorithm SHA256).Hash.ToLowerInvariant()
    $FragmentMetadata.Add(("INDEX={0};NAME={1};LENGTH={2};SHA256={3}" -f $Index, $InstallerFragments[$Index].Name, $FragmentText.Length, $FragmentSha))
}
[IO.File]::WriteAllLines((Join-Path $DiagnosticLogRoot 'jungle-v3-installer-fragments.log'), $FragmentMetadata, $Utf8NoBom)
$ExpectedInstallerSha = '8caff1745534e6c73ae21b86a6cb5ee035bad5512741a1c5561bc3b649a348d5'
if ($InstallerSha -ne $ExpectedInstallerSha) {
    throw "V3 installer hash mismatch: expected=$ExpectedInstallerSha actual=$InstallerSha"
}
Write-Host "JUNGLE_V3_INSTALLER_UTF8_BYTES=$InstallerByteLength"
Write-Host "JUNGLE_V3_INSTALLER_CANONICAL_SHA256=$InstallerSha"
'@

$PatchedText = $SourceText.Substring(0, $StartIndex) + $Replacement.TrimEnd() + "`n" + $SourceText.Substring($EndIndex)

$PreflightStartMarker = @'
$OwnedProjectRoot = 'C:\Users\Lauri\Desktop\ProceduralJungle58'
'@.Trim()
$BuildMarker = @'
$BuildPath = Join-Path $ExtractRoot 'scripts\procedural_jungle\build-procedural-jungle.ps1'
'@.Trim()
$PreflightStart = $PatchedText.IndexOf($PreflightStartMarker, [StringComparison]::Ordinal)
$BuildIndex = $PatchedText.IndexOf($BuildMarker, [StringComparison]::Ordinal)
if ($PreflightStart -lt 0 -or $BuildIndex -le $PreflightStart) { throw 'Safe-preflight replacement markers are invalid' }
if ($PatchedText.IndexOf($PreflightStartMarker, $PreflightStart + 1, [StringComparison]::Ordinal) -ge 0) { throw 'Preflight start marker is not unique' }
if ($PatchedText.IndexOf($BuildMarker, $BuildIndex + 1, [StringComparison]::Ordinal) -ge 0) { throw 'Build marker is not unique' }

$CoexistencePreflight = @'
$OwnedProjectRoot = 'C:\Users\Lauri\Desktop\ProceduralJungle58'
$ProtectedProjectPath = 'C:\Users\Lauri\Desktop\UnrealAITest58\UnrealAITest58.uproject'
$ProcessLogRoot = 'C:\AI\ProceduralJungle\20260804\logs'
New-Item -ItemType Directory -Path $ProcessLogRoot -Force | Out-Null
$ProcessLogPath = Join-Path $ProcessLogRoot 'unreal_process_preflight.log'
$ExistingUnreal = @(Get-CimInstance Win32_Process -Filter "Name = 'UnrealEditor.exe'" -ErrorAction Stop)
$PipelineOwned = New-Object Collections.Generic.List[object]
$ProtectedResponsive = New-Object Collections.Generic.List[object]
$UnknownUnreal = New-Object Collections.Generic.List[object]
$ProcessLines = New-Object Collections.Generic.List[string]
foreach ($Entry in $ExistingUnreal) {
    $NativeProcess = Get-Process -Id $Entry.ProcessId -ErrorAction SilentlyContinue
    $Responding = if ($NativeProcess) { [bool]$NativeProcess.Responding } else { $false }
    $WindowTitle = if ($NativeProcess) { [string]$NativeProcess.MainWindowTitle } else { '' }
    $Command = [string]$Entry.CommandLine
    $NormalizedCommand = $Command.Replace('/', '\')
    $Class = 'UNKNOWN'
    if ($NormalizedCommand.IndexOf($OwnedProjectRoot, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
        $PipelineOwned.Add($Entry)
        $Class = 'PIPELINE_OWNED'
    }
    elseif ($Responding -and
            $NormalizedCommand.IndexOf($ProtectedProjectPath, [StringComparison]::OrdinalIgnoreCase) -ge 0 -and
            $WindowTitle.StartsWith('UnrealAITest58', [StringComparison]::OrdinalIgnoreCase)) {
        $ProtectedResponsive.Add($Entry)
        $Class = 'PROTECTED_RESPONSIVE_COEXISTENCE'
    }
    else {
        $UnknownUnreal.Add($Entry)
    }
    $ProcessLines.Add(("PID={0};CLASS={1};CREATED={2};RESPONDING={3};TITLE={4};PATH={5};COMMAND={6}" -f $Entry.ProcessId, $Class, $Entry.CreationDate, $Responding, $WindowTitle, $Entry.ExecutablePath, $Entry.CommandLine))
}

$NvidiaCandidates = @(
    $(if (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue) { (Get-Command nvidia-smi.exe).Source } else { $null }),
    'C:\Windows\System32\nvidia-smi.exe',
    'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
$NvidiaSmi = $NvidiaCandidates | Select-Object -First 1
if (-not $NvidiaSmi) { throw 'nvidia-smi.exe is required for protected-editor coexistence gating' }
$GpuLine = @(& $NvidiaSmi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)[0].Trim()
if ($LASTEXITCODE -ne 0 -or $GpuLine -notmatch '^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*$') {
    throw "Unable to parse NVIDIA coexistence metrics: $GpuLine"
}
$GpuUsedMiB = [int]$Matches[1]
$GpuTotalMiB = [int]$Matches[2]
$GpuUtilPercent = [int]$Matches[3]
$ProcessLines.Add(("GPU_USED_MIB={0};GPU_TOTAL_MIB={1};GPU_UTIL_PERCENT={2}" -f $GpuUsedMiB, $GpuTotalMiB, $GpuUtilPercent))
[IO.File]::WriteAllLines($ProcessLogPath, $ProcessLines, (New-Object Text.UTF8Encoding($false)))

if ($UnknownUnreal.Count -gt 0) {
    $Ids = @($UnknownUnreal | ForEach-Object { $_.ProcessId }) -join ','
    throw "Unknown or unresponsive Unreal Editor process(es) are running. PID(s): $Ids. Evidence: $ProcessLogPath"
}
if ($GpuUsedMiB -gt 4400 -or $GpuUtilPercent -gt 85) {
    throw "Insufficient GPU headroom for safe coexistence: used=${GpuUsedMiB}MiB/${GpuTotalMiB}MiB util=${GpuUtilPercent}%. Evidence: $ProcessLogPath"
}

foreach ($Entry in $PipelineOwned) {
    $NativeProcess = Get-Process -Id $Entry.ProcessId -ErrorAction SilentlyContinue
    if (-not $NativeProcess) { continue }
    $RequestedClose = $false
    if ($NativeProcess.MainWindowHandle -ne 0) { $RequestedClose = $NativeProcess.CloseMainWindow() }
    $Deadline = (Get-Date).AddSeconds(20)
    while ((Get-Date) -lt $Deadline -and (Get-Process -Id $Entry.ProcessId -ErrorAction SilentlyContinue)) {
        Start-Sleep -Milliseconds 500
    }
    if (Get-Process -Id $Entry.ProcessId -ErrorAction SilentlyContinue) {
        throw "Pipeline-owned Unreal Editor PID $($Entry.ProcessId) did not close gracefully; force termination is not authorized. CloseRequested=$RequestedClose Evidence=$ProcessLogPath"
    }
}
Write-Host 'JUNGLE_PROTECTED_UNREAL_COEXISTENCE_V1=PROVEN'
Write-Host "JUNGLE_PROTECTED_UNREAL_COUNT=$($ProtectedResponsive.Count)"
Write-Host "JUNGLE_PIPELINE_OWNED_UNREAL_COUNT=$($PipelineOwned.Count)"
Write-Host "JUNGLE_GPU_PREBUILD_USED_MIB=$GpuUsedMiB"
Write-Host "JUNGLE_GPU_PREBUILD_UTIL_PERCENT=$GpuUtilPercent"

'@
$PatchedText = $PatchedText.Substring(0, $PreflightStart) + $CoexistencePreflight.TrimEnd() + "`n" + $PatchedText.Substring($BuildIndex)

if ($PatchedText -match 'InstallerText\.Length -ne 46901' -or $PatchedText -match 'InstallerText\.Length -eq 46900') { throw 'Rejected character-count installer logic remains' }
if ($PatchedText -notmatch 'InstallerByteLength -ne 46901') { throw 'UTF-8 byte-length gate is missing' }
if ($PatchedText -notmatch 'JUNGLE_PROTECTED_UNREAL_COEXISTENCE_V1') { throw 'Protected Unreal coexistence preflight is missing' }
if ($PatchedText -match 'Unrelated Unreal Editor process\(es\) are running; refusing to interfere') { throw 'Old blanket Unreal rejection remains' }

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-v3-coexist-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedWorker = Join-Path $TempRoot 'run-procedural-jungle-v3-coexistence.ps1'
[IO.File]::WriteAllText($PatchedWorker, $PatchedText, (New-Object Text.UTF8Encoding($false)))

Write-Host "JUNGLE_V3_SAFE_PREFLIGHT_SOURCE=$SourceCommit"
Write-Host 'JUNGLE_V3_INSTALLER_ASSEMBLY_BLOCK_REPLACEMENT=PROVEN'
Write-Host 'JUNGLE_V3_KNOWN_EDITOR_COEXISTENCE_PATCH=PROVEN'
& powershell -NoProfile -ExecutionPolicy Bypass -File $PatchedWorker -ExpectedBranch $ExpectedBranch
if ($LASTEXITCODE -ne 0) { throw "Coexistence-patched V3 worker failed with exit code $LASTEXITCODE" }
