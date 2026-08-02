<#
.SYNOPSIS
    Run the canonical lighthouse image through the isolated MoGe-2 Small Normal probe.

.DESCRIPTION
    Creates a persistent dedicated Python environment without modifying ComfyUI or the
    production control environment. The lighthouse source is resolved by exact SHA-256
    across normal user and benchmark folders, so moving or renaming the image does not
    break the run. Raw maps and compact previews are retained for proof.
#>
[CmdletBinding()]
param(
    [string]$ImagePath = "$env:USERPROFILE\Downloads\ChatGPT Image 30.7.2026 klo 00.40.32.png",
    [string]$ExpectedSha256 = "e8ea9e307327169d998df9fd6757db718e5647ac46fc8235f971416e132df6ba",
    [string]$BasePython = "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\control\Scripts\python.exe",
    [string]$EnvironmentRoot = "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\image-world-moge",
    [string]$OutputRoot = "$env:RUNNER_TEMP\image-world-lighthouse-moge",
    [int]$NumTokens = 1200,
    [int]$InputLongEdge = 768,
    [int]$MaxGpuMemoryMb = 5600,
    [bool]$AllowDownload = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Python = Join-Path $EnvironmentRoot "Scripts\python.exe"
$Marker = Join-Path $EnvironmentRoot "image-world-moge-environment.json"
$EnvironmentContract = "moge2-vits-normal-torch2.8.0-cu128-b942f00-v1"
$SetupReport = Join-Path $OutputRoot "setup-report.json"
$RequestedImagePath = $ImagePath
$ResolvedImagePath = $null
$SourceSearchRoots = @()
$SourceCandidatesChecked = 0

function Assert-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing ${Label}: $Path"
    }
    if ((Get-Item -LiteralPath $Path).Length -le 0) {
        throw "Empty ${Label}: $Path"
    }
}

function Test-ExpectedSourceHash([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $Path).Length -le 0) { return $false }
    $script:SourceCandidatesChecked += 1
    try {
        $hash = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($hash -eq $ExpectedSha256.ToLowerInvariant()) {
            $script:ResolvedImagePath = (Resolve-Path -LiteralPath $Path).Path
            return $true
        }
    } catch {
        Write-Warning "Could not hash candidate '$Path': $($_.Exception.Message)"
    }
    return $false
}

function Resolve-LighthouseSource([string]$RequestedPath) {
    if (Test-ExpectedSourceHash $RequestedPath) {
        return $script:ResolvedImagePath
    }

    $roots = @(
        (Split-Path -Parent $RequestedPath),
        (Join-Path $env:USERPROFILE "Downloads"),
        (Join-Path $env:USERPROFILE "Desktop"),
        (Join-Path $env:USERPROFILE "Pictures"),
        (Join-Path $env:USERPROFILE "Documents"),
        (Join-Path $env:USERPROFILE "OneDrive\Downloads"),
        (Join-Path $env:USERPROFILE "OneDrive\Desktop"),
        (Join-Path $env:USERPROFILE "OneDrive\Pictures"),
        "C:\AI\LowVRAM3D-benchmarks",
        "C:\AI\LowVRAM3D-benchmarks\inputs",
        "C:\AI\LowVRAM3D-benchmarks\sources"
    ) | Where-Object {
        $_ -and (Test-Path -LiteralPath $_ -PathType Container)
    } | Select-Object -Unique
    $script:SourceSearchRoots = @($roots)

    $seen = [System.Collections.Generic.HashSet[string]]::new(
        [System.StringComparer]::OrdinalIgnoreCase
    )
    $candidates = [System.Collections.Generic.List[string]]::new()
    function Add-Candidate([string]$CandidatePath) {
        if ($CandidatePath -and $seen.Add($CandidatePath)) {
            $candidates.Add($CandidatePath)
        }
    }

    $requestedName = [IO.Path]::GetFileName($RequestedPath)
    foreach ($root in $roots) {
        foreach ($file in Get-ChildItem -LiteralPath $root -File -Recurse -Filter $requestedName -ErrorAction SilentlyContinue) {
            Add-Candidate $file.FullName
        }
    }
    foreach ($root in $roots) {
        foreach ($pattern in @("*00.40.32*.png", "*lighthouse*.png", "*majakka*.png")) {
            foreach ($file in Get-ChildItem -Path (Join-Path $root $pattern) -File -Recurse -ErrorAction SilentlyContinue) {
                Add-Candidate $file.FullName
            }
        }
    }

    # If the name changed, scan image files only in ordinary user folders. Benchmark
    # roots can be very large, so they remain limited to the targeted name search above.
    $broadRoots = @($roots | Where-Object {
        $_ -notlike "C:\AI\LowVRAM3D-benchmarks*"
    })
    foreach ($root in $broadRoots) {
        foreach ($file in Get-ChildItem -LiteralPath $root -File -Recurse -ErrorAction SilentlyContinue) {
            if ($file.Extension.ToLowerInvariant() -in @(".png", ".jpg", ".jpeg", ".webp")) {
                Add-Candidate $file.FullName
            }
        }
    }

    Write-Host "LIGHTHOUSE_SOURCE_CANDIDATES=$($candidates.Count)"
    foreach ($candidate in $candidates) {
        if (Test-ExpectedSourceHash $candidate) {
            return $script:ResolvedImagePath
        }
    }

    throw (
        "Canonical lighthouse image was not found by SHA-256. " +
        "Checked $SourceCandidatesChecked image candidates under: $($roots -join '; ')"
    )
}

function Write-SetupReport([string]$Status, [string[]]$Errors = @()) {
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
    [ordered]@{
        status = $Status
        environment_contract = $EnvironmentContract
        base_python = $BasePython
        environment_python = $Python
        environment_root = $EnvironmentRoot
        source_requested = $RequestedImagePath
        source_resolved = $ResolvedImagePath
        source_search_roots = $SourceSearchRoots
        source_candidates_checked = $SourceCandidatesChecked
        source_sha256_expected = $ExpectedSha256
        num_tokens = $NumTokens
        input_long_edge = $InputLongEdge
        max_gpu_memory_mb = $MaxGpuMemoryMb
        allow_download = $AllowDownload
        errors = $Errors
        recorded_at = (Get-Date).ToUniversalTime().ToString("o")
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $SetupReport -Encoding utf8
}

try {
    if (Test-Path -LiteralPath $OutputRoot) {
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

    $ImagePath = Resolve-LighthouseSource $RequestedImagePath
    Assert-File $ImagePath "canonical lighthouse source"
    $ActualHash = (Get-FileHash -LiteralPath $ImagePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualHash -ne $ExpectedSha256.ToLowerInvariant()) {
        throw "Resolved lighthouse source SHA-256 mismatch. Expected $ExpectedSha256, got $ActualHash"
    }
    Write-Host "LIGHTHOUSE_SOURCE_RESOLVED=$ImagePath"
    Assert-File $BasePython "LowVRAM3D control Python"

    $EnvironmentReady = $false
    if ((Test-Path -LiteralPath $Python -PathType Leaf) -and (Test-Path -LiteralPath $Marker -PathType Leaf)) {
        try {
            $MarkerData = Get-Content -LiteralPath $Marker -Raw | ConvertFrom-Json
            if ($MarkerData.environment_contract -eq $EnvironmentContract) {
                & $Python -c "import cv2, numpy, torch; from moge.model.v2 import MoGeModel; assert torch.cuda.is_available(); print(torch.__version__)"
                $EnvironmentReady = ($LASTEXITCODE -eq 0)
            }
        } catch {
            $EnvironmentReady = $false
        }
    }

    if (-not $EnvironmentReady) {
        if (Test-Path -LiteralPath $EnvironmentRoot) {
            Remove-Item -LiteralPath $EnvironmentRoot -Recurse -Force
        }
        & $BasePython -m venv $EnvironmentRoot
        if ($LASTEXITCODE -ne 0) { throw "Could not create MoGe environment" }
        Assert-File $Python "new MoGe environment Python"

        & $Python -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
        if ($LASTEXITCODE -ne 0) { throw "Could not bootstrap MoGe environment" }
        & $Python -m pip install --disable-pip-version-check `
            --index-url https://download.pytorch.org/whl/cu128 `
            "torch==2.8.0" "torchvision==0.23.0"
        if ($LASTEXITCODE -ne 0) { throw "Could not install CUDA 12.8 PyTorch" }
        & $Python -m pip install --disable-pip-version-check `
            "git+https://github.com/microsoft/MoGe.git@b942f00"
        if ($LASTEXITCODE -ne 0) { throw "Could not install pinned MoGe-2 code" }

        & $Python -c "import cv2, numpy, torch; from moge.model.v2 import MoGeModel; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)"
        if ($LASTEXITCODE -ne 0) { throw "MoGe environment import/CUDA verification failed" }
        [ordered]@{
            environment_contract = $EnvironmentContract
            created_at = (Get-Date).ToUniversalTime().ToString("o")
            base_python = $BasePython
        } | ConvertTo-Json | Set-Content -LiteralPath $Marker -Encoding utf8
    }

    $env:PYTHONPATH = @((Join-Path $RepoRoot "src"), $RepoRoot) -join [IO.Path]::PathSeparator
    $env:PYTHONUNBUFFERED = "1"
    $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
    $env:CUDA_MODULE_LOADING = "LAZY"
    $env:HF_HOME = "$env:LOCALAPPDATA\LowVRAM3DStudio\cache\huggingface"
    $env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
    New-Item -ItemType Directory -Path $env:HF_HUB_CACHE -Force | Out-Null

    Write-SetupReport "READY"
    $WorkerArgs = @(
        (Join-Path $RepoRoot "workers\image_world\moge_probe.py"),
        "--input", $ImagePath,
        "--output", $OutputRoot,
        "--model", "Ruicheng/moge-2-vits-normal",
        "--num-tokens", "$NumTokens",
        "--input-long-edge", "$InputLongEdge",
        "--max-gpu-memory-mb", "$MaxGpuMemoryMb",
        "--lock-path", (Join-Path $env:LOCALAPPDATA "LowVRAM3DStudio\locks\image-world-heavy-gpu.lock")
    )
    if ($AllowDownload) { $WorkerArgs += "--allow-download" }

    & $Python @WorkerArgs
    $ProbeExit = $LASTEXITCODE
    Assert-File (Join-Path $OutputRoot "moge-probe-report.json") "MoGe probe report"
    if ($ProbeExit -ne 0) {
        throw "MoGe probe failed with exit code $ProbeExit"
    }

    & $Python (Join-Path $RepoRoot "workers\image_world\render_moge_previews.py") `
        --geometry (Join-Path $OutputRoot "geometry") `
        --output (Join-Path $OutputRoot "previews")
    if ($LASTEXITCODE -ne 0) { throw "MoGe preview rendering failed" }

    $Report = Get-Content -LiteralPath (Join-Path $OutputRoot "moge-probe-report.json") -Raw | ConvertFrom-Json
    if ($Report.status -ne "PASS") {
        throw "MoGe report status is $($Report.status)"
    }
    if ($Report.source_sha256 -ne $ExpectedSha256.ToLowerInvariant()) {
        throw "MoGe report source hash does not match the canonical lighthouse"
    }
    foreach ($Required in @(
        "geometry\points.npy",
        "geometry\depth.npy",
        "geometry\mask.npy",
        "geometry\normal.npy",
        "geometry\intrinsics.npy",
        "previews\depth.png",
        "previews\mask.png",
        "previews\normal.png",
        "previews\preview-summary.json"
    )) {
        Assert-File (Join-Path $OutputRoot $Required) $Required
    }
    Write-Host "IMAGE_WORLD_LIGHTHOUSE_MOGE_PROBE_PASS"
    Write-Host "SOURCE: $ImagePath"
    Write-Host "OUTPUT: $OutputRoot"
    Write-Host "PEAK_RESERVED_MB: $($Report.peak_gpu_reserved_mb)"
    exit 0
} catch {
    $Message = "$(($_.Exception.GetType().Name)): $($_.Exception.Message)"
    Write-SetupReport "FAILED" @($Message, $_.ScriptStackTrace)
    Write-Error $Message
    exit 1
}
