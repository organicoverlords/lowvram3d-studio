param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][string]$Repository,
    [Parameter(Mandatory = $true)][string]$Branch,
    [Parameter(Mandatory = $true)][string]$HeadSha
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-StrictMode -Version Latest

$Root = "C:\AI\LowVRAM3D-benchmarks\production\panda_face_surface_ownership_v2\$RunId"
$Upload = Join-Path $Root "upload"
$Logs = Join-Path $Root "logs"
$Candidate = Join-Path $Root "candidate_a"
$Renders = Join-Path $Root "renders"
$BaselineRender = Join-Path $Renders "baseline"
$CandidateRender = Join-Path $Renders "candidate_a"
$Package = Join-Path $Root "package"
$BaselineGlb = "C:\AI\panda_support_local_2048\candidate_2048\panda_atlas_support_fixed_2048.glb"
$BaselineAtlas = "C:\AI\panda_support_local_2048\candidate_2048\atlas_2048_nearest.png"
$SourceImage = "C:\AI\LowVRAM3D-benchmarks\images\red_panda_character.png"
$Fixture = Join-Path $PSScriptRoot "..\fixtures\panda_face_source_fixture_20260806.json"

function Write-Json([object]$Value, [string]$Path) {
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $Path -Encoding utf8
}

function Hash-Lower([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLower()
}

function Resolve-Python {
    $candidates = @(
        "C:\AI\HY3D2\python_standalone\python.exe",
        "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\control\Scripts\python.exe"
    )
    $pathPython = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($pathPython) {
        $candidates += [string]$pathPython.Source
    }
    foreach ($candidate in $candidates) {
        if (-not $candidate -or -not (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            continue
        }
        $previousPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & $candidate -c "import cv2, numpy, PIL, pytest" 2>$null
            $exitCode = [int]$LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousPreference
        }
        if ($exitCode -eq 0) {
            return [string]$candidate
        }
    }
    throw "No suitable Python environment was found"
}

function Resolve-Blender {
    $candidates = @(
        "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
        "C:\Program Files\Blender Foundation\Blender 5.1\blender.exe",
        "C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
    )
    $pathBlender = Get-Command blender.exe -ErrorAction SilentlyContinue
    if ($pathBlender) {
        $candidates += [string]$pathBlender.Source
    }
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return [string]$candidate
        }
    }
    throw "Blender was not found"
}

function Run-Native(
    [string]$Executable,
    [string[]]$Arguments,
    [string]$Stdout,
    [string]$Stderr
) {
    $stdoutParent = Split-Path -Parent $Stdout
    $stderrParent = Split-Path -Parent $Stderr
    foreach ($directory in @($stdoutParent, $stderrParent)) {
        if ($directory) {
            New-Item -ItemType Directory -Path $directory -Force | Out-Null
        }
    }

    # Windows PowerShell 5.1 converts any native stderr output into a
    # NativeCommandError when ErrorActionPreference is Stop. Blender writes
    # harmless warnings to stderr even on exit code 0. Temporarily allow native
    # stderr, then use the process exit code as the sole execution result.
    $previousPreference = $ErrorActionPreference
    $exitCode = -1
    try {
        $ErrorActionPreference = "Continue"
        & $Executable @Arguments 1> $Stdout 2> $Stderr
        $exitCode = [int]$LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    return $exitCode
}

if (Test-Path -LiteralPath $Root) {
    Remove-Item -LiteralPath $Root -Recurse -Force
}
foreach ($directory in @($Root, $Upload, $Logs, $Candidate, $Renders, $BaselineRender, $CandidateRender)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

$Python = Resolve-Python
$Blender = Resolve-Blender
$env:PYTHONPATH = "$PWD\workers;$PWD"

$expected = @{
    $BaselineGlb = "1c4bec32a31ce90766f12f095e2d3c9c7e57e0b34e4f8ffc79c9a29ea8e7a817"
    $BaselineAtlas = "88ee2bf03bfe27b6ee28720734c6abd94fc03242f27a2f33529bc3db0f08383e"
    $SourceImage = "fbda3719b42366477f77e94d87cb1d32a5bf6adc4b6f29782194cc154471159b"
}
$inputRecords = @()
foreach ($path in $expected.Keys) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required input is missing: $path"
    }
    $hash = Hash-Lower $path
    if ($hash -ne $expected[$path]) {
        throw "Input hash mismatch: $path = $hash"
    }
    $inputRecords += [ordered]@{ path = $path; sha256 = $hash }
}
if (-not (Test-Path -LiteralPath $Fixture -PathType Leaf)) {
    throw "Source fixture is missing: $Fixture"
}
$inputRecords += [ordered]@{ path = $Fixture; sha256 = (Hash-Lower $Fixture) }
Write-Json $inputRecords (Join-Path $Root "input_hashes.json")

$testStdout = Join-Path $Logs "focused_tests.stdout.log"
$testStderr = Join-Path $Logs "focused_tests.stderr.log"
$testArgs = @(
    "-m", "pytest",
    "tests\test_face_surface_ownership_core.py",
    "tests\test_face_surface_candidate.py",
    "tests\test_face_patch_texture_contract.py",
    "tests\test_face_surface_candidate_v2.py",
    "-q"
)
$testExit = Run-Native $Python $testArgs $testStdout $testStderr
Write-Json ([ordered]@{
    schema = "face_v2_test_execution_v1"
    exit_code = $testExit
    stdout = $testStdout
    stderr = $testStderr
}) (Join-Path $Root "test_execution.json")

$candidateExit = -1
if ($testExit -eq 0) {
    $candidateStdout = Join-Path $Logs "candidate.stdout.log"
    $candidateStderr = Join-Path $Logs "candidate.stderr.log"
    $candidateArgs = @(
        "workers\face_surface_candidate_v2.py",
        "--baseline-glb", $BaselineGlb,
        "--baseline-atlas", $BaselineAtlas,
        "--source-image", $SourceImage,
        "--source-fixture", $Fixture,
        "--output-dir", $Candidate,
        "--ray-stride", "3",
        "--minimum-alpha", "0.30"
    )
    $candidateExit = Run-Native $Python $candidateArgs $candidateStdout $candidateStderr
}
$candidateGlb = Join-Path $Candidate "panda_face_surface_owned_2048.glb"
$candidateAtlas = Join-Path $Candidate "atlas_face_surface_owned_2048.png"
$candidateReport = Join-Path $Candidate "report.json"
Write-Json ([ordered]@{
    schema = "face_v2_candidate_execution_v1"
    exit_code = $candidateExit
    glb_exists = (Test-Path -LiteralPath $candidateGlb -PathType Leaf)
    atlas_exists = (Test-Path -LiteralPath $candidateAtlas -PathType Leaf)
    report_exists = (Test-Path -LiteralPath $candidateReport -PathType Leaf)
}) (Join-Path $Root "candidate_execution.json")

$baselineBlenderLog = Join-Path $Logs "baseline_blender.stdout.log"
$baselineBlenderErr = Join-Path $Logs "baseline_blender.stderr.log"
$baselineArgs = @(
    "--background", "--factory-startup",
    "--python", "$PWD\blender\render_glb_diagnostic_set.py", "--",
    "--input", $BaselineGlb,
    "--out-dir", $BaselineRender,
    "--label", "baseline_2048",
    "--resolution", "768"
)
$baselineExit = Run-Native $Blender $baselineArgs $baselineBlenderLog $baselineBlenderErr
$baselineSheetExit = -1
if ($baselineExit -eq 0) {
    $baselineSheetExit = Run-Native $Python @(
        "workers\build_contact_sheet.py",
        "--render-dir", $BaselineRender,
        "--output", (Join-Path $BaselineRender "contact_sheet.png")
    ) (Join-Path $Logs "baseline_sheet.stdout.log") (Join-Path $Logs "baseline_sheet.stderr.log")
}

$candidateRenderExit = -1
$candidateSheetExit = -1
$comparisonExit = -1
if (Test-Path -LiteralPath $candidateGlb -PathType Leaf) {
    $candidateRenderExit = Run-Native $Blender @(
        "--background", "--factory-startup",
        "--python", "$PWD\blender\render_glb_diagnostic_set.py", "--",
        "--input", $candidateGlb,
        "--out-dir", $CandidateRender,
        "--label", "face_surface_owned_2048_v2",
        "--resolution", "768"
    ) (Join-Path $Logs "candidate_blender.stdout.log") (Join-Path $Logs "candidate_blender.stderr.log")
    if ($candidateRenderExit -eq 0) {
        $candidateSheetExit = Run-Native $Python @(
            "workers\build_contact_sheet.py",
            "--render-dir", $CandidateRender,
            "--output", (Join-Path $CandidateRender "contact_sheet.png")
        ) (Join-Path $Logs "candidate_sheet.stdout.log") (Join-Path $Logs "candidate_sheet.stderr.log")
    }
    if ($candidateSheetExit -eq 0 -and $baselineSheetExit -eq 0) {
        $comparisonExit = Run-Native $Python @(
            "workers\build_render_comparison.py",
            "--baseline-dir", $BaselineRender,
            "--candidate-dir", $CandidateRender,
            "--output", (Join-Path $Renders "baseline_vs_candidate.png")
        ) (Join-Path $Logs "comparison.stdout.log") (Join-Path $Logs "comparison.stderr.log")
    }
}
Write-Json ([ordered]@{
    schema = "face_v2_render_execution_v1"
    baseline_blender_exit = $baselineExit
    baseline_contact_sheet_exit = $baselineSheetExit
    candidate_blender_exit = $candidateRenderExit
    candidate_contact_sheet_exit = $candidateSheetExit
    comparison_exit = $comparisonExit
    baseline_contact_sheet_exists = (Test-Path -LiteralPath (Join-Path $BaselineRender "contact_sheet.png"))
    candidate_contact_sheet_exists = (Test-Path -LiteralPath (Join-Path $CandidateRender "contact_sheet.png"))
    comparison_exists = (Test-Path -LiteralPath (Join-Path $Renders "baseline_vs_candidate.png"))
}) (Join-Path $Root "render_execution.json")

if (Test-Path -LiteralPath $Package) {
    Remove-Item -LiteralPath $Package -Recurse -Force
}
New-Item -ItemType Directory -Path $Package -Force | Out-Null
foreach ($name in @("candidate_a", "renders", "logs")) {
    $source = Join-Path $Root $name
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $Package $name) -Recurse -Force
    }
}
foreach ($name in @("input_hashes.json", "test_execution.json", "candidate_execution.json", "render_execution.json")) {
    $source = Join-Path $Root $name
    if (Test-Path -LiteralPath $source -PathType Leaf) {
        Copy-Item -LiteralPath $source -Destination $Package -Force
    }
}
Copy-Item -LiteralPath $SourceImage -Destination (Join-Path $Package "authoritative_source.png") -Force
Copy-Item -LiteralPath $BaselineAtlas -Destination (Join-Path $Package "baseline_atlas_2048.png") -Force
Copy-Item -LiteralPath $Fixture -Destination (Join-Path $Package "source_fixture.json") -Force

$gpu = "not_available"
$nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($nvidia) {
    $gpu = (& $nvidia.Source --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1) -join "`n"
}
$success = (
    $testExit -eq 0 -and
    $candidateExit -eq 0 -and
    (Test-Path -LiteralPath $candidateGlb -PathType Leaf) -and
    $baselineExit -eq 0 -and
    $baselineSheetExit -eq 0 -and
    $candidateRenderExit -eq 0 -and
    $candidateSheetExit -eq 0 -and
    $comparisonExit -eq 0
)
Write-Json ([ordered]@{
    schema = "panda_face_surface_worker_receipt_v2"
    repository = $Repository
    branch = $Branch
    head_sha = $HeadSha
    workflow_run_id = $RunId
    runner_name = "$env:RUNNER_NAME"
    python = (& $Python --version 2>&1) -join " "
    blender = (& $Blender --version 2>&1 | Select-Object -First 1) -join " "
    gpu = $gpu
    tests_exit = $testExit
    candidate_exit = $candidateExit
    baseline_render_exit = $baselineExit
    candidate_render_exit = $candidateRenderExit
    comparison_exit = $comparisonExit
    execution_success = $success
    baseline_overwritten = $false
    neural_views_regenerated = $false
    geometry_regenerated = $false
    promotion_authorized = $false
    classification = $(if ($success) { "CANDIDATE_REQUIRES_VISUAL_REVIEW" } else { "EVIDENCE_PACKAGED_AFTER_FAILURE" })
    recorded_at_utc = (Get-Date).ToUniversalTime().ToString("o")
}) (Join-Path $Package "worker_receipt.json")

Get-ChildItem -LiteralPath $Package -File -Recurse | ForEach-Object {
    $hash = Hash-Lower $_.FullName
    $relative = $_.FullName.Substring($Package.Length).TrimStart("\", "/").Replace("\", "/")
    "$hash  $relative"
} | Sort-Object | Set-Content -LiteralPath (Join-Path $Package "SHA256SUMS.txt") -Encoding ascii

$Zip = Join-Path $Upload "panda_face_surface_ownership_v2_$RunId.zip"
Compress-Archive -LiteralPath $Package -DestinationPath $Zip -CompressionLevel Optimal -Force
$zipHash = Hash-Lower $Zip
"$zipHash  $([IO.Path]::GetFileName($Zip))" | Set-Content -LiteralPath (Join-Path $Upload "ZIP_SHA256SUMS.txt") -Encoding ascii
Copy-Item -LiteralPath (Join-Path $Package "worker_receipt.json") -Destination $Upload -Force
Copy-Item -LiteralPath (Join-Path $Package "SHA256SUMS.txt") -Destination $Upload -Force

Write-Host "FACE_V2_ZIP=$Zip"
Write-Host "FACE_V2_ZIP_SHA256=$zipHash"
Write-Host "FACE_V2_SUCCESS=$success"
if (-not $success) { exit 1 }
exit 0
