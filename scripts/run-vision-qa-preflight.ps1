[CmdletBinding()]
param(
    [string]$RepoRoot = (Get-Location).Path,
    [string]$ExpectedRemote = "organicoverlords/lowvram3d-studio",
    [string]$ExpectedBranch = "agent/vision-qa-prework-20260803",
    [string]$Python = "python",
    [string]$OutputDir = "proof/vision-qa/preflight"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Run([string]$Exe, [string[]]$Arguments) {
    $text = & $Exe @Arguments 2>&1 | Out-String
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($Arguments -join ' ') failed:`n$text" }
    return $text.Trim()
}

$RepoRoot = (Resolve-Path $RepoRoot).Path
Set-Location $RepoRoot
$gitDir = Run git @("rev-parse", "--show-toplevel")
if ((Resolve-Path $gitDir).Path -ne $RepoRoot) { throw "RepoRoot mismatch: $gitDir" }
$branch = Run git @("branch", "--show-current")
$head = Run git @("rev-parse", "HEAD")
$remote = Run git @("remote", "get-url", "origin")
$status = Run git @("status", "--porcelain=v1")
if ($branch -ne $ExpectedBranch) { throw "Branch mismatch: expected $ExpectedBranch, got $branch" }
if ($remote -notmatch [regex]::Escape($ExpectedRemote)) { throw "Remote mismatch: $remote" }
if ($status) { throw "Working tree is dirty; refusing preflight:`n$status" }

$started = Get-Date
Run $Python @("-m", "compileall", "-q", "src/lowvram3d/vision_qa", "workers/vision_qa") | Out-Null
$pytest = & $Python -m pytest -q tests/test_vision_qa_contracts.py tests/test_vision_qa_policy.py tests/test_vision_qa_evidence.py tests/test_vision_qa_prompting.py tests/test_vision_qa_registry.py 2>&1 | Out-String
$pytestExit = $LASTEXITCODE
if ($pytestExit -ne 0) { throw "Focused tests failed:`n$pytest" }

$out = Join-Path $RepoRoot $OutputDir
New-Item -ItemType Directory -Force -Path $out | Out-Null
$receipt = [ordered]@{
    schema = "vision_qa_preflight_v1"
    classification = "VISION_QA_PREWORK_CPU_PROVEN"
    repo = $RepoRoot
    branch = $branch
    head = $head
    remote = $remote
    clean = $true
    gpu_work_started = $false
    model_download_started = $false
    compileall = "PROVEN"
    focused_tests = "PROVEN"
    pytest_output = $pytest.Trim()
    started_utc = $started.ToUniversalTime().ToString("o")
    finished_utc = (Get-Date).ToUniversalTime().ToString("o")
    not_proven = @(
        "Qwen3.5-2B target-GPU compatibility",
        "MiniCPM-V 4.6 target-GPU compatibility",
        "Florence-2 output quality",
        "EdgeTAM output quality",
        "Depth Anything 3 output quality",
        "vision-controlled pipeline retries"
    )
}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -Encoding UTF8 (Join-Path $out "preflight-receipt.json")
$receipt | ConvertTo-Json -Depth 8
