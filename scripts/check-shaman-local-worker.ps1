[CmdletBinding()]
param(
    [string]$Repository = "organicoverlords/lowvram3d-studio",
    [string]$Branch = "infra/windows-self-hosted-runner-20260731",
    [string]$WorkflowPath = ".github/workflows/shaman-local-worker.yml"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Resolve-GitHubCli {
    $command = Get-Command gh.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $candidate = "$env:ProgramFiles\GitHub CLI\gh.exe"
    if (Test-Path $candidate) { return $candidate }
    throw "GitHub CLI is not installed."
}

$gh = Resolve-GitHubCli
& $gh auth status --hostname github.com
if ($LASTEXITCODE -ne 0) { throw "GitHub CLI is not authenticated." }

$encodedBranch = [Uri]::EscapeDataString($Branch)
$runsPayload = & $gh api "repos/$Repository/actions/runs?branch=$encodedBranch&event=push&per_page=100"
if ($LASTEXITCODE -ne 0 -or -not $runsPayload) {
    throw "Could not list GitHub Actions runs through the REST API."
}

$runs = @((ConvertFrom-Json $runsPayload).workflow_runs)
$run = $runs |
    Where-Object {
        $_.head_branch -eq $Branch -and
        $_.path -eq $WorkflowPath
    } |
    Sort-Object created_at -Descending |
    Select-Object -First 1

if (-not $run) {
    throw "No push-triggered shaman workflow run exists for branch '$Branch' and path '$WorkflowPath'."
}

Write-Host "SHAMAN_WORKFLOW_FOUND" -ForegroundColor Green
Write-Host "run_id=$($run.id)"
Write-Host "status=$($run.status)"
Write-Host "conclusion=$($run.conclusion)"
Write-Host "head_sha=$($run.head_sha)"
Write-Host "url=$($run.html_url)"

$jobsPayload = & $gh api "repos/$Repository/actions/runs/$($run.id)/jobs?per_page=100"
if ($LASTEXITCODE -eq 0 -and $jobsPayload) {
    $jobs = @((ConvertFrom-Json $jobsPayload).jobs)
    foreach ($job in $jobs) {
        Write-Host "job=$($job.name) status=$($job.status) conclusion=$($job.conclusion) runner=$($job.runner_name)"
    }
}

switch ($run.status) {
    "queued" {
        Write-Host "SHAMAN_WORKFLOW_QUEUED runner should accept it when capacity is available." -ForegroundColor Yellow
        exit 0
    }
    "in_progress" {
        Write-Host "SHAMAN_IMAGE_TO_3D_RUNNING" -ForegroundColor Green
        exit 0
    }
    "completed" {
        if ($run.conclusion -eq "success") {
            Write-Host "SHAMAN_IMAGE_TO_3D_COMPLETED_SUCCESSFULLY" -ForegroundColor Green
            exit 0
        }
        Write-Host "SHAMAN_IMAGE_TO_3D_FAILED conclusion=$($run.conclusion)" -ForegroundColor Red
        & $gh run view $run.id --repo $Repository --log-failed
        exit 1
    }
    default {
        Write-Host "SHAMAN_WORKFLOW_STATUS_$($run.status)" -ForegroundColor Yellow
        exit 0
    }
}
