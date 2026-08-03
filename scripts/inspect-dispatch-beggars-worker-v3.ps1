[CmdletBinding()]
param(
    [int]$RegistrationWaitSeconds = 60
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Repository = 'organicoverlords/lowvram3d-studio'
$Branch = 'agent/blender-beggars-scene-20260804'
$RunnerRoot = 'C:\actions-runner-lowvram3d'
$TaskName = 'LowVRAM3D-GitHub-Runner'
$ReportRoot = 'C:\AI\LowVRAM3D-benchmarks\beggars-scene'
$BuildWorkflow = 'beggars-scene-local-worker.yml'
$DiagnosticWorkflow = 'beggars-scene-diagnostic.yml'
$BuildPath = '.github/workflows/beggars-scene-local-worker.yml'
$DiagnosticPath = '.github/workflows/beggars-scene-diagnostic.yml'

function Get-GhPath {
    $command = Get-Command gh.exe -ErrorAction SilentlyContinue
    if ($command -and (Test-Path -LiteralPath $command.Source)) {
        return $command.Source
    }
    $candidate = "$env:ProgramFiles\GitHub CLI\gh.exe"
    if (Test-Path -LiteralPath $candidate) {
        return $candidate
    }
    throw 'GitHub CLI is missing.'
}

function Invoke-GhJson {
    param([Parameter(Mandatory)][string]$Endpoint)
    $raw = & $script:Gh api $Endpoint 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "gh api failed for $Endpoint`n$($raw | Out-String)"
    }
    return ($raw | Out-String | ConvertFrom-Json)
}

function Get-RemoteHead {
    $encodedBranch = [Uri]::EscapeDataString($Branch)
    $ref = Invoke-GhJson "repos/$Repository/git/ref/heads/$encodedBranch"
    return [string]$ref.object.sha
}

function Get-RunnerSnapshot {
    $response = Invoke-GhJson "repos/$Repository/actions/runners?per_page=100"
    return @($response.runners | Where-Object {
        $labels = @($_.labels | ForEach-Object { [string]$_.name })
        $labels -contains 'self-hosted' -and
        $labels -contains 'Windows' -and
        $labels -contains 'X64'
    } | ForEach-Object {
        [ordered]@{
            id = [long]$_.id
            name = [string]$_.name
            status = [string]$_.status
            busy = [bool]$_.busy
            labels = @($_.labels | ForEach-Object { [string]$_.name })
        }
    })
}

function Start-RunnerIfOffline {
    param([array]$Snapshot)

    if (@($Snapshot | Where-Object { $_.status -eq 'online' }).Count -gt 0) {
        return 'ALREADY_ONLINE'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $RunnerRoot '.runner'))) {
        throw "Runner registration is missing: $RunnerRoot\.runner"
    }

    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) {
        if ($task.State -ne 'Running') {
            Start-ScheduledTask -TaskName $TaskName
            $action = 'SCHEDULED_TASK_STARTED'
        }
        else {
            $action = 'SCHEDULED_TASK_ALREADY_RUNNING'
        }
    }
    else {
        $runCmd = Join-Path $RunnerRoot 'run.cmd'
        if (-not (Test-Path -LiteralPath $runCmd)) {
            throw "Runner command is missing: $runCmd"
        }
        $listener = @(Get-CimInstance Win32_Process | Where-Object {
            $_.ExecutablePath -and
            $_.ExecutablePath -like "$RunnerRoot*" -and
            $_.Name -match 'Runner.Listener|Runner.Worker'
        })
        if ($listener.Count -eq 0) {
            Start-Process -FilePath "$env:SystemRoot\System32\cmd.exe" `
                -ArgumentList "/d /c `"`"$runCmd`"`"" `
                -WorkingDirectory $RunnerRoot `
                -WindowStyle Hidden
            $action = 'RUN_CMD_STARTED'
        }
        else {
            $action = 'RUNNER_PROCESS_ALREADY_PRESENT'
        }
    }

    Start-Sleep -Seconds 8
    return $action
}

function Get-RelevantRuns {
    $encodedBranch = [Uri]::EscapeDataString($Branch)
    $response = Invoke-GhJson "repos/$Repository/actions/runs?branch=$encodedBranch&per_page=50"
    return @($response.workflow_runs | Where-Object {
        $_.path -in @($BuildPath, $DiagnosticPath)
    } | Sort-Object created_at -Descending | ForEach-Object {
        [ordered]@{
            id = [long]$_.id
            name = [string]$_.name
            display_title = [string]$_.display_title
            path = [string]$_.path
            event = [string]$_.event
            status = [string]$_.status
            conclusion = if ($null -eq $_.conclusion) { $null } else { [string]$_.conclusion }
            head_sha = [string]$_.head_sha
            run_attempt = [int]$_.run_attempt
            created_at = [string]$_.created_at
            updated_at = [string]$_.updated_at
            url = [string]$_.html_url
        }
    })
}

function Get-LatestForPath {
    param(
        [array]$Runs,
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Head
    )
    return @($Runs | Where-Object {
        $_.path -eq $Path -and $_.head_sha -eq $Head
    } | Select-Object -First 1)
}

function Dispatch-Workflow {
    param([Parameter(Mandatory)][string]$Workflow)

    $output = & $script:Gh workflow run $Workflow --repo $Repository --ref $Branch 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Could not dispatch $Workflow`n$($output | Out-String)"
    }
    Write-Host "DISPATCHED=$Workflow"
}

function Get-RunJobs {
    param([Parameter(Mandatory)][long]$RunId)

    $response = Invoke-GhJson "repos/$Repository/actions/runs/$RunId/jobs?filter=latest&per_page=100"
    return @($response.jobs | ForEach-Object {
        [ordered]@{
            id = [long]$_.id
            name = [string]$_.name
            status = [string]$_.status
            conclusion = if ($null -eq $_.conclusion) { $null } else { [string]$_.conclusion }
            started_at = [string]$_.started_at
            completed_at = [string]$_.completed_at
            runner_name = [string]$_.runner_name
            runner_group_name = [string]$_.runner_group_name
            steps = @($_.steps | ForEach-Object {
                [ordered]@{
                    number = [int]$_.number
                    name = [string]$_.name
                    status = [string]$_.status
                    conclusion = if ($null -eq $_.conclusion) { $null } else { [string]$_.conclusion }
                    started_at = [string]$_.started_at
                    completed_at = [string]$_.completed_at
                }
            })
        }
    })
}

function Get-RunArtifacts {
    param([Parameter(Mandatory)][long]$RunId)

    $response = Invoke-GhJson "repos/$Repository/actions/runs/$RunId/artifacts?per_page=100"
    return @($response.artifacts | ForEach-Object {
        [ordered]@{
            id = [long]$_.id
            name = [string]$_.name
            size_in_bytes = [long]$_.size_in_bytes
            expired = [bool]$_.expired
            created_at = [string]$_.created_at
            expires_at = [string]$_.expires_at
        }
    })
}

function Get-FailedLogTail {
    param([Parameter(Mandatory)][long]$RunId)

    $lines = @(& $script:Gh run view $RunId --repo $Repository --log-failed 2>&1)
    if ($LASTEXITCODE -ne 0 -and $lines.Count -eq 0) {
        return @('FAILED_LOG_UNAVAILABLE')
    }
    $maximum = 180
    if ($lines.Count -le $maximum) {
        return $lines
    }
    return @($lines | Select-Object -Last $maximum)
}

New-Item -ItemType Directory -Path $ReportRoot -Force | Out-Null
$script:Gh = Get-GhPath
& $script:Gh auth status --hostname github.com 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'GitHub CLI is not authenticated.'
}

$profile = Invoke-GhJson 'user'
if ([string]$profile.login -ne 'organicoverlords') {
    throw "Authenticated GitHub user mismatch: $($profile.login)"
}

$startedAt = (Get-Date).ToUniversalTime()
$head = Get-RemoteHead
$runnersBefore = Get-RunnerSnapshot
$runnerAction = Start-RunnerIfOffline -Snapshot $runnersBefore
$runnersAfter = Get-RunnerSnapshot
if (@($runnersAfter | Where-Object { $_.status -eq 'online' }).Count -eq 0) {
    throw 'The registered Windows GitHub runner is offline after recovery.'
}

$runs = Get-RelevantRuns
$build = Get-LatestForPath -Runs $runs -Path $BuildPath -Head $head
$diagnostic = Get-LatestForPath -Runs $runs -Path $DiagnosticPath -Head $head

if ($build.Count -eq 0) {
    Dispatch-Workflow -Workflow $BuildWorkflow
}
if ($diagnostic.Count -eq 0) {
    Dispatch-Workflow -Workflow $DiagnosticWorkflow
}

$deadline = (Get-Date).AddSeconds([Math]::Max(10, $RegistrationWaitSeconds))
do {
    $runs = Get-RelevantRuns
    $build = Get-LatestForPath -Runs $runs -Path $BuildPath -Head $head
    $diagnostic = Get-LatestForPath -Runs $runs -Path $DiagnosticPath -Head $head
    if ($build.Count -gt 0 -and $diagnostic.Count -gt 0) {
        break
    }
    Start-Sleep -Seconds 5
} while ((Get-Date) -lt $deadline)

if ($build.Count -eq 0 -and $diagnostic.Count -eq 0) {
    throw "No relevant workflow runs registered for HEAD $head"
}

$selectedRuns = @($build + $diagnostic)
$runDetails = @()
foreach ($run in $selectedRuns) {
    if ($null -eq $run) { continue }
    $jobs = Get-RunJobs -RunId $run.id
    $artifacts = @()
    if ($run.status -eq 'completed') {
        $artifacts = Get-RunArtifacts -RunId $run.id
    }
    $failedLogTail = @()
    if ($run.status -eq 'completed' -and $run.conclusion -notin @('success', 'skipped', 'neutral')) {
        $failedLogTail = Get-FailedLogTail -RunId $run.id
    }
    $runDetails += [ordered]@{
        run = $run
        jobs = $jobs
        artifacts = $artifacts
        failed_log_tail = $failedLogTail
    }
}

$buildRun = if ($build.Count -gt 0) { $build[0] } else { $null }
$classification = if ($null -eq $buildRun) {
    'BUILD_RUN_MISSING_DIAGNOSTIC_ONLY'
}
elseif ($buildRun.status -ne 'completed') {
    'BUILD_QUEUED_OR_RUNNING'
}
elseif ($buildRun.conclusion -eq 'success') {
    'BUILD_AUTOMATION_SUCCEEDED_USER_REVIEW_REQUIRED'
}
else {
    'BUILD_AUTOMATION_FAILED_DIAGNOSIS_CAPTURED'
}

$report = [ordered]@{
    classification = $classification
    repository = $Repository
    branch = $Branch
    head_sha = $head
    authenticated_user = [string]$profile.login
    runner_action = $runnerAction
    runners_before = $runnersBefore
    runners_after = $runnersAfter
    selected_runs = $runDetails
    source_media_read_by_this_script = $false
    source_media_uploaded_by_this_script = $false
    blender_started_directly_by_this_script = $false
    started_at = $startedAt.ToString('o')
    finished_at = (Get-Date).ToUniversalTime().ToString('o')
}

$reportPath = Join-Path $ReportRoot ('worker-status-v3-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json')
$report | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $reportPath -Encoding utf8

Write-Host 'BEGGARS_WORKER_STATUS_V3: COMPLETE'
Write-Host "CLASSIFICATION=$classification"
Write-Host "HEAD_SHA=$head"
Write-Host "RUNNER_ACTION=$runnerAction"
Write-Host "RUNNER_ONLINE_COUNT=$(@($runnersAfter | Where-Object { $_.status -eq 'online' }).Count)"
Write-Host "REPORT=$reportPath"

foreach ($detail in $runDetails) {
    $run = $detail.run
    Write-Host "RUN=$($run.id) WORKFLOW=$($run.path) EVENT=$($run.event) STATUS=$($run.status) CONCLUSION=$($run.conclusion) URL=$($run.url)"
    foreach ($job in $detail.jobs) {
        Write-Host "JOB=$($job.id) NAME=$($job.name) STATUS=$($job.status) CONCLUSION=$($job.conclusion) RUNNER=$($job.runner_name)"
        foreach ($step in @($job.steps | Where-Object { $_.conclusion -eq 'failure' })) {
            Write-Host "FAILED_STEP=$($step.number) $($step.name)"
        }
    }
    foreach ($artifact in $detail.artifacts) {
        Write-Host "ARTIFACT=$($artifact.id) NAME=$($artifact.name) BYTES=$($artifact.size_in_bytes) EXPIRED=$($artifact.expired)"
    }
    if ($detail.failed_log_tail.Count -gt 0) {
        Write-Host 'FAILED_LOG_TAIL_BEGIN'
        $detail.failed_log_tail | ForEach-Object { Write-Host $_ }
        Write-Host 'FAILED_LOG_TAIL_END'
    }
}

exit 0
