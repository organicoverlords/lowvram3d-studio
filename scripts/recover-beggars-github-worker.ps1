[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Repository = 'organicoverlords/lowvram3d-studio'
$Branch = 'agent/blender-beggars-scene-20260804'
$ExpectedLocalRepo = 'C:\Users\Lauri\Desktop\lowvram3d-repo'
$RunnerRoot = 'C:\actions-runner-lowvram3d'
$TaskName = 'LowVRAM3D-GitHub-Runner'
$ReportRoot = 'C:\AI\LowVRAM3D-benchmarks\beggars-scene'
$DispatchClone = 'C:\AI\LowVRAM3D-worker-dispatch'
$StartedAt = (Get-Date).ToUniversalTime()

function Invoke-GhJson {
    param([Parameter(Mandatory)][string]$Endpoint)
    $text = & $script:Gh api $Endpoint 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "gh api failed for $Endpoint`n$text"
    }
    return ($text | Out-String | ConvertFrom-Json)
}

function Get-GhPath {
    $candidates = @(
        (Get-Command gh.exe -ErrorAction SilentlyContinue).Source,
        "$env:ProgramFiles\GitHub CLI\gh.exe"
    ) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
    if (-not $candidates) { throw 'GitHub CLI is missing.' }
    return $candidates[0]
}

function Get-RemoteHead {
    $encoded = [Uri]::EscapeDataString($Branch)
    $ref = Invoke-GhJson "repos/$Repository/git/ref/heads/$encoded"
    return [string]$ref.object.sha
}

function Get-RunnerSnapshot {
    $response = Invoke-GhJson "repos/$Repository/actions/runners?per_page=100"
    $matches = @($response.runners | Where-Object {
        $labels = @($_.labels | ForEach-Object { [string]$_.name })
        $labels -contains 'self-hosted' -and
        $labels -contains 'Windows' -and
        $labels -contains 'X64'
    })
    return @($matches | ForEach-Object {
        [ordered]@{
            id = $_.id
            name = $_.name
            status = $_.status
            busy = [bool]$_.busy
            labels = @($_.labels | ForEach-Object { $_.name })
        }
    })
}

function Get-BranchRuns {
    $encoded = [Uri]::EscapeDataString($Branch)
    $response = Invoke-GhJson "repos/$Repository/actions/runs?branch=$encoded&per_page=50"
    return @($response.workflow_runs | Where-Object {
        $_.path -in @(
            '.github/workflows/beggars-scene-local-worker.yml',
            '.github/workflows/beggars-scene-diagnostic.yml'
        )
    } | Sort-Object created_at -Descending | ForEach-Object {
        [ordered]@{
            id = $_.id
            name = $_.name
            path = $_.path
            event = $_.event
            status = $_.status
            conclusion = $_.conclusion
            head_sha = $_.head_sha
            created_at = $_.created_at
            updated_at = $_.updated_at
            html_url = $_.html_url
        }
    })
}

function Read-LocalRepoIdentity {
    if (-not (Test-Path -LiteralPath (Join-Path $ExpectedLocalRepo '.git'))) {
        return [ordered]@{ exists = $false; path = $ExpectedLocalRepo }
    }
    $remote = (& git -C $ExpectedLocalRepo remote get-url origin 2>&1 | Out-String).Trim()
    $branch = (& git -C $ExpectedLocalRepo branch --show-current 2>&1 | Out-String).Trim()
    $head = (& git -C $ExpectedLocalRepo rev-parse HEAD 2>&1 | Out-String).Trim()
    $dirty = @(& git -C $ExpectedLocalRepo status --short 2>&1)
    if ($remote -notmatch 'organicoverlords/lowvram3d-studio') {
        throw "Local repo identity mismatch: $remote"
    }
    return [ordered]@{
        exists = $true
        path = $ExpectedLocalRepo
        remote = $remote
        branch = $branch
        head = $head
        dirty = ($dirty.Count -gt 0)
        dirty_entries = $dirty.Count
        edited = $false
    }
}

function Start-RunnerSafely {
    if (-not (Test-Path -LiteralPath (Join-Path $RunnerRoot '.runner'))) {
        throw "Runner registration is missing: $RunnerRoot\.runner"
    }
    $runCmd = Join-Path $RunnerRoot 'run.cmd'
    if (-not (Test-Path -LiteralPath $runCmd)) {
        throw "Runner executable is missing: $runCmd"
    }

    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    $action = 'NONE'
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
        $listener = @(Get-CimInstance Win32_Process | Where-Object {
            $_.ExecutablePath -and $_.ExecutablePath -like "$RunnerRoot*" -and
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

function Push-BoundedDispatch {
    param([Parameter(Mandatory)][string]$ExpectedRemoteHead)

    if (Test-Path -LiteralPath $DispatchClone) {
        Remove-Item -LiteralPath $DispatchClone -Recurse -Force
    }
    & git clone --quiet --single-branch --branch $Branch "https://github.com/$Repository.git" $DispatchClone
    if ($LASTEXITCODE -ne 0) { throw 'Could not create isolated dispatch clone.' }

    $remote = (& git -C $DispatchClone remote get-url origin | Out-String).Trim()
    $branch = (& git -C $DispatchClone branch --show-current | Out-String).Trim()
    $head = (& git -C $DispatchClone rev-parse HEAD | Out-String).Trim()
    $dirtyBefore = @(& git -C $DispatchClone status --short)
    if ($remote -notmatch 'organicoverlords/lowvram3d-studio') { throw "Dispatch clone remote mismatch: $remote" }
    if ($branch -ne $Branch) { throw "Dispatch clone branch mismatch: $branch" }
    if ($head -ne $ExpectedRemoteHead) { throw "Dispatch clone HEAD $head does not equal remote $ExpectedRemoteHead" }
    if ($dirtyBefore.Count -gt 0) { throw 'Dispatch clone is dirty before edit.' }

    $nonce = (Get-Date).ToUniversalTime().ToString('o')
    $configPath = Join-Path $DispatchClone 'configs\scene\beggars_banquet_recreation_v1.json'
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $config.worker_policy | Add-Member -NotePropertyName dispatch_nonce_utc -NotePropertyValue $nonce -Force
    $config | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $configPath -Encoding utf8

    $diagPath = Join-Path $DispatchClone '.github\workflows\beggars-scene-diagnostic.yml'
    Add-Content -LiteralPath $diagPath -Value "`n# dispatch $nonce" -Encoding utf8

    & git -C $DispatchClone config user.name 'organicoverlords'
    & git -C $DispatchClone config user.email 'organicoverlords@users.noreply.github.com'
    & git -C $DispatchClone add -- $configPath $diagPath
    $changes = @(& git -C $DispatchClone status --short)
    if ($changes.Count -ne 2) { throw "Unexpected dispatch change count: $($changes.Count)" }
    & git -C $DispatchClone commit -m "request(scene): dispatch beggars local worker $nonce"
    if ($LASTEXITCODE -ne 0) { throw 'Could not commit dispatch nonce.' }
    & git -C $DispatchClone push origin "HEAD:$Branch"
    if ($LASTEXITCODE -ne 0) { throw 'Could not push dispatch nonce.' }
    return (& git -C $DispatchClone rev-parse HEAD | Out-String).Trim()
}

New-Item -ItemType Directory -Path $ReportRoot -Force | Out-Null
$script:Gh = Get-GhPath
& $script:Gh auth status --hostname github.com 2>$null
if ($LASTEXITCODE -ne 0) { throw 'GitHub CLI is not authenticated.' }

$profile = Invoke-GhJson 'user'
if ([string]$profile.login -ne 'organicoverlords') {
    throw "Authenticated GitHub user mismatch: $($profile.login)"
}

$localRepo = Read-LocalRepoIdentity
$remoteHeadBefore = Get-RemoteHead
$runnersBefore = Get-RunnerSnapshot
$startAction = Start-RunnerSafely
$runnersAfterStart = Get-RunnerSnapshot
$runsBefore = Get-BranchRuns

$matchingRun = @($runsBefore | Where-Object {
    $_.head_sha -eq $remoteHeadBefore -and $_.status -in @('queued', 'in_progress', 'completed')
})

$dispatchHead = $null
if ($matchingRun.Count -eq 0) {
    $dispatchHead = Push-BoundedDispatch -ExpectedRemoteHead $remoteHeadBefore
    Start-Sleep -Seconds 8
}

$remoteHeadAfter = Get-RemoteHead
$runsAfter = Get-BranchRuns
$runnersFinal = Get-RunnerSnapshot
$relevant = @($runsAfter | Where-Object { $_.head_sha -eq $remoteHeadAfter })

$classification = if ($relevant.Count -gt 0) {
    'GITHUB_WORKER_DISPATCH_PROVEN'
}
elseif (@($runnersFinal | Where-Object { $_.status -eq 'online' }).Count -gt 0) {
    'RUNNER_ONLINE_DISPATCH_NOT_PROVEN'
}
else {
    'RUNNER_OFFLINE_OR_UNAVAILABLE'
}

$report = [ordered]@{
    classification = $classification
    repository = $Repository
    branch = $Branch
    authenticated_user = [string]$profile.login
    local_repo = $localRepo
    runner_root = $RunnerRoot
    runner_start_action = $startAction
    runners_before = $runnersBefore
    runners_after_start = $runnersAfterStart
    runners_final = $runnersFinal
    remote_head_before = $remoteHeadBefore
    dispatch_commit = $dispatchHead
    remote_head_after = $remoteHeadAfter
    workflow_runs_before = $runsBefore
    workflow_runs_after = $runsAfter
    matching_latest_head_runs = $relevant
    source_media_touched = $false
    blender_started_by_this_script = $false
    started_at = $StartedAt.ToString('o')
    finished_at = (Get-Date).ToUniversalTime().ToString('o')
}

$reportPath = Join-Path $ReportRoot ('runner-recovery-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json')
$report | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath $reportPath -Encoding utf8

Write-Host 'BEGGARS_GITHUB_WORKER_RECOVERY: COMPLETE'
Write-Host "CLASSIFICATION=$classification"
Write-Host "REMOTE_HEAD=$remoteHeadAfter"
Write-Host "RUNNER_START_ACTION=$startAction"
Write-Host "RUNNER_ONLINE_COUNT=$(@($runnersFinal | Where-Object { $_.status -eq 'online' }).Count)"
Write-Host "LATEST_HEAD_RUN_COUNT=$($relevant.Count)"
Write-Host "REPORT=$reportPath"
foreach ($run in $relevant) {
    Write-Host "RUN=$($run.id) STATUS=$($run.status) CONCLUSION=$($run.conclusion) URL=$($run.html_url)"
}
