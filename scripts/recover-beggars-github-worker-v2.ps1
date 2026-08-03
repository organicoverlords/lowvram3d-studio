[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Repository = 'organicoverlords/lowvram3d-studio'
$Branch = 'agent/blender-beggars-scene-20260804'
$LocalRepo = 'C:\Users\Lauri\Desktop\lowvram3d-repo'
$RunnerRoot = 'C:\actions-runner-lowvram3d'
$TaskName = 'LowVRAM3D-GitHub-Runner'
$DispatchClone = 'C:\AI\LowVRAM3D-beggars-dispatch'
$ReportRoot = 'C:\AI\LowVRAM3D-benchmarks\beggars-scene'

function Get-GhPath {
    $command = Get-Command gh.exe -ErrorAction SilentlyContinue
    if ($command -and (Test-Path -LiteralPath $command.Source)) { return $command.Source }
    $candidate = "$env:ProgramFiles\GitHub CLI\gh.exe"
    if (Test-Path -LiteralPath $candidate) { return $candidate }
    throw 'GitHub CLI is missing.'
}

function Invoke-GhJson {
    param([Parameter(Mandatory)][string]$Endpoint)
    $raw = & $script:Gh api $Endpoint 2>&1
    if ($LASTEXITCODE -ne 0) { throw "gh api failed for $Endpoint`n$raw" }
    return ($raw | Out-String | ConvertFrom-Json)
}

function Get-RunnerSnapshot {
    $response = Invoke-GhJson "repos/$Repository/actions/runners?per_page=100"
    return @($response.runners | Where-Object {
        $labels = @($_.labels | ForEach-Object { [string]$_.name })
        $labels -contains 'self-hosted' -and $labels -contains 'Windows' -and $labels -contains 'X64'
    } | ForEach-Object {
        [ordered]@{
            id = $_.id
            name = $_.name
            status = $_.status
            busy = [bool]$_.busy
            labels = @($_.labels | ForEach-Object { $_.name })
        }
    })
}

function Get-RemoteHead {
    $encodedBranch = [Uri]::EscapeDataString($Branch)
    $ref = Invoke-GhJson "repos/$Repository/git/ref/heads/$encodedBranch"
    return [string]$ref.object.sha
}

function Get-WorkflowRuns {
    $encodedBranch = [Uri]::EscapeDataString($Branch)
    $response = Invoke-GhJson "repos/$Repository/actions/runs?branch=$encodedBranch&per_page=30"
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
            status = $_.status
            conclusion = $_.conclusion
            head_sha = $_.head_sha
            created_at = $_.created_at
            updated_at = $_.updated_at
            url = $_.html_url
        }
    })
}

function Get-LocalRepoSnapshot {
    if (-not (Test-Path -LiteralPath (Join-Path $LocalRepo '.git'))) {
        return [ordered]@{ exists = $false; path = $LocalRepo; edited = $false }
    }
    $remote = (& git -C $LocalRepo remote get-url origin 2>&1 | Out-String).Trim()
    $branch = (& git -C $LocalRepo branch --show-current 2>&1 | Out-String).Trim()
    $head = (& git -C $LocalRepo rev-parse HEAD 2>&1 | Out-String).Trim()
    $dirty = @(& git -C $LocalRepo status --short 2>&1)
    if ($remote -notmatch 'organicoverlords/lowvram3d-studio') {
        throw "Local repo identity mismatch: $remote"
    }
    return [ordered]@{
        exists = $true
        path = $LocalRepo
        remote = $remote
        branch = $branch
        head = $head
        dirty = ($dirty.Count -gt 0)
        dirty_entries = $dirty.Count
        edited = $false
    }
}

function Start-RunnerIfNeeded {
    param([array]$Before)
    if (@($Before | Where-Object { $_.status -eq 'online' }).Count -gt 0) {
        return 'ALREADY_ONLINE'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $RunnerRoot '.runner'))) {
        throw "Runner registration is missing: $RunnerRoot\.runner"
    }
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) {
        Start-ScheduledTask -TaskName $TaskName
        Start-Sleep -Seconds 8
        return 'SCHEDULED_TASK_STARTED'
    }
    $runCmd = Join-Path $RunnerRoot 'run.cmd'
    if (-not (Test-Path -LiteralPath $runCmd)) { throw "Runner command is missing: $runCmd" }
    Start-Process -FilePath "$env:SystemRoot\System32\cmd.exe" `
        -ArgumentList "/d /c `"`"$runCmd`"`"" `
        -WorkingDirectory $RunnerRoot `
        -WindowStyle Hidden
    Start-Sleep -Seconds 8
    return 'RUN_CMD_STARTED'
}

function Push-DispatchNonce {
    param([Parameter(Mandatory)][string]$ExpectedHead)

    if (Test-Path -LiteralPath $DispatchClone) {
        Remove-Item -LiteralPath $DispatchClone -Recurse -Force
    }
    & git clone --quiet --single-branch --branch $Branch "https://github.com/$Repository.git" $DispatchClone
    if ($LASTEXITCODE -ne 0) { throw 'Could not clone the dispatch branch.' }

    $remote = (& git -C $DispatchClone remote get-url origin | Out-String).Trim()
    $branch = (& git -C $DispatchClone branch --show-current | Out-String).Trim()
    $head = (& git -C $DispatchClone rev-parse HEAD | Out-String).Trim()
    $dirty = @(& git -C $DispatchClone status --short)
    if ($remote -notmatch 'organicoverlords/lowvram3d-studio') { throw "Dispatch remote mismatch: $remote" }
    if ($branch -ne $Branch) { throw "Dispatch branch mismatch: $branch" }
    if ($head -ne $ExpectedHead) { throw "Dispatch HEAD $head does not equal remote $ExpectedHead" }
    if ($dirty.Count -gt 0) { throw 'Dispatch clone is dirty before edit.' }

    $nonce = (Get-Date).ToUniversalTime().ToString('o')
    $configRelative = 'configs/scene/beggars_banquet_recreation_v1.json'
    $diagnosticRelative = '.github/workflows/beggars-scene-diagnostic.yml'
    $configPath = Join-Path $DispatchClone ($configRelative -replace '/', '\')
    $diagnosticPath = Join-Path $DispatchClone ($diagnosticRelative -replace '/', '\')

    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $config.worker_policy | Add-Member -NotePropertyName dispatch_nonce_utc -NotePropertyValue $nonce -Force
    $config | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $configPath -Encoding utf8
    Add-Content -LiteralPath $diagnosticPath -Value "`n# dispatch $nonce" -Encoding utf8

    & git -C $DispatchClone config user.name 'organicoverlords'
    & git -C $DispatchClone config user.email 'organicoverlords@users.noreply.github.com'
    & git -C $DispatchClone add -- $configRelative $diagnosticRelative
    if ($LASTEXITCODE -ne 0) { throw 'Could not stage the bounded dispatch edit.' }
    $changes = @(& git -C $DispatchClone status --short)
    if ($changes.Count -ne 2) { throw "Unexpected dispatch change count: $($changes.Count)" }

    & git -C $DispatchClone commit -m "request(scene): dispatch beggars local worker $nonce"
    if ($LASTEXITCODE -ne 0) { throw 'Could not commit the dispatch nonce.' }
    & git -C $DispatchClone push origin "HEAD:$Branch"
    if ($LASTEXITCODE -ne 0) { throw 'Could not push the dispatch nonce.' }
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

$startedAt = (Get-Date).ToUniversalTime()
$localSnapshot = Get-LocalRepoSnapshot
$remoteHeadBefore = Get-RemoteHead
$runnersBefore = Get-RunnerSnapshot
$runnerAction = Start-RunnerIfNeeded -Before $runnersBefore
$runnersAfter = Get-RunnerSnapshot

if (@($runnersAfter | Where-Object { $_.status -eq 'online' }).Count -eq 0) {
    throw 'The registered Windows GitHub runner is still offline after recovery.'
}

$dispatchHead = Push-DispatchNonce -ExpectedHead $remoteHeadBefore
Start-Sleep -Seconds 10
$remoteHeadAfter = Get-RemoteHead
if ($remoteHeadAfter -ne $dispatchHead) {
    throw "Remote head mismatch after dispatch: expected $dispatchHead, found $remoteHeadAfter"
}

$runs = Get-WorkflowRuns
$matchingRuns = @($runs | Where-Object { $_.head_sha -eq $dispatchHead })
$classification = if ($matchingRuns.Count -ge 2) {
    'BUILD_AND_DIAGNOSTIC_DISPATCH_PROVEN'
}
elseif ($matchingRuns.Count -eq 1) {
    'PARTIAL_DISPATCH_PROVEN'
}
else {
    'DISPATCH_NOT_PROVEN'
}

$report = [ordered]@{
    classification = $classification
    repository = $Repository
    branch = $Branch
    authenticated_user = [string]$profile.login
    local_repo = $localSnapshot
    remote_head_before = $remoteHeadBefore
    dispatch_commit = $dispatchHead
    remote_head_after = $remoteHeadAfter
    runner_action = $runnerAction
    runners_before = $runnersBefore
    runners_after = $runnersAfter
    matching_runs = $matchingRuns
    all_relevant_runs = $runs
    source_media_touched = $false
    blender_started_directly = $false
    started_at = $startedAt.ToString('o')
    finished_at = (Get-Date).ToUniversalTime().ToString('o')
}

$reportPath = Join-Path $ReportRoot ('runner-recovery-v2-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json')
$report | ConvertTo-Json -Depth 15 | Set-Content -LiteralPath $reportPath -Encoding utf8

Write-Host 'BEGGARS_GITHUB_WORKER_RECOVERY_V2: COMPLETE'
Write-Host "CLASSIFICATION=$classification"
Write-Host "DISPATCH_COMMIT=$dispatchHead"
Write-Host "RUNNER_ACTION=$runnerAction"
Write-Host "MATCHING_RUN_COUNT=$($matchingRuns.Count)"
Write-Host "REPORT=$reportPath"
foreach ($run in $matchingRuns) {
    Write-Host "RUN=$($run.id) STATUS=$($run.status) CONCLUSION=$($run.conclusion) URL=$($run.url)"
}
