[CmdletBinding()]
param(
    [int]$MaximumWatchSeconds = 3300
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$env:GH_PAGER = 'cat'

$RepositorySlug = 'organicoverlords/lowvram3d-studio'
$BranchName = 'agent/blender-beggars-scene-20260804'
$BuildWorkflowName = 'beggars-scene-local-worker.yml'
$DiagnosticWorkflowName = 'beggars-scene-diagnostic.yml'
$BuildWorkflowPath = '.github/workflows/beggars-scene-local-worker.yml'
$CloneRoot = 'C:\AI\LowVRAM3D-beggars-yunet-fix'
$ReportRoot = 'C:\AI\LowVRAM3D-benchmarks\beggars-scene'
$OldYuNetUrl = 'https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
$PinnedYuNetUrl = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
$PinnedYuNetSha256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
$PinnedYuNetBytes = 232589
$StartedAt = (Get-Date).ToUniversalTime()

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
    $raw = @(& $script:Gh api $Endpoint 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "gh api failed for $Endpoint`n$($raw -join [Environment]::NewLine)"
    }
    return (($raw -join [Environment]::NewLine) | ConvertFrom-Json)
}

function Get-RemoteHead {
    $encodedBranch = [Uri]::EscapeDataString($BranchName)
    $ref = Invoke-GhJson "repos/$RepositorySlug/git/ref/heads/$encodedBranch"
    return [string]$ref.object.sha
}

function Get-CompatibleRunners {
    $response = Invoke-GhJson "repos/$RepositorySlug/actions/runners?per_page=100"
    return @(@($response.runners) | Where-Object {
        $labels = @(@($_.labels) | ForEach-Object { [string]$_.name })
        $labels -contains 'self-hosted' -and
        $labels -contains 'Windows' -and
        $labels -contains 'X64'
    })
}

function Get-RunsForHead {
    param([Parameter(Mandatory)][string]$HeadSha)
    $encodedBranch = [Uri]::EscapeDataString($BranchName)
    $response = Invoke-GhJson "repos/$RepositorySlug/actions/runs?branch=$encodedBranch&per_page=50"
    return @(@($response.workflow_runs) | Where-Object {
        [string]$_.path -eq $BuildWorkflowPath -and [string]$_.head_sha -eq $HeadSha
    } | Sort-Object created_at -Descending)
}

function Get-Run {
    param([Parameter(Mandatory)][long]$RunId)
    return Invoke-GhJson "repos/$RepositorySlug/actions/runs/$RunId"
}

function Get-RunJobs {
    param([Parameter(Mandatory)][long]$RunId)
    $response = Invoke-GhJson "repos/$RepositorySlug/actions/runs/$RunId/jobs?filter=latest&per_page=100"
    return @(@($response.jobs))
}

function Get-RunArtifacts {
    param([Parameter(Mandatory)][long]$RunId)
    $response = Invoke-GhJson "repos/$RepositorySlug/actions/runs/$RunId/artifacts?per_page=100"
    return @(@($response.artifacts))
}

function Write-Utf8NoBom {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Text
    )
    $encoding = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($Path, $Text, $encoding)
}

function Ensure-YuNetWorkflowFix {
    $remoteHeadBefore = Get-RemoteHead

    if (Test-Path -LiteralPath $CloneRoot) {
        Remove-Item -LiteralPath $CloneRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path (Split-Path $CloneRoot) -Force | Out-Null

    & git clone --quiet --single-branch --branch $BranchName "https://github.com/$RepositorySlug.git" $CloneRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not create the isolated branch clone.'
    }

    $remote = (& git -C $CloneRoot remote get-url origin | Out-String).Trim()
    $branch = (& git -C $CloneRoot branch --show-current | Out-String).Trim()
    $head = (& git -C $CloneRoot rev-parse HEAD | Out-String).Trim()
    $dirtyBefore = @(& git -C $CloneRoot status --short)

    if ($remote -notmatch 'organicoverlords/lowvram3d-studio') {
        throw "Isolated clone remote mismatch: $remote"
    }
    if ($branch -ne $BranchName) {
        throw "Isolated clone branch mismatch: $branch"
    }
    if ($head -ne $remoteHeadBefore) {
        throw "Isolated clone HEAD $head does not equal remote $remoteHeadBefore"
    }
    if (@($dirtyBefore).Count -gt 0) {
        throw 'Isolated clone is dirty before the bounded edit.'
    }

    $workflowFile = Join-Path $CloneRoot ($BuildWorkflowPath -replace '/', '\')
    $text = [System.IO.File]::ReadAllText($workflowFile).Replace("`r`n", "`n")

    $oldUrlCount = ([regex]::Matches($text, [regex]::Escape($OldYuNetUrl))).Count
    $newUrlCount = ([regex]::Matches($text, [regex]::Escape($PinnedYuNetUrl))).Count

    $oldValidation = @'
          if (-not (Test-Path -LiteralPath $yunet) -or (Get-Item -LiteralPath $yunet).Length -lt 100000) {
            throw 'YuNet model download failed or produced an implausibly small file'
          }
'@.TrimEnd()

    $newValidation = @'
          if (-not (Test-Path -LiteralPath $yunet)) {
            throw 'YuNet model download failed'
          }
          $yunetInfo = Get-Item -LiteralPath $yunet
          $yunetHash = (Get-FileHash -LiteralPath $yunet -Algorithm SHA256).Hash.ToLowerInvariant()
          if ($yunetInfo.Length -ne 232589 -or $yunetHash -ne '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4') {
            throw "YuNet integrity mismatch: bytes=$($yunetInfo.Length) sha256=$yunetHash"
          }
'@.TrimEnd()

    $alreadyFixed = $newUrlCount -eq 1 -and $text.Contains($newValidation)
    if (-not $alreadyFixed) {
        if ($oldUrlCount -ne 1) {
            throw "Expected exactly one old YuNet URL, found $oldUrlCount"
        }
        if (-not $text.Contains($oldValidation)) {
            throw 'The expected YuNet size-validation block was not found.'
        }
        $text = $text.Replace($OldYuNetUrl, $PinnedYuNetUrl)
        $text = $text.Replace($oldValidation, $newValidation)
        Write-Utf8NoBom -Path $workflowFile -Text ($text.TrimEnd() + "`n")
    }

    $changed = @(& git -C $CloneRoot status --short)
    if (@($changed).Count -eq 0) {
        Write-Host 'YUNET_WORKFLOW_ALREADY_FIXED=TRUE'
        return $head
    }
    if (@($changed).Count -ne 1 -or [string]$changed[0] -notmatch 'beggars-scene-local-worker\.yml$') {
        throw "Unexpected bounded-edit scope: $($changed -join '; ')"
    }

    & git -C $CloneRoot diff --check
    if ($LASTEXITCODE -ne 0) {
        throw 'Workflow patch failed git diff --check.'
    }

    & git -C $CloneRoot config user.name 'organicoverlords'
    & git -C $CloneRoot config user.email 'organicoverlords@users.noreply.github.com'
    & git -C $CloneRoot add -- $BuildWorkflowPath
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not stage the bounded YuNet workflow fix.'
    }

    & git -C $CloneRoot commit --quiet -m 'fix(scene): download and verify YuNet Git LFS media'
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not commit the bounded YuNet workflow fix.'
    }
    $fixHead = (& git -C $CloneRoot rev-parse HEAD | Out-String).Trim()

    & git -C $CloneRoot push --quiet origin "HEAD:$BranchName"
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not push the bounded YuNet workflow fix.'
    }

    $remoteHeadAfter = Get-RemoteHead
    if ($remoteHeadAfter -ne $fixHead) {
        throw "Remote verification failed: expected $fixHead, found $remoteHeadAfter"
    }

    Write-Host 'YUNET_WORKFLOW_FIX=PROVEN'
    Write-Host "FIX_HEAD=$fixHead"
    return $fixHead
}

function Wait-ForRunRegistration {
    param([Parameter(Mandatory)][string]$HeadSha)
    $deadline = (Get-Date).AddSeconds(120)
    do {
        $matches = @(Get-RunsForHead -HeadSha $HeadSha)
        if (@($matches).Count -gt 0) {
            return $matches[0]
        }
        Start-Sleep -Seconds 5
    } while ((Get-Date) -lt $deadline)
    return $null
}

function Wait-ForRunCompletion {
    param(
        [Parameter(Mandatory)][long]$RunId,
        [Parameter(Mandatory)][int]$MaximumSeconds
    )
    $deadline = (Get-Date).AddSeconds([Math]::Max(30, $MaximumSeconds))
    do {
        $run = Get-Run -RunId $RunId
        Write-Host "RUN_STATUS=$($run.status) CONCLUSION=$($run.conclusion) UPDATED=$($run.updated_at)"
        if ([string]$run.status -eq 'completed') {
            return $run
        }
        Start-Sleep -Seconds 15
    } while ((Get-Date) -lt $deadline)
    return Get-Run -RunId $RunId
}

function Write-RunEvidence {
    param([Parameter(Mandatory)]$Run)
    $runId = [long]$Run.id
    $jobs = @(Get-RunJobs -RunId $runId)
    $artifacts = @()
    if ([string]$Run.status -eq 'completed') {
        $artifacts = @(Get-RunArtifacts -RunId $runId)
    }

    Write-Host 'BUILD_EVIDENCE_BEGIN'
    Write-Host "RUN_ID=$runId"
    Write-Host "RUN_STATUS=$($Run.status)"
    Write-Host "RUN_CONCLUSION=$($Run.conclusion)"
    Write-Host "RUN_URL=$($Run.html_url)"

    foreach ($job in $jobs) {
        Write-Host "JOB=$($job.id) NAME=$($job.name) STATUS=$($job.status) CONCLUSION=$($job.conclusion) RUNNER=$($job.runner_name)"
        foreach ($step in @(@($job.steps))) {
            Write-Host "STEP=$($step.number) STATUS=$($step.status) CONCLUSION=$($step.conclusion) NAME=$($step.name)"
        }
    }

    foreach ($artifact in $artifacts) {
        Write-Host "ARTIFACT=$($artifact.id) NAME=$($artifact.name) BYTES=$($artifact.size_in_bytes) EXPIRED=$($artifact.expired)"
    }

    if ([string]$Run.status -eq 'completed' -and [string]$Run.conclusion -ne 'success') {
        Write-Host 'FAILED_LOG_BEGIN'
        & $script:Gh run view $runId --repo $RepositorySlug --log-failed
        Write-Host 'FAILED_LOG_END'
    }
    Write-Host 'BUILD_EVIDENCE_END'

    return [ordered]@{
        run_id = $runId
        status = [string]$Run.status
        conclusion = if ($null -eq $Run.conclusion) { $null } else { [string]$Run.conclusion }
        url = [string]$Run.html_url
        jobs = @($jobs | ForEach-Object {
            [ordered]@{
                id = [long]$_.id
                name = [string]$_.name
                status = [string]$_.status
                conclusion = if ($null -eq $_.conclusion) { $null } else { [string]$_.conclusion }
                runner_name = [string]$_.runner_name
                steps = @(@($_.steps) | ForEach-Object {
                    [ordered]@{
                        number = [int]$_.number
                        name = [string]$_.name
                        status = [string]$_.status
                        conclusion = if ($null -eq $_.conclusion) { $null } else { [string]$_.conclusion }
                    }
                })
            }
        })
        artifacts = @($artifacts | ForEach-Object {
            [ordered]@{
                id = [long]$_.id
                name = [string]$_.name
                bytes = [long]$_.size_in_bytes
                expired = [bool]$_.expired
            }
        })
    }
}

New-Item -ItemType Directory -Path $ReportRoot -Force | Out-Null
$script:Gh = Get-GhPath
& $script:Gh auth status --hostname github.com
if ($LASTEXITCODE -ne 0) {
    throw 'GitHub CLI is not authenticated.'
}

$profile = Invoke-GhJson 'user'
if ([string]$profile.login -ne 'organicoverlords') {
    throw "Authenticated GitHub user mismatch: $($profile.login)"
}

$runners = @(Get-CompatibleRunners)
$onlineRunners = @($runners | Where-Object { [string]$_.status -eq 'online' })
if (@($onlineRunners).Count -eq 0) {
    throw 'No compatible self-hosted Windows x64 runner is online.'
}
Write-Host "RUNNER_ONLINE=$($onlineRunners[0].name) BUSY=$($onlineRunners[0].busy)"

$fixHead = Ensure-YuNetWorkflowFix
$registeredRun = Wait-ForRunRegistration -HeadSha $fixHead
if ($null -eq $registeredRun) {
    Write-Host 'PUSH_TRIGGER_NOT_REGISTERED_WITHIN_120_SECONDS=TRUE'
    & $script:Gh workflow run $BuildWorkflowName --repo $RepositorySlug --ref $BranchName
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not dispatch the corrected build workflow.'
    }
    $registeredRun = Wait-ForRunRegistration -HeadSha $fixHead
}
if ($null -eq $registeredRun) {
    throw "No corrected build run registered for HEAD $fixHead"
}

Write-Host "CORRECTED_BUILD_RUN=$($registeredRun.id)"
Write-Host "CORRECTED_BUILD_URL=$($registeredRun.html_url)"

$completedRun = Wait-ForRunCompletion -RunId ([long]$registeredRun.id) -MaximumSeconds $MaximumWatchSeconds
$evidence = Write-RunEvidence -Run $completedRun

if ([string]$completedRun.status -eq 'completed') {
    & $script:Gh workflow run $DiagnosticWorkflowName --repo $RepositorySlug --ref $BranchName
    if ($LASTEXITCODE -eq 0) {
        Write-Host 'POST_BUILD_DIAGNOSTIC_DISPATCHED=TRUE'
    }
    else {
        Write-Host "POST_BUILD_DIAGNOSTIC_DISPATCH_FAILED=$LASTEXITCODE"
    }
}

$classification = if ([string]$completedRun.status -ne 'completed') {
    'BUILD_STILL_RUNNING_AFTER_BOUNDED_WATCH'
}
elseif ([string]$completedRun.conclusion -eq 'success' -and @($evidence.artifacts).Count -gt 0) {
    'BLENDER_BUILD_PROVEN_USER_VISUAL_REVIEW_REQUIRED'
}
elseif ([string]$completedRun.conclusion -eq 'success') {
    'BUILD_SUCCEEDED_ARTIFACT_NOT_PROVEN'
}
else {
    'BUILD_REJECTED_FAILURE_CAPTURED'
}

$report = [ordered]@{
    classification = $classification
    repository = $RepositorySlug
    branch = $BranchName
    fixed_head = $fixHead
    pinned_yunet = [ordered]@{
        url = $PinnedYuNetUrl
        bytes = $PinnedYuNetBytes
        sha256 = $PinnedYuNetSha256
    }
    runner = [ordered]@{
        name = [string]$onlineRunners[0].name
        status = [string]$onlineRunners[0].status
        busy_at_start = [bool]$onlineRunners[0].busy
    }
    build = $evidence
    source_media_read_by_this_repair_script = $false
    blender_started_directly_by_this_repair_script = $false
    started_at = $StartedAt.ToString('o')
    finished_at = (Get-Date).ToUniversalTime().ToString('o')
}

$reportPath = Join-Path $ReportRoot ('yunet-fix-and-build-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json')
$report | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $reportPath -Encoding utf8

Write-Host "CLASSIFICATION=$classification"
Write-Host "REPORT=$reportPath"
exit 0
