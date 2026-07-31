[CmdletBinding()]
param(
    [string]$Repository = "organicoverlords/lowvram3d-studio",
    [string]$RunnerRoot = "C:\actions-runner-lowvram3d",
    [string]$RunnerName = "LOWVRAM3D-$env:COMPUTERNAME"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Require-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        $args = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`" -Repository `"$Repository`" -RunnerRoot `"$RunnerRoot`" -RunnerName `"$RunnerName`""
        Start-Process powershell.exe -Verb RunAs -ArgumentList $args
        exit 0
    }
}

function Ensure-GitHubCli {
    $gh = Get-Command gh.exe -ErrorAction SilentlyContinue
    if ($gh) { return $gh.Source }
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) { throw "GitHub CLI is missing and winget is unavailable." }
    & $winget.Source install --id GitHub.cli --exact --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "GitHub CLI installation failed." }
    $candidate = "$env:ProgramFiles\GitHub CLI\gh.exe"
    if (-not (Test-Path $candidate)) { throw "GitHub CLI installed but gh.exe was not found." }
    return $candidate
}

function Ensure-GitHubLogin([string]$Gh) {
    & $Gh auth status --hostname github.com 2>$null
    if ($LASTEXITCODE -eq 0) { return }
    Write-Host "A GitHub browser sign-in is required once." -ForegroundColor Yellow
    & $Gh auth login --hostname github.com --git-protocol https --web
    if ($LASTEXITCODE -ne 0) { throw "GitHub sign-in failed." }
}

function Get-LatestRunnerAsset {
    $headers = @{ Accept = "application/vnd.github+json"; "User-Agent" = "LowVRAM3DStudio" }
    $release = Invoke-RestMethod "https://api.github.com/repos/actions/runner/releases/latest" -Headers $headers
    $asset = $release.assets | Where-Object { $_.name -match '^actions-runner-win-x64-[0-9.]+\.zip$' } | Select-Object -First 1
    if (-not $asset) { throw "No official Windows x64 Actions runner asset was found." }
    return $asset
}

Require-Administrator
$gh = Ensure-GitHubCli
Ensure-GitHubLogin $gh

$permission = & $gh api "repos/$Repository/collaborators/organicoverlords/permission" --jq ".permission" 2>$null
if ($LASTEXITCODE -ne 0 -or $permission -notin @("admin", "maintain")) {
    throw "The signed-in GitHub account does not have repository administration permission."
}

$token = & $gh api --method POST "repos/$Repository/actions/runners/registration-token" --jq ".token"
if ($LASTEXITCODE -ne 0 -or -not $token) { throw "Could not create a runner registration token." }

New-Item -ItemType Directory -Path $RunnerRoot -Force | Out-Null
if (-not (Test-Path (Join-Path $RunnerRoot "config.cmd"))) {
    $asset = Get-LatestRunnerAsset
    $zip = Join-Path $env:TEMP $asset.name
    Invoke-WebRequest $asset.browser_download_url -OutFile $zip -UseBasicParsing
    if ($asset.digest) {
        $expected = ([string]$asset.digest -replace '^sha256:', '').ToLowerInvariant()
        $actual = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($expected -ne $actual) { throw "Official runner archive SHA-256 mismatch." }
    }
    Expand-Archive $zip $RunnerRoot -Force
    Remove-Item $zip -Force
}

$taskName = "LowVRAM3D-GitHub-Runner"
$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existingTask) {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}

Push-Location $RunnerRoot
try {
    if (-not (Test-Path ".runner")) {
        & .\config.cmd --unattended --url "https://github.com/$Repository" --token $token.Trim() --name $RunnerName --labels "lowvram3d-pc,gtx1660s,blender52,windows10" --work "_work" --replace
        if ($LASTEXITCODE -ne 0) { throw "GitHub runner registration failed." }
    }
} finally { Pop-Location }

$user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$runCmd = Join-Path $RunnerRoot "run.cmd"
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\cmd.exe" -Argument "/d /c `"`"$runCmd`"`"" -WorkingDirectory $RunnerRoot
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 5

Write-Host "GitHub worker registered and started: $RunnerName" -ForegroundColor Green
Write-Host "Labels: self-hosted, Windows, X64, lowvram3d-pc, gtx1660s, blender52, windows10" -ForegroundColor Green
Write-Host "The queued shaman image-to-3D workflow can now be accepted by this PC." -ForegroundColor Cyan
