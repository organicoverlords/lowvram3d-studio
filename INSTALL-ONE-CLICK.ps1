[CmdletBinding()]
param(
  [string]$InstallRoot = "$env:LOCALAPPDATA\LowVRAM3DStudio",
  [switch]$SkipModels,
  [switch]$SkipPrerequisites,
  [switch]$RetryOptional
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
Set-StrictMode -Version Latest
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

$PackageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppRoot = Join-Path $InstallRoot 'app'
$ThirdParty = Join-Path $InstallRoot 'thirdparty'
$Envs = Join-Path $InstallRoot 'envs'
$Logs = Join-Path $InstallRoot 'install-logs'
$Proof = Join-Path $InstallRoot 'proof'
$StateRoot = Join-Path $InstallRoot 'install-state\stages'
$PackageVersion = (Get-Content (Join-Path $PackageRoot 'VERSION') -Raw).Trim()
New-Item -ItemType Directory -Force -Path $InstallRoot,$AppRoot,$ThirdParty,$Envs,$Logs,$Proof,$StateRoot | Out-Null
. (Join-Path $PackageRoot 'scripts\windows\install-checkpoints.ps1')
Start-Transcript -Path (Join-Path $Logs 'install.log') -Append | Out-Null

function Note([string]$Text) { Write-Host "`n== $Text ==" -ForegroundColor Cyan }
function Has([string]$Command) { return [bool](Get-Command $Command -ErrorAction SilentlyContinue) }

function Write-Utf8NoBom {
  param(
    [Parameter(Mandatory=$true)][string]$Path,
    [Parameter(Mandatory=$true)][AllowEmptyString()][string]$Text
  )
  $parent = Split-Path -Parent $Path
  if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
  [System.IO.File]::WriteAllText($Path, $Text, (New-Object System.Text.UTF8Encoding($false)))
}

function Set-ObjectProperty {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory=$true)][object]$Object,
    [Parameter(Mandatory=$true)][string]$Name,
    [AllowNull()][object]$Value
  )
  if ($null -eq $Object) { throw "Cannot set JSON property '$Name' on a null object." }
  $Object | Add-Member -NotePropertyName $Name -NotePropertyValue $Value -Force
}

function Test-WingetPackageInstalled([string]$Id) {
  if (-not (Has 'winget')) { return $false }
  $output = (& winget list --id $Id --exact --accept-source-agreements 2>&1 | Out-String)
  return ($LASTEXITCODE -eq 0 -and $output -match [regex]::Escape($Id))
}

function Ensure-WingetPackage([string]$Id, [string]$Name) {
  if (-not (Has 'winget')) { throw "winget is required to install $Name automatically." }
  if (Test-WingetPackageInstalled $Id) { Note "$Name is already installed"; return }
  Note "Installing $Name"
  & winget install --id $Id --exact --silent --accept-package-agreements --accept-source-agreements
  $installExit = $LASTEXITCODE
  if ($installExit -ne 0 -and -not (Test-WingetPackageInstalled $Id)) {
    throw "winget failed to install $Name (exit code $installExit)"
  }
  if (-not (Test-WingetPackageInstalled $Id)) { throw "$Name was not found after winget installation." }
}

function Find-BlenderExecutable {
  $command = Get-Command blender -ErrorAction SilentlyContinue
  if ($command) { return $command.Source }
  $roots = @()
  if ($env:ProgramFiles) { $roots += (Join-Path $env:ProgramFiles 'Blender Foundation') }
  if (${env:ProgramFiles(x86)}) { $roots += (Join-Path ${env:ProgramFiles(x86)} 'Blender Foundation') }
  if ($env:LOCALAPPDATA) { $roots += (Join-Path $env:LOCALAPPDATA 'Programs\Blender Foundation') }
  $matches = foreach ($root in ($roots | Where-Object { Test-Path $_ })) {
    Get-ChildItem -Path $root -Filter blender.exe -Recurse -File -ErrorAction SilentlyContinue
  }
  $best = $matches | Sort-Object FullName -Descending | Select-Object -First 1
  if ($best) { return $best.FullName }
  return $null
}

function Refresh-Path {
  $machine = [Environment]::GetEnvironmentVariable('Path','Machine')
  $user = [Environment]::GetEnvironmentVariable('Path','User')
  $env:Path = "$machine;$user"
}

function Get-NodeVersion {
  if (-not (Has 'node')) { return $null }
  $raw = (& node --version 2>$null | Out-String).Trim().TrimStart('v')
  if (-not $raw) { return $null }
  try { return [Version]$raw } catch { return $null }
}

function Test-NodeCompatible {
  $version = Get-NodeVersion
  if (-not $version) { return $false }
  if ($version.Major -eq 20) { return $version.Minor -ge 19 }
  if ($version.Major -eq 22) { return $version.Minor -ge 12 }
  return $version.Major -gt 22
}

function Ensure-NodeCompatible {
  if (Test-NodeCompatible) { return }
  if (-not (Has 'winget')) {
    throw '3D Gen Studio uses Vite 8 and requires Node.js 20.19+, 22.12+, or newer. Install a compatible Node.js LTS release.'
  }
  Note 'Installing or upgrading compatible Node.js LTS'
  if (Test-WingetPackageInstalled 'OpenJS.NodeJS.LTS') {
    & winget upgrade --id OpenJS.NodeJS.LTS --exact --silent --accept-package-agreements --accept-source-agreements
  } else {
    & winget install --id OpenJS.NodeJS.LTS --exact --silent --accept-package-agreements --accept-source-agreements
  }
  $nodeExit = $LASTEXITCODE
  Refresh-Path
  if (-not (Test-NodeCompatible)) {
    throw "Compatible Node.js installation failed (winget exit $nodeExit). Vite 8 requires Node.js 20.19+, 22.12+, or newer."
  }
}

function Invoke-GitRetry {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory=$true)][string[]]$Arguments,
    [int]$Attempts = 4
  )
  for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
    & git -c http.version=HTTP/1.1 -c core.compression=0 @Arguments
    if ($LASTEXITCODE -eq 0) { return }
    if ($attempt -eq $Attempts) { throw "git failed after $Attempts attempts: git $($Arguments -join ' ')" }
    $delay = [int][Math]::Min(30, [Math]::Pow(2, $attempt))
    Write-Warning "git attempt $attempt failed; retrying in $delay seconds."
    Start-Sleep -Seconds $delay
  }
}

function Get-GitHead([string]$Path) {
  if (-not (Test-Path (Join-Path $Path '.git'))) { return '' }
  $value = (& git -C $Path rev-parse HEAD 2>$null | Out-String).Trim()
  if ($LASTEXITCODE -ne 0) { return '' }
  return $value
}

function Test-GitCheckout([string]$Path, [string]$Commit, [string[]]$RequiredFiles=@()) {
  if ((Get-GitHead $Path) -ne $Commit) { return $false }
  & git -C $Path cat-file -e "$Commit^{commit}" 2>$null
  if ($LASTEXITCODE -ne 0) { return $false }
  foreach ($relative in $RequiredFiles) {
    if (-not (Test-Path (Join-Path $Path $relative))) { return $false }
  }
  return $true
}

function Ensure-GitRepo([string]$Url, [string]$Path, [string]$Commit, [string[]]$RequiredFiles=@()) {
  if (Test-GitCheckout -Path $Path -Commit $Commit -RequiredFiles $RequiredFiles) { return }
  if (-not (Test-Path (Join-Path $Path '.git'))) {
    if (Test-Path $Path) { Remove-Item $Path -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
    Invoke-GitRetry -Arguments @('-C',$Path,'init')
    Invoke-GitRetry -Arguments @('-C',$Path,'remote','add','origin',$Url)
  } else {
    $origin = (& git -C $Path remote get-url origin 2>$null | Out-String).Trim()
    if (-not $origin) { Invoke-GitRetry -Arguments @('-C',$Path,'remote','add','origin',$Url) }
    elseif ($origin -ne $Url) { Invoke-GitRetry -Arguments @('-C',$Path,'remote','set-url','origin',$Url) }
  }
  Invoke-GitRetry -Arguments @('-C',$Path,'fetch','--depth=1','--no-tags','origin',$Commit)
  Invoke-GitRetry -Arguments @('-C',$Path,'checkout','--detach','--force','FETCH_HEAD')
  if (-not (Test-GitCheckout -Path $Path -Commit $Commit -RequiredFiles $RequiredFiles)) {
    throw "Pinned checkout failed validation for $Url at $Commit"
  }
}

function Ensure-Uv {
  if (Has 'uv') { return }
  Note 'Installing uv Python toolchain manager'
  Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
  if (-not (Has 'uv')) { throw 'uv installation failed.' }
}

function Ensure-Venv([string]$Path, [string]$PythonVersion) {
  uv python install $PythonVersion
  if ($LASTEXITCODE -ne 0) { throw "uv could not install Python $PythonVersion" }
  if (-not (Test-Path (Join-Path $Path 'Scripts\python.exe'))) {
    uv venv $Path --python $PythonVersion
    if ($LASTEXITCODE -ne 0) { throw "uv could not create environment $Path" }
  }
}

function UvPip {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(Mandatory=$true)][string[]]$Arguments
  )
  if (-not $Arguments -or $Arguments.Count -lt 1) { throw 'UvPip requires at least one argument.' }
  $uvArgs = @('pip', $Arguments[0], '--python', $Python)
  if ($Arguments.Count -gt 1) { $uvArgs += $Arguments[1..($Arguments.Count - 1)] }
  Write-Host ("uv " + ($uvArgs -join ' ')) -ForegroundColor DarkGray
  for ($attempt = 1; $attempt -le 3; $attempt++) {
    & uv @uvArgs
    if ($LASTEXITCODE -eq 0) { return }
    if ($attempt -eq 3) { throw "uv command failed for $Python after 3 attempts: $($uvArgs -join ' ')" }
    $delay = 5 * $attempt
    Write-Warning "uv attempt $attempt failed; retrying in $delay seconds without deleting the environment."
    Start-Sleep -Seconds $delay
  }
}

function Get-FileFingerprint([string]$Path) {
  if (-not (Test-Path $Path)) { return 'missing' }
  return (Get-FileHash -Algorithm SHA256 $Path).Hash.ToLowerInvariant()
}

function Test-PythonCommand([string]$Python,[string]$Code,[string[]]$Arguments=@()) {
  if (-not (Test-Path $Python)) { return $false }
  & $Python -c $Code @Arguments *> $null
  return ($LASTEXITCODE -eq 0)
}

function Test-JsonDocument([string]$Path) {
  if (-not (Test-Path $Path)) { return $false }
  try { Get-Content $Path -Raw | ConvertFrom-Json | Out-Null; return $true } catch { return $false }
}

function Test-JsonStatus([string]$Path,[string]$Expected) {
  if (-not (Test-JsonDocument $Path)) { return $false }
  try { return ([string](Get-Content $Path -Raw | ConvertFrom-Json).status -eq $Expected) } catch { return $false }
}

function Test-StudioJsDependencies([string]$Path) {
  if (-not (Test-Path (Join-Path $Path 'node_modules'))) { return $false }
  Push-Location $Path
  try {
    & node -e "require('express'); require('vite'); console.log('studio-js-ready')" *> $null
    return ($LASTEXITCODE -eq 0)
  } finally { Pop-Location }
}

function Test-StudioNodeRuntime([string]$Path) {
  if (-not (Test-StudioJsDependencies $Path)) { return $false }
  Push-Location $Path
  try {
    & node -e "require('express'); require('sqlite3'); console.log('studio-node-runtime-ok')" *> $null
    return ($LASTEXITCODE -eq 0)
  } finally { Pop-Location }
}

try {
  Invoke-InstallStage -StateRoot $StateRoot -Name '01-prerequisites' -AdoptExisting -Fingerprint 'prerequisites-v4-node-vite8' -Test {
    (Has 'git') -and (Test-NodeCompatible) -and (Has 'npm') -and [bool](Find-BlenderExecutable) -and (Has 'uv')
  } -Action {
    Note 'Checking prerequisites'
    if (-not $SkipPrerequisites) {
      if (-not (Has 'git')) { Ensure-WingetPackage 'Git.Git' 'Git' }
      Ensure-NodeCompatible
      if (-not (Find-BlenderExecutable)) { Ensure-WingetPackage 'BlenderFoundation.Blender' 'Blender' }
      Refresh-Path
    }
    if (-not (Has 'git')) { throw 'Git is missing.' }
    if (-not (Test-NodeCompatible)) { throw 'Node.js is missing or incompatible. Vite 8 requires Node.js 20.19+, 22.12+, or newer.' }
    if (-not (Has 'npm')) { throw 'npm is missing.' }
    if (-not (Find-BlenderExecutable)) { throw 'Blender is missing or blender.exe could not be discovered.' }
    Ensure-Uv
  }
  Write-Host "Blender: $(Find-BlenderExecutable)" -ForegroundColor DarkGray

  Note 'Synchronizing LowVRAM stack payload'
  robocopy $PackageRoot $AppRoot /MIR /XD '.git' 'thirdparty' 'envs' 'jobs' 'install-logs' 'proof' '__pycache__' /XF '*.pyc' 'local.json' | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }

  $ControlVenv = Join-Path $Envs 'control'
  $ControlPython = Join-Path $ControlVenv 'Scripts\python.exe'
  $ControlRequirements = Join-Path $AppRoot 'requirements-control.txt'
  Invoke-InstallStage -StateRoot $StateRoot -Name '02-control-environment' -AdoptExisting -Fingerprint "control-$((Get-FileFingerprint $ControlRequirements))" -Test {
    Test-PythonCommand $ControlPython 'import fastapi, uvicorn, trimesh, PIL, psutil, cv2, httpx'
  } -Action {
    Note 'Creating control service environment'
    Ensure-Venv $ControlVenv '3.11'
    UvPip -Python $ControlPython -Arguments @('install','-r',$ControlRequirements)
  }

  $Discovery = Join-Path $Proof 'discovery.json'
  Invoke-InstallStage -StateRoot $StateRoot -Name '03-local-discovery' -Fingerprint "discovery-$PackageVersion" -Test {
    if (-not (Test-JsonDocument $Discovery)) { return $false }
    $probe = Get-Content $Discovery -Raw | ConvertFrom-Json
    return [bool]($probe.blender_path -and (Test-Path $probe.blender_path))
  } -Action {
    Note 'Discovering existing ComfyUI and Blender'
    $discoveryJson = (& $ControlPython (Join-Path $AppRoot 'scripts\discover_local.py') | Out-String)
    if ($LASTEXITCODE -ne 0) { throw 'Local software discovery failed.' }
    Write-Utf8NoBom -Path $Discovery -Text $discoveryJson
  }
  $discovered = Get-Content $Discovery -Raw | ConvertFrom-Json

  $StudioRoot = Join-Path $ThirdParty '3DGenStudio'
  $StudioCommit = '98bdc4a3aa5e898251a2172c227f0d5f756d21d0'
  Invoke-InstallStage -StateRoot $StateRoot -Name '04-studio-source' -AdoptExisting -Fingerprint $StudioCommit -Test {
    Test-GitCheckout -Path $StudioRoot -Commit $StudioCommit -RequiredFiles @('package.json','server.js','python-server\main.py')
  } -Action {
    Note 'Installing pinned 3D Gen Studio source'
    Ensure-GitRepo 'https://github.com/visualbruno/3DGenStudio.git' $StudioRoot $StudioCommit @('package.json','server.js','python-server\main.py')
  }

  $StudioLock = Join-Path $StudioRoot 'package-lock.json'
  $NodeVersion = (& node --version | Out-String).Trim()
  $StudioNodeFingerprint = "studio-node-$StudioCommit-$NodeVersion-$((Get-FileFingerprint $StudioLock))"
  Invoke-InstallStage -StateRoot $StateRoot -Name '05-studio-node-runtime' -AdoptExisting -Fingerprint $StudioNodeFingerprint -Test {
    Test-StudioNodeRuntime $StudioRoot
  } -Action {
    Note 'Installing 3D Gen Studio Node runtime'
    Push-Location $StudioRoot
    try {
      if (-not (Test-StudioJsDependencies $StudioRoot)) {
        if (Test-Path 'package-lock.json') { npm ci --no-audit --no-fund --ignore-scripts }
        else { npm install --no-audit --no-fund --ignore-scripts }
        if ($LASTEXITCODE -ne 0) { throw '3D Gen Studio npm dependency installation failed.' }
      } else {
        Write-Host '[resume] Existing node_modules is intact; rebuilding only sqlite3.' -ForegroundColor DarkGreen
      }
      npm rebuild sqlite3 --foreground-scripts --ignore-scripts=false
      if ($LASTEXITCODE -ne 0) { throw '3D Gen Studio sqlite3 native rebuild failed.' }
    } finally { Pop-Location }
  }

  Invoke-InstallStage -StateRoot $StateRoot -Name '06-studio-frontend-build' -AdoptExisting -Fingerprint "studio-build-$StudioCommit-$((Get-FileFingerprint $StudioLock))" -Test {
    Test-Path (Join-Path $StudioRoot 'dist\index.html')
  } -Action {
    Note 'Building 3D Gen Studio frontend'
    Push-Location $StudioRoot
    try { npm run build; if ($LASTEXITCODE -ne 0) { throw '3D Gen Studio frontend build failed.' } }
    finally { Pop-Location }
  }

  $MeshRoot = Join-Path $StudioRoot 'python-server'
  $MeshVenv = Join-Path $MeshRoot '.venv'
  $MeshPython = Join-Path $MeshVenv 'Scripts\python.exe'
  $MeshRequirements = Join-Path $MeshRoot 'requirements.txt'
  $MeshFingerprint = "mesh-tools-$StudioCommit-$((Get-FileFingerprint $MeshRequirements))"
  Invoke-InstallStage -StateRoot $StateRoot -Name '07-mesh-tools-environment' -AdoptExisting -Fingerprint $MeshFingerprint -Test {
    Test-PythonCommand $MeshPython 'import fastapi, trimesh, numpy'
  } -Action {
    Note 'Installing 3D Gen Studio mesh tools in CPU-only mode'
    Ensure-Venv $MeshVenv '3.13'
    UvPip -Python $MeshPython -Arguments @('install','-r',$MeshRequirements)
  }

  $MvRoot = Join-Path $ThirdParty 'MV-Adapter'
  $MvCommit = '4277e0018232bac82bb2c103caf0893cedb711be'
  $MvVenv = Join-Path $Envs 'mv-adapter'
  $MvPython = Join-Path $MvVenv 'Scripts\python.exe'
  $MvProof = Join-Path $Proof 'mv-adapter-readiness.json'
  $MvReadinessLog = Join-Path $Logs 'mv-adapter-readiness.log'
  $MvVerifier = Join-Path $AppRoot 'scripts\verify_mv_adapter_env.py'
  $TestMvAdapter = {
    if (-not (Test-Path $MvPython) -or -not (Test-Path $MvVerifier)) { return $false }
    & $MvPython $MvVerifier --repo $MvRoot --proof $MvProof *> $MvReadinessLog
    return ($LASTEXITCODE -eq 0)
  }
  Invoke-InstallStage -StateRoot $StateRoot -Name '08-mv-adapter-environment' -AdoptExisting -Fingerprint "mv-adapter-$MvCommit-torch251-cu124-avatar-v5-single-opencv" -Test $TestMvAdapter -Action {
    Note 'Installing MV-Adapter SD2.1 isolated worker (Lane A)'
    Ensure-GitRepo 'https://github.com/huanngzh/MV-Adapter.git' $MvRoot $MvCommit @('mvadapter\__init__.py','scripts\inference_t2mv_sd.py')
    Ensure-Venv $MvVenv '3.11'
    UvPip -Python $MvPython -Arguments @('install','torch==2.5.1','torchvision==0.20.1','--index-url','https://download.pytorch.org/whl/cu124')
    UvPip -Python $MvPython -Arguments @('install','diffusers==0.31.0','transformers==4.46.3','accelerate==1.1.1','huggingface_hub==0.27.1','peft','safetensors','Pillow','numpy<2','omegaconf','einops','timm','kornia','scikit-image','sentencepiece','psutil','mediapipe==0.10.21','prettytable','tqdm')

    # All OpenCV wheel variants install the same cv2 namespace. Keeping more than
    # one is explicitly unsupported and can produce Windows DLL/import failures.
    $InstalledOpenCv = @(& $MvPython -c "import importlib.metadata as m; names=('opencv-python','opencv-python-headless','opencv-contrib-python','opencv-contrib-python-headless'); print(' '.join(n for n in names if any(d.metadata.get('Name','').lower()==n for d in m.distributions())))")
    if ($LASTEXITCODE -eq 0 -and $InstalledOpenCv.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace(($InstalledOpenCv -join ' '))) {
      $OpenCvNames = (($InstalledOpenCv -join ' ').Trim() -split '\s+')
      if ($OpenCvNames.Count -gt 0) { UvPip -Python $MvPython -Arguments (@('uninstall') + $OpenCvNames) }
    }
    UvPip -Python $MvPython -Arguments @('install','opencv-contrib-python==4.11.0.86')

    $MvSitePackages = (& $MvPython -c "import site; print(site.getsitepackages()[0])").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $MvSitePackages) { throw 'Could not resolve MV-Adapter site-packages directory.' }
    & $MvPython -c "from pathlib import Path; import sys; Path(sys.argv[1], 'lowvram3d_mv_adapter_repo.pth').write_text(sys.argv[2] + '\n', encoding='utf-8')" $MvSitePackages $MvRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not register MV-Adapter repository path.' }

    & $MvPython $MvVerifier --repo $MvRoot --proof $MvProof 2>&1 | Tee-Object -FilePath $MvReadinessLog
    if ($LASTEXITCODE -ne 0) { throw "MV-Adapter readiness verification failed. See $MvReadinessLog and $MvProof" }
  }

  $TripoRoot = Join-Path $ThirdParty 'TripoSR'
  $TripoCommit = '107cefdc244c39106fa830359024f6a2f1c78871'
  $TripoVenv = Join-Path $Envs 'triposr'
  $TripoPython = Join-Path $TripoVenv 'Scripts\python.exe'
  $TripoReady = Invoke-InstallStage -StateRoot $StateRoot -Name '09-triposr-optional-environment' -AdoptExisting -RetryDegraded:$RetryOptional -Fingerprint "triposr-$TripoCommit-cpu-mc-v3" -Optional -Test {
    Test-PythonCommand -Python $TripoPython -Code 'import sys; sys.path.insert(0, sys.argv[1]); import torch, tsr; from torchmcubes import marching_cubes; assert torch.cuda.is_available()' -Arguments @($TripoRoot)
  } -Action {
    Note 'Installing TripoSR emergency fallback (Lane C)'
    Ensure-GitRepo 'https://github.com/VAST-AI-Research/TripoSR.git' $TripoRoot $TripoCommit @('tsr\system.py','run.py')
    Ensure-Venv $TripoVenv '3.10'
    UvPip -Python $TripoPython -Arguments @('install','torch==2.2.2','torchvision==0.17.2','--index-url','https://download.pytorch.org/whl/cu121')
    UvPip -Python $TripoPython -Arguments @('install','numpy==1.26.4','scikit-image==0.24.0','omegaconf==2.3.0','Pillow==10.1.0','einops==0.7.0','transformers==4.35.0','trimesh==4.0.5','rembg','onnxruntime==1.19.2','huggingface-hub','imageio[ffmpeg]','xatlas==0.0.9','moderngl==5.10.0')
    & $TripoPython (Join-Path $AppRoot 'scripts\install_torchmcubes_cpu_shim.py') --proof (Join-Path $Proof 'triposr-marching-cubes.json')
    if ($LASTEXITCODE -ne 0) { throw 'TripoSR CPU marching-cubes verification failed.' }
  }
  $TripoCheckpoint = Read-StageCheckpoint -StateRoot $StateRoot -Name '09-triposr-optional-environment'
  $TripoFailure = if ($TripoReady) { '' } elseif ($TripoCheckpoint) { [string]$TripoCheckpoint.error } else { 'not installed' }
  if ($TripoFailure) { $TripoFailure | Set-Content -Encoding UTF8 (Join-Path $Proof 'TRIPOSR_UNAVAILABLE.txt') }

  $ConfigPath = Join-Path $AppRoot 'config\local.json'
  Invoke-InstallStage -StateRoot $StateRoot -Name '10-local-configuration' -Fingerprint "config-$PackageVersion-$TripoReady" -Test {
    Test-JsonDocument $ConfigPath
  } -Action {
    Note 'Writing local configuration'
    & $ControlPython (Join-Path $AppRoot 'scripts\write_local_config.py') --root $InstallRoot --discovery $Discovery --output $ConfigPath | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Local configuration generation failed.' }
    $cfg = Get-Content $ConfigPath -Raw | ConvertFrom-Json
    if ($null -eq $cfg.PSObject.Properties['extra'] -or $null -eq $cfg.extra) {
      Set-ObjectProperty -Object $cfg -Name 'extra' -Value ([pscustomobject]@{})
    }
    $tripoPythonValue = if ($TripoReady) { $TripoPython } else { '' }
    $tripoRootValue = if ($TripoReady) { $TripoRoot } else { '' }
    $tripoBackend = if ($TripoReady) { 'scikit-image-lewiner-cpu' } else { 'unavailable' }
    Set-ObjectProperty -Object $cfg -Name 'tripo_python' -Value $tripoPythonValue
    Set-ObjectProperty -Object $cfg -Name 'tripo_root' -Value $tripoRootValue
    Set-ObjectProperty -Object $cfg.extra -Name 'triposr_ready' -Value ([bool]$TripoReady)
    Set-ObjectProperty -Object $cfg.extra -Name 'triposr_backend' -Value $tripoBackend
    Set-ObjectProperty -Object $cfg.extra -Name 'triposr_failure' -Value $TripoFailure
    Set-ObjectProperty -Object $cfg -Name 'mv_adapter_python' -Value $MvPython
    Set-ObjectProperty -Object $cfg -Name 'mv_adapter_root' -Value $MvRoot
    Set-ObjectProperty -Object $cfg.extra -Name 'studio_root' -Value $StudioRoot
    Write-Utf8NoBom -Path $ConfigPath -Text ($cfg | ConvertTo-Json -Depth 20)
  }

  $SmokeProof = Join-Path $Proof 'service-smoke.json'
  Invoke-InstallStage -StateRoot $StateRoot -Name '11-runtime-service-smoke' -Fingerprint "service-smoke-$PackageVersion-$StudioNodeFingerprint-$MeshFingerprint" -Test {
    Test-JsonStatus $SmokeProof 'passed'
  } -Action {
    Note 'Smoke-testing installed local services'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $AppRoot 'scripts\windows\SMOKE-SERVICES.ps1') `
      -InstallRoot $InstallRoot -AppRoot $AppRoot -StudioRoot $StudioRoot `
      -ControlPython $ControlPython -MeshPython $MeshPython -ConfigPath $ConfigPath -OutputPath $SmokeProof
    if ($LASTEXITCODE -ne 0) { throw 'Installed service smoke test failed.' }
  }

  $ModelReceipt = Join-Path $Proof 'model-downloads.json'
  $ModelFingerprint = if ($SkipModels) { 'models-skipped-v2' } else { "models-sd21-mvadapter-birefnet-e2bf8e4-v4-tripo-$TripoReady" }
  Invoke-InstallStage -StateRoot $StateRoot -Name '12-model-cache' -AdoptExisting -RetryDegraded:$RetryOptional -Fingerprint $ModelFingerprint -Optional -Test {
    if ($SkipModels) { return (Test-JsonStatus $ModelReceipt 'skipped') }
    if (-not (Test-JsonStatus $ModelReceipt 'passed')) { return $false }
    $env:HF_HOME = Join-Path $InstallRoot 'models\huggingface'
    $verifyArgs = @((Join-Path $AppRoot 'scripts\prefetch_models.py'),'--cache',$env:HF_HOME,'--verify-only')
    if ($TripoReady) { $verifyArgs += '--include-triposr' }
    & $MvPython @verifyArgs *> $null
    return ($LASTEXITCODE -eq 0)
  } -Action {
    Note 'Prefetching model weights'
    $env:HF_HOME = Join-Path $InstallRoot 'models\huggingface'
    $modelArgs = @((Join-Path $AppRoot 'scripts\prefetch_models.py'),'--cache',$env:HF_HOME)
    if ($SkipModels) { $modelArgs += '--skip' }
    elseif ($TripoReady) { $modelArgs += '--include-triposr' }
    $modelJson = (& $MvPython @modelArgs | Out-String)
    Write-Utf8NoBom -Path $ModelReceipt -Text $modelJson
    if ($LASTEXITCODE -ne 0) { throw 'Model prefetch failed.' }
  }

  Invoke-InstallStage -StateRoot $StateRoot -Name '13-comfyui-bridge' -RetryDegraded:$RetryOptional -Fingerprint "comfy-bridge-$PackageVersion" -Optional -Test {
    if (-not $discovered.comfyui_path) { return $true }
    Test-Path (Join-Path $discovered.comfyui_path 'custom_nodes\LowVRAM3DStudio\nodes.py')
  } -Action {
    Note 'Installing optional ComfyUI bridge'
    if ($discovered.comfyui_path) {
      $nodeTarget = Join-Path $discovered.comfyui_path 'custom_nodes\LowVRAM3DStudio'
      New-Item -ItemType Directory -Force -Path $nodeTarget | Out-Null
      Copy-Item (Join-Path $AppRoot 'comfyui_nodes\LowVRAM3DStudio\*') $nodeTarget -Recurse -Force
    }
  }

  $Preflight = Join-Path $Proof 'preflight.json'
  Invoke-InstallStage -StateRoot $StateRoot -Name '14-package-verification' -Fingerprint "verify-$PackageVersion" -Test {
    Test-JsonDocument $Preflight
  } -Action {
    Note 'Running package verification'
    # tests import the top-level 'service', 'workers' and 'scripts' packages as well as
    # 'lowvram3d' from src, but 'discover -s tests' puts only the tests directory on sys.path.
    # Both roots must be on PYTHONPATH. (-t $AppRoot is not usable: it would require
    # tests/__init__.py, which the package deliberately does not ship.)
    $env:PYTHONPATH = ((Join-Path $AppRoot 'src') + [IO.Path]::PathSeparator + $AppRoot)
    & $ControlPython -m compileall -q (Join-Path $AppRoot 'src') (Join-Path $AppRoot 'service') (Join-Path $AppRoot 'workers') (Join-Path $AppRoot 'scripts')
    if ($LASTEXITCODE -ne 0) { throw 'Python compile verification failed.' }
    & $ControlPython -m unittest discover -s (Join-Path $AppRoot 'tests') -v
    if ($LASTEXITCODE -ne 0) { throw 'Unit verification failed.' }
    $env:LOWVRAM3D_PREFLIGHT = $Preflight
    & $ControlPython (Join-Path $AppRoot 'scripts\preflight.py')
    if ($LASTEXITCODE -ne 0) { throw 'Preflight verification failed.' }
  }

  $desktop = [Environment]::GetFolderPath('Desktop')
  Invoke-InstallStage -StateRoot $StateRoot -Name '15-shortcuts' -Fingerprint "shortcuts-$PackageVersion" -Test {
    (Test-Path (Join-Path $desktop 'LowVRAM 3D Studio.lnk')) -and (Test-Path (Join-Path $desktop 'Stop LowVRAM 3D Studio.lnk')) -and (Test-Path (Join-Path $desktop 'Resume Last 3D Job.lnk'))
  } -Action {
    Note 'Creating desktop shortcuts'
    $shell = New-Object -ComObject WScript.Shell
    foreach ($item in @(
      @{Name='LowVRAM 3D Studio'; Target=(Join-Path $AppRoot 'START-STUDIO.cmd')},
      @{Name='Stop LowVRAM 3D Studio'; Target=(Join-Path $AppRoot 'STOP-STUDIO.cmd')},
      @{Name='Resume Last 3D Job'; Target=(Join-Path $AppRoot 'RESUME-LAST-JOB.cmd')}
    )) {
      $shortcut = $shell.CreateShortcut((Join-Path $desktop ($item.Name + '.lnk')))
      $shortcut.TargetPath = $item.Target
      $shortcut.WorkingDirectory = $AppRoot
      $shortcut.Save()
    }
  }

  Set-Content -Encoding UTF8 (Join-Path $Proof 'INSTALL_OK.txt') "Installed $(Get-Date -Format o)`nVersion: $PackageVersion`nConfig: $ConfigPath"
  Remove-Item (Join-Path $Proof 'INSTALL_FAILED.txt') -Force -ErrorAction SilentlyContinue
  Write-InstallSummary -StateRoot $StateRoot -OutputPath (Join-Path $Proof 'install-summary.json')
  Note 'Installation complete'
  Write-Host 'Future reruns skip every verified completed stage and every recorded optional degradation.' -ForegroundColor Green
  Write-Host 'Launch from the desktop shortcut: LowVRAM 3D Studio' -ForegroundColor Green
} catch {
  $_ | Out-String | Set-Content -Encoding UTF8 (Join-Path $Proof 'INSTALL_FAILED.txt')
  Write-InstallSummary -StateRoot $StateRoot -OutputPath (Join-Path $Proof 'install-summary.json')
  Write-Host "Run CONTINUE-INSTALL.cmd again. Only the failed or unverified stage will run." -ForegroundColor Cyan
  Write-Host "Checkpoint status: $Proof\install-summary.json" -ForegroundColor Cyan
  Write-Error $_
  exit 1
} finally {
  Stop-Transcript | Out-Null
}
