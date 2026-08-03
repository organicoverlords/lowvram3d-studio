[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$controlPython = "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\control\Scripts\python.exe"
$venvRoot = 'C:\AI\LowVRAM3D-cache\beggars-scene-v2\venv'
$venvPython = Join-Path $venvRoot 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $controlPython)) {
    throw "Control Python is missing: $controlPython"
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    New-Item -ItemType Directory -Path (Split-Path $venvRoot) -Force | Out-Null
    & $controlPython -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not create the cached reconstruction environment.'
    }
}

$baseHealthy = $false
& $venvPython -c "import cv2,numpy,onnxruntime,onnx,scipy,skimage,yaml,gdown,imageio_ffmpeg; print('BASE_RUNTIME=PROVEN')" 2>$null
$baseHealthy = ($LASTEXITCODE -eq 0)

if (-not $baseHealthy) {
    & $venvPython -m pip install --disable-pip-version-check --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not bootstrap the cached reconstruction environment.'
    }
    & $venvPython -m pip install --disable-pip-version-check `
        numpy==1.26.4 `
        opencv-python-headless==4.10.0.84 `
        pyyaml==6.0.2 `
        onnxruntime==1.19.2 `
        onnx==1.16.2 `
        scipy==1.13.1 `
        scikit-image==0.24.0 `
        yt-dlp==2026.7.4 `
        gdown==5.2.0 `
        imageio-ffmpeg==0.5.1
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not install the base reconstruction dependencies.'
    }
}

& $venvPython -m pip install --disable-pip-version-check `
    matplotlib==3.9.2 `
    imageio==2.36.1 `
    tqdm==4.67.1 `
    cython==3.0.11
if ($LASTEXITCODE -ne 0) {
    throw 'Could not install the pinned 3DDFA runtime dependencies.'
}

$env:MPLBACKEND = 'Agg'
if ($env:GITHUB_ENV) {
    'MPLBACKEND=Agg' | Add-Content -LiteralPath $env:GITHUB_ENV
}

& $venvPython -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot, imageio, tqdm, Cython; print('THREEDDFA_RUNTIME=PROVEN')"
if ($LASTEXITCODE -ne 0) {
    throw 'Pinned 3DDFA runtime import validation failed.'
}

Write-Host 'BEGGARS_V2_RUNTIME_DEPENDENCIES=PROVEN'
