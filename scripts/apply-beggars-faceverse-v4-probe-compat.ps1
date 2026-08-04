[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'scripts\run-beggars-faceverse-v4-preflight.ps1'
if (-not (Test-Path -LiteralPath $target)) {
    throw "FaceVerse preflight script is missing: $target"
}
$text = Get-Content -LiteralPath $target -Raw

$newProbeFunctions = @'
function Invoke-PythonProbe {
    param(
        [Parameter(Mandatory)][string]$Python,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [string]$SuccessMarker
    )
    if (-not (Test-Path -LiteralPath $Python)) { return $false }
    $previousPreference = $ErrorActionPreference
    $output = @()
    $exitCode = -1
    try {
        $ErrorActionPreference = 'SilentlyContinue'
        $output = @(& $Python @ArgumentList 2>&1)
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -eq 0) {
        foreach ($line in $output) { Write-Host ([string]$line) }
        if ($SuccessMarker) { Write-Host $SuccessMarker }
        return $true
    }
    Write-Host "PYTHON_PROBE_REJECTED=$Python EXIT=$exitCode"
    foreach ($line in $output | Select-Object -Last 12) {
        Write-Host ([string]$line)
    }
    return $false
}

function Test-Environment {
    return (Invoke-PythonProbe -Python $VenvPython -ArgumentList @(
        '-c',
        "import cv2,mediapipe,numpy,PIL,torch,torchvision,Cython; assert torch.__version__.startswith('2.5.1'); print('FACEVERSE_ENV_IMPORTS=PROVEN')"
    ))
}

function Test-Pip {
    return (Invoke-PythonProbe -Python $VenvPython -ArgumentList @('-m','pip','--version'))
}

function Test-FaceVerseModel {
    if (-not (Test-Path -LiteralPath $ModelPath)) { return $false }
    return (Invoke-PythonProbe -Python $VenvPython -ArgumentList @(
        '-c',
        "import numpy as np; d=np.load(r'$ModelPath',allow_pickle=True).item(); req={'meanshape','idBase','exBase','texBase','meantex','tri','ver_inds'}; missing=sorted(req-set(d)); assert not missing, missing; assert d['meanshape'].shape[0] > 10000; print('FACEVERSE_MODEL_STRUCTURE=PROVEN',d['meanshape'].shape,d['tri'].shape)"
    ))
}

function Test-FaceVerseCheckpoint {
    if (-not (Test-Path -LiteralPath $CheckpointPath)) { return $false }
    return (Invoke-PythonProbe -Python $VenvPython -ArgumentList @(
        '-c',
        "import torch; d=torch.load(r'$CheckpointPath',map_location='cpu'); assert isinstance(d,dict); state=d.get('state_dict',d); assert isinstance(state,dict) and len(state)>100; print('FACEVERSE_CHECKPOINT_STRUCTURE=PROVEN',len(state))"
    ))
}
'@

$probePattern = '(?s)function Test-Environment \{.*?\r?\n\}\r?\n\r?\nfunction Test-FaceVerseModel \{.*?\r?\n\}\r?\n\r?\nfunction Test-FaceVerseCheckpoint \{.*?\r?\n\}'
if (-not [regex]::IsMatch($text, $probePattern)) {
    throw 'Could not locate the FaceVerse probe functions for replacement.'
}
$text = [regex]::Replace($text, $probePattern, $newProbeFunctions.TrimEnd(), 1)

$venvPattern = '(?s)    \$basePython = Resolve-Python39\r?\n    if \(-not \(Test-Path -LiteralPath \$VenvPython\)\) \{.*?\r?\n    \}\r?\n    if \(-not \(Test-Environment\)\) \{'
$newVenv = @'
    $basePython = Resolve-Python39
    $uv = Get-Uv
    if (-not $uv) {
        throw 'uv.exe is unavailable for FaceVerse environment creation.'
    }
    if (-not (Test-Pip)) {
        if (Test-Path -LiteralPath $VenvRoot) {
            Remove-Item -LiteralPath $VenvRoot -Recurse -Force
        }
        New-Item -ItemType Directory -Path (Split-Path $VenvRoot) -Force | Out-Null
        Invoke-Native -FilePath $uv -ArgumentList @('venv','--seed','--python','3.9.21',$VenvRoot) -FailureMessage 'uv could not create and seed the FaceVerse Python 3.9 environment'
        if (-not (Test-Pip)) {
            throw "uv created the FaceVerse environment but pip is not functional: $VenvPython"
        }
        Write-Host "FACEVERSE_VENV_SEED=PROVEN PYTHON=$VenvPython"
    }
    if (-not (Test-Environment)) {
'@
if (-not [regex]::IsMatch($text, $venvPattern)) {
    throw 'Could not locate the FaceVerse environment block for seeded-venv replacement.'
}
$text = [regex]::Replace($text, $venvPattern, $newVenv.TrimEnd(), 1)

[System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $target), $text, [System.Text.UTF8Encoding]::new($false))

$tokens = $null
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path -LiteralPath $target), [ref]$tokens, [ref]$errors)
if (@($errors).Count -gt 0) {
    throw "Patched FaceVerse preflight failed PowerShell parsing: $($errors[0].Message)"
}
Write-Host 'FACEVERSE_PROBE_AND_SEED_COMPAT=PROVEN'
