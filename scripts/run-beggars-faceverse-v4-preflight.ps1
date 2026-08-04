[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$ExpectedRepository = 'organicoverlords/lowvram3d-studio'
$ExpectedBranch = 'agent/blender-beggars-scene-20260804'
$FaceVerseCommit = '19c67cc4d7234b1ea7d55a185a2cb55fd49bb877'
$FaceVerseArchiveUrl = "https://codeload.github.com/LizhenWangT/FaceVerse_v4/zip/$FaceVerseCommit"
$FaceVerseModelShare = 'https://1drv.ms/u/c/b8eab7b1820a6fa4/EWJOsgGxPMZDkl8xJ_QZB30BpcjNoMVGK9mnUPq5n9-lyw?e=4GvEs9'
$FaceVerseCheckpointShare = 'https://1drv.ms/u/c/b8eab7b1820a6fa4/ETfT_C9Oz1FFlykJdtj3h6MBR1KvQb5BYwesxFykH-7BZA?e=7ti1yj'
$LandmarkerUrl = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
$ReferenceClipUrl = 'https://media.tenor.com/e8wT4qAP6zwAAAPo/somebody-get-these-beggars-out-of-here-the-odyssey.mp4'
$GpuAbortThresholdMiB = 4800

$RunId = if ($env:GITHUB_RUN_ID) { $env:GITHUB_RUN_ID } else { (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss') }
$OutputRoot = "C:\AI\LowVRAM3D-benchmarks\beggars-scene\faceverse-v4-preflight-$RunId"
$PrivateRoot = Join-Path $OutputRoot '_reference_private'
$ArtifactRoot = Join-Path $OutputRoot 'artifact'
$CacheRoot = 'C:\AI\LowVRAM3D-cache\faceverse-v4'
$SourceRoot = Join-Path $CacheRoot "source-$FaceVerseCommit"
$VenvRoot = Join-Path $CacheRoot 'venv-py39-cu118'
$VenvPython = Join-Path $VenvRoot 'Scripts\python.exe'
$ModelRoot = Join-Path $CacheRoot 'models'
$ModelPath = Join-Path $ModelRoot 'faceverse_v4_2.npy'
$CheckpointPath = Join-Path $ModelRoot 'faceverse_resnet50.pth'
$LandmarkerPath = Join-Path $ModelRoot 'face_landmarker.task'
$Status = 'REJECTED'
$Failure = $null

function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [Parameter(Mandatory)][string]$FailureMessage,
        [string]$WorkingDirectory
    )
    $previous = Get-Location
    try {
        if ($WorkingDirectory) {
            Set-Location -LiteralPath $WorkingDirectory
        }
        & $FilePath @ArgumentList
        if ($LASTEXITCODE -ne 0) {
            throw "$FailureMessage (exit $LASTEXITCODE)"
        }
    }
    finally {
        Set-Location -LiteralPath $previous
    }
}

function Get-Curl {
    $command = Get-Command curl.exe -ErrorAction SilentlyContinue
    if (-not $command) {
        throw 'curl.exe is unavailable'
    }
    return $command.Source
}

function Get-Uv {
    $candidates = @(
        (Get-Command uv.exe -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source -ErrorAction SilentlyContinue),
        "$env:USERPROFILE\.local\bin\uv.exe",
        "$env:LOCALAPPDATA\Programs\uv\uv.exe",
        "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv*\uv.exe"
    ) | Where-Object { $_ }
    foreach ($candidate in $candidates) {
        if ($candidate -like '*`**') {
            $resolved = Get-ChildItem -Path $candidate -File -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($resolved) { return $resolved.FullName }
        }
        elseif (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function Resolve-Python39 {
    $launcher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($launcher) {
        $resolved = (& $launcher.Source -3.9 -c 'import sys; print(sys.executable)' 2>$null | Out-String).Trim()
        if ($LASTEXITCODE -eq 0 -and $resolved -and (Test-Path -LiteralPath $resolved)) {
            return $resolved
        }
    }

    $uv = Get-Uv
    if (-not $uv) {
        throw 'Python 3.9 is unavailable and uv.exe was not found to provision it.'
    }
    Invoke-Native -FilePath $uv -ArgumentList @('python','install','3.9.21') -FailureMessage 'uv could not install Python 3.9.21'
    $resolved = (& $uv python find 3.9 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $resolved -or -not (Test-Path -LiteralPath $resolved)) {
        throw "uv installed Python 3.9 but could not resolve its executable: $resolved"
    }
    return $resolved
}

function Get-OneDriveApiUrl {
    param([Parameter(Mandatory)][string]$ShareUrl)
    $bytes = [System.Text.Encoding]::UTF8.GetBytes($ShareUrl)
    $encoded = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('/','_').Replace('+','-')
    return "https://api.onedrive.com/v1.0/shares/u!$encoded/root/content"
}

function Download-File {
    param(
        [Parameter(Mandatory)][string[]]$Urls,
        [Parameter(Mandatory)][string]$Destination,
        [Parameter(Mandatory)][long]$MinimumBytes,
        [Parameter(Mandatory)][string]$Label
    )
    $curl = Get-Curl
    New-Item -ItemType Directory -Path (Split-Path $Destination) -Force | Out-Null
    $temporary = "$Destination.partial"
    Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
    foreach ($url in $Urls) {
        Write-Host "DOWNLOAD_ATTEMPT=$Label URL=$url"
        & $curl -L --fail --retry 3 --retry-delay 2 --connect-timeout 30 --output $temporary $url
        if ($LASTEXITCODE -ne 0) {
            Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
            continue
        }
        $length = (Get-Item -LiteralPath $temporary).Length
        if ($length -lt $MinimumBytes) {
            Write-Host "DOWNLOAD_REJECTED=$Label BYTES=$length"
            Remove-Item -LiteralPath $temporary -Force -ErrorAction SilentlyContinue
            continue
        }
        Move-Item -LiteralPath $temporary -Destination $Destination -Force
        Write-Host "DOWNLOAD_PROVEN=$Label BYTES=$length"
        return
    }
    throw "Could not download a valid $Label file"
}

function Test-FaceVerseSource {
    foreach ($relative in @(
        'run.py',
        'faceversev4\__init__.py',
        'faceversev4\FaceVerse_networks.py',
        'faceversev4\FaceVerseModel_torch.py',
        'Sim3DR\renderer.py',
        'Sim3DR\Sim3DR_Cython.cp39-win_amd64.pyd'
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $SourceRoot $relative))) {
            return $false
        }
    }
    return $true
}

function Test-Environment {
    if (-not (Test-Path -LiteralPath $VenvPython)) { return $false }
    & $VenvPython -c "import cv2,mediapipe,numpy,PIL,torch,torchvision,Cython; assert torch.__version__.startswith('2.5.1'); print('FACEVERSE_ENV_IMPORTS=PROVEN')" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Test-FaceVerseModel {
    if (-not (Test-Path -LiteralPath $ModelPath)) { return $false }
    & $VenvPython -c "import numpy as np; d=np.load(r'$ModelPath',allow_pickle=True).item(); req={'meanshape','idBase','exBase','texBase','meantex','tri','ver_inds'}; missing=sorted(req-set(d)); assert not missing, missing; assert d['meanshape'].shape[0] > 10000; print('FACEVERSE_MODEL_STRUCTURE=PROVEN',d['meanshape'].shape,d['tri'].shape)" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Test-FaceVerseCheckpoint {
    if (-not (Test-Path -LiteralPath $CheckpointPath)) { return $false }
    & $VenvPython -c "import torch; d=torch.load(r'$CheckpointPath',map_location='cpu'); assert isinstance(d,dict) and len(d)>100; print('FACEVERSE_CHECKPOINT_STRUCTURE=PROVEN',len(d))" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Write-FinalReceipt {
    New-Item -ItemType Directory -Path $ArtifactRoot -Force | Out-Null
    [ordered]@{
        classification = $Status
        failure = $Failure
        route = 'FACEVERSE_V4_RESNET50_SINGLE_FRAME_FULL_HEAD'
        repository = $env:GITHUB_REPOSITORY
        branch = $env:GITHUB_REF_NAME
        head_sha = $env:GITHUB_SHA
        workflow_run_id = $env:GITHUB_RUN_ID
        workflow_run_url = if ($env:GITHUB_RUN_ID) { "https://github.com/$env:GITHUB_REPOSITORY/actions/runs/$env:GITHUB_RUN_ID" } else { $null }
        output_root = $OutputRoot
        source_commit = $FaceVerseCommit
        reference_media_upload_scope = 'EXCLUDED'
        reference_media_deleted = -not (Test-Path -LiteralPath $PrivateRoot)
        recorded_at = (Get-Date).ToUniversalTime().ToString('o')
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $ArtifactRoot 'workflow_final_receipt.json') -Encoding utf8
}

try {
    if ($env:GITHUB_REPOSITORY -and $env:GITHUB_REPOSITORY -ne $ExpectedRepository) {
        throw "Repository mismatch: $env:GITHUB_REPOSITORY"
    }
    if ($env:GITHUB_REF_NAME -and $env:GITHUB_REF_NAME -ne $ExpectedBranch) {
        throw "Branch mismatch: $env:GITHUB_REF_NAME"
    }

    if (Test-Path -LiteralPath $OutputRoot) {
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $PrivateRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $ArtifactRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $CacheRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $ModelRoot -Force | Out-Null
    if ($env:GITHUB_ENV) {
        "FACEVERSE_V4_OUTPUT=$ArtifactRoot" | Add-Content -LiteralPath $env:GITHUB_ENV
    }

    $gpuLine = (& nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>$null | Out-String).Trim()
    $gpuUsed = 0
    if ($gpuLine) {
        foreach ($line in $gpuLine -split "`r?`n") {
            $value = 0
            if ([int]::TryParse($line.Trim(), [ref]$value)) { $gpuUsed += $value }
        }
    }
    Write-Host "GPU_USED_MIB=$gpuUsed"
    Write-Host "GPU_ABORT_THRESHOLD_MIB=$GpuAbortThresholdMiB"
    if ($gpuUsed -ge $GpuAbortThresholdMiB) {
        throw "GPU_BUSY: used memory $gpuUsed MiB is at or above the $GpuAbortThresholdMiB MiB threshold"
    }

    if (-not (Test-FaceVerseSource)) {
        if (Test-Path -LiteralPath $SourceRoot) {
            Remove-Item -LiteralPath $SourceRoot -Recurse -Force
        }
        $archive = Join-Path $PrivateRoot 'faceverse-v4-source.zip'
        $extractRoot = Join-Path $PrivateRoot 'faceverse-v4-extract'
        Download-File -Urls @($FaceVerseArchiveUrl) -Destination $archive -MinimumBytes 1000000 -Label 'FaceVerse v4 source archive'
        Expand-Archive -LiteralPath $archive -DestinationPath $extractRoot -Force
        $extracted = Get-ChildItem -LiteralPath $extractRoot -Directory | Select-Object -First 1
        if (-not $extracted) { throw 'FaceVerse v4 source archive did not contain a source directory' }
        Move-Item -LiteralPath $extracted.FullName -Destination $SourceRoot
        if (-not (Test-FaceVerseSource)) {
            throw 'Pinned FaceVerse v4 source cache failed validation after extraction'
        }
    }
    Write-Host "FACEVERSE_SOURCE_COMMIT=$FaceVerseCommit"
    Write-Host 'FACEVERSE_SOURCE_CACHE=PROVEN'

    $basePython = Resolve-Python39
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        New-Item -ItemType Directory -Path (Split-Path $VenvRoot) -Force | Out-Null
        Invoke-Native -FilePath $basePython -ArgumentList @('-m','venv',$VenvRoot) -FailureMessage 'Could not create FaceVerse Python 3.9 environment'
    }
    if (-not (Test-Environment)) {
        Invoke-Native -FilePath $VenvPython -ArgumentList @('-m','pip','install','--disable-pip-version-check','--upgrade','pip','setuptools','wheel') -FailureMessage 'Could not bootstrap FaceVerse Python environment'
        Invoke-Native -FilePath $VenvPython -ArgumentList @('-m','pip','install','--disable-pip-version-check','--index-url','https://download.pytorch.org/whl/cu118','torch==2.5.1','torchvision==0.20.1') -FailureMessage 'Could not install pinned CUDA PyTorch for FaceVerse'
        Invoke-Native -FilePath $VenvPython -ArgumentList @('-m','pip','install','--disable-pip-version-check','numpy==1.26.4','cython==3.0.11','opencv-python==4.10.0.84','Pillow==10.2.0','mediapipe==0.10.20') -FailureMessage 'Could not install pinned FaceVerse runtime dependencies'
        if (-not (Test-Environment)) {
            throw 'Pinned FaceVerse Python environment failed import validation after installation'
        }
    }
    Write-Host 'FACEVERSE_RUNTIME_ENV=PROVEN'

    & $VenvPython -c "import sys; sys.path.insert(0,r'$SourceRoot'); import Sim3DR,faceversev4; print('FACEVERSE_SOURCE_IMPORT=PROVEN')"
    if ($LASTEXITCODE -ne 0) {
        Invoke-Native -FilePath $VenvPython -ArgumentList @('setup.py','build_ext','--inplace') -FailureMessage 'Could not compile Sim3DR for FaceVerse' -WorkingDirectory (Join-Path $SourceRoot 'Sim3DR')
        & $VenvPython -c "import sys; sys.path.insert(0,r'$SourceRoot'); import Sim3DR,faceversev4; print('FACEVERSE_SOURCE_IMPORT=PROVEN')"
        if ($LASTEXITCODE -ne 0) { throw 'FaceVerse source import still fails after Sim3DR build' }
    }

    if (-not (Test-FaceVerseModel)) {
        Remove-Item -LiteralPath $ModelPath -Force -ErrorAction SilentlyContinue
        $apiUrl = Get-OneDriveApiUrl -ShareUrl $FaceVerseModelShare
        Download-File -Urls @("$FaceVerseModelShare&download=1", $apiUrl) -Destination $ModelPath -MinimumBytes 1000000 -Label 'FaceVerse v4 model'
        if (-not (Test-FaceVerseModel)) { throw 'FaceVerse v4 model failed structural validation' }
    }
    if (-not (Test-FaceVerseCheckpoint)) {
        Remove-Item -LiteralPath $CheckpointPath -Force -ErrorAction SilentlyContinue
        $apiUrl = Get-OneDriveApiUrl -ShareUrl $FaceVerseCheckpointShare
        Download-File -Urls @("$FaceVerseCheckpointShare&download=1", $apiUrl) -Destination $CheckpointPath -MinimumBytes 1000000 -Label 'FaceVerse v4 ResNet50 checkpoint'
        if (-not (Test-FaceVerseCheckpoint)) { throw 'FaceVerse v4 checkpoint failed structural validation' }
    }
    if (-not (Test-Path -LiteralPath $LandmarkerPath) -or (Get-Item -LiteralPath $LandmarkerPath).Length -lt 1000000) {
        Remove-Item -LiteralPath $LandmarkerPath -Force -ErrorAction SilentlyContinue
        Download-File -Urls @($LandmarkerUrl) -Destination $LandmarkerPath -MinimumBytes 1000000 -Label 'MediaPipe face landmarker'
    }
    Write-Host 'FACEVERSE_MODEL_FILES=PROVEN'

    $clipPath = Join-Path $PrivateRoot 'reference_clip.mp4'
    $keyframePath = Join-Path $PrivateRoot 'selected_keyframe.png'
    Download-File -Urls @($ReferenceClipUrl) -Destination $clipPath -MinimumBytes 50000 -Label 'bounded meme reference clip'
    $extractCode = @"
import cv2
clip=r'$clipPath'
out=r'$keyframePath'
cap=cv2.VideoCapture(clip)
if not cap.isOpened(): raise SystemExit('OpenCV could not open reference clip')
cap.set(cv2.CAP_PROP_POS_FRAMES,19)
ok,frame=cap.read()
cap.release()
if not ok or frame is None: raise SystemExit('Could not extract source frame 19')
if not cv2.imwrite(out,frame): raise SystemExit('Could not write selected keyframe')
print('FACEVERSE_PRIVATE_KEYFRAME=PROVEN',frame.shape)
"@
    & $VenvPython -c $extractCode
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $keyframePath)) {
        throw 'Could not extract the bounded private FaceVerse keyframe'
    }

    $env:FACEVERSE_SOURCE_COMMIT = $FaceVerseCommit
    Invoke-Native -FilePath $VenvPython -ArgumentList @(
        'tools\beggars_scene\run_faceverse_v4_preflight.py',
        '--faceverse-root',$SourceRoot,
        '--model-npy',$ModelPath,
        '--checkpoint',$CheckpointPath,
        '--landmarker',$LandmarkerPath,
        '--input-image',$keyframePath,
        '--output-dir',$ArtifactRoot,
        '--device','auto'
    ) -FailureMessage 'FaceVerse v4 single-frame proof failed'

    foreach ($name in @(
        'faceverse_v4_render.png',
        'faceverse_v4_depth.png',
        'faceverse_v4_colored_head.ply',
        'faceverse_v4_coefficients.npz',
        'faceverse_v4_report.json'
    )) {
        $path = Join-Path $ArtifactRoot $name
        if (-not (Test-Path -LiteralPath $path) -or (Get-Item -LiteralPath $path).Length -le 0) {
            throw "FaceVerse v4 proof output is missing or empty: $name"
        }
    }

    $Status = 'USER_VISUAL_REVIEW_REQUIRED'
    Write-Host 'FACEVERSE_V4_SINGLE_FRAME_PROOF=PROVEN'
}
catch {
    $Failure = $_.Exception.Message
    Write-Error $Failure
}
finally {
    if (Test-Path -LiteralPath $PrivateRoot) {
        Remove-Item -LiteralPath $PrivateRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-FinalReceipt
}

if ($Failure) { exit 1 }
exit 0
