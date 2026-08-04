[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$ExpectedRepository = 'organicoverlords/lowvram3d-studio'
$ExpectedBranch = 'agent/blender-beggars-scene-20260804'
$ThreeDdfaCommit = '1b6c67601abffc1e9f248b291708aef0e43b55ae'
$ThreeDdfaArchiveUrl = "https://codeload.github.com/cleardusk/3DDFA_V2/zip/$ThreeDdfaCommit"
$ThreeDdfaModelUrl = 'https://drive.google.com/uc?id=1YpO1KfXvJHRmCBkErNa62dHm-CUjsoIk'
$YuNetUrl = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
$YuNetBytes = 232589L
$YuNetSha256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'

$RunId = if ($env:GITHUB_RUN_ID) { $env:GITHUB_RUN_ID } else { (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss') }
$OutputRoot = "C:\AI\LowVRAM3D-benchmarks\beggars-scene\canonical-diagnostic-$RunId"
$PrivateRoot = Join-Path $OutputRoot '_reference_private'
$ArtifactRoot = Join-Path $OutputRoot 'artifact'
$CacheRoot = 'C:\AI\LowVRAM3D-cache\beggars-scene-v2'
$ModelCacheRoot = Join-Path $CacheRoot 'models'
$ThreeDdfaRoot = Join-Path $ModelCacheRoot "3DDFA_V2-$ThreeDdfaCommit"
$YuNetPath = Join-Path $ModelCacheRoot 'face_detection_yunet_2023mar.onnx'
$VenvPython = Join-Path $CacheRoot 'venv\Scripts\python.exe'
$Blender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$Status = 'REJECTED'
$Failure = $null

function Invoke-Native {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [Parameter(Mandatory)][string]$FailureMessage
    )
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "$FailureMessage (exit $LASTEXITCODE)"
    }
}

function Test-ThreeDdfaCache {
    foreach ($relative in @(
        'TDDFA_ONNX.py',
        'configs\mb1_120x120.yml',
        'configs\param_mean_std_62d_120x120.pkl',
        'configs\bfm_noneck_v3.pkl',
        'configs\tri.pkl',
        'weights\mb1_120x120.onnx'
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $ThreeDdfaRoot $relative))) {
            return $false
        }
    }
    return ((Get-Item -LiteralPath (Join-Path $ThreeDdfaRoot 'weights\mb1_120x120.onnx')).Length -gt 1000000)
}

function Write-FinalReceipt {
    New-Item -ItemType Directory -Path $ArtifactRoot -Force | Out-Null
    [ordered]@{
        classification = $Status
        failure = $Failure
        repository = $env:GITHUB_REPOSITORY
        branch = $env:GITHUB_REF_NAME
        head_sha = $env:GITHUB_SHA
        workflow_run_id = $env:GITHUB_RUN_ID
        workflow_run_url = if ($env:GITHUB_RUN_ID) { "https://github.com/$env:GITHUB_REPOSITORY/actions/runs/$env:GITHUB_RUN_ID" } else { $null }
        output_root = $OutputRoot
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
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw "Cached reconstruction Python is missing: $VenvPython"
    }
    if (-not (Test-Path -LiteralPath $Blender)) {
        throw "Blender is missing: $Blender"
    }
    $curlCommand = Get-Command curl.exe -ErrorAction SilentlyContinue
    if (-not $curlCommand) {
        throw 'curl.exe is unavailable'
    }
    $curl = $curlCommand.Source

    if (Test-Path -LiteralPath $OutputRoot) {
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $PrivateRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $ArtifactRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $ModelCacheRoot -Force | Out-Null
    "CANONICAL_DIAGNOSTIC_OUTPUT=$ArtifactRoot" | Add-Content -LiteralPath $env:GITHUB_ENV

    Invoke-Native -FilePath $VenvPython -ArgumentList @(
        '-m','py_compile',
        'tools\beggars_scene\prepare_reference_sequence.py',
        'tools\beggars_scene\build_canonical_face_sequence.py',
        'blender\diagnose_beggars_canonical_face.py'
    ) -FailureMessage 'Canonical diagnostic Python compilation failed'

    if (-not (Test-ThreeDdfaCache)) {
        if (Test-Path -LiteralPath $ThreeDdfaRoot) {
            Remove-Item -LiteralPath $ThreeDdfaRoot -Recurse -Force
        }
        $archive = Join-Path $PrivateRoot '3ddfa.zip'
        $extractRoot = Join-Path $PrivateRoot '3ddfa_extract'
        Invoke-Native -FilePath $curl -ArgumentList @('-L','--fail','--retry','3','--output',$archive,$ThreeDdfaArchiveUrl) -FailureMessage 'Could not download pinned 3DDFA source archive'
        if ((Get-Item -LiteralPath $archive).Length -lt 500000) {
            throw '3DDFA source archive is implausibly small'
        }
        Expand-Archive -LiteralPath $archive -DestinationPath $extractRoot -Force
        $extracted = Get-ChildItem -LiteralPath $extractRoot -Directory | Select-Object -First 1
        if (-not $extracted) {
            throw '3DDFA archive did not contain a source directory'
        }
        Move-Item -LiteralPath $extracted.FullName -Destination $ThreeDdfaRoot
        New-Item -ItemType Directory -Path (Join-Path $ThreeDdfaRoot 'weights') -Force | Out-Null
        Invoke-Native -FilePath $VenvPython -ArgumentList @('-m','gdown',$ThreeDdfaModelUrl,'-O',(Join-Path $ThreeDdfaRoot 'weights\mb1_120x120.onnx')) -FailureMessage 'Could not download pinned 3DDFA ONNX model'
        if (-not (Test-ThreeDdfaCache)) {
            throw 'Pinned 3DDFA cache failed validation after installation'
        }
    }
    Write-Host 'CANONICAL_3DDFA_CACHE=PROVEN'

    $yunetHealthy = $false
    if (Test-Path -LiteralPath $YuNetPath) {
        $yunetInfo = Get-Item -LiteralPath $YuNetPath
        $yunetHash = (Get-FileHash -LiteralPath $YuNetPath -Algorithm SHA256).Hash.ToLowerInvariant()
        $yunetHealthy = ($yunetInfo.Length -eq $YuNetBytes -and $yunetHash -eq $YuNetSha256)
    }
    if (-not $yunetHealthy) {
        Remove-Item -LiteralPath $YuNetPath -Force -ErrorAction SilentlyContinue
        Invoke-Native -FilePath $curl -ArgumentList @('-L','--fail','--retry','3','--output',$YuNetPath,$YuNetUrl) -FailureMessage 'Could not download YuNet model'
        $yunetInfo = Get-Item -LiteralPath $YuNetPath
        $yunetHash = (Get-FileHash -LiteralPath $YuNetPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($yunetInfo.Length -ne $YuNetBytes -or $yunetHash -ne $YuNetSha256) {
            throw "YuNet integrity mismatch: bytes=$($yunetInfo.Length) sha256=$yunetHash"
        }
    }
    Write-Host 'CANONICAL_YUNET_CACHE=PROVEN'

    $ffmpeg = (& $VenvPython -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" | Out-String).Trim()
    if (-not (Test-Path -LiteralPath $ffmpeg)) {
        throw "FFmpeg is missing: $ffmpeg"
    }

    $configPath = 'configs\scene\beggars_banquet_recreation_v1.json'
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $downloadRoot = Join-Path $PrivateRoot 'download'
    New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null
    $accepted = $null
    $acceptedTitle = $null
    foreach ($candidate in @($config.source_context.reference_clip_candidates)) {
        Get-ChildItem -LiteralPath $downloadRoot -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        Write-Host "CANONICAL_REFERENCE_ATTEMPT=$candidate"
        & $VenvPython -m yt_dlp --no-playlist --max-downloads 1 --match-filter 'duration > 1 & duration < 30' --ffmpeg-location $ffmpeg --merge-output-format mp4 --write-info-json --no-write-thumbnail --format 'bv*[height<=720]+ba/b[height<=720]/b' --output (Join-Path $downloadRoot 'source.%(ext)s') $candidate
        if ($LASTEXITCODE -ne 0) {
            Write-Host "CANONICAL_REFERENCE_ATTEMPT_FAILED=$candidate"
            continue
        }
        $video = Get-ChildItem -LiteralPath $downloadRoot -File | Where-Object { $_.Extension -in '.mp4','.mkv','.webm','.mov' } | Sort-Object Length -Descending | Select-Object -First 1
        $infoFile = Get-ChildItem -LiteralPath $downloadRoot -Filter '*.info.json' -File | Select-Object -First 1
        if (-not $video -or -not $infoFile) { continue }
        $info = Get-Content -LiteralPath $infoFile.FullName -Raw | ConvertFrom-Json
        $searchable = "$($info.title) $($info.description) $($info.uploader)"
        $duration = [double]$info.duration
        if ($duration -gt 1 -and $duration -lt 30 -and $searchable -match '(?i)(beggar|odyssey|pattinson|antinous|reaction)') {
            $accepted = $video.FullName
            $acceptedTitle = [string]$info.title
            break
        }
    }
    if (-not $accepted) {
        throw 'REFERENCE_CLIP_BLOCKED: no bounded candidate could be downloaded and identity-validated'
    }
    Write-Host 'CANONICAL_REFERENCE_CLIP=PROVEN'
    Write-Host "CANONICAL_REFERENCE_TITLE=$acceptedTitle"

    $frames = Join-Path $PrivateRoot 'frames'
    $tracked = Join-Path $PrivateRoot 'face_sequence.npz'
    $keyframe = Join-Path $PrivateRoot 'selected_keyframe.png'
    $trackedReport = Join-Path $ArtifactRoot 'reference_reconstruction_report.json'
    Invoke-Native -FilePath $VenvPython -ArgumentList @(
        'tools\beggars_scene\prepare_reference_sequence.py',
        '--clip',$accepted,
        '--third-party-root',$ThreeDdfaRoot,
        '--yunet-model',$YuNetPath,
        '--frames-dir',$frames,
        '--output-npz',$tracked,
        '--output-report',$trackedReport,
        '--keyframe-output',$keyframe,
        '--target-fps','12',
        '--max-seconds','8',
        '--smoothing','0.68'
    ) -FailureMessage 'Tracked face reconstruction failed'
    if (-not (Test-Path -LiteralPath $tracked) -or (Get-Item -LiteralPath $tracked).Length -lt 1000000) {
        throw 'Tracked face sequence is missing or implausibly small'
    }
    Write-Host 'CANONICAL_TRACKED_RECONSTRUCTION=PROVEN'

    $canonical = Join-Path $PrivateRoot 'canonical_face_sequence.npz'
    $canonicalReport = Join-Path $ArtifactRoot 'canonical_report.json'
    Invoke-Native -FilePath $VenvPython -ArgumentList @(
        'tools\beggars_scene\build_canonical_face_sequence.py',
        '--input-npz',$tracked,
        '--third-party-root',$ThreeDdfaRoot,
        '--bfm-pkl',(Join-Path $ThreeDdfaRoot 'configs\bfm_noneck_v3.pkl'),
        '--output-npz',$canonical,
        '--output-report',$canonicalReport
    ) -FailureMessage 'Canonical pose-decoupling failed'
    if (-not (Test-Path -LiteralPath $canonical) -or (Get-Item -LiteralPath $canonical).Length -lt 1000000) {
        throw 'Canonical face sequence is missing or implausibly small'
    }
    Write-Host 'CANONICAL_POSE_DECOUPLING=PROVEN'

    Invoke-Native -FilePath $Blender -ArgumentList @(
        '--background','--factory-startup',
        '--python','blender\diagnose_beggars_canonical_face.py',
        '--',
        '--sequence',$canonical,
        '--output-dir',$ArtifactRoot
    ) -FailureMessage 'Canonical Blender still diagnostic failed'

    foreach ($name in @(
        'canonical_01_face_only.png',
        'canonical_02_features_no_hair.png',
        'canonical_03_complete_neutral.png',
        'canonical_04_complete_pose.png',
        'canonical_05_complete_grin.png',
        'diagnostic.json',
        'canonical_report.json',
        'reference_reconstruction_report.json'
    )) {
        if (-not (Test-Path -LiteralPath (Join-Path $ArtifactRoot $name))) {
            throw "Canonical diagnostic output is missing: $name"
        }
    }

    $Status = 'USER_VISUAL_REVIEW_REQUIRED'
    Write-Host 'CANONICAL_STILL_DIAGNOSTIC=PROVEN'
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

if ($Failure) {
    exit 1
}
exit 0
