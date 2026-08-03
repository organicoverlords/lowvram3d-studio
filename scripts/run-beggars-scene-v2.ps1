[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$ExpectedRepository = 'organicoverlords/lowvram3d-studio'
$ExpectedBranch = 'agent/blender-beggars-scene-20260804'
$ThreeDdfaCommit = '1b6c67601abffc1e9f248b291708aef0e43b55ae'
$ThreeDdfaArchiveUrl = "https://codeload.github.com/cleardusk/3DDFA_V2/zip/$ThreeDdfaCommit"
$YuNetUrl = 'https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
$YuNetBytes = 232589L
$YuNetSha256 = '8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4'
$OutputRoot = "C:\AI\LowVRAM3D-benchmarks\beggars-scene\run-$env:GITHUB_RUN_ID"
$PrivateRoot = Join-Path $OutputRoot '_reference_private'
$ArtifactRoot = Join-Path $OutputRoot 'artifact'
$CacheRoot = 'C:\AI\LowVRAM3D-cache\beggars-scene-v2'
$VenvRoot = Join-Path $CacheRoot 'venv'
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

function Write-FinalReceipt {
    New-Item -ItemType Directory -Path $ArtifactRoot -Force | Out-Null
    [ordered]@{
        classification = $Status
        failure = $Failure
        repository = $env:GITHUB_REPOSITORY
        branch = $env:GITHUB_REF_NAME
        head_sha = $env:GITHUB_SHA
        workflow_run_id = $env:GITHUB_RUN_ID
        workflow_run_url = "https://github.com/$env:GITHUB_REPOSITORY/actions/runs/$env:GITHUB_RUN_ID"
        reference_media_upload_scope = 'EXCLUDED'
        recorded_at = (Get-Date).ToUniversalTime().ToString('o')
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $ArtifactRoot 'workflow_final_receipt.json') -Encoding utf8
}

try {
    if ($env:GITHUB_REPOSITORY -ne $ExpectedRepository) {
        throw "Repository mismatch: $env:GITHUB_REPOSITORY"
    }
    if ($env:GITHUB_REF_NAME -ne $ExpectedBranch) {
        throw "Branch mismatch: $env:GITHUB_REF_NAME"
    }
    $remote = (& git remote get-url origin | Out-String).Trim()
    $head = (& git rev-parse HEAD | Out-String).Trim()
    if ($remote -notmatch 'organicoverlords/lowvram3d-studio') {
        throw "Remote mismatch: $remote"
    }
    if ($head -ne $env:GITHUB_SHA) {
        throw "Checkout HEAD $head does not equal workflow SHA $env:GITHUB_SHA"
    }
    if (@(& git status --short).Count -gt 0) {
        throw 'Checkout is dirty before execution'
    }

    $blender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
    if (-not (Test-Path -LiteralPath $blender)) {
        throw "Blender is missing: $blender"
    }
    $controlPython = "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\control\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $controlPython)) {
        throw "Control Python is missing: $controlPython"
    }
    $curl = (Get-Command curl.exe -ErrorAction SilentlyContinue).Source
    if (-not $curl) {
        throw 'curl.exe is unavailable'
    }
    $nvidia = (Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue).Source
    if (-not $nvidia) {
        throw 'nvidia-smi.exe is unavailable'
    }

    if (Test-Path -LiteralPath $OutputRoot) {
        Remove-Item -LiteralPath $OutputRoot -Recurse -Force
    }
    New-Item -ItemType Directory -Path $PrivateRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $ArtifactRoot -Force | Out-Null
    New-Item -ItemType Directory -Path $CacheRoot -Force | Out-Null
    "BEGGARS_V2_OUTPUT=$ArtifactRoot" | Add-Content -LiteralPath $env:GITHUB_ENV

    $configPath = 'configs\scene\beggars_banquet_recreation_v1.json'
    Invoke-Native -FilePath $controlPython -ArgumentList @('-m','py_compile','tools\beggars_scene\prepare_reference_sequence.py','tools\beggars_scene\package_results.py','blender\build_beggars_meme_scene.py') -FailureMessage 'Python compilation failed'
    Invoke-Native -FilePath $controlPython -ArgumentList @('-c',"import json,pathlib; json.loads(pathlib.Path(r'$configPath').read_text(encoding='utf-8')); print('CONFIG_JSON=PROVEN')") -FailureMessage 'Scene configuration JSON is invalid'
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json

    $usedValues = @(& $nvidia --query-gpu=memory.used --format=csv,noheader,nounits | ForEach-Object { [int]($_.Trim()) })
    $used = [int](($usedValues | Measure-Object -Sum).Sum)
    $threshold = [int]$config.worker_policy.gpu_memory_abort_threshold_mib
    Write-Host "GPU_USED_MIB=$used"
    Write-Host "GPU_ABORT_THRESHOLD_MIB=$threshold"
    if ($used -ge $threshold) {
        throw "BLOCKED_GPU_BUSY: $used MiB is already in use"
    }

    $venvPython = Join-Path $VenvRoot 'Scripts\python.exe'
    $venvHealthy = $false
    if (Test-Path -LiteralPath $venvPython) {
        & $venvPython -c "import cv2,numpy,onnxruntime,onnx,scipy,skimage,yaml,gdown,imageio_ffmpeg; print('VENV_HEALTH=PROVEN')" 2>$null
        $venvHealthy = ($LASTEXITCODE -eq 0)
    }
    if (-not $venvHealthy) {
        if (Test-Path -LiteralPath $VenvRoot) {
            Remove-Item -LiteralPath $VenvRoot -Recurse -Force
        }
        Invoke-Native -FilePath $controlPython -ArgumentList @('-m','venv',$VenvRoot) -FailureMessage 'Could not create cached reconstruction environment'
        $venvPython = Join-Path $VenvRoot 'Scripts\python.exe'
        Invoke-Native -FilePath $venvPython -ArgumentList @('-m','pip','install','--disable-pip-version-check','--upgrade','pip','setuptools','wheel') -FailureMessage 'Could not bootstrap pip'
        Invoke-Native -FilePath $venvPython -ArgumentList @('-m','pip','install','--disable-pip-version-check','numpy==1.26.4','opencv-python-headless==4.10.0.84','pyyaml==6.0.2','onnxruntime==1.19.2','onnx==1.16.2','scipy==1.13.1','scikit-image==0.24.0','yt-dlp==2026.7.4','gdown==5.2.0','imageio-ffmpeg==0.5.1') -FailureMessage 'Could not install pinned reconstruction dependencies'
    }
    $ffmpeg = (& $venvPython -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" | Out-String).Trim()
    if (-not (Test-Path -LiteralPath $ffmpeg)) {
        throw "ffmpeg is missing: $ffmpeg"
    }

    $archive = Join-Path $PrivateRoot '3ddfa.zip'
    Invoke-Native -FilePath $curl -ArgumentList @('-L','--fail','--retry','3','--output',$archive,$ThreeDdfaArchiveUrl) -FailureMessage 'Could not download pinned 3DDFA source archive'
    if ((Get-Item -LiteralPath $archive).Length -lt 500000) {
        throw '3DDFA source archive is implausibly small'
    }
    $extractRoot = Join-Path $PrivateRoot '3ddfa_extract'
    Expand-Archive -LiteralPath $archive -DestinationPath $extractRoot -Force
    $extracted = Get-ChildItem -LiteralPath $extractRoot -Directory | Select-Object -First 1
    if (-not $extracted) {
        throw '3DDFA archive did not contain a source directory'
    }
    $thirdParty = Join-Path $PrivateRoot '3DDFA_V2'
    Move-Item -LiteralPath $extracted.FullName -Destination $thirdParty
    Remove-Item -LiteralPath $archive -Force
    Remove-Item -LiteralPath $extractRoot -Recurse -Force
    foreach ($requiredSource in @('TDDFA_ONNX.py','configs\mb1_120x120.yml','configs\param_mean_std_62d_120x120.pkl','configs\bfm_noneck_v3.pkl')) {
        $requiredPath = Join-Path $thirdParty $requiredSource
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "Pinned 3DDFA archive is missing $requiredSource"
        }
    }

    $onnxPath = Join-Path $thirdParty 'weights\mb1_120x120.onnx'
    New-Item -ItemType Directory -Path (Split-Path $onnxPath) -Force | Out-Null
    Invoke-Native -FilePath $venvPython -ArgumentList @('-m','gdown','https://drive.google.com/uc?id=1YpO1KfXvJHRmCBkErNa62dHm-CUjsoIk','-O',$onnxPath) -FailureMessage 'Could not download pinned 3DDFA ONNX model'
    if ((Get-Item -LiteralPath $onnxPath).Length -lt 1000000) {
        throw 'Downloaded 3DDFA ONNX model is implausibly small'
    }

    $yunet = Join-Path $PrivateRoot 'face_detection_yunet_2023mar.onnx'
    Invoke-Native -FilePath $curl -ArgumentList @('-L','--fail','--retry','3','--output',$yunet,$YuNetUrl) -FailureMessage 'Could not download YuNet model'
    $yunetInfo = Get-Item -LiteralPath $yunet
    $yunetHash = (Get-FileHash -LiteralPath $yunet -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($yunetInfo.Length -ne $YuNetBytes -or $yunetHash -ne $YuNetSha256) {
        throw "YuNet integrity mismatch: bytes=$($yunetInfo.Length) sha256=$yunetHash"
    }
    Write-Host 'FACE_MODELS=PROVEN'

    $downloadRoot = Join-Path $PrivateRoot 'download'
    New-Item -ItemType Directory -Path $downloadRoot -Force | Out-Null
    $accepted = $null
    $acceptedTitle = $null
    $acceptedCandidate = $null
    foreach ($candidate in @($config.source_context.reference_clip_candidates)) {
        Get-ChildItem -LiteralPath $downloadRoot -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
        Write-Host "REFERENCE_ATTEMPT=$candidate"
        & $venvPython -m yt_dlp --no-playlist --max-downloads 1 --match-filter 'duration > 1 & duration < 30' --ffmpeg-location $ffmpeg --merge-output-format mp4 --write-info-json --no-write-thumbnail --format 'bv*[height<=720]+ba/b[height<=720]/b' --output (Join-Path $downloadRoot 'source.%(ext)s') $candidate
        if ($LASTEXITCODE -ne 0) {
            Write-Host "REFERENCE_ATTEMPT_FAILED=$candidate"
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
            $acceptedCandidate = [string]$candidate
            break
        }
    }
    if (-not $accepted) {
        throw 'REFERENCE_CLIP_BLOCKED: no bounded candidate could be downloaded and identity-validated'
    }
    Write-Host "REFERENCE_CLIP=PROVEN"
    Write-Host "REFERENCE_TITLE=$acceptedTitle"

    $frames = Join-Path $PrivateRoot 'frames'
    $sequence = Join-Path $PrivateRoot 'face_sequence.npz'
    $keyframe = Join-Path $PrivateRoot 'selected_keyframe.png'
    $reconstructionReport = Join-Path $ArtifactRoot 'reference_reconstruction_report.json'
    Invoke-Native -FilePath $venvPython -ArgumentList @('tools\beggars_scene\prepare_reference_sequence.py','--clip',$accepted,'--third-party-root',$thirdParty,'--yunet-model',$yunet,'--frames-dir',$frames,'--output-npz',$sequence,'--output-report',$reconstructionReport,'--keyframe-output',$keyframe,'--target-fps','12','--max-seconds','8','--smoothing','0.68') -FailureMessage 'Tracked 3D face reconstruction failed'
    if (-not (Test-Path -LiteralPath $sequence) -or (Get-Item -LiteralPath $sequence).Length -lt 1000000) {
        throw 'Tracked face sequence is missing or implausibly small'
    }
    Write-Host 'TRACKED_FACE_RECONSTRUCTION=PROVEN'

    Invoke-Native -FilePath $blender -ArgumentList @('--background','--factory-startup','--python','blender\build_beggars_meme_scene.py','--','--sequence',$sequence,'--config',$configPath,'--output-dir',$ArtifactRoot,'--render-engine','eevee') -FailureMessage 'Blender scene build or render failed'
    foreach ($name in @('beggars_photoreal_recreation.blend','hero_clean_render.png','wide_scene_proof.png','beggars_photoreal_recreation_silent.mp4','scene_receipt.json')) {
        if (-not (Test-Path -LiteralPath (Join-Path $ArtifactRoot $name))) {
            throw "Missing Blender output: $name"
        }
    }
    Write-Host 'BLENDER_BUILD_SAVE_RELOAD_RENDER=PROVEN'

    $voice = Join-Path $OutputRoot 'synthetic_voice.wav'
    Add-Type -AssemblyName System.Speech
    $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
    try {
        $synth.Rate = -1
        $synth.Volume = 100
        $synth.SetOutputToWaveFile($voice)
        $synth.Speak('Somebody get these beggars out of here!')
    }
    finally {
        $synth.Dispose()
    }
    if (-not (Test-Path -LiteralPath $voice) -or (Get-Item -LiteralPath $voice).Length -lt 10000) {
        throw 'Synthetic voice generation failed'
    }
    $durationSeconds = [double]$config.creative_target.duration_frames / [double]$config.creative_target.fps
    $silent = Join-Path $ArtifactRoot 'beggars_photoreal_recreation_silent.mp4'
    $final = Join-Path $ArtifactRoot 'beggars_photoreal_recreation.mp4'
    $filter = "[1:a]adelay=520|520,apad=pad_dur=$durationSeconds[a]"
    Invoke-Native -FilePath $ffmpeg -ArgumentList @('-y','-i',$silent,'-i',$voice,'-filter_complex',$filter,'-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-b:a','192k','-t',[string]$durationSeconds,$final) -FailureMessage 'Could not mux synthetic voice into final MP4'
    if (-not (Test-Path -LiteralPath $final) -or (Get-Item -LiteralPath $final).Length -lt 100000) {
        throw 'Final MP4 is missing or implausibly small'
    }

    [ordered]@{
        classification = 'PROVEN'
        repository = $env:GITHUB_REPOSITORY
        branch = $env:GITHUB_REF_NAME
        head_sha = $env:GITHUB_SHA
        workflow_run_id = $env:GITHUB_RUN_ID
        blender = $blender
        python = $venvPython
        gpu_used_mib_at_start = $used
        reference_title = $acceptedTitle
        reference_candidate = $acceptedCandidate
        reference_media_uploaded = $false
        actor_voice_cloned = $false
        render_scope = 'TRACKED_3D_FACE_PLUS_PROCEDURAL_BLENDER_SCENE'
        recorded_at = (Get-Date).ToUniversalTime().ToString('o')
    } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $ArtifactRoot 'worker_receipt.json') -Encoding utf8

    Invoke-Native -FilePath $venvPython -ArgumentList @('tools\beggars_scene\package_results.py','--output-dir',$ArtifactRoot,'--final-video',$final,'--workflow-run-id',$env:GITHUB_RUN_ID,'--head-sha',$env:GITHUB_SHA) -FailureMessage 'Output packaging or validation failed'
    Remove-Item -LiteralPath $silent -Force

    $evidence = Join-Path $PWD 'evidence\latest-beggars-scene'
    if (Test-Path -LiteralPath $evidence) {
        Remove-Item -LiteralPath $evidence -Recurse -Force
    }
    New-Item -ItemType Directory -Path $evidence -Force | Out-Null
    foreach ($name in @('scene_receipt.json','artifact_manifest.json','reference_reconstruction_report.json','worker_receipt.json')) {
        Copy-Item -LiteralPath (Join-Path $ArtifactRoot $name) -Destination $evidence
    }
    git config user.name 'github-actions[bot]'
    git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
    git add -- evidence/latest-beggars-scene
    if (@(& git status --porcelain -- evidence/latest-beggars-scene).Count -gt 0) {
        Invoke-Native -FilePath 'git.exe' -ArgumentList @('commit','-m',"evidence(scene): record beggars meme v2 run $env:GITHUB_RUN_ID") -FailureMessage 'Could not commit compact scene receipts'
        Invoke-Native -FilePath 'git.exe' -ArgumentList @('push','origin',"HEAD:$ExpectedBranch") -FailureMessage 'Could not push compact scene receipts'
    }

    $Status = 'PROVEN'
    Write-Host 'BEGGARS_SCENE_V2=PROVEN_USER_VISUAL_REVIEW_REQUIRED'
}
catch {
    $Failure = $_.Exception.Message
    Write-Error $Failure
    throw
}
finally {
    if (Test-Path -LiteralPath $PrivateRoot) {
        Remove-Item -LiteralPath $PrivateRoot -Recurse -Force -ErrorAction SilentlyContinue
    }
    Write-FinalReceipt
}
