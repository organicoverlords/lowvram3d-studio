[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [Parameter(Mandatory)][string]$FailureMessage
    )
    & $FilePath @ArgumentList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$FailureMessage (exit $exitCode)"
    }
}

$runId = if ($env:GITHUB_RUN_ID) { $env:GITHUB_RUN_ID } else { 'local' }
$output = "C:\AI\LowVRAM3D-benchmarks\beggars-scene\blender-faceverse-v11-preflight-$runId"
$private = Join-Path $output '_private'
$clip = Join-Path $private 'public_reference.mp4'
$sequence = Join-Path $private 'faceverse_model_space_sequence.npz'
$sequenceReport = Join-Path $output 'faceverse_model_space_sequence_report.json'
$projection = Join-Path $private 'faceverse_projection_texture.npz'
$projectionImage = Join-Path $output 'projected_keyframe_031.png'
$projectionReport = Join-Path $output 'projection_texture_report.json'

$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
$source = 'C:\AI\LowVRAM3D-cache\faceverse-v4\source-19c67cc4d7234b1ea7d55a185a2cb55fd49bb877'
$model = 'C:\AI\LowVRAM3D-cache\faceverse-v4\models\faceverse_v4_2.npy'
$checkpoint = 'C:\AI\LowVRAM3D-cache\faceverse-v4\models\faceverse_resnet50.pth'
$coefficients = 'C:\AI\LowVRAM3D-benchmarks\beggars-scene\faceverse-shared-identity-fit-v2-30875541045\faceverse_shared_identity_fit_coefficients.npz'
$blender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'

foreach ($required in @($python, $source, $model, $checkpoint, $coefficients, $blender)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Blender v11 input missing: $required"
    }
}

$gpuLines = @(& nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>$null)
$gpuUsed = 0
foreach ($line in $gpuLines) {
    $value = 0
    if ([int]::TryParse(([string]$line).Trim(), [ref]$value)) {
        $gpuUsed += $value
    }
}
Write-Host "GPU_USED_MIB=$gpuUsed"
if ($gpuUsed -ge 4800) {
    throw "GPU_BUSY: $gpuUsed MiB"
}

Invoke-NativeChecked -FilePath $python -ArgumentList @(
    '-I','-u','-X','faulthandler','-c',
    "import torch; assert torch.__version__.startswith('2.5.1'), torch.__version__; assert torch.cuda.is_available(); print('BLENDER_V11_FACEVERSE_RUNTIME=PROVEN',torch.__version__,torch.cuda.get_device_name(0))"
) -FailureMessage 'Blender v11 FaceVerse runtime validation failed'

Invoke-NativeChecked -FilePath $python -ArgumentList @(
    '-m','py_compile',
    'tools\beggars_scene\export_faceverse_v4_blender_model_sequence.py',
    'tools\beggars_scene\export_faceverse_v4_projection_texture.py',
    'blender\build_beggars_meme_scene_faceverse_v11.py'
) -FailureMessage 'Blender v11 sources failed compilation'

New-Item -ItemType Directory -Path $private -Force | Out-Null
try {
    Invoke-NativeChecked -FilePath 'curl.exe' -ArgumentList @(
        '-L','--fail','--retry','3','--retry-delay','2',
        '--output',$clip,
        'https://media.tenor.com/e8wT4qAP6zwAAAPo/somebody-get-these-beggars-out-of-here-the-odyssey.mp4'
    ) -FailureMessage 'Could not download public reference clip'

    $previousNoUser = $env:PYTHONNOUSERSITE
    $previousPythonPath = $env:PYTHONPATH
    try {
        $env:PYTHONNOUSERSITE = '1'
        $env:PYTHONPATH = ''
        Invoke-NativeChecked -FilePath $python -ArgumentList @(
            '-I','-u','-X','faulthandler',
            'tools\beggars_scene\export_faceverse_v4_blender_model_sequence.py',
            '--coefficients',$coefficients,
            '--faceverse-root',$source,
            '--model-npy',$model,
            '--checkpoint',$checkpoint,
            '--clip',$clip,
            '--output-npz',$sequence,
            '--output-report',$sequenceReport,
            '--anchors','21,31,35',
            '--device','auto'
        ) -FailureMessage 'Blender v11 model-space sequence export failed'

        Invoke-NativeChecked -FilePath $python -ArgumentList @(
            '-I','-u','-X','faulthandler',
            'tools\beggars_scene\export_faceverse_v4_projection_texture.py',
            '--coefficients',$coefficients,
            '--faceverse-root',$source,
            '--model-npy',$model,
            '--checkpoint',$checkpoint,
            '--clip',$clip,
            '--frame','31',
            '--output-npz',$projection,
            '--output-image',$projectionImage,
            '--output-report',$projectionReport,
            '--device','auto'
        ) -FailureMessage 'Blender v11 projection-texture export failed'
    }
    finally {
        $env:PYTHONNOUSERSITE = $previousNoUser
        $env:PYTHONPATH = $previousPythonPath
        Remove-Item -LiteralPath $clip -Force -ErrorAction SilentlyContinue
    }

    foreach ($path in @($sequence, $projection, $projectionImage, $projectionReport)) {
        if (-not (Test-Path -LiteralPath $path) -or (Get-Item -LiteralPath $path).Length -le 0) {
            throw "Blender v11 generated input missing or empty: $path"
        }
    }

    $env:B10_PROJECTION_NPZ = $projection
    $env:B10_PROJECTION_IMAGE = $projectionImage
    Invoke-NativeChecked -FilePath $blender -ArgumentList @(
        '--background',
        '--factory-startup',
        '--python','blender\build_beggars_meme_scene_faceverse_v11.py',
        '--',
        '--sequence',$sequence,
        '--config','configs\scene\beggars_banquet_recreation_v1.json',
        '--output-dir',$output,
        '--render-engine','eevee'
    ) -FailureMessage 'Exact-UV Blender v11 preflight failed'

    foreach ($name in @(
        'beggars_photoreal_recreation.blend',
        'hero_clean_render.png',
        'wide_scene_proof.png',
        'proof_frame_01.png',
        'proof_frame_02.png',
        'proof_frame_03.png',
        'projection_blend_045.png',
        'projection_blend_060.png',
        'projection_blend_075.png',
        'scene_receipt.json',
        'faceverse_model_space_sequence_report.json',
        'projection_texture_report.json',
        'projected_keyframe_031.png'
    )) {
        $path = Join-Path $output $name
        if (-not (Test-Path -LiteralPath $path) -or (Get-Item -LiteralPath $path).Length -le 0) {
            throw "Exact-UV Blender v11 output missing or empty: $name"
        }
    }
    Write-Host 'BLENDER_FACEVERSE_V11_SAVE_RELOAD_STILLS=PROVEN'

    $sheetScript = Join-Path $output 'make_contact_sheet.py'
    $sheetLines = @(
        'from pathlib import Path',
        'from PIL import Image, ImageDraw',
        'import os',
        'root = Path(os.environ["B11_OUTPUT"])',
        'names = ["projected_keyframe_031.png", "projection_blend_045.png", "projection_blend_060.png", "projection_blend_075.png", "hero_clean_render.png", "wide_scene_proof.png"]',
        'images = []',
        'for name in names:',
        '    image = Image.open(root / name).convert("RGB")',
        '    image.thumbnail((640, 360), Image.Resampling.LANCZOS)',
        '    tile = Image.new("RGB", (640, 396), "black")',
        '    tile.paste(image, ((640-image.width)//2, 0))',
        '    ImageDraw.Draw(tile).text((12, 368), name, fill="white")',
        '    images.append(tile)',
        'sheet = Image.new("RGB", (1280, 1188), "black")',
        'for index, image in enumerate(images):',
        '    sheet.paste(image, ((index % 2)*640, (index // 2)*396))',
        'sheet.save(root / "contact_sheet.png", quality=95)'
    )
    [System.IO.File]::WriteAllLines(
        $sheetScript,
        $sheetLines,
        [System.Text.UTF8Encoding]::new($false)
    )
    $env:B11_OUTPUT = $output
    Invoke-NativeChecked -FilePath $python -ArgumentList @($sheetScript) -FailureMessage 'Could not build Blender v11 contact sheet'
    Remove-Item -LiteralPath $sheetScript -Force -ErrorAction SilentlyContinue

    $contactSheet = Join-Path $output 'contact_sheet.png'
    if (-not (Test-Path -LiteralPath $contactSheet) -or (Get-Item -LiteralPath $contactSheet).Length -lt 100000) {
        throw 'Blender v11 contact sheet is missing or implausibly small'
    }
}
finally {
    Remove-Item -LiteralPath $sequence -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $projection -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $private) {
        Remove-Item -LiteralPath $private -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($env:GITHUB_ENV) {
    "B11_OUTPUT=$output" | Add-Content -LiteralPath $env:GITHUB_ENV
}
Write-Host "BLENDER_FACEVERSE_V11_OUTPUT=$output"
Write-Host 'BLENDER_FACEVERSE_V11_RUNNER=PROVEN'
exit 0
