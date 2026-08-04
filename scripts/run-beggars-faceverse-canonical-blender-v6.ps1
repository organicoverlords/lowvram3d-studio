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
$output = "C:\AI\LowVRAM3D-benchmarks\beggars-scene\faceverse-canonical-blender-v6-$runId"
$private = Join-Path $output '_private'
$clip = Join-Path $private 'public_reference.mp4'
$canonical = Join-Path $private 'faceverse_canonical_v6.npz'
$exportReport = Join-Path $output 'canonical_export_report.json'

$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
$source = 'C:\AI\LowVRAM3D-cache\faceverse-v4\source-19c67cc4d7234b1ea7d55a185a2cb55fd49bb877'
$model = 'C:\AI\LowVRAM3D-cache\faceverse-v4\models\faceverse_v4_2.npy'
$checkpoint = 'C:\AI\LowVRAM3D-cache\faceverse-v4\models\faceverse_resnet50.pth'
$landmarker = 'C:\AI\LowVRAM3D-cache\faceverse-v4\models\face_landmarker.task'
$coefficients = 'C:\AI\LowVRAM3D-benchmarks\beggars-scene\faceverse-landmark-refine-30876280215\faceverse_landmark_refine_coefficients.npz'
$blender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'

foreach ($required in @($python,$source,$model,$checkpoint,$landmarker,$coefficients,$blender)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Canonical Blender v6 input missing: $required"
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
    "import torch; assert torch.__version__.startswith('2.5.1'), torch.__version__; assert torch.version.cuda == '11.8'; assert torch.cuda.is_available(); print('CANONICAL_BLENDER_V6_CUDA_RUNTIME=PROVEN',torch.__version__,torch.cuda.get_device_name(0))"
) -FailureMessage 'Canonical Blender v6 CUDA runtime validation failed'

Invoke-NativeChecked -FilePath $python -ArgumentList @(
    '-m','py_compile',
    'tools\beggars_scene\export_faceverse_v4_blender_canonical.py',
    'blender\diagnose_beggars_faceverse_canonical_v6.py'
) -FailureMessage 'Canonical Blender v6 source compilation failed'

New-Item -ItemType Directory -Path $private -Force | Out-Null
try {
    Invoke-NativeChecked -FilePath 'curl.exe' -ArgumentList @(
        '-L','--fail','--retry','3','--retry-delay','2',
        '--output',$clip,
        'https://media.tenor.com/e8wT4qAP6zwAAAPo/somebody-get-these-beggars-out-of-here-the-odyssey.mp4'
    ) -FailureMessage 'Could not download public reference clip'

    $previousNoUser = $env:PYTHONNOUSERSITE
    $previousPath = $env:PYTHONPATH
    $previousUnbuffered = $env:PYTHONUNBUFFERED
    $previousPipNoIndex = $env:PIP_NO_INDEX
    $previousUvOffline = $env:UV_OFFLINE
    try {
        $env:PYTHONNOUSERSITE = '1'
        $env:PYTHONPATH = ''
        $env:PYTHONUNBUFFERED = '1'
        $env:PIP_NO_INDEX = '1'
        $env:UV_OFFLINE = '1'
        Invoke-NativeChecked -FilePath $python -ArgumentList @(
            '-I','-u','-X','faulthandler',
            'tools\beggars_scene\export_faceverse_v4_blender_canonical.py',
            '--coefficients',$coefficients,
            '--variant','flexible',
            '--faceverse-root',$source,
            '--model-npy',$model,
            '--checkpoint',$checkpoint,
            '--landmarker',$landmarker,
            '--clip',$clip,
            '--output-npz',$canonical,
            '--output-report',$exportReport,
            '--anchors','21,31,35',
            '--device','auto'
        ) -FailureMessage 'Canonical FaceVerse export failed'
    }
    finally {
        $env:PYTHONNOUSERSITE = $previousNoUser
        $env:PYTHONPATH = $previousPath
        $env:PYTHONUNBUFFERED = $previousUnbuffered
        $env:PIP_NO_INDEX = $previousPipNoIndex
        $env:UV_OFFLINE = $previousUvOffline
    }

    if (-not (Test-Path -LiteralPath $canonical) -or (Get-Item -LiteralPath $canonical).Length -lt 500000) {
        throw 'Canonical FaceVerse export is missing or implausibly small'
    }

    Invoke-NativeChecked -FilePath $blender -ArgumentList @(
        '--background','--factory-startup',
        '--python','blender\diagnose_beggars_faceverse_canonical_v6.py',
        '--','--canonical',$canonical,'--output-dir',$output
    ) -FailureMessage 'Canonical Blender v6 diagnostic failed'

    foreach ($name in @(
        '01_raw_canonical.png',
        '02_cranium_corrected.png',
        '03_corrected_groomed.png',
        '04_corrected_groomed_uniform_teeth.png',
        'canonical_blender_diagnostic.json',
        'canonical_export_report.json'
    )) {
        $path = Join-Path $output $name
        if (-not (Test-Path -LiteralPath $path) -or (Get-Item -LiteralPath $path).Length -le 0) {
            throw "Canonical Blender v6 output missing or empty: $name"
        }
    }

    $sheetScript = Join-Path $output 'make_contact_sheet.py'
    $sheetSource = @'
from pathlib import Path
from PIL import Image, ImageDraw
import os
root = Path(os.environ["CB6_OUTPUT"])
names = [
    "01_raw_canonical.png",
    "02_cranium_corrected.png",
    "03_corrected_groomed.png",
    "04_corrected_groomed_uniform_teeth.png",
]
tiles = []
for name in names:
    image = Image.open(root / name).convert("RGB")
    image.thumbnail((720, 405), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (720, 445), "black")
    tile.paste(image, ((720-image.width)//2, 0))
    ImageDraw.Draw(tile).text((12, 416), name, fill="white")
    tiles.append(tile)
sheet = Image.new("RGB", (1440, 890), "black")
for index, tile in enumerate(tiles):
    sheet.paste(tile, ((index % 2) * 720, (index // 2) * 445))
sheet.save(root / "canonical_blender_contact_sheet.jpg", quality=95)
'@
    [System.IO.File]::WriteAllText(
        $sheetScript,
        $sheetSource,
        [System.Text.UTF8Encoding]::new($false)
    )
    $env:CB6_OUTPUT = $output
    Invoke-NativeChecked -FilePath $python -ArgumentList @($sheetScript) -FailureMessage 'Canonical Blender v6 contact-sheet generation failed'
    Remove-Item -LiteralPath $sheetScript -Force -ErrorAction SilentlyContinue

    $sheet = Join-Path $output 'canonical_blender_contact_sheet.jpg'
    if (-not (Test-Path -LiteralPath $sheet) -or (Get-Item -LiteralPath $sheet).Length -lt 100000) {
        throw 'Canonical Blender v6 contact sheet is missing or implausibly small'
    }
}
finally {
    Remove-Item -LiteralPath $clip -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $canonical -Force -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $private) {
        Remove-Item -LiteralPath $private -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if ($env:GITHUB_ENV) {
    "CB6_OUTPUT=$output" | Add-Content -LiteralPath $env:GITHUB_ENV
}
Write-Host "CANONICAL_BLENDER_V6_OUTPUT=$output"
Write-Host 'CANONICAL_BLENDER_V6_STILLS=PROVEN'
exit 0
