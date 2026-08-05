<#
.SYNOPSIS
    Install the optional tiny visual-QA environment and pre-download the judge model.

.DESCRIPTION
    Creates a dedicated venv so torch/transformers cannot destabilise the existing pipeline
    environments, then downloads the model ONCE into a local directory. This is the only place
    that is allowed to fetch weights; production runs are offline and point at --model-dir.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\install-tiny-visual-qa.ps1
#>
[CmdletBinding()]
param(
    [string]$EnvRoot = "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\visualqa",
    [string]$ModelDir = "$env:LOCALAPPDATA\LowVRAM3DStudio\models\SmolVLM-256M-Instruct",
    [string]$ModelId = "HuggingFaceTB/SmolVLM-256M-Instruct",
    # CPU by default on purpose: a 256M judge runs fast on CPU, and the 6 GB GPU stays free for
    # Blender and the generation pipeline. Pass -Cuda only if you want the judge on the GPU.
    [switch]$Cuda
)

$ErrorActionPreference = 'Stop'

Write-Host "=== tiny visual QA install ==="
Write-Host "env    : $EnvRoot"
Write-Host "model  : $ModelId -> $ModelDir"

$basePython = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
if (-not $basePython) { throw "python.exe was not found on PATH" }

if (-not (Test-Path -LiteralPath $EnvRoot)) {
    Write-Host "`n[1/4] creating virtual environment"
    & $basePython -m venv $EnvRoot
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed" }
} else {
    Write-Host "`n[1/4] reusing existing virtual environment"
}

$python = Join-Path $EnvRoot "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "interpreter missing at $python" }

Write-Host "`n[2/4] installing dependencies"
& $python -m pip install --upgrade pip --quiet
if ($Cuda) {
    & $python -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cu124
} else {
    & $python -m pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { throw "torch installation failed" }
& $python -m pip install --quiet "transformers>=4.46" pillow accelerate num2words
if ($LASTEXITCODE -ne 0) { throw "transformers installation failed" }

Write-Host "`n[3/4] downloading model weights (the only permitted download)"
$fetch = @"
import sys, time
from pathlib import Path
import transformers
from transformers import AutoProcessor
AutoModelForVision2Seq = getattr(transformers, 'AutoModelForImageTextToText', None) or transformers.AutoModelForVision2Seq
model_id, target = sys.argv[1], Path(sys.argv[2])
target.mkdir(parents=True, exist_ok=True)
started = time.monotonic()
AutoProcessor.from_pretrained(model_id).save_pretrained(target)
AutoModelForVision2Seq.from_pretrained(model_id).save_pretrained(target)
size = sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
print(f'MODEL_READY dir={target} bytes={size} mib={size/1048576:.1f} seconds={time.monotonic()-started:.1f}')
"@
$fetchScript = Join-Path $env:TEMP "lowvram3d_fetch_vqa.py"
[IO.File]::WriteAllText($fetchScript, $fetch, [Text.UTF8Encoding]::new($false))
& $python $fetchScript $ModelId $ModelDir
if ($LASTEXITCODE -ne 0) { throw "model download failed" }

Write-Host "`n[4/4] verifying offline load"
& $python -c "import os,torch,transformers;os.environ['HF_HUB_OFFLINE']='1';print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('transformers',transformers.__version__)"

Write-Host "`nInstalled. Set this for production runs:"
Write-Host "  `$env:LOWVRAM3D_TINY_VQA_MODEL_DIR = '$ModelDir'"
