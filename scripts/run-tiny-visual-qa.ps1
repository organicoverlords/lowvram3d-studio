<#
.SYNOPSIS
    Run the tiny visual-QA judge over one or more manifests.

.DESCRIPTION
    Uses the optional visual-QA environment and an already-installed local model. Never downloads:
    if the model directory is absent the judge reports "unavailable", which in auto mode preserves
    the baseline and lets the pipeline continue.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\run-tiny-visual-qa.ps1 `
        -Manifest evidence\visual-qa\staff_hole_rejected.json -ReceiptDir out\vqa -Mode auto
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string[]]$Manifest,
    [Parameter(Mandatory = $true)][string]$ReceiptDir,
    [ValidateSet('off', 'auto', 'required')][string]$Mode = 'auto',
    [ValidateSet('auto', 'cuda', 'cpu')][string]$Device = 'auto',
    [string]$EnvRoot = "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\visualqa",
    [string]$ModelDir = $(if ($env:LOWVRAM3D_TINY_VQA_MODEL_DIR) { $env:LOWVRAM3D_TINY_VQA_MODEL_DIR } else { "$env:LOCALAPPDATA\LowVRAM3DStudio\models\SmolVLM-256M-Instruct" }),
    [int]$Repeat = 1,
    [ValidateSet('true', 'false')][string]$HardGatesPassed = 'true'
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot

if ($Mode -eq 'off') {
    Write-Host '{"status":"unavailable","mode":"off","reason":"visual QA disabled"}'
    exit 0
}

$python = Join-Path $EnvRoot "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    $message = "visual QA environment is not installed at $EnvRoot"
    if ($Mode -eq 'required') { throw $message }
    Write-Warning "$message - continuing, baseline preserved (mode=auto)"
    exit 0
}
if (-not (Test-Path -LiteralPath $ModelDir)) {
    $message = "visual QA model is not installed at $ModelDir"
    if ($Mode -eq 'required') { throw $message }
    Write-Warning "$message - continuing, baseline preserved (mode=auto)"
    exit 0
}

$env:PYTHONPATH = "$repo\src;$repo\workers;$repo"
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

$arguments = @()
foreach ($item in $Manifest) { $arguments += @('--manifest', $item) }
$arguments += @(
    '--receipt-dir', $ReceiptDir,
    '--mode', $Mode,
    '--device', $Device,
    '--model-dir', $ModelDir,
    '--repeat', $Repeat,
    '--hard-gates-passed', $HardGatesPassed
)

& $python (Join-Path $repo "workers\tiny_visual_qa.py") @arguments
$code = $LASTEXITCODE

# In auto mode an inconclusive or failed optional check must never fail the asset pipeline.
if ($Mode -eq 'auto' -and $code -ne 0) {
    Write-Warning "visual QA reported a non-zero result ($code); baseline preserved, continuing"
    exit 0
}
exit $code
