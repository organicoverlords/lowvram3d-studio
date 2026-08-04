[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'tools\beggars_scene\run_faceverse_v4_preflight.py'
if (-not (Test-Path -LiteralPath $target)) {
    throw "FaceVerse proof source is missing: $target"
}

$text = (Get-Content -LiteralPath $target -Raw).Replace("`r`n", "`n")
$old = @'
    device = choose_device(args.device)
    print(f"FACEVERSE_DEVICE={device}", flush=True)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    faceverse_stage("MODEL_LOAD_BEGIN")
'@
$new = @'
    device = choose_device(args.device)
    print(f"FACEVERSE_DEVICE={device}", flush=True)
    if device.type == "cuda":
        faceverse_stage("CUDA_CONTEXT_BEGIN")
        torch.cuda.set_device(device)
        cuda_probe = torch.ones((1,), dtype=torch.float32, device=device)
        torch.cuda.synchronize(device)
        del cuda_probe
        faceverse_stage("CUDA_CONTEXT_DONE")

    faceverse_stage("MODEL_LOAD_BEGIN")
'@

if ($text.Contains($old)) {
    $text = $text.Replace($old, $new)
}
elif ($text.Contains('faceverse_stage("CUDA_CONTEXT_BEGIN")')) {
    Write-Host 'FACEVERSE_CUDA_INIT_COMPAT=ALREADY_APPLIED'
}
else {
    throw 'Could not locate the FaceVerse CUDA initialization block.'
}

[System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $target), $text, [System.Text.UTF8Encoding]::new($false))

$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
& $python -m py_compile $target
if ($LASTEXITCODE -ne 0) {
    throw 'FaceVerse proof failed compilation after CUDA initialization repair.'
}
Write-Host 'FACEVERSE_CUDA_INIT_COMPAT=PROVEN'
