[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'scripts\run-beggars-faceverse-v4-preflight.ps1'
if (-not (Test-Path -LiteralPath $target)) {
    throw "FaceVerse preflight script is missing: $target"
}

$text = Get-Content -LiteralPath $target -Raw

$oldInvocation = @'
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
'@

$newInvocation = @'
    $sitePackages = Join-Path $VenvRoot 'Lib\site-packages'
    $proofScript = (Resolve-Path -LiteralPath 'tools\beggars_scene\run_faceverse_v4_preflight.py').Path
    if (-not (Test-Path -LiteralPath $sitePackages)) {
        throw "FaceVerse site-packages directory is missing: $sitePackages"
    }

    $isolationAudit = @"
import json, pathlib, site, sys
root = pathlib.Path(r'$sitePackages')
records = []
for pattern in ('*.pth', 'sitecustomize.py', 'usercustomize.py'):
    for path in sorted(root.glob(pattern)):
        text = path.read_text(encoding='utf-8', errors='replace') if path.suffix in {'.pth', '.py'} else ''
        records.append({'path': str(path), 'bytes': path.stat().st_size, 'mentions_package_mutation': any(token in text.lower() for token in ('pip', 'uv ', 'torch==', 'onnxscript', 'subprocess'))})
print('FACEVERSE_STARTUP_HOOK_AUDIT=' + json.dumps(records, sort_keys=True), flush=True)
print('FACEVERSE_USER_SITE=' + str(site.getusersitepackages()), flush=True)
print('FACEVERSE_ISOLATED_SYS_PATH=' + json.dumps(sys.path), flush=True)
"@
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $VenvPython -I -u -c $isolationAudit
        $isolationAuditExit = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($isolationAuditExit -ne 0) {
        throw "FaceVerse startup-hook isolation audit failed with exit code $isolationAuditExit"
    }

    $torchGuard = @"
import json, torch
assert torch.__version__.startswith('2.5.1'), torch.__version__
assert torch.version.cuda == '11.8', torch.version.cuda
assert torch.cuda.is_available(), 'CUDA is unavailable'
print('FACEVERSE_CUDA_TORCH_GUARD=' + json.dumps({'version': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0)}, sort_keys=True), flush=True)
"@
    & $VenvPython -I -u -X faulthandler -c $torchGuard
    if ($LASTEXITCODE -ne 0) {
        throw 'FaceVerse CUDA Torch guard failed before isolated proof.'
    }

    $proofBootstrap = @"
import runpy, sys, torch
torch.cuda.empty_cache = lambda: None
torch.cuda.reset_peak_memory_stats = lambda *args, **kwargs: None
print('FACEVERSE_OPTIONAL_CUDA_MEMORY_HOOKS=DISABLED', flush=True)
sys.argv = [
    r'$proofScript',
    '--faceverse-root', r'$SourceRoot',
    '--model-npy', r'$ModelPath',
    '--checkpoint', r'$CheckpointPath',
    '--landmarker', r'$LandmarkerPath',
    '--input-image', r'$keyframePath',
    '--output-dir', r'$ArtifactRoot',
    '--device', 'auto',
]
runpy.run_path(r'$proofScript', run_name='__main__')
"@

    $previousPythonNoUserSite = $env:PYTHONNOUSERSITE
    $previousPythonPath = $env:PYTHONPATH
    $previousPythonSafePath = $env:PYTHONSAFEPATH
    $previousPythonUnbuffered = $env:PYTHONUNBUFFERED
    $previousPipNoIndex = $env:PIP_NO_INDEX
    $previousPipIndexUrl = $env:PIP_INDEX_URL
    $previousUvOffline = $env:UV_OFFLINE
    try {
        $env:PYTHONNOUSERSITE = '1'
        $env:PYTHONPATH = ''
        $env:PYTHONSAFEPATH = '1'
        $env:PYTHONUNBUFFERED = '1'
        $env:PIP_NO_INDEX = '1'
        $env:PIP_INDEX_URL = 'http://127.0.0.1:9/disabled'
        $env:UV_OFFLINE = '1'
        Invoke-Native -FilePath $VenvPython -ArgumentList @('-I','-u','-X','faulthandler','-c',$proofBootstrap) -FailureMessage 'FaceVerse v4 isolated single-frame proof failed'
    }
    finally {
        $env:PYTHONNOUSERSITE = $previousPythonNoUserSite
        $env:PYTHONPATH = $previousPythonPath
        $env:PYTHONSAFEPATH = $previousPythonSafePath
        $env:PYTHONUNBUFFERED = $previousPythonUnbuffered
        $env:PIP_NO_INDEX = $previousPipNoIndex
        $env:PIP_INDEX_URL = $previousPipIndexUrl
        $env:UV_OFFLINE = $previousUvOffline
    }

    & $VenvPython -I -u -X faulthandler -c $torchGuard
    if ($LASTEXITCODE -ne 0) {
        throw 'FaceVerse CUDA Torch guard failed after isolated proof; the environment was mutated.'
    }
    Write-Host 'FACEVERSE_ISOLATED_PROOF_ENVIRONMENT=PROVEN'
'@

if ($text.Contains($oldInvocation)) {
    $text = $text.Replace($oldInvocation, $newInvocation)
}
elseif ($text.Contains('FACEVERSE_ISOLATED_PROOF_ENVIRONMENT=PROVEN')) {
    Write-Host 'FACEVERSE_ISOLATED_PROOF_PATCH=ALREADY_APPLIED'
}
else {
    throw 'Could not locate the FaceVerse proof invocation for isolated replacement.'
}

[System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $target), $text, [System.Text.UTF8Encoding]::new($false))

$tokens = $null
$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path -LiteralPath $target), [ref]$tokens, [ref]$errors)
if (@($errors).Count -gt 0) {
    throw "Isolated FaceVerse preflight patch failed PowerShell parsing: $($errors[0].Message)"
}
Write-Host 'FACEVERSE_ISOLATED_PROOF_PATCH=PROVEN'
