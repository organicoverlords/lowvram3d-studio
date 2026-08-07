param(
    [string]$Output = "evidence/rigging/vendor_preflight.json",
    [string]$ExpectedCommit = "69ee59dc459d2da7cb0291930c1f944886c31d7c"
)

$ErrorActionPreference = "Stop"

function Add-Candidate([System.Collections.Generic.List[string]]$List, [string]$Path) {
    if (-not $Path) { return }
    try { $full = [System.IO.Path]::GetFullPath($Path) } catch { return }
    if (-not $List.Contains($full)) { $List.Add($full) }
}

$candidates = New-Object 'System.Collections.Generic.List[string]'
if ($env:GENSTUDIO_DATA_ROOT) {
    Add-Candidate $candidates (Join-Path $env:GENSTUDIO_DATA_ROOT "comfyui")
}

foreach ($base in @($env:APPDATA, $env:LOCALAPPDATA)) {
    if (-not $base -or -not (Test-Path -LiteralPath $base)) { continue }
    Get-ChildItem -LiteralPath $base -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        Add-Candidate $candidates (Join-Path $_.FullName "comfyui")
    }
}

foreach ($path in @("C:\AI\ComfyUI", "C:\ComfyUI")) {
    Add-Candidate $candidates $path
}

$comfyRoot = $null
foreach ($candidate in $candidates) {
    if (Test-Path -LiteralPath (Join-Path $candidate "main.py")) {
        $comfyRoot = $candidate
        break
    }
}

$python = $null
$pythonCandidates = New-Object 'System.Collections.Generic.List[string]'
if ($comfyRoot) {
    $dataRoot = Split-Path -Parent $comfyRoot
    foreach ($candidate in @(
        (Join-Path $dataRoot "comfy-venv\Scripts\python.exe"),
        (Join-Path $comfyRoot "venv\Scripts\python.exe"),
        (Join-Path $comfyRoot "python_embeded\python.exe")
    )) { Add-Candidate $pythonCandidates $candidate }
}
try {
    $systemPython = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
    Add-Candidate $pythonCandidates $systemPython
} catch {}
foreach ($candidate in $pythonCandidates) {
    if (Test-Path -LiteralPath $candidate) { $python = $candidate; break }
}

$packRoot = if ($comfyRoot) { Join-Path $comfyRoot "custom_nodes\ComfyUI-UniRig" } else { $null }
$packPresent = [bool]($packRoot -and (Test-Path -LiteralPath $packRoot))
$packCommit = $null
$packCommitProven = $false
if ($packPresent -and (Test-Path -LiteralPath (Join-Path $packRoot ".git"))) {
    try {
        $packCommit = (& git -C $packRoot rev-parse HEAD 2>$null).Trim()
        $packCommitProven = [bool]$packCommit
    } catch {}
}

$workflowPaths = @{}
foreach ($name in @("mia_humanoid.json", "unirig_humanoid.json", "unirig_bird.json", "apply_animation.json")) {
    $workflowPaths[$name] = [bool]($packPresent -and (Test-Path -LiteralPath (Join-Path $packRoot ("workflows\" + $name))))
}

$lightImportsOk = $false
$lightImportsDetail = $null
if ($python) {
    try {
        $probe = & $python -c "import importlib.util; mods=['trimesh','comfy_env']; missing=[m for m in mods if importlib.util.find_spec(m) is None]; print('OK' if not missing else 'MISSING:'+','.join(missing))" 2>&1
        $lightImportsDetail = ($probe | Out-String).Trim()
        $lightImportsOk = ($LASTEXITCODE -eq 0 -and $lightImportsDetail -eq "OK")
    } catch {
        $lightImportsDetail = $_.Exception.Message
    }
}

$server = $null
$serverNodes = @{}
foreach ($port in 8188..8205) {
    try {
        $uri = "http://127.0.0.1:$port/object_info"
        $info = Invoke-RestMethod -Uri $uri -Method Get -TimeoutSec 1
        $server = "http://127.0.0.1:$port"
        foreach ($node in @("UniRigLoadMesh", "MIALoadModel", "MIAAutoRig", "UniRigPreviewRiggedMesh", "UniRigApplyAnimation")) {
            $serverNodes[$node] = [bool]($info.PSObject.Properties.Name -contains $node)
        }
        break
    } catch {}
}

$status = "BLOCKED_NO_COMFY"
if ($comfyRoot) {
    if (-not $python) { $status = "BLOCKED_NO_PYTHON" }
    elseif (-not $packPresent) { $status = "INSTALL_REQUIRED" }
    elseif ($packCommitProven -and $packCommit -ne $ExpectedCommit) { $status = "PIN_MISMATCH" }
    elseif (-not $workflowPaths["mia_humanoid.json"]) { $status = "PACK_INCOMPLETE" }
    elseif ($server -and $serverNodes["MIAAutoRig"] -and $serverNodes["MIALoadModel"]) { $status = "READY_RUNNING" }
    else { $status = "READY_OFFLINE" }
}

$report = [ordered]@{
    schema = "lowvram3d_rig_vendor_preflight_v1"
    timestamp_utc = [DateTime]::UtcNow.ToString("o")
    status = $status
    expected = [ordered]@{
        repository = "PozzettiAndrea/ComfyUI-UniRig"
        commit = $ExpectedCommit
        first_workflow = "workflows/mia_humanoid.json"
        first_asset = "assets/realistic_male_character.glb"
    }
    comfy = [ordered]@{
        root = $comfyRoot
        python = $python
        candidates = @($candidates)
        running_server = $server
    }
    vendor_pack = [ordered]@{
        root = $packRoot
        present = $packPresent
        commit = $packCommit
        commit_proven = $packCommitProven
        exact_pin = [bool]($packCommitProven -and $packCommit -eq $ExpectedCommit)
        workflows = $workflowPaths
        lightweight_imports_ok = $lightImportsOk
        lightweight_imports_detail = $lightImportsDetail
        server_nodes = $serverNodes
    }
    gpu_touched = $false
    torch_imported_by_preflight = $false
    next_action = switch ($status) {
        "BLOCKED_NO_COMFY" { "Locate or install a ComfyUI instance before altering rig code." }
        "BLOCKED_NO_PYTHON" { "Resolve the ComfyUI Python environment." }
        "INSTALL_REQUIRED" { "Install the stock ComfyUI-UniRig pack at the pinned upstream commit; do not patch it." }
        "PIN_MISMATCH" { "Create an isolated pinned stock test environment instead of modifying the installed pack in place." }
        "PACK_INCOMPLETE" { "Reinstall the unmodified pinned upstream pack." }
        "READY_OFFLINE" { "Start this ComfyUI unchanged, then run the vendor MIA workflow on the vendor asset." }
        "READY_RUNNING" { "Run the vendor MIA workflow on the vendor-bundled humanoid before testing our generated mesh." }
        default { "Review preflight evidence." }
    }
}

$parent = Split-Path -Parent $Output
if ($parent) { New-Item -ItemType Directory -Force -Path $parent | Out-Null }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Output -Encoding utf8
$report | ConvertTo-Json -Depth 8
exit 0
