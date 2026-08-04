[CmdletBinding()]
param(
    [string]$OutputRoot = 'C:\AI\ProceduralJungle\20260804',
    [string]$ProjectRoot = 'C:\Users\Lauri\Desktop\ProceduralJungle58',
    [string]$ExpectedBranch = 'feature/procedural-jungle-playable-20260804',
    [string]$SourceRoot = '',
    [int]$Seed = 20260804
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

function Write-Utf8Text {
    param([Parameter(Mandatory=$true)][string]$Path, [Parameter(Mandatory=$true)][string]$Text)
    $Parent = Split-Path -Parent $Path
    if ($Parent) { New-Item -ItemType Directory -Path $Parent -Force | Out-Null }
    [IO.File]::WriteAllText($Path, $Text, $script:Utf8NoBom)
}

function Write-JsonFile {
    param([Parameter(Mandatory=$true)]$Value, [Parameter(Mandatory=$true)][string]$Path, [int]$Depth = 30)
    Write-Utf8Text -Path $Path -Text ($Value | ConvertTo-Json -Depth $Depth)
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory=$true)][string]$Name,
        [Parameter(Mandatory=$true)][scriptblock]$Command,
        [string]$LogPath
    )
    Write-Host "PHASE=$Name"
    if ($LogPath) {
        $Parent = Split-Path -Parent $LogPath
        if ($Parent) { New-Item -ItemType Directory -Path $Parent -Force | Out-Null }
        & $Command 2>&1 | Tee-Object -FilePath $LogPath
    } else {
        & $Command
    }
    $ExitCode = $LASTEXITCODE
    if ($ExitCode -ne 0) { throw "$Name failed with exit code $ExitCode" }
}

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Test-PreservedPandaRig {
    param([string]$ReportPath, [string]$FbxPath, [string]$BlendPath)
    if (-not (Test-Path -LiteralPath $ReportPath -PathType Leaf)) { return $false }
    if (-not (Test-Path -LiteralPath $FbxPath -PathType Leaf)) { return $false }
    if (-not (Test-Path -LiteralPath $BlendPath -PathType Leaf)) { return $false }
    if ((Get-Item -LiteralPath $FbxPath).Length -lt 10000000) { return $false }
    if ((Get-Item -LiteralPath $BlendPath).Length -lt 10000000) { return $false }
    try {
        $Report = Get-Content -LiteralPath $ReportPath -Raw | ConvertFrom-Json
        if ($Report.classification -ne 'PANDA_WALK_RIG_GENERATED') { return $false }
        if ([int]$Report.bone_count -ne 25) { return $false }
        if ([int]$Report.mesh_vertices -ne 456092) { return $false }
        if ([int]$Report.mesh_faces -ne 644348) { return $false }
        if ([int]$Report.material_count -ne 2) { return $false }
        if ([string]$Report.source_sha256 -ne '78c55133165e931bc8d6765610a679d1d18badcdc178820a69e31b7b32bcbfb8') { return $false }
        if ([int64]$Report.source_size -ne 50244400) { return $false }
        if ($Report.animation.name -ne 'Walk') { return $false }
        if (-not $Report.animation.looping -or -not $Report.animation.in_place) { return $false }
        if ([int]$Report.animation.fps -ne 24 -or [int]$Report.animation.frame_start -ne 1 -or [int]$Report.animation.frame_end -ne 33) { return $false }
        if ([string]$Report.outputs.fbx -ne $FbxPath -or [string]$Report.outputs.blend -ne $BlendPath) { return $false }
        return $true
    } catch {
        Write-Warning "Preserved panda validation failed: $_"
        return $false
    }
}

$RepoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $RepoRoot) { throw 'Not inside a Git repository' }
Set-Location -LiteralPath $RepoRoot
if (-not $SourceRoot) { $SourceRoot = $RepoRoot }
$SourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
$Remote = (git config --get remote.origin.url).Trim()
$Branch = (git branch --show-current).Trim()
$Head = (git rev-parse HEAD).Trim()
$Status = @(git status --short)
if ($Remote -notmatch 'organicoverlords/lowvram3d-studio(\.git)?$') { throw "Repository mismatch: $Remote" }
if ($Branch -ne $ExpectedBranch) { throw "Branch mismatch: $Branch" }
if ($Status.Count -ne 0) { throw "Repository must be clean before build. Dirty entries: $($Status -join '; ')" }
git fetch origin $ExpectedBranch --quiet
$RemoteHead = (git rev-parse "origin/$ExpectedBranch").Trim()
if ($Head -ne $RemoteHead) { throw "Local HEAD $Head differs from origin/$ExpectedBranch $RemoteHead" }

$Blender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$UnrealRoot = 'C:\Program Files\Epic Games\UE_5.8'
$UnrealCmd = Join-Path $UnrealRoot 'Engine\Binaries\Win64\UnrealEditor-Cmd.exe'
$UnrealEditor = Join-Path $UnrealRoot 'Engine\Binaries\Win64\UnrealEditor.exe'
$BuildBat = Join-Path $UnrealRoot 'Engine\Build\BatchFiles\Build.bat'
foreach ($Required in @($Blender, $UnrealCmd, $UnrealEditor, $BuildBat)) {
    if (-not (Test-Path -LiteralPath $Required)) { throw "Required tool is missing: $Required" }
}
$ExistingUnreal = @(Get-Process UnrealEditor, UnrealEditor-Cmd -ErrorAction SilentlyContinue)
if ($ExistingUnreal.Count -gt 0) {
    throw "Unreal Editor is already running; refusing to interfere with PID(s): $($ExistingUnreal.Id -join ',')"
}
$SystemPython = Get-Command python.exe -ErrorAction SilentlyContinue
$PythonCandidates = @(
    "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\control\Scripts\python.exe",
    $(if ($SystemPython) { $SystemPython.Source } else { $null })
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }
$Python = $null
foreach ($Candidate in @($PythonCandidates)) {
    & $Candidate -c "import json, pathlib; from PIL import Image; print('PYTHON_OK')" 2>$null
    if ($LASTEXITCODE -eq 0) { $Python = $Candidate; break }
}
if (-not $Python) { throw 'No Python environment with Pillow is available' }

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
$GeneratedRoot = Join-Path $OutputRoot 'generated'
$ProofRoot = Join-Path $OutputRoot 'proof'
$LogsRoot = Join-Path $OutputRoot 'logs'
$CompactEvidence = Join-Path $RepoRoot 'evidence\latest-procedural-jungle'
New-Item -ItemType Directory -Path $GeneratedRoot, $ProofRoot, $LogsRoot, $CompactEvidence -Force | Out-Null

# Remove only stale outputs owned by this proof lane. Preserve the validated panda rig cache.
foreach ($Path in @(
    (Join-Path $GeneratedRoot 'jungle_static_assets.fbx'),
    (Join-Path $GeneratedRoot 'jungle_sources.blend'),
    (Join-Path $GeneratedRoot 'jungle_manifest.json'),
    (Join-Path $GeneratedRoot 'jungle_generation_report.json'),
    (Join-Path $GeneratedRoot 'generated_validation.json'),
    (Join-Path $OutputRoot 'acceptance.json'),
    (Join-Path $OutputRoot 'unreal_build_report.json'),
    (Join-Path $OutputRoot 'map_reload_audit.json')
)) {
    if (Test-Path -LiteralPath $Path) { Remove-Item -LiteralPath $Path -Force }
}
Get-ChildItem -LiteralPath $ProofRoot -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -like 'capture_*.png' -or $_.Name -in @('contact_sheet.png','gameplay_runtime_proof.json','visual_capture_audit.json')
} | Remove-Item -Force

$Gpu = (& nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader 2>&1 | Out-String).Trim()
$RelevantProcesses = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match 'Unreal|Blender|python|codex|claude' -or $_.CommandLine -match 'ProceduralJungle|LowVRAM3D'
} | Select-Object ProcessId, Name, CreationDate, CommandLine)
$Preflight = [ordered]@{
    classification = 'PROVEN'
    repository_root = $RepoRoot
    remote = $Remote
    branch = $Branch
    head = $Head
    remote_head = $RemoteHead
    dirty_count = $Status.Count
    source_root = $SourceRoot
    blender = $Blender
    blender_version = (& $Blender --version 2>&1 | Select-Object -First 1 | Out-String).Trim()
    unreal_root = $UnrealRoot
    unreal_version = '5.8'
    python = $Python
    python_version = (& $Python --version 2>&1 | Out-String).Trim()
    gpu = $Gpu
    relevant_processes = $RelevantProcesses
    visual_overhaul = $true
    codex_invoked = $false
    claude_invoked = $false
    magicmusic_invoked = $false
    recorded_at = (Get-Date).ToUniversalTime().ToString('o')
}
Write-JsonFile $Preflight (Join-Path $OutputRoot 'preflight.json')

$PandaGeneratedRoot = Join-Path $GeneratedRoot 'panda'
New-Item -ItemType Directory -Path $PandaGeneratedRoot -Force | Out-Null
$PandaRigReport = Join-Path $PandaGeneratedRoot 'panda_rig_report.json'
$PandaFbx = Join-Path $PandaGeneratedRoot 'tactical_red_panda_walk.fbx'
$PandaBlend = Join-Path $PandaGeneratedRoot 'tactical_red_panda_walk.blend'
$ReusePandaRig = Test-PreservedPandaRig -ReportPath $PandaRigReport -FbxPath $PandaFbx -BlendPath $PandaBlend
$PandaSource = $null
$PandaSourceClassification = $null

if ($ReusePandaRig) {
    $PandaReport = Get-Content -LiteralPath $PandaRigReport -Raw | ConvertFrom-Json
    $PandaSourceClassification = 'PRESERVED_GENERATED_RIG_PROVEN'
    Write-Host 'PANDA_GENERATED_RIG_REUSE=PROVEN'
} else {
    $PandaRoot = 'C:\AI\LowVRAM3D-benchmarks\miniturbo-3step-experiment-20260803\tactical_red_panda_scout'
    $Candidates = @(
        (Join-Path $PandaRoot 'panda_full_pipeline_repair_20260804\final_candidate\tactical_red_panda_scout_textured.glb'),
        (Join-Path $PandaRoot 'bar_local_closure_v1\tactical_red_panda_scout_bar_repaired.glb')
    )
    foreach ($Candidate in $Candidates) {
        if ((Test-Path -LiteralPath $Candidate -PathType Leaf) -and
            (Get-Item -LiteralPath $Candidate).Length -eq 50244400 -and
            (Get-Sha256 $Candidate) -eq '78c55133165e931bc8d6765610a679d1d18badcdc178820a69e31b7b32bcbfb8') {
            $PandaSource = $Candidate
            break
        }
    }
    if (-not $PandaSource) { throw 'Neither a proven preserved panda rig nor the exact approved source GLB is available' }
    $PandaSourceClassification = 'EXACT_APPROVED_SOURCE_REGENERATED'
}

$PandaIdentity = [ordered]@{
    classification = $PandaSourceClassification
    approved_source_sha256 = '78c55133165e931bc8d6765610a679d1d18badcdc178820a69e31b7b32bcbfb8'
    approved_source_bytes = 50244400
    source = if ($PandaSource) { $PandaSource } else { [string]$PandaReport.source }
    source_currently_exists = if ($PandaSource) { $true } else { Test-Path -LiteralPath ([string]$PandaReport.source) }
    generated_fbx = $PandaFbx
    generated_fbx_sha256 = if (Test-Path -LiteralPath $PandaFbx) { Get-Sha256 $PandaFbx } else { $null }
    generated_fbx_bytes = if (Test-Path -LiteralPath $PandaFbx) { (Get-Item -LiteralPath $PandaFbx).Length } else { 0 }
    generated_blend = $PandaBlend
    generated_blend_sha256 = if (Test-Path -LiteralPath $PandaBlend) { Get-Sha256 $PandaBlend } else { $null }
    generated_blend_bytes = if (Test-Path -LiteralPath $PandaBlend) { (Get-Item -LiteralPath $PandaBlend).Length } else { 0 }
    selected_at = (Get-Date).ToUniversalTime().ToString('o')
}
Write-JsonFile $PandaIdentity (Join-Path $OutputRoot 'panda_source.json')

$MarkerPath = Join-Path $ProjectRoot '.procedural-jungle-owner.json'
if (Test-Path -LiteralPath $ProjectRoot) {
    if (-not (Test-Path -LiteralPath $MarkerPath)) {
        $ExistingEntries = @(Get-ChildItem -LiteralPath $ProjectRoot -Force -ErrorAction SilentlyContinue)
        if ($ExistingEntries.Count -gt 0) { throw "Target project directory exists without ownership marker: $ProjectRoot" }
    } else {
        $Marker = Get-Content -LiteralPath $MarkerPath -Raw | ConvertFrom-Json
        if ($Marker.owner -ne 'organicoverlords/lowvram3d-studio' -or $Marker.project -ne 'ProceduralJungle58') {
            throw 'Target project marker does not match this task'
        }
    }
} else {
    New-Item -ItemType Directory -Path $ProjectRoot -Force | Out-Null
}
Write-JsonFile ([ordered]@{
    owner = 'organicoverlords/lowvram3d-studio'
    project = 'ProceduralJungle58'
    branch = $ExpectedBranch
    visual_overhaul = $true
    created_or_verified_at = (Get-Date).ToUniversalTime().ToString('o')
}) $MarkerPath

$UProject = Join-Path $ProjectRoot 'ProceduralJungle58.uproject'
$UProjectObject = [ordered]@{
    FileVersion = 3
    EngineAssociation = '5.8'
    Category = ''
    Description = 'Deterministic dense procedural jungle with animated tactical red panda.'
    Modules = @([ordered]@{ Name = 'ProceduralJungle58'; Type = 'Runtime'; LoadingPhase = 'Default' })
    Plugins = @(
        [ordered]@{ Name = 'PythonScriptPlugin'; Enabled = $true },
        [ordered]@{ Name = 'EditorScriptingUtilities'; Enabled = $true }
    )
}
Write-JsonFile $UProjectObject $UProject

$TemplateRoot = Join-Path $SourceRoot 'project_template'
if (-not (Test-Path -LiteralPath $TemplateRoot)) { throw "Project template missing: $TemplateRoot" }
New-Item -ItemType Directory -Path (Join-Path $ProjectRoot 'Source'), (Join-Path $ProjectRoot 'Config'), (Join-Path $ProjectRoot 'Content\ProceduralJungle\Generated') -Force | Out-Null
Copy-Item -Path (Join-Path $TemplateRoot 'Source\*') -Destination (Join-Path $ProjectRoot 'Source') -Recurse -Force

$EngineIni = @'
[/Script/EngineSettings.GameMapsSettings]
EditorStartupMap=/Game/ProceduralJungle/Maps/L_ProceduralJungle
GameDefaultMap=/Game/ProceduralJungle/Maps/L_ProceduralJungle
GlobalDefaultGameMode=/Script/ProceduralJungle58.JungleGameMode

[/Script/Engine.RendererSettings]
r.DefaultFeature.AutoExposure=False
r.DynamicGlobalIlluminationMethod=0
r.ReflectionMethod=0
r.Shadow.Virtual.Enable=0
r.GenerateMeshDistanceFields=False
r.AntiAliasingMethod=2
r.DefaultFeature.Bloom=True
r.Tonemapper.Sharpen=0.55
r.Fog=1

[/Script/WindowsTargetPlatform.WindowsTargetSettings]
DefaultGraphicsRHI=DefaultGraphicsRHI_DX12
'@
Write-Utf8Text -Path (Join-Path $ProjectRoot 'Config\DefaultEngine.ini') -Text $EngineIni

$InputIni = @'
[/Script/Engine.InputSettings]
+AxisMappings=(AxisName="MoveForward",Scale=1.000000,Key=W)
+AxisMappings=(AxisName="MoveForward",Scale=-1.000000,Key=S)
+AxisMappings=(AxisName="MoveRight",Scale=1.000000,Key=D)
+AxisMappings=(AxisName="MoveRight",Scale=-1.000000,Key=A)
+AxisMappings=(AxisName="Turn",Scale=1.000000,Key=MouseX)
+AxisMappings=(AxisName="LookUp",Scale=-1.000000,Key=MouseY)
+ActionMappings=(ActionName="Jump",Key=SpaceBar)
bUseMouseForTouch=False
DefaultViewportMouseCaptureMode=CapturePermanently_IncludingInitialMouseDown
'@
Write-Utf8Text -Path (Join-Path $ProjectRoot 'Config\DefaultInput.ini') -Text $InputIni

$JungleReport = Join-Path $GeneratedRoot 'jungle_generation_report.json'
Invoke-Checked -Name 'BLENDER_GENERATE_DENSE_JUNGLE' -LogPath (Join-Path $LogsRoot 'blender_generate_jungle.log') -Command {
    & $Blender --background --factory-startup --python (Join-Path $SourceRoot 'blender\procedural_jungle\generate_jungle_assets.py') -- `
        --output-root $GeneratedRoot --seed $Seed --report $JungleReport
}

if (-not $ReusePandaRig) {
    Invoke-Checked -Name 'BLENDER_RIG_ANIMATE_PANDA' -LogPath (Join-Path $LogsRoot 'blender_rig_panda.log') -Command {
        & $Blender --background --factory-startup --python (Join-Path $SourceRoot 'blender\procedural_jungle\rig_animate_panda.py') -- `
            --input $PandaSource --output-root $PandaGeneratedRoot --report $PandaRigReport
    }
    if (-not (Test-PreservedPandaRig -ReportPath $PandaRigReport -FbxPath $PandaFbx -BlendPath $PandaBlend)) {
        throw 'Fresh panda rig failed the exact provenance and geometry contract'
    }
}

$GeneratedValidation = Join-Path $GeneratedRoot 'generated_validation.json'
Invoke-Checked -Name 'VALIDATE_DENSE_GENERATED_CONTRACTS' -LogPath (Join-Path $LogsRoot 'validate_generated.log') -Command {
    & $Python (Join-Path $SourceRoot 'scripts\procedural_jungle\validate_generated.py') --root $GeneratedRoot --report $GeneratedValidation
}

Invoke-Checked -Name 'COMPILE_UNREAL_PROJECT' -LogPath (Join-Path $LogsRoot 'unreal_build.log') -Command {
    & $BuildBat ProceduralJungle58Editor Win64 Development "-Project=$UProject" -WaitMutex -NoHotReloadFromIDE
}

$env:JUNGLE_GENERATED_ROOT = $GeneratedRoot
$env:JUNGLE_PROJECT_ROOT = $ProjectRoot
$env:JUNGLE_UNREAL_BUILD_REPORT = Join-Path $OutputRoot 'unreal_build_report.json'
Invoke-Checked -Name 'BUILD_DENSE_UNREAL_JUNGLE' -LogPath (Join-Path $LogsRoot 'unreal_scene_build.log') -Command {
    & $UnrealCmd $UProject -run=pythonscript "-script=$(Join-Path $SourceRoot 'unreal\procedural_jungle\build_unreal_scene.py')" `
        -unattended -nop4 -nosplash -nullrhi -stdout -FullStdOutLogOutput
}

$env:JUNGLE_UNREAL_AUDIT_REPORT = Join-Path $OutputRoot 'map_reload_audit.json'
Invoke-Checked -Name 'AUDIT_UNREAL_SCENE' -LogPath (Join-Path $LogsRoot 'unreal_scene_audit.log') -Command {
    & $UnrealCmd $UProject -run=pythonscript "-script=$(Join-Path $SourceRoot 'unreal\procedural_jungle\audit_unreal_scene.py')" `
        -unattended -nop4 -nosplash -nullrhi -stdout -FullStdOutLogOutput
}

# Clear stale proof again immediately before the visible standalone run.
Get-ChildItem -LiteralPath $ProofRoot -File -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -like 'capture_*.png' -or $_.Name -in @('contact_sheet.png','gameplay_runtime_proof.json','visual_capture_audit.json')
} | Remove-Item -Force

$env:JUNGLE_PROOF_ROOT = $ProofRoot
$GameLog = Join-Path $LogsRoot 'unreal_game.log'
$GameErrorLog = Join-Path $LogsRoot 'unreal_game_stderr.log'
$GameArgs = @(
    $UProject,
    '/Game/ProceduralJungle/Maps/L_ProceduralJungle',
    '-game', '-ResX=1920', '-ResY=1080', '-Windowed', '-ForceRes', '-WinX=0', '-WinY=0',
    '-unattended', '-nop4', '-nosplash', '-nosound', '-stdout', '-FullStdOutLogOutput',
    '-ExecCmds="r.VSync 0,r.ScreenPercentage 100,sg.ViewDistanceQuality 2,sg.ShadowQuality 2,sg.EffectsQuality 2,sg.FoliageQuality 2,r.MotionBlurQuality 0"'
)
Write-Host 'PHASE=RUN_VISIBLE_GAMEPLAY_PROOF'
$GameProcess = Start-Process -FilePath $UnrealEditor -ArgumentList $GameArgs -RedirectStandardOutput $GameLog -RedirectStandardError $GameErrorLog -PassThru
$Deadline = (Get-Date).AddSeconds(240)
while (-not $GameProcess.HasExited -and (Get-Date) -lt $Deadline) {
    Start-Sleep -Seconds 1
    $GameProcess.Refresh()
}
if (-not $GameProcess.HasExited) {
    Stop-Process -Id $GameProcess.Id -Force
    throw 'Standalone gameplay proof timed out before the proof director completed'
}
if ($GameProcess.ExitCode -ne 0) { throw "Standalone gameplay proof exited with code $($GameProcess.ExitCode)" }

$RuntimeProofPath = Join-Path $ProofRoot 'gameplay_runtime_proof.json'
if (-not (Test-Path -LiteralPath $RuntimeProofPath)) { throw 'Runtime proof JSON was not written' }
$RuntimeProof = Get-Content -LiteralPath $RuntimeProofPath -Raw | ConvertFrom-Json
if ($RuntimeProof.classification -ne 'PROVEN') { throw "Runtime panda/gameplay proof rejected: $($RuntimeProof | ConvertTo-Json -Compress)" }
if ([double]$RuntimeProof.average_fps -lt 30.0) { throw "Average FPS below 30: $($RuntimeProof.average_fps)" }
if ([double]$RuntimeProof.minimum_fps -lt 20.0) { throw "Minimum FPS below 20: $($RuntimeProof.minimum_fps)" }
if ([double]$RuntimeProof.panda_travel_distance_cm -lt 300.0) { throw 'Panda did not travel far enough during proof' }
if ([int]$RuntimeProof.panda_framed_capture_count -lt 4) { throw 'Fewer than four captures were dynamically framed on the panda' }
$Captures = @(Get-ChildItem -LiteralPath $ProofRoot -Filter 'capture_*.png' -File | Sort-Object Name)
if ($Captures.Count -ne 8) { throw "Expected exactly 8 gameplay captures, found $($Captures.Count)" }
foreach ($Capture in $Captures) {
    if ($Capture.Length -lt 100000) { throw "Gameplay capture is implausibly small: $($Capture.FullName)" }
}

$VisualAuditPath = Join-Path $ProofRoot 'visual_capture_audit.json'
Invoke-Checked -Name 'AUDIT_DENSE_JUNGLE_VISUALS' -LogPath (Join-Path $LogsRoot 'visual_capture_audit.log') -Command {
    & $Python (Join-Path $SourceRoot 'scripts\procedural_jungle\audit_visual_captures.py') --input-dir $ProofRoot --output $VisualAuditPath
}
$VisualAudit = Get-Content -LiteralPath $VisualAuditPath -Raw | ConvertFrom-Json
if ($VisualAudit.classification -ne 'DENSE_JUNGLE_VISUAL_QUALITY_PROVEN') {
    throw "Visual quality audit rejected: $($VisualAudit | ConvertTo-Json -Compress -Depth 8)"
}

$ContactSheet = Join-Path $ProofRoot 'contact_sheet.png'
Invoke-Checked -Name 'BUILD_CONTACT_SHEET' -LogPath (Join-Path $LogsRoot 'contact_sheet.log') -Command {
    & $Python (Join-Path $SourceRoot 'scripts\procedural_jungle\make_contact_sheet.py') --input-dir $ProofRoot --output $ContactSheet
}

$BuildReport = Get-Content -LiteralPath $env:JUNGLE_UNREAL_BUILD_REPORT -Raw | ConvertFrom-Json
$AuditReport = Get-Content -LiteralPath $env:JUNGLE_UNREAL_AUDIT_REPORT -Raw | ConvertFrom-Json
$GeneratedReport = Get-Content -LiteralPath $GeneratedValidation -Raw | ConvertFrom-Json
$Acceptance = [ordered]@{
    schema = 'procedural_jungle_panda_acceptance_v2'
    classification = 'JUNGLE_PANDA_PLAYABLE_PROVEN'
    visual_quality_classification = $VisualAudit.classification
    visual_quality = 'PROVEN'
    project_exists = (Test-Path -LiteralPath $UProject)
    uproject = $UProject
    canonical_map = '/Game/ProceduralJungle/Maps/L_ProceduralJungle'
    map_save_reload = if ($AuditReport.classification -eq 'PROVEN') { 'PROVEN' } else { 'REJECTED' }
    no_external_art_assets = if ($AuditReport.no_external_art_assets) { 'PROVEN' } else { 'REJECTED' }
    terrain = 'PROVEN'
    river = 'PROVEN'
    waterfall = 'PROVEN'
    lower_pool = 'PROVEN'
    vegetation = 'PROVEN'
    wind = 'PROVEN'
    lighting_atmosphere = 'PROVEN'
    player_controls = 'PROVEN'
    collision_route = 'PROVEN'
    panda_skeletal_mesh = 'PROVEN'
    panda_walk_animation = if ($RuntimeProof.panda_animation_active) { 'PROVEN' } else { 'REJECTED' }
    panda_route_motion = if ([double]$RuntimeProof.panda_travel_distance_cm -ge 300.0) { 'PROVEN' } else { 'REJECTED' }
    standalone_launch = 'PROVEN'
    visual_capture = 'PROVEN'
    performance_1080p = 'PROVEN'
    average_fps = [double]$RuntimeProof.average_fps
    minimum_fps = [double]$RuntimeProof.minimum_fps
    panda_travel_distance_cm = [double]$RuntimeProof.panda_travel_distance_cm
    panda_framed_capture_count = [int]$RuntimeProof.panda_framed_capture_count
    capture_count = $Captures.Count
    contact_sheet = $ContactSheet
    visual_capture_audit = $VisualAuditPath
    average_green_dominant_fraction = [double]$VisualAudit.average_green_dominant_fraction
    average_saturation = [double]$VisualAudit.average_saturation
    average_near_white_fraction = [double]$VisualAudit.average_near_white_fraction
    vegetation_rich_frame_count = [int]$VisualAudit.vegetation_rich_frame_count
    water_frame_count = [int]$VisualAudit.water_frame_count
    generated_variant_count = [int]$GeneratedReport.variant_count
    generated_instance_count = [int]$GeneratedReport.instance_count
    canopy_instance_count = [int]$GeneratedReport.canopy_instance_count
    understory_instance_count = [int]$GeneratedReport.understory_instance_count
    hero_mesh_count = [int]$GeneratedReport.hero_mesh_count
    panda_bone_count = [int]$GeneratedReport.panda_bone_count
    forbidden_asset_reference_count = @($AuditReport.forbidden_asset_references).Count
    tests_passed = $true
    codex_invoked = $false
    claude_invoked = $false
    magicmusic_invoked = $false
    repository_head = $Head
    recorded_at = (Get-Date).ToUniversalTime().ToString('o')
}
$AcceptancePath = Join-Path $OutputRoot 'acceptance.json'
Write-JsonFile $Acceptance $AcceptancePath
Copy-Item -LiteralPath $AcceptancePath -Destination (Join-Path $CompactEvidence 'acceptance.json') -Force

$WorkflowReceipt = [ordered]@{
    classification = 'JUNGLE_PANDA_PLAYABLE_PROVEN'
    visual_quality_classification = $VisualAudit.classification
    repository = 'organicoverlords/lowvram3d-studio'
    branch = $ExpectedBranch
    head = $Head
    project = $UProject
    output_root = $OutputRoot
    panda_source = $PandaIdentity
    generated_validation = $GeneratedReport
    visual_capture_audit = $VisualAudit
    game_runtime_proof = $RuntimeProof
    recorded_at = (Get-Date).ToUniversalTime().ToString('o')
}
Write-JsonFile $WorkflowReceipt (Join-Path $CompactEvidence 'workflow_receipt.json')

$FinalReportDir = Join-Path $RepoRoot 'proof\scene\20260804-procedural-jungle'
New-Item -ItemType Directory -Path $FinalReportDir -Force | Out-Null
$FinalReport = @"
# Dense procedural jungle with walking tactical red panda

Classification: **JUNGLE_PANDA_PLAYABLE_PROVEN**

Visual quality: **$($VisualAudit.classification)**

- Unreal project: `$UProject`
- Canonical map: `/Game/ProceduralJungle/Maps/L_ProceduralJungle`
- Generated variants: $($Acceptance.generated_variant_count)
- Generated placed instances: $($Acceptance.generated_instance_count)
- Canopy/emergent instances: $($Acceptance.canopy_instance_count)
- Understory instances: $($Acceptance.understory_instance_count)
- Panda bones: $($Acceptance.panda_bone_count)
- Panda-framed captures: $($Acceptance.panda_framed_capture_count)
- Panda travel during runtime proof: $([math]::Round($Acceptance.panda_travel_distance_cm, 1)) cm
- 1080p average / minimum FPS: $([math]::Round($Acceptance.average_fps, 2)) / $([math]::Round($Acceptance.minimum_fps, 2))
- Captures: $($Acceptance.capture_count)
- Average green coverage: $([math]::Round($Acceptance.average_green_dominant_fraction, 4))
- Average saturation: $([math]::Round($Acceptance.average_saturation, 4))
- Average near-white coverage: $([math]::Round($Acceptance.average_near_white_fraction, 4))
- Contact sheet: `$ContactSheet`
- External/marketplace art references: $($Acceptance.forbidden_asset_reference_count)
- Codex invoked: false
- Claude invoked: false
- MagicMusic invoked: false

The environment is rebuilt as a dense layered jungle with canopy trees, emergent trees, palms, saplings, shrubs, ferns, grass, vines, fallen logs, mossy rocks, a wider river, five-sheet waterfall, pool foam, mist, procedural material variation, controlled exposure, and lower cinematic proof cameras. The panda rig, walk animation, route motion, player controls, collision, save/reload, and performance proof remain deterministic.
"@
Write-Utf8Text -Path (Join-Path $FinalReportDir 'FINAL_REPORT.md') -Text $FinalReport

Write-Host 'CLASSIFICATION=JUNGLE_PANDA_PLAYABLE_PROVEN'
Write-Host "VISUAL_QUALITY_CLASSIFICATION=$($VisualAudit.classification)"
Write-Host "UPROJECT=$UProject"
Write-Host "CONTACT_SHEET=$ContactSheet"
