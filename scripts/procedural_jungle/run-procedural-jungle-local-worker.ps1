[CmdletBinding()]
param(
    [string]$ExpectedBranch = 'feature/procedural-jungle-playable-20260804'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (git rev-parse --show-toplevel).Trim()
if (-not $RepoRoot) { throw 'Not inside the expected Git repository' }
Set-Location -LiteralPath $RepoRoot
$Remote = (git config --get remote.origin.url).Trim()
$Branch = (git branch --show-current).Trim()
$Head = (git rev-parse HEAD).Trim()
if ($Remote -notmatch 'organicoverlords/lowvram3d-studio(\.git)?$') { throw "Repository mismatch: $Remote" }
if ($Branch -ne $ExpectedBranch) { throw "Branch mismatch: $Branch" }

$ParentCommit = '9d293ff814c5d6cd5d96196b40abd8d7c56d8b93'
$ParentBlob = 'a3f5721d5f4b086152ba4ac8fb2591cc49bfd799'
$WrapperPath = 'scripts/procedural_jungle/run-procedural-jungle-local-worker.ps1'
& git merge-base --is-ancestor $ParentCommit HEAD
if ($LASTEXITCODE -ne 0) { throw "Parent recovery commit is not an ancestor: $ParentCommit" }
$ActualParentBlob = (& git rev-parse "$ParentCommit`:$WrapperPath").Trim()
if ($LASTEXITCODE -ne 0 -or $ActualParentBlob -ne $ParentBlob) {
    throw "Parent recovery identity mismatch: expected=$ParentBlob actual=$ActualParentBlob"
}

$TempRoot = Join-Path $env:RUNNER_TEMP "procedural-jungle-runtime-fix-$Head"
if (Test-Path -LiteralPath $TempRoot) { Remove-Item -LiteralPath $TempRoot -Recurse -Force }
New-Item -ItemType Directory -Path $TempRoot -Force | Out-Null
$PatchedParent = Join-Path $TempRoot 'run-procedural-jungle-runtime-fixed.ps1'
$ParentLines = @(& git show "$ParentCommit`:$WrapperPath")
if ($LASTEXITCODE -ne 0 -or $ParentLines.Count -lt 10) { throw 'Could not read parent recovery wrapper' }
$ParentText = $ParentLines -join "`n"

$ParentInsertion = '$RecoveryText = $RecoveryText.Replace($OldRead, $NewRead).Replace($OldWrite, $NewWrite)'
if ([regex]::Matches($ParentText, [regex]::Escape($ParentInsertion)).Count -ne 1) {
    throw 'Could not prove unique parent runtime-fix insertion point'
}

$ParentAugmentation = @'
$RuntimeSourceMarker = '# The approved source GLB disappeared after the rig was generated. Reuse is allowed only when'
if ([regex]::Matches($RecoveryText, [regex]::Escape($RuntimeSourceMarker)).Count -ne 1) {
    throw 'Could not prove unique runtime source-fix marker in recovery wrapper'
}
$RuntimeSourceFix = @'
# Runtime run 30867542882 proved that the panda animation played but the unpossessed
# CharacterMovement component accumulated zero travel. Allow movement physics without
# a controller while retaining the deterministic route and collision sweep.
$PandaCpp = Join-Path $SourceRoot 'project_template\Source\ProceduralJungle58\PandaWalkerCharacter.cpp'
if (-not (Test-Path -LiteralPath $PandaCpp -PathType Leaf)) { throw "Panda runtime source missing: $PandaCpp" }
$PandaCppText = Get-Content -LiteralPath $PandaCpp -Raw
$MovementAnchor = '    GetCharacterMovement()->bOrientRotationToMovement = false;'
if ([regex]::Matches($PandaCppText, [regex]::Escape($MovementAnchor)).Count -ne 1) {
    throw 'Could not prove unique controller-free panda movement anchor'
}
$PandaCppText = $PandaCppText.Replace($MovementAnchor, $MovementAnchor + "`r`n" + '    GetCharacterMovement()->bRunPhysicsWithNoController = true;')
Set-Content -LiteralPath $PandaCpp -Value $PandaCppText -Encoding utf8
Write-Host 'PANDA_CONTROLLER_FREE_MOVEMENT_PATCH=PROVEN'

# Request each screenshot on the frame after the player view target changes. The rejected
# run requested it in the same frame and captured a black pre-camera backbuffer eight times.
$ProofCpp = Join-Path $SourceRoot 'project_template\Source\ProceduralJungle58\JungleProofDirector.cpp'
if (-not (Test-Path -LiteralPath $ProofCpp -PathType Leaf)) { throw "Proof director source missing: $ProofCpp" }
$ProofText = Get-Content -LiteralPath $ProofCpp -Raw
$OldCapture = @'
    APlayerController* PC = UGameplayStatics::GetPlayerController(GetWorld(), 0);
    if (PC)
    {
        PC->SetViewTarget(Cameras[CaptureIndex % Cameras.Num()]);
    }
    const FString Filename = FPaths::Combine(ProofRoot, FString::Printf(TEXT("capture_%02d.png"), CaptureIndex));
    FScreenshotRequest::RequestScreenshot(Filename, false, false);
    UE_LOG(LogTemp, Display, TEXT("JUNGLE_PROOF_CAPTURE=%s"), *Filename);
    ++CaptureIndex;
'@
$NewCapture = @'
    APlayerController* PC = UGameplayStatics::GetPlayerController(GetWorld(), 0);
    if (!PC)
    {
        UE_LOG(LogTemp, Error, TEXT("JUNGLE_PROOF_PLAYER_CONTROLLER_MISSING"));
        FinishProof();
        return;
    }
    ACameraActor* Camera = Cameras[CaptureIndex % Cameras.Num()];
    PC->SetViewTarget(Camera);
    const FString Filename = FPaths::Combine(ProofRoot, FString::Printf(TEXT("capture_%02d.png"), CaptureIndex));
    GetWorldTimerManager().SetTimerForNextTick(FTimerDelegate::CreateWeakLambda(this, [this, Filename, Camera]()
    {
        FScreenshotRequest::RequestScreenshot(Filename, false, false);
        UE_LOG(LogTemp, Display, TEXT("JUNGLE_PROOF_CAPTURE=%s CAMERA=%s LOCATION=%s"), *Filename, *GetNameSafe(Camera), *Camera->GetActorLocation().ToString());
        ++CaptureIndex;
    }));
'@
$ProofNormalized = $ProofText -replace "`r`n", "`n"
$OldCaptureNormalized = $OldCapture -replace "`r`n", "`n"
if ([regex]::Matches($ProofNormalized, [regex]::Escape($OldCaptureNormalized)).Count -ne 1) {
    throw 'Could not prove unique same-frame screenshot block'
}
$ProofNormalized = $ProofNormalized.Replace($OldCaptureNormalized, ($NewCapture -replace "`r`n", "`n"))
Set-Content -LiteralPath $ProofCpp -Value $ProofNormalized -Encoding utf8
Write-Host 'JUNGLE_DELAYED_VIEWTARGET_CAPTURE_PATCH=PROVEN'

# The rejected images were exact RGB zero. Add an actual daylight sky and physically
# plausible sun intensity, and mark generated materials for HISM runtime use.
$RuntimeSceneScript = Join-Path $SourceRoot 'unreal\procedural_jungle\build_unreal_scene.py'
$RuntimeSceneText = (Get-Content -LiteralPath $RuntimeSceneScript -Raw) -replace "`r`n", "`n"
$SunOld = "    set_light_component(sun, intensity=7.5, cast_shadows=True)"
$SunNew = "    set_light_component(sun, intensity=75000.0, cast_shadows=True, atmosphere_sun_light=True)"
if ([regex]::Matches($RuntimeSceneText, [regex]::Escape($SunOld)).Count -ne 1) { throw 'Could not prove unique jungle sun intensity line' }
$RuntimeSceneText = $RuntimeSceneText.Replace($SunOld, $SunNew)
$SunAppend = "    actors.append(sun)"
$AtmosphereBlock = @'
    actors.append(sun)
    atmosphere = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.SkyAtmosphere, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
    atmosphere.set_actor_label('JungleSkyAtmosphere')
    actors.append(atmosphere)
'@
if ([regex]::Matches($RuntimeSceneText, [regex]::Escape($SunAppend)).Count -ne 1) { throw 'Could not prove unique sun append line' }
$RuntimeSceneText = $RuntimeSceneText.Replace($SunAppend, ($AtmosphereBlock -replace "`r`n", "`n").TrimEnd("`n"))
$SkyIntensityOld = "        comp.set_editor_property('intensity_scale', 0.75)"
$SkyIntensityNew = "        comp.set_editor_property('intensity_scale', 1.0)"
if ([regex]::Matches($RuntimeSceneText, [regex]::Escape($SkyIntensityOld)).Count -ne 1) { throw 'Could not prove unique skylight intensity line' }
$RuntimeSceneText = $RuntimeSceneText.Replace($SkyIntensityOld, $SkyIntensityNew)
$MaterialAnchor = "    material.set_editor_property('two_sided', bool(wind or translucent))"
$MaterialUsage = @'
    material.set_editor_property('two_sided', bool(wind or translucent))
    try:
        material.set_editor_property('used_with_instanced_static_meshes', True)
    except Exception as exc:
        log(f'instanced material usage warning for {name}: {exc}')
'@
if ([regex]::Matches($RuntimeSceneText, [regex]::Escape($MaterialAnchor)).Count -ne 1) { throw 'Could not prove unique material usage anchor' }
$RuntimeSceneText = $RuntimeSceneText.Replace($MaterialAnchor, ($MaterialUsage -replace "`r`n", "`n").TrimEnd("`n"))
Set-Content -LiteralPath $RuntimeSceneScript -Value $RuntimeSceneText -Encoding utf8

$AuditScript = Join-Path $SourceRoot 'unreal\procedural_jungle\audit_unreal_scene.py'
$AuditText = (Get-Content -LiteralPath $AuditScript -Raw) -replace "`r`n", "`n"
$AuditOld = "        'JungleProofDirector', 'JungleSun', 'JungleSkyLight', 'JungleHeightFog'"
$AuditNew = "        'JungleProofDirector', 'JungleSun', 'JungleSkyAtmosphere', 'JungleSkyLight', 'JungleHeightFog'"
if ([regex]::Matches($AuditText, [regex]::Escape($AuditOld)).Count -ne 1) { throw 'Could not prove unique atmosphere audit line' }
$AuditText = $AuditText.Replace($AuditOld, $AuditNew)
Set-Content -LiteralPath $AuditScript -Value $AuditText -Encoding utf8
Write-Host 'JUNGLE_VISIBLE_DAYLIGHT_ATMOSPHERE_PATCH=PROVEN'
'@
$RecoveryText = $RecoveryText.Replace($RuntimeSourceMarker, $RuntimeSourceFix + "`n`n" + $RuntimeSourceMarker)

$RuntimeBuildMarker = '$BuildText = $BuildText.Replace($RigOld, $RigNew)'
if ([regex]::Matches($RecoveryText, [regex]::Escape($RuntimeBuildMarker)).Count -ne 1) {
    throw 'Could not prove unique runtime build-text marker in recovery wrapper'
}
$RuntimeBuildFix = @'

# Render through a real 1080p game viewport. Run 30867542882 proved -RenderOffScreen ignored
# the requested resolution and produced eight byte-identical 888x500 black frames.
$OldGameArgs = "    '-game', '-RenderOffScreen', '-ResX=1920', '-ResY=1080', '-Windowed',"
$NewGameArgs = "    '-game', '-ResX=1920', '-ResY=1080', '-ForceRes', '-Windowed',"
if ([regex]::Matches($BuildText, [regex]::Escape($OldGameArgs)).Count -ne 1) {
    throw 'Could not prove unique offscreen gameplay argument block'
}
$BuildText = $BuildText.Replace($OldGameArgs, $NewGameArgs)

# Do not accept merely existing PNGs. Require diverse images with nonzero luminance and range.
$ContactSheetMarker = '$ContactSheet = Join-Path $ProofRoot ''contact_sheet.png'''
if ([regex]::Matches($BuildText, [regex]::Escape($ContactSheetMarker)).Count -ne 1) {
    throw 'Could not prove unique contact-sheet insertion marker'
}
$VisualAudit = @'
$CaptureHashes = @($Captures | ForEach-Object { (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash })
if (@($CaptureHashes | Sort-Object -Unique).Count -lt 6) { throw 'Gameplay captures are insufficiently distinct' }
$CaptureAuditCode = @"
from pathlib import Path
from PIL import Image, ImageStat
import sys
root = Path(sys.argv[1])
paths = sorted(root.glob('capture_*.png'))
if len(paths) < 8:
    raise SystemExit('capture count below 8')
for path in paths:
    with Image.open(path) as image:
        rgb = image.convert('RGB')
        stat = ImageStat.Stat(rgb)
        mean = sum(stat.mean) / 3.0
        lo, hi = rgb.getextrema()[0][0], max(channel[1] for channel in rgb.getextrema())
        if mean <= 1.0 or hi <= 16 or hi - lo <= 8:
            raise SystemExit(f'capture is visually blank: {path.name} mean={mean:.3f} range={lo}-{hi}')
print('JUNGLE_CAPTURE_VISUAL_AUDIT=PROVEN')
"@
& $Python -c $CaptureAuditCode $ProofRoot
if ($LASTEXITCODE -ne 0) { throw 'Gameplay capture visual audit failed' }

'@
$BuildText = $BuildText.Replace($ContactSheetMarker, $VisualAudit + $ContactSheetMarker)
'@
$RecoveryText = $RecoveryText.Replace($RuntimeBuildMarker, $RuntimeBuildMarker + $RuntimeBuildFix)
'@

$PatchedParentText = $ParentText.Replace($ParentInsertion, $ParentInsertion + "`n" + $ParentAugmentation)
Set-Content -LiteralPath $PatchedParent -Value $PatchedParentText -Encoding utf8
Write-Host "PARENT_RECOVERY_COMMIT=$ParentCommit"
Write-Host "PARENT_RECOVERY_BLOB=$ParentBlob"
Write-Host 'JUNGLE_RUNTIME_MOVEMENT_CAPTURE_FIX_INJECTION=PROVEN'

& $PatchedParent -ExpectedBranch $ExpectedBranch
$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 0) { throw "Runtime-fixed direct worker failed with exit code $ExitCode" }
