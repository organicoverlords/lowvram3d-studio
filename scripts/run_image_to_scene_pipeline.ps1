param(
    [Parameter(Mandatory=$true)][string]$Image,
    [Parameter(Mandatory=$true)][string]$Project,
    [Parameter(Mandatory=$true)][string]$SceneId,
    [Parameter(Mandatory=$false)][string]$OutputRoot='/Game/GeneratedScenes',
    [Parameter(Mandatory=$false)][string]$OutputMap,
    [Parameter(Mandatory=$false)][switch]$Resume,
    [Parameter(Mandatory=$false)][ValidateSet('smoke','preview','quality')][string]$QualityTier='smoke',
    [Parameter(Mandatory=$false)][string]$EvidenceRoot,
    [Parameter(Mandatory=$false)][string]$SceneSpec,
    [Parameter(Mandatory=$false)][string]$RunId='run-current',
    [Parameter(Mandatory=$false)][int]$MaxVramMb=6144,
    [Parameter(Mandatory=$false)][int]$MaxTriangles=1500000,
    [Parameter(Mandatory=$false)][switch]$DisableNeural,
    [Parameter(Mandatory=$false)][switch]$EnablePcg
)
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$arguments = @('-m','lowvram3d.image_to_scene_pipeline','--image',$Image,'--project',$Project,'--scene-id',$SceneId,'--run-id',$RunId,'--output-root',$OutputRoot,'--quality-tier',$QualityTier,'--max-vram-mb',$MaxVramMb,'--max-triangles',$MaxTriangles)
if ($OutputMap) { $arguments += @('--output-map',$OutputMap) }
if ($EvidenceRoot) { $arguments += @('--evidence-root',$EvidenceRoot) }
if ($SceneSpec) { $arguments += @('--scene-spec',$SceneSpec) }
if ($Resume) { $arguments += '--resume' }
if ($DisableNeural) { $arguments += '--disable-neural' }
if ($EnablePcg) { $arguments += '--enable-pcg' }
python @arguments
exit $LASTEXITCODE
