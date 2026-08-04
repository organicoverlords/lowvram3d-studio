param(
    [Parameter(Mandatory=$true)][string]$Image,
    [Parameter(Mandatory=$true)][string]$Project,
    [Parameter(Mandatory=$true)][string]$SceneId,
    [Parameter(Mandatory=$true)][string]$OutputMap,
    [Parameter(Mandatory=$false)][switch]$Resume,
    [Parameter(Mandatory=$false)][ValidateSet('smoke','preview','quality')][string]$QualityTier='smoke',
    [Parameter(Mandatory=$true)][string]$EvidenceRoot
)
$env:PYTHONPATH = (Join-Path (Get-Location) 'src')
$arguments = @('-m','lowvram3d.image_to_scene_pipeline','--image',$Image,'--project',$Project,'--scene-id',$SceneId,'--output-map',$OutputMap,'--quality-tier',$QualityTier,'--evidence-root',$EvidenceRoot)
if ($Resume) { $arguments += '--resume' }
python @arguments
exit $LASTEXITCODE
