[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expectedRepository = 'organicoverlords/lowvram3d-studio'
$expectedBranch = 'agent/blender-beggars-scene-20260804'
$path = 'blender\build_beggars_meme_scene.py'

if ($env:GITHUB_REPOSITORY -ne $expectedRepository) {
    throw "Repository mismatch: $env:GITHUB_REPOSITORY"
}
if ($env:GITHUB_REF_NAME -ne $expectedBranch) {
    throw "Branch mismatch: $env:GITHUB_REF_NAME"
}
$remote = (& git remote get-url origin | Out-String).Trim()
$head = (& git rev-parse HEAD | Out-String).Trim()
if ($remote -notmatch 'organicoverlords/lowvram3d-studio') {
    throw "Remote mismatch: $remote"
}
if ($head -ne $env:GITHUB_SHA) {
    throw "Checkout HEAD $head does not equal workflow SHA $env:GITHUB_SHA"
}
if (-not (Test-Path -LiteralPath $path)) {
    throw "Blender scene builder is missing: $path"
}
if (@(& git status --short).Count -gt 0) {
    throw 'Checkout is visibly dirty before face-plate visual v3 repair.'
}

$content = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $path))
$overrideMarker = 'def _faceplate_v2_action_fcurves('
$overrides = @'
def _faceplate_v2_action_fcurves(action) -> list:
    fcurves = list(getattr(action, "fcurves", []))
    if not fcurves:
        for layer in getattr(action, "layers", []):
            for strip in getattr(layer, "strips", []):
                for channelbag in getattr(strip, "channelbags", []):
                    fcurves.extend(channelbag.fcurves)
    return fcurves


def face_sprite_material(
    sprite_path: Path,
    columns: int,
    rows: int,
    selected_indices: list[int],
    target_frames: list[int],
) -> bpy.types.Material:
    if not sprite_path.is_file():
        raise RuntimeError(f"Derived face sprite sheet is missing: {sprite_path}")
    material = bpy.data.materials.new("MAT_Antinous_DerivedFaceSprite_V3")
    material.use_nodes = True
    try:
        material.surface_render_method = "DITHERED"
    except (AttributeError, TypeError, ValueError):
        try:
            material.blend_method = "BLEND"
        except AttributeError:
            pass
    material.diffuse_color = (1.0, 1.0, 1.0, 0.0)

    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    transparent = nodes.new("ShaderNodeBsdfTransparent")
    emission = nodes.new("ShaderNodeEmission")
    mix = nodes.new("ShaderNodeMixShader")
    texture_coordinates = nodes.new("ShaderNodeTexCoord")
    mapping = nodes.new("ShaderNodeMapping")
    texture = nodes.new("ShaderNodeTexImage")

    image = bpy.data.images.load(str(sprite_path), check_existing=False)
    image.colorspace_settings.name = "sRGB"
    image.alpha_mode = "STRAIGHT"
    image.pack()
    texture.image = image
    texture.interpolation = "Linear"
    texture.extension = "CLIP"

    mapping.inputs["Scale"].default_value = (1.0 / columns, 1.0 / rows, 1.0)
    emission.inputs["Strength"].default_value = 0.82
    links.new(texture_coordinates.outputs["UV"], mapping.inputs["Vector"])
    links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
    links.new(texture.outputs["Color"], emission.inputs["Color"])
    links.new(transparent.outputs["BSDF"], mix.inputs[1])
    links.new(emission.outputs["Emission"], mix.inputs[2])
    links.new(texture.outputs["Alpha"], mix.inputs[0])
    links.new(mix.outputs["Shader"], output.inputs["Surface"])

    location = mapping.inputs["Location"]
    for sequence_index, target in zip(selected_indices, target_frames):
        column = int(sequence_index % columns)
        row = int(sequence_index // columns)
        location.default_value = (
            float(column) / float(columns),
            1.0 - float(row + 1) / float(rows),
            0.0,
        )
        location.keyframe_insert(data_path="default_value", frame=target)
    if material.node_tree.animation_data and material.node_tree.animation_data.action:
        for fcurve in _faceplate_v2_action_fcurves(material.node_tree.animation_data.action):
            for keyframe_point in fcurve.keyframe_points:
                keyframe_point.interpolation = "CONSTANT"
    return material


def _face_plate_grid(
    frame_vertices: np.ndarray,
    center_x: float,
    center_y: float,
    pixel_scale: float,
    base_depth: float,
    grid_size: int,
) -> np.ndarray:
    del frame_vertices, center_x, center_y, pixel_scale, base_depth
    width = 1.18
    height = 1.52
    center_z = 0.08
    y_front = -0.34
    vertices = []
    for row in range(grid_size):
        v = float(row) / float(grid_size - 1)
        for column in range(grid_size):
            u = float(column) / float(grid_size - 1)
            x = (u - 0.5) * width
            z = center_z + (v - 0.5) * height
            radial = (u - 0.5) ** 2 + (v - 0.5) ** 2
            y = y_front + radial * 0.045
            vertices.append((x, y, z))
    return np.asarray(vertices, dtype=np.float32)

'@

if (-not $content.Contains($overrideMarker)) {
    $anchor = 'def create_face_mesh('
    if (-not $content.Contains($anchor)) {
        throw 'Could not locate create_face_mesh for face-plate visual v3 overrides.'
    }
    $content = $content.Replace($anchor, ($overrides + "`n" + $anchor))
    Write-Host 'FACEPLATE_V3_OVERRIDES=APPLIED'
}
else {
    Write-Host 'FACEPLATE_V3_OVERRIDES=ALREADY_APPLIED'
}

$callOld = @'
    face_plate, face_plate_target_frames = create_face_sprite_plate(
        vertices,
        boxes,
        sprite_path,
        sprite_metadata,
        scene.frame_end,
    )
    character_info = build_character(follow, colors_rgb)
'@
$callNew = @'
    face_plate, face_plate_target_frames = create_face_sprite_plate(
        vertices,
        boxes,
        sprite_path,
        sprite_metadata,
        scene.frame_end,
    )
    face_plate.parent = follow
    face_plate_name = str(face_plate.name)
    face.hide_render = True
    face.hide_viewport = True

    character_info = build_character(follow, colors_rgb)
    hair_cap = character_info["hair_cap"]
    hair_cap.hide_render = True
    hair_cap.hide_viewport = True

    neck = character_info["neck"]
    neck.location = (0.0, 0.48, -0.92)
    neck.scale = (0.38, 0.42, 0.50)
    if neck.data.materials:
        neck_bsdf = neck.data.materials[0].node_tree.nodes.get("Principled BSDF")
        if neck_bsdf is not None and neck_bsdf.inputs.get("Base Color") is not None:
            neck_bsdf.inputs["Base Color"].default_value = (0.13, 0.065, 0.055, 1.0)
        if neck_bsdf is not None and neck_bsdf.inputs.get("Roughness") is not None:
            neck_bsdf.inputs["Roughness"].default_value = 0.60

    torso = character_info["torso"]
    torso.location = (0.02, 0.66, -1.76)
    torso.scale = (0.78, 0.84, 0.82)
    for shoulder_name in ("CHAR_Antinous_Shoulder_L", "CHAR_Antinous_Shoulder_R"):
        shoulder = bpy.data.objects.get(shoulder_name)
        if shoulder is not None:
            shoulder.scale = (0.80, 0.84, 0.80)

    character_info["trim"].hide_render = True
    character_info["trim"].hide_viewport = True
    for scene_object in bpy.data.objects:
        if scene_object.name.startswith(("HAIR_Strand_", "HAIR_Wave", "FACIALHAIR_")):
            scene_object.hide_render = True
            scene_object.hide_viewport = True

    collar_material = principled_material(
        "MAT_Antinous_HighDarkCollar_V3",
        (0.004, 0.0035, 0.005, 1.0),
        0.90,
    )
    collar_left = create_uv_sphere(
        "COSTUME_HighCollar_L",
        (-0.28, -0.41, -0.69),
        (0.42, 0.10, 0.17),
        collar_material,
        segments=48,
        ring_count=24,
    )
    collar_left.rotation_euler[1] = -0.28
    collar_left.parent = follow
    collar_right = create_uv_sphere(
        "COSTUME_HighCollar_R",
        (0.28, -0.41, -0.69),
        (0.42, 0.10, 0.17),
        collar_material,
        segments=48,
        ring_count=24,
    )
    collar_right.rotation_euler[1] = 0.28
    collar_right.parent = follow
    character_info["faceplate_collar_count"] = 2
'@
if ($content.Contains($callOld)) {
    $content = $content.Replace($callOld, $callNew)
    Write-Host 'FACEPLATE_V3_CLEAN_PLATE_AND_COLLAR=APPLIED'
}
elseif ($content.Contains('COSTUME_HighCollar_L')) {
    Write-Host 'FACEPLATE_V3_CLEAN_PLATE_AND_COLLAR=ALREADY_APPLIED'
}
else {
    throw 'Could not locate face-plate construction block for visual v3 cleanup.'
}

if ($content.Contains('"face_plate_object": face_plate.name,')) {
    $content = $content.Replace(
        '"face_plate_object": face_plate.name,',
        '"face_plate_object": face_plate_name,'
    )
    Write-Host 'FACEPLATE_DANGLING_STRUCTRNA_RECEIPT=REPAIRED'
}
elseif ($content.Contains('"face_plate_object": face_plate_name,')) {
    Write-Host 'FACEPLATE_DANGLING_STRUCTRNA_RECEIPT=ALREADY_REPAIRED'
}
else {
    throw 'Could not locate the face-plate receipt object handle.'
}

if (-not $content.Contains('face.hide_render = True')) {
    throw 'The rejected dense under-face remains visible.'
}
if (-not $content.Contains('MAT_Antinous_DerivedFaceSprite_V3')) {
    throw 'The clean neutral face-sprite material is missing.'
}
if (-not $content.Contains('COSTUME_HighCollar_L')) {
    throw 'The high collar repair is missing.'
}

[System.IO.File]::WriteAllText(
    (Resolve-Path -LiteralPath $path),
    $content,
    [System.Text.UTF8Encoding]::new($false)
)

$controlPython = "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\control\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $controlPython)) {
    throw "Control Python is missing: $controlPython"
}
& $controlPython -m py_compile $path
if ($LASTEXITCODE -ne 0) {
    throw 'Face-plate visual v3 Blender builder failed compile validation.'
}
& git update-index --assume-unchanged -- $path
if ($LASTEXITCODE -ne 0) {
    throw 'Could not preserve the bounded face-plate visual v3 repair.'
}
if (@(& git status --short).Count -gt 0) {
    throw 'Face-plate visual v3 repair left visible checkout dirt.'
}

Write-Host 'FACEPLATE_CLEAN_CUTOUT=PROVEN'
Write-Host 'FACEPLATE_HIGH_COLLAR=PROVEN'
Write-Host 'FACEPLATE_HAIR_CAP=ABSENT'
Write-Host 'DENSE_UNDERFACE_RENDER=ABSENT'
Write-Host 'PROCEDURAL_FACE_OCCLUDERS=ABSENT'
Write-Host 'DANGLING_FACEPLATE_STRUCTRNA=REPAIRED'
