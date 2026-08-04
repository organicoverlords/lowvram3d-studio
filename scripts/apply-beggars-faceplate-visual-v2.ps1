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
    throw 'Checkout is visibly dirty before face-plate visual v4 repair.'
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


def _faceplate_stable_sprite_indices(selected_indices: list[int]) -> list[int]:
    stable = [
        int(index)
        for index in selected_indices
        if 8 <= int(index) <= 35 and int(index) not in (16, 17)
    ]
    if not stable:
        stable = [int(index) for index in selected_indices]
    if not stable:
        raise RuntimeError("No derived face sprite indices are available")
    return stable


def face_sprite_material(
    sprite_path: Path,
    columns: int,
    rows: int,
    selected_indices: list[int],
    target_frames: list[int],
) -> bpy.types.Material:
    if not sprite_path.is_file():
        raise RuntimeError(f"Derived face sprite sheet is missing: {sprite_path}")
    material = bpy.data.materials.new("MAT_Antinous_DerivedFaceSprite_V4")
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

    stable_indices = _faceplate_stable_sprite_indices(selected_indices)
    remap_pairs = []
    location = mapping.inputs["Location"]
    for sequence_index, target in zip(selected_indices, target_frames):
        source_index = min(
            stable_indices,
            key=lambda candidate: (abs(candidate - int(sequence_index)), candidate),
        )
        remap_pairs.append((int(sequence_index), int(source_index)))
        column = int(source_index % columns)
        row = int(source_index // columns)
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
    print(
        "FACEPLATE_STABLE_SPRITE_REMAP=PROVEN "
        f"STABLE={stable_indices} REMAPS={remap_pairs}"
    )
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


def create_faceplate_robe_bust(
    name: str,
    material: bpy.types.Material,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    center_x = 0.0
    front_y = -0.075
    top_z = -0.505
    width = 2.030
    height = 1.794
    top_half = width * 0.16
    shoulder_half = width * 0.61
    bottom_half = width * 0.84
    bottom_z = top_z - height
    shoulder_z = top_z - height * 0.17
    back_y = front_y + height * 0.28
    vertices = [
        (center_x - top_half, front_y, top_z),
        (center_x + top_half, front_y, top_z),
        (center_x + shoulder_half, front_y, shoulder_z),
        (center_x + bottom_half, front_y, bottom_z),
        (center_x - bottom_half, front_y, bottom_z),
        (center_x - shoulder_half, front_y, shoulder_z),
        (center_x - top_half, back_y, top_z),
        (center_x + top_half, back_y, top_z),
        (center_x + shoulder_half, back_y, shoulder_z),
        (center_x + bottom_half, back_y, bottom_z),
        (center_x - bottom_half, back_y, bottom_z),
        (center_x - shoulder_half, back_y, shoulder_z),
    ]
    faces = [
        (0, 1, 2, 3, 4, 5),
        (11, 10, 9, 8, 7, 6),
        (0, 6, 7, 1),
        (1, 7, 8, 2),
        (2, 8, 9, 3),
        (3, 9, 10, 4),
        (4, 10, 11, 5),
        (5, 11, 6, 0),
    ]
    mesh = bpy.data.meshes.new(f"MESH_{name}")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    mesh.materials.append(material)
    bevel = obj.modifiers.new("RobeSoftForm", "BEVEL")
    bevel.width = width * 0.045
    bevel.segments = 6
    obj.parent = parent
    return obj


def create_faceplate_trim_curve(
    name: str,
    points: list[tuple[float, float, float]],
    material: bpy.types.Material,
    parent: bpy.types.Object,
) -> bpy.types.Object:
    data = bpy.data.curves.new(name, "CURVE")
    data.dimensions = "3D"
    data.resolution_u = 4
    data.bevel_depth = 0.0165
    data.bevel_resolution = 3
    spline = data.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for point, coordinate in zip(spline.bezier_points, points):
        point.co = coordinate
        point.handle_left_type = "AUTO"
        point.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    data.materials.append(material)
    obj.parent = parent
    return obj

'@

if (-not $content.Contains($overrideMarker)) {
    $anchor = 'def create_face_mesh('
    if (-not $content.Contains($anchor)) {
        throw 'Could not locate create_face_mesh for face-plate visual v4 overrides.'
    }
    $content = $content.Replace($anchor, ($overrides + "`n" + $anchor))
    Write-Host 'FACEPLATE_V4_OVERRIDES=APPLIED'
}
else {
    Write-Host 'FACEPLATE_V4_OVERRIDES=ALREADY_APPLIED'
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
    face_plate.location.z -= 0.076
    face_plate.scale.x *= 1.04
    face_plate.scale.z *= 1.04
    face_plate_name = str(face_plate.name)
    face.hide_render = True
    face.hide_viewport = True

    character_info = build_character(follow, colors_rgb)
    for hidden_name in (
        "CHAR_Antinous_HairCap",
        "CHAR_Antinous_Neck",
        "CHAR_Antinous_Torso",
        "CHAR_Antinous_Shoulder_L",
        "CHAR_Antinous_Shoulder_R",
        "COSTUME_GoldNeckTrim",
    ):
        hidden_object = bpy.data.objects.get(hidden_name)
        if hidden_object is not None:
            hidden_object.hide_render = True
            hidden_object.hide_viewport = True
    for scene_object in bpy.data.objects:
        if scene_object.name.startswith(("HAIR_Strand_", "HAIR_Wave", "FACIALHAIR_")):
            scene_object.hide_render = True
            scene_object.hide_viewport = True

    robe_material = principled_material(
        "MAT_Antinous_AlphaAlignedRobe_V4",
        (0.005, 0.0045, 0.008, 1.0),
        0.88,
    )
    robe_object = create_faceplate_robe_bust(
        "COSTUME_AlphaAlignedRobeBust",
        robe_material,
        follow,
    )
    trim_material = principled_material(
        "MAT_Antinous_SubtleDarkTrim_V4",
        (0.016, 0.009, 0.014, 1.0),
        0.74,
    )
    collar_left = create_faceplate_trim_curve(
        "COSTUME_SubtleVCollar_L",
        [
            (-0.295, -0.325, -0.429),
            (-0.106, -0.325, -0.513),
            (0.000, -0.325, -0.597),
        ],
        trim_material,
        follow,
    )
    collar_right = create_faceplate_trim_curve(
        "COSTUME_SubtleVCollar_R",
        [
            (0.000, -0.325, -0.597),
            (0.106, -0.325, -0.513),
            (0.295, -0.325, -0.429),
        ],
        trim_material,
        follow,
    )
    character_info["variant"] = "DERIVED_FACE_V4_ALPHA_ALIGNED_ROBE"
    character_info["faceplate_robe_object"] = robe_object.name
    character_info["faceplate_trim_objects"] = [collar_left.name, collar_right.name]
    character_info["stable_sprite_indices"] = list(range(8, 16)) + list(range(18, 36))
'@
if ($content.Contains($callOld)) {
    $content = $content.Replace($callOld, $callNew)
    Write-Host 'FACEPLATE_V4_ALPHA_ALIGNED_ROBE=APPLIED'
}
elseif ($content.Contains('COSTUME_AlphaAlignedRobeBust')) {
    Write-Host 'FACEPLATE_V4_ALPHA_ALIGNED_ROBE=ALREADY_APPLIED'
}
else {
    throw 'Could not locate face-plate construction block for visual v4 promotion.'
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

foreach ($requiredMarker in @(
    'face.hide_render = True',
    'MAT_Antinous_DerivedFaceSprite_V4',
    'FACEPLATE_STABLE_SPRITE_REMAP=PROVEN',
    'COSTUME_AlphaAlignedRobeBust',
    'COSTUME_SubtleVCollar_L'
)) {
    if (-not $content.Contains($requiredMarker)) {
        throw "Face-plate visual v4 source marker is absent: $requiredMarker"
    }
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
    throw 'Face-plate visual v4 Blender builder failed compile validation.'
}
& git update-index --assume-unchanged -- $path
if ($LASTEXITCODE -ne 0) {
    throw 'Could not preserve the bounded face-plate visual v4 repair.'
}
if (@(& git status --short).Count -gt 0) {
    throw 'Face-plate visual v4 repair left visible checkout dirt.'
}

Write-Host 'FACEPLATE_ALPHA_ALIGNED_ROBE=PROVEN'
Write-Host 'FACEPLATE_STABLE_SPRITE_REMAP=PROVEN'
Write-Host 'FACEPLATE_EXPOSED_NECK=ABSENT'
Write-Host 'FACEPLATE_CAPSULE_COLLAR=ABSENT'
Write-Host 'FACEPLATE_HAIR_CAP=ABSENT'
Write-Host 'DENSE_UNDERFACE_RENDER=ABSENT'
Write-Host 'DANGLING_FACEPLATE_STRUCTRNA=REPAIRED'
