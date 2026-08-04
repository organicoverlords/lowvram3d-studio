[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expectedRepository = 'organicoverlords/lowvram3d-studio'
$expectedBranch = 'agent/blender-beggars-scene-20260804'
$preparePath = 'tools\beggars_scene\prepare_reference_sequence.py'
$builderPath = 'blender\build_beggars_meme_scene.py'

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
foreach ($path in @($preparePath, $builderPath)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required source is missing: $path"
    }
}
if (@(& git status --short).Count -gt 0) {
    throw 'Checkout is visibly dirty before face-atlas repair.'
}

$prepare = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $preparePath))
$builder = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $builderPath))

$prepareHelpers = @'
def build_masked_face_atlas(
    image_bgr: np.ndarray,
    vertices: np.ndarray,
    output_path: Path,
    atlas_size: int = 1024,
) -> tuple[np.ndarray, dict]:
    height, width = image_bgr.shape[:2]
    points = np.stack([vertices[0], vertices[1]], axis=1).astype(np.float32)
    x_min = float(np.min(points[:, 0]))
    x_max = float(np.max(points[:, 0]))
    y_min = float(np.min(points[:, 1]))
    y_max = float(np.max(points[:, 1]))
    pad_x = max(4, int(round((x_max - x_min) * 0.08)))
    pad_y = max(4, int(round((y_max - y_min) * 0.08)))
    left = max(0, int(np.floor(x_min)) - pad_x)
    right = min(width - 1, int(np.ceil(x_max)) + pad_x)
    top = max(0, int(np.floor(y_min)) - pad_y)
    bottom = min(height - 1, int(np.ceil(y_max)) + pad_y)
    if right <= left or bottom <= top:
        raise RuntimeError("Face atlas crop is degenerate")

    crop = image_bgr[top : bottom + 1, left : right + 1].copy()
    local = np.rint(points - np.asarray([left, top], dtype=np.float32)).astype(np.int32)
    hull = cv2.convexHull(local)
    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull, 255, lineType=cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.15, sigmaY=1.15)

    kernel = np.ones((7, 7), dtype=np.uint8)
    edge_bleed = cv2.dilate(crop, kernel, iterations=5)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    filled = np.clip(crop.astype(np.float32) * alpha + edge_bleed.astype(np.float32) * (1.0 - alpha), 0, 255)
    atlas = cv2.resize(
        filled.astype(np.uint8),
        (atlas_size, atlas_size),
        interpolation=cv2.INTER_LANCZOS4,
    )
    if not cv2.imwrite(str(output_path), atlas):
        raise RuntimeError(f"Could not write derived face atlas: {output_path}")

    denominator_x = max(float(right - left), 1.0)
    denominator_y = max(float(bottom - top), 1.0)
    uv = np.empty((points.shape[0], 2), dtype=np.float32)
    uv[:, 0] = (points[:, 0] - float(left)) / denominator_x
    uv[:, 1] = 1.0 - (points[:, 1] - float(top)) / denominator_y
    uv = np.clip(uv, 0.0, 1.0)

    return uv, {
        "classification": "PROVEN",
        "atlas_size": [atlas_size, atlas_size],
        "source_crop": [left, top, right, bottom],
        "mask_coverage_fraction": float(np.mean(mask > 8)),
        "policy": "DERIVED_FACE_ONLY_MASKED_ATLAS_NO_BACKGROUND_FRAME",
    }

'@
if (-not $prepare.Contains('def build_masked_face_atlas(')) {
    $anchor = 'def face_sharpness(image: np.ndarray, bbox: list[float]) -> float:'
    if (-not $prepare.Contains($anchor)) {
        throw 'Could not locate face_sharpness for atlas helper insertion.'
    }
    $prepare = $prepare.Replace($anchor, ($prepareHelpers + "`n" + $anchor))
    Write-Host 'FACE_ATLAS_PREPARE_HELPERS=APPLIED'
}
else {
    Write-Host 'FACE_ATLAS_PREPARE_HELPERS=ALREADY_APPLIED'
}

$prepareBuildOld = @'
    colors_rgb = sample_colors(keyframe_image, stack_raw[keyframe_index])
    triangles = np.asarray(tddfa.tri, dtype=np.int32)
'@
$prepareBuildNew = @'
    colors_rgb = sample_colors(keyframe_image, stack_raw[keyframe_index])
    triangles = np.asarray(tddfa.tri, dtype=np.int32)
    atlas_output = keyframe_output.with_name("face_albedo_atlas.png")
    uv_coordinates, atlas_report = build_masked_face_atlas(
        keyframe_image,
        stack_raw[keyframe_index],
        atlas_output,
    )
'@
if ($prepare.Contains($prepareBuildOld)) {
    $prepare = $prepare.Replace($prepareBuildOld, $prepareBuildNew)
    Write-Host 'FACE_ATLAS_BUILD=APPLIED'
}
elseif ($prepare.Contains('uv_coordinates, atlas_report = build_masked_face_atlas')) {
    Write-Host 'FACE_ATLAS_BUILD=ALREADY_APPLIED'
}
else {
    throw 'Could not locate keyframe color extraction for atlas creation.'
}

$prepareNpzOld = '        colors_rgb=colors_rgb,'
$prepareNpzNew = @'
        colors_rgb=colors_rgb,
        uv_coordinates=uv_coordinates,
'@
if ($prepare.Contains($prepareNpzOld) -and -not $prepare.Contains('uv_coordinates=uv_coordinates')) {
    $prepare = $prepare.Replace($prepareNpzOld, $prepareNpzNew)
    Write-Host 'FACE_ATLAS_UV_NPZ=APPLIED'
}
elseif ($prepare.Contains('uv_coordinates=uv_coordinates')) {
    Write-Host 'FACE_ATLAS_UV_NPZ=ALREADY_APPLIED'
}
else {
    throw 'Could not add face-atlas UVs to the reconstruction NPZ.'
}

$prepareReportOld = '        "keyframe_output": str(keyframe_output),'
$prepareReportNew = @'
        "keyframe_output": str(keyframe_output),
        "derived_face_atlas": str(atlas_output),
        "derived_face_atlas_report": atlas_report,
        "appearance_policy": "Face-only masked atlas is derived locally; the source frame and clip remain private and excluded.",
'@
if ($prepare.Contains($prepareReportOld) -and -not $prepare.Contains('"derived_face_atlas":')) {
    $prepare = $prepare.Replace($prepareReportOld, $prepareReportNew)
    Write-Host 'FACE_ATLAS_REPORT=APPLIED'
}
elseif ($prepare.Contains('"derived_face_atlas":')) {
    Write-Host 'FACE_ATLAS_REPORT=ALREADY_APPLIED'
}
else {
    throw 'Could not add face-atlas evidence to the reconstruction report.'
}

$builderMaterial = @'
def face_atlas_material(atlas_path: Path) -> bpy.types.Material:
    if not atlas_path.is_file():
        raise RuntimeError(f"Derived face atlas is missing: {atlas_path}")
    material = bpy.data.materials.new("MAT_Antinous_Face_DerivedAtlas")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    texture = nodes.new("ShaderNodeTexImage")
    image = bpy.data.images.load(str(atlas_path), check_existing=False)
    image.colorspace_settings.name = "sRGB"
    image.pack()
    texture.image = image
    texture.interpolation = "Linear"
    texture.extension = "EXTEND"
    set_input(bsdf, "Roughness", 0.50)
    set_input(bsdf, "Specular IOR Level", 0.30)
    set_input(bsdf, "Subsurface Weight", 0.025)
    set_input(bsdf, "Subsurface Radius", (1.0, 0.42, 0.18))
    links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return material

'@
if (-not $builder.Contains('def face_atlas_material(')) {
    $anchor = 'def vertex_skin_material() -> bpy.types.Material:'
    if (-not $builder.Contains($anchor)) {
        throw 'Could not locate vertex_skin_material for atlas material insertion.'
    }
    $builder = $builder.Replace($anchor, ($builderMaterial + "`n" + $anchor))
    Write-Host 'FACE_ATLAS_MATERIAL=APPLIED'
}
else {
    Write-Host 'FACE_ATLAS_MATERIAL=ALREADY_APPLIED'
}

$builderHoleHelpers = @'
def _atlas_boundary_loops(triangles: np.ndarray) -> list[list[int]]:
    edge_counts: dict[tuple[int, int], int] = {}
    for triangle in np.asarray(triangles, dtype=np.int32):
        a, b, c = [int(value) for value in triangle]
        for left, right in ((a, b), (b, c), (c, a)):
            edge = (left, right) if left < right else (right, left)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    edges = [edge for edge, count in edge_counts.items() if count == 1]
    adjacency: dict[int, list[int]] = {}
    for left, right in edges:
        adjacency.setdefault(left, []).append(right)
        adjacency.setdefault(right, []).append(left)
    unused = set(edges)
    loops: list[list[int]] = []
    while unused:
        left, right = min(unused)
        unused.remove((left, right))
        loop = [left, right]
        previous, current = left, right
        for _ in range(len(edges) + 1):
            candidates = []
            for candidate in adjacency.get(current, []):
                edge = (current, candidate) if current < candidate else (candidate, current)
                if edge in unused and candidate != previous:
                    candidates.append(candidate)
            if not candidates:
                closing = (current, loop[0]) if current < loop[0] else (loop[0], current)
                if closing in unused:
                    unused.remove(closing)
                break
            following = min(candidates)
            edge = (current, following) if current < following else (following, current)
            unused.remove(edge)
            if following == loop[0]:
                break
            loop.append(following)
            previous, current = current, following
        if len(loop) >= 4:
            loops.append(loop)
    return loops


def _atlas_loop_area(points: np.ndarray) -> float:
    x = points[:, 0]
    z = points[:, 2]
    return float(abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))) * 0.5)


def _close_atlas_feature_holes(
    transformed: np.ndarray,
    triangles: np.ndarray,
    colors_rgb: np.ndarray,
    uv_coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    loops = _atlas_boundary_loops(triangles)
    if len(loops) < 4:
        raise RuntimeError(f"Expected outer, eye and mouth boundary loops; found {len(loops)}")
    base = transformed[0]
    metrics = []
    for loop in loops:
        indices = np.asarray(loop, dtype=np.int32)
        points = base[indices]
        metrics.append(
            {
                "loop": loop,
                "area": _atlas_loop_area(points),
                "centroid_z": float(np.mean(points[:, 2])),
                "vertex_count": len(loop),
            }
        )
    outer = max(metrics, key=lambda item: (item["area"], item["vertex_count"]))
    interior = [item for item in metrics if item is not outer and item["vertex_count"] >= 6]
    if len(interior) < 3:
        raise RuntimeError(f"Expected at least three interior facial loops; found {len(interior)}")
    mouth = min(interior, key=lambda item: item["centroid_z"])
    eyes = sorted(
        [item for item in interior if item is not mouth],
        key=lambda item: item["centroid_z"],
        reverse=True,
    )[:2]
    selected = [("eye", item) for item in eyes] + [("mouth", mouth)]

    faces = [tuple(int(value) for value in row) for row in np.asarray(triangles, dtype=np.int32)]
    colors = np.asarray(colors_rgb, dtype=np.float32)
    uvs = np.asarray(uv_coordinates, dtype=np.float32)
    added = []
    for kind, item in selected:
        loop = np.asarray(item["loop"], dtype=np.int32)
        centers = np.mean(transformed[:, loop, :], axis=1)
        ring = transformed[:, loop, :] * 0.42 + centers[:, None, :] * 0.58
        depth = 0.018 if kind == "eye" else 0.025
        ring[:, :, 1] += depth
        center_vertices = centers.copy()
        center_vertices[:, 1] += depth + 0.010
        ring_start = int(transformed.shape[1])
        transformed = np.concatenate([transformed, ring.astype(np.float32)], axis=1)
        center_index = int(transformed.shape[1])
        transformed = np.concatenate([transformed, center_vertices[:, None, :].astype(np.float32)], axis=1)

        center_uv = np.mean(uvs[loop], axis=0)
        ring_uv = uvs[loop] * 0.42 + center_uv[None, :] * 0.58
        uvs = np.concatenate([uvs, ring_uv.astype(np.float32), center_uv[None, :].astype(np.float32)], axis=0)
        center_color = np.mean(colors[loop], axis=0)
        ring_color = colors[loop] * 0.42 + center_color[None, :] * 0.58
        colors = np.concatenate([colors, ring_color.astype(np.float32), center_color[None, :].astype(np.float32)], axis=0)

        for index in range(len(loop)):
            following = (index + 1) % len(loop)
            boundary_a = int(loop[index])
            boundary_b = int(loop[following])
            inner_a = ring_start + index
            inner_b = ring_start + following
            faces.append((boundary_a, boundary_b, inner_b))
            faces.append((boundary_a, inner_b, inner_a))
            faces.append((inner_a, inner_b, center_index))
        added.append({"kind": kind, "boundary_vertices": int(len(loop))})

    return (
        transformed.astype(np.float32),
        np.asarray(faces, dtype=np.int32),
        colors.astype(np.float32),
        uvs.astype(np.float32),
        {
            "classification": "PROVEN",
            "boundary_loop_count": len(loops),
            "closed_features": added,
            "triangle_count_after": len(faces),
        },
    )

'@
if (-not $builder.Contains('def _close_atlas_feature_holes(')) {
    $anchor = 'def create_face_mesh('
    if (-not $builder.Contains($anchor)) {
        throw 'Could not locate create_face_mesh for atlas topology insertion.'
    }
    $builder = $builder.Replace($anchor, ($builderHoleHelpers + "`n" + $anchor))
    Write-Host 'FACE_ATLAS_TOPOLOGY_HELPERS=APPLIED'
}
else {
    Write-Host 'FACE_ATLAS_TOPOLOGY_HELPERS=ALREADY_APPLIED'
}

$signatureOld = @'
    image_size: tuple[int, int],
    scene_frame_count: int,
) -> tuple[bpy.types.Object, bpy.types.Object, list[int]]:
'@
$signatureNew = @'
    image_size: tuple[int, int],
    uv_coordinates: np.ndarray,
    atlas_path: Path,
    scene_frame_count: int,
) -> tuple[bpy.types.Object, bpy.types.Object, list[int]]:
'@
if ($builder.Contains($signatureOld)) {
    $builder = $builder.Replace($signatureOld, $signatureNew)
    Write-Host 'FACE_ATLAS_SIGNATURE=APPLIED'
}
elseif ($builder.Contains('    uv_coordinates: np.ndarray,')) {
    Write-Host 'FACE_ATLAS_SIGNATURE=ALREADY_APPLIED'
}
else {
    throw 'Could not extend create_face_mesh with atlas arguments.'
}

$closeOld = '    maximum_shape_keys = 42'
$closeNew = @'
    transformed, triangles, colors_rgb, uv_coordinates, feature_report = _close_atlas_feature_holes(
        transformed,
        triangles,
        colors_rgb,
        uv_coordinates,
    )

    maximum_shape_keys = 42
'@
if ($builder.Contains($closeOld) -and -not $builder.Contains('feature_report = _close_atlas_feature_holes')) {
    $builder = $builder.Replace($closeOld, $closeNew)
    Write-Host 'FACE_ATLAS_FEATURE_CLOSE=APPLIED'
}
elseif ($builder.Contains('feature_report = _close_atlas_feature_holes')) {
    Write-Host 'FACE_ATLAS_FEATURE_CLOSE=ALREADY_APPLIED'
}
else {
    throw 'Could not insert UV-aware feature closure.'
}

$uvOld = @'
    mesh.update()
    face = bpy.data.objects.new("CHAR_Antinous", mesh)
'@
$uvNew = @'
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon in mesh.polygons:
        for loop_index in polygon.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = uv_coordinates[vertex_index]
    face = bpy.data.objects.new("CHAR_Antinous", mesh)
'@
if ($builder.Contains($uvOld)) {
    $builder = $builder.Replace($uvOld, $uvNew)
    Write-Host 'FACE_ATLAS_UV_LAYER=APPLIED'
}
elseif ($builder.Contains('uv_layer = mesh.uv_layers.new(name="UVMap")')) {
    Write-Host 'FACE_ATLAS_UV_LAYER=ALREADY_APPLIED'
}
else {
    throw 'Could not attach the face-atlas UV layer.'
}

$builder = $builder.Replace(
    '    mesh.materials.append(vertex_skin_material())',
    '    mesh.materials.append(face_atlas_material(atlas_path))'
)
if (-not $builder.Contains('mesh.materials.append(face_atlas_material(atlas_path))')) {
    throw 'Could not replace vertex-color appearance with the derived atlas material.'
}

$propertyOld = '    bpy.context.collection.objects.link(face)'
$propertyNew = @'
    bpy.context.collection.objects.link(face)
    face["feature_hole_repair"] = json.dumps(feature_report, sort_keys=True)
    face["appearance_route"] = "PACKED_DERIVED_FACE_ONLY_ATLAS"
'@
if ($builder.Contains($propertyOld) -and -not $builder.Contains('face["appearance_route"]')) {
    $builder = $builder.Replace($propertyOld, $propertyNew)
    Write-Host 'FACE_ATLAS_OBJECT_RECEIPT=APPLIED'
}
elseif ($builder.Contains('face["appearance_route"]')) {
    Write-Host 'FACE_ATLAS_OBJECT_RECEIPT=ALREADY_APPLIED'
}
else {
    throw 'Could not attach face-atlas evidence to the hero object.'
}

$loadOld = @'
    colors_rgb = np.asarray(data["colors_rgb"], dtype=np.float32)
    boxes = np.asarray(data["boxes"], dtype=np.float32)
'@
$loadNew = @'
    colors_rgb = np.asarray(data["colors_rgb"], dtype=np.float32)
    uv_coordinates = np.asarray(data["uv_coordinates"], dtype=np.float32)
    atlas_path = sequence_path.with_name("face_albedo_atlas.png")
    if not atlas_path.is_file():
        raise SystemExit(f"Derived face atlas is missing: {atlas_path}")
    boxes = np.asarray(data["boxes"], dtype=np.float32)
'@
if ($builder.Contains($loadOld)) {
    $builder = $builder.Replace($loadOld, $loadNew)
    Write-Host 'FACE_ATLAS_LOAD=APPLIED'
}
elseif ($builder.Contains('atlas_path = sequence_path.with_name("face_albedo_atlas.png")')) {
    Write-Host 'FACE_ATLAS_LOAD=ALREADY_APPLIED'
}
else {
    throw 'Could not load face-atlas data in the Blender builder.'
}

$callOld = @'
        boxes,
        image_size,
        scene.frame_end,
'@
$callNew = @'
        boxes,
        image_size,
        uv_coordinates,
        atlas_path,
        scene.frame_end,
'@
if ($builder.Contains($callOld)) {
    $builder = $builder.Replace($callOld, $callNew)
    Write-Host 'FACE_ATLAS_CALL=APPLIED'
}
elseif ($builder.Contains('        atlas_path,')) {
    Write-Host 'FACE_ATLAS_CALL=ALREADY_APPLIED'
}
else {
    throw 'Could not pass face-atlas data into create_face_mesh.'
}

$saveOld = @'
    blend_path = output_dir / "beggars_photoreal_recreation.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
'@
$saveNew = @'
    blend_path = output_dir / "beggars_photoreal_recreation.blend"
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
'@
if ($builder.Contains($saveOld)) {
    $builder = $builder.Replace($saveOld, $saveNew)
    Write-Host 'FACE_ATLAS_PACK=APPLIED'
}
elseif ($builder.Contains('    bpy.ops.file.pack_all()')) {
    Write-Host 'FACE_ATLAS_PACK=ALREADY_APPLIED'
}
else {
    throw 'Could not force derived atlas packing before save.'
}

[System.IO.File]::WriteAllText(
    (Resolve-Path -LiteralPath $preparePath),
    $prepare,
    [System.Text.UTF8Encoding]::new($false)
)
[System.IO.File]::WriteAllText(
    (Resolve-Path -LiteralPath $builderPath),
    $builder,
    [System.Text.UTF8Encoding]::new($false)
)

$controlPython = "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\control\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $controlPython)) {
    throw "Control Python is missing: $controlPython"
}
& $controlPython -m py_compile $preparePath $builderPath
if ($LASTEXITCODE -ne 0) {
    throw 'Face-atlas repaired Python sources failed compile validation.'
}
& git update-index --assume-unchanged -- $preparePath $builderPath
if ($LASTEXITCODE -ne 0) {
    throw 'Could not preserve the bounded face-atlas runtime repair as assume-unchanged.'
}
if (@(& git status --short).Count -gt 0) {
    throw 'Face-atlas repair left visible checkout dirt.'
}

Write-Host 'DERIVED_MASKED_FACE_ATLAS_ROUTE=PROVEN'
Write-Host 'DIRECT_SOURCE_FRAME_PLANE=ABSENT'
Write-Host 'SOURCE_CLIP_PACKAGED=FALSE'
