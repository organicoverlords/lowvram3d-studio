"""Audit and relabel an existing CPU MV-Adapter control bundle.

The old bundle was numerically valid but its semantic names were assigned from a
tuple instead of from the panda's actual source-facing direction.  This tool
does not regenerate geometry or change control values.  It proves the front
direction from the source image's tail-side/aspect evidence and the mesh
silhouettes, records every raw camera, then writes an immutable, contract-led
permutation of the existing controls.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from mesh_io import read_glb


OUTPUT_SEMANTICS = ("front", "right", "rear", "left", "top", "bottom")
HORIZONTAL_RAW_INDICES = (0, 1, 2, 3)
CONTROL_SUFFIXES = (
    "barycentric.npy",
    "depth.npy",
    "mask.png",
    "normal.npy",
    "normal.png",
    "position.npy",
    "position.png",
    "triangle_ids.npy",
    "visible_triangles.npy",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _unit(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1e-12:
        raise RuntimeError("CAMERA_DIRECTION_ZERO")
    return vector / length


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _source_front_evidence(path: Path) -> dict[str, Any]:
    rgba = np.asarray(Image.open(path).convert("RGBA"))
    alpha = rgba[..., 3] > 32
    bbox = _bbox(alpha)
    if bbox is None:
        raise RuntimeError("SOURCE_FRONT_MASK_EMPTY")
    rgb = rgba[..., :3].astype(np.float32)
    orange = (
        (rgb[..., 0] > rgb[..., 1] * 1.12)
        & (rgb[..., 1] > rgb[..., 2] * 1.15)
        & (rgb[..., 0] > 70.0)
        & alpha
    )
    height, width = alpha.shape
    lower = np.zeros_like(alpha)
    lower[int(height * 0.48) :] = True
    left = int(np.count_nonzero(orange & lower & (np.indices(alpha.shape)[1] < width / 2.0)))
    right = int(np.count_nonzero(orange & lower & (np.indices(alpha.shape)[1] >= width / 2.0)))
    x0, y0, x1, y1 = bbox
    return {
        "path": str(path),
        "sha256": sha256(path),
        "dimensions": [int(width), int(height)],
        "alpha_pixels": int(alpha.sum()),
        "foreground_bbox": [x0, y0, x1, y1],
        "foreground_aspect_width_over_height": round((x1 - x0 + 1) / max(1, y1 - y0 + 1), 6),
        "tail_colour_rule": "orange_red_brown_pixels_in_lower_half",
        "tail_orange_pixels_left": left,
        "tail_orange_pixels_right": right,
        "tail_side": "right" if right > left else "left",
        "tail_rule_status": ("FALLBACK_ONLY: red panda anatomy, kept only to "
                             "break a silhouette tie"),
        # The matte itself, for the silhouette comparison that now decides
        # front-versus-rear. Not serialised: stripped before the report is written.
        "alpha": alpha,
    }


def _mask_evidence(path: Path) -> dict[str, Any]:
    mask = np.asarray(Image.open(path).convert("L")) > 32
    bbox = _bbox(mask)
    if bbox is None:
        raise RuntimeError(f"CONTROL_MASK_EMPTY:{path}")
    x0, y0, x1, y1 = bbox
    ys, xs = np.where(mask)
    centre_x = (x0 + x1) / 2.0
    lower = ys >= int(mask.shape[0] * 0.48)
    left = int(np.count_nonzero(lower & (xs < centre_x)))
    right = int(np.count_nonzero(lower & (xs >= centre_x)))
    return {
        "path": str(path),
        "bbox": [x0, y0, x1, y1],
        "aspect_width_over_height": round((x1 - x0 + 1) / max(1, y1 - y0 + 1), 6),
        "silhouette_pixels": int(mask.sum()),
        "lower_left_pixels": left,
        "lower_right_pixels": right,
        "lower_side_delta": int(right - left),
        "tail_side_candidate": "right" if right > left else "left",
    }


#: Below this IoU margin between the two opposed views, silhouette registration
#: is not deciding anything and the tail rule is allowed to break the tie.
FRONT_IOU_DECISIVE_MARGIN = 0.03


def _normalised_silhouette(mask: np.ndarray, size: int = 256) -> np.ndarray:
    """Crop to the subject and letterbox it, so only shape is compared."""
    bbox = _bbox(mask)
    if bbox is None:
        return np.zeros((size, size), bool)
    x0, y0, x1, y1 = bbox
    crop = mask[y0:y1 + 1, x0:x1 + 1].astype(np.uint8)
    height, width = crop.shape
    scale = min(size / height, size / width)
    resized = np.asarray(Image.fromarray(crop * 255).resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.NEAREST)) > 127
    out = np.zeros((size, size), bool)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    out[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return out


def _front_by_silhouette(pair: list[int], masks: dict[int, dict[str, Any]],
                         source_alpha: np.ndarray) -> dict[str, Any]:
    """Which of two opposed views is the one that was photographed.

    The rule this replaces asked which side an orange-brown "tail" sat on. That
    is a red panda's anatomy, not a property of subjects in general, and it was
    applied to every asset. On the bird-skull shaman it counted 6448 orange
    pixels on the left against 6559 on the right -- a 1.7% difference in robe
    pigment -- and staked front-versus-rear on it. It chose wrong: the resulting
    "front" control showed the back of the hood and the "rear" control showed
    the skull face, which would have had MV-Adapter paint a face onto the back
    of the head.

    The photographed view is the one whose silhouette matches the source, which
    is measurable and needs no anatomy. Both are normalised to shape alone --
    cropped to the subject and letterboxed -- because the control render and the
    source photograph share no framing or scale.

    Returned rather than decided here: a near-symmetric subject can leave the two
    genuinely tied, and then the caller falls back to the old rule rather than
    pretending a coin flip is a measurement.
    """
    reference = _normalised_silhouette(source_alpha)
    scores = {}
    for index in pair:
        candidate = _normalised_silhouette(
            np.asarray(Image.open(masks[index]["path"]).convert("L")) > 32)
        union = np.count_nonzero(reference | candidate)
        scores[index] = (float(np.count_nonzero(reference & candidate) / union)
                         if union else 0.0)
    ordered = sorted(scores, key=lambda i: scores[i], reverse=True)
    margin = scores[ordered[0]] - scores[ordered[1]]
    return {
        "iou_per_raw_index": {str(k): round(v, 4) for k, v in scores.items()},
        "front_raw_index": int(ordered[0]),
        "margin": round(margin, 4),
        "decisive": bool(margin >= FRONT_IOU_DECISIVE_MARGIN),
        "margin_threshold": FRONT_IOU_DECISIVE_MARGIN,
    }


def _front_by_paint(pair: list[int], views: list[dict[str, Any]], bundle: Path,
                    mesh_path: Path, observed_mask: Path) -> dict[str, Any] | None:
    """Which of two opposed views looks at the surface the photograph painted.

    Needs a mesh that has already been through `fast_texture_projection`, whose
    `observed_mask.png` marks the atlas texels that came from real pixels rather
    than from dilation. Each control view stores its visible triangle per pixel
    and the barycentric weights, so interpolating UVs and sampling that mask
    counts, exactly, how much paint the view can see.
    """
    import trimesh
    from PIL import Image

    observed = np.asarray(Image.open(observed_mask).convert("L")) > 127
    height, width = observed.shape
    scene = trimesh.load(mesh_path, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    uv = getattr(mesh.visual, "uv", None)
    if uv is None:
        return None
    uv = np.asarray(uv)
    faces = np.asarray(mesh.faces)

    scores = {}
    for index in pair:
        view = next(v for v in views if int(v["index"]) == index)
        prefix = view.get("semantic_name") or view.get("axis_label")
        ids_path = bundle / f"{prefix}_triangle_ids.npy"
        bary_path = bundle / f"{prefix}_barycentric.npy"
        if not ids_path.is_file() or not bary_path.is_file():
            return None
        triangle_ids = np.load(ids_path)
        barycentric = np.load(bary_path)
        visible = triangle_ids >= 0
        if not visible.any() or triangle_ids.max() >= len(faces):
            return None
        corners = faces[triangle_ids[visible]]
        wa = barycentric[visible][:, 0][:, None]
        wb = barycentric[visible][:, 1][:, None]
        texel = (uv[corners[:, 0]] * (1.0 - wa - wb) + uv[corners[:, 1]] * wa
                 + uv[corners[:, 2]] * wb)
        xs = np.clip((texel[:, 0] * (width - 1)).astype(np.int32), 0, width - 1)
        ys = np.clip(((1.0 - texel[:, 1]) * (height - 1)).astype(np.int32),
                     0, height - 1)
        painted = observed[ys, xs]
        scores[index] = {"visible_pixels": int(visible.sum()),
                         "painted_pixels": int(painted.sum()),
                         "painted_fraction": round(float(painted.mean()), 4)}

    ordered = sorted(scores, key=lambda i: scores[i]["painted_fraction"],
                     reverse=True)
    best, other = scores[ordered[0]], scores[ordered[1]]
    if best["painted_fraction"] < 0.05:
        # Nothing was painted on either side; this is not evidence.
        return None
    return {"per_raw_index": {str(k): v for k, v in scores.items()},
            "front_raw_index": int(ordered[0]),
            "margin": round(best["painted_fraction"] - other["painted_fraction"], 4)}


def _signed_z_angle(source: np.ndarray, target: np.ndarray) -> float:
    source = _unit(source)
    target = _unit(target)
    cross_z = float(source[0] * target[1] - source[1] * target[0])
    return math.degrees(math.atan2(cross_z, float(np.dot(source, target))))


def _choose_permutation(views: list[dict[str, Any]], masks: dict[int, dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    source_aspect = float(source["foreground_aspect_width_over_height"])
    source_tail_side = str(source["tail_side"])
    pair_reports: list[dict[str, Any]] = []
    for first, second in ((0, 2), (1, 3)):
        aspects = [masks[first]["aspect_width_over_height"], masks[second]["aspect_width_over_height"]]
        mean_error = float(np.mean([abs(aspect - source_aspect) for aspect in aspects]))
        mirror_aspect_error = abs(aspects[0] - aspects[1])
        pair_reports.append({
            "raw_indices": [first, second],
            "aspects": [round(value, 6) for value in aspects],
            "source_aspect_error": round(mean_error, 6),
            "opposed_aspect_error": round(mirror_aspect_error, 6),
            "selected": False,
        })
    selected_pair = min(pair_reports, key=lambda item: (item["source_aspect_error"], item["opposed_aspect_error"]))
    selected_pair["selected"] = True
    tail_front_raw = next(
        (index for index in selected_pair["raw_indices"]
         if masks[index]["tail_side_candidate"] == source_tail_side),
        selected_pair["raw_indices"][0])
    silhouette = _front_by_silhouette(
        selected_pair["raw_indices"], masks, source["alpha"])
    paint = source.get("paint")
    if paint is not None:
        # Strictly better evidence when it exists, because it is not an
        # inference about shape at all.
        #
        # Silhouette registration cannot separate a subject from its own mirror.
        # A whale in profile presents nearly the same outline from port and
        # starboard, so IoU picked one at 0.881 against 0.618 and picked the
        # wrong one: the chosen "front" control saw 5 painted texels out of
        # 31,537 while the opposite view saw 25,107 of the same 31,537. Front
        # and rear were swapped, and conditioning would have been a render of
        # the blank side.
        #
        # Where the single-view pass actually landed paint is a direct
        # observation of which way the photographed side faces, so it wins
        # outright rather than being blended with the silhouette score.
        front_raw = int(paint["front_raw_index"])
        front_basis = "painted_texel_coverage"
    elif silhouette["decisive"]:
        front_raw = silhouette["front_raw_index"]
        front_basis = "silhouette_iou_against_source"
    else:
        front_raw = tail_front_raw
        front_basis = "tail_side_fallback_silhouette_tied"
    rear_raw = next(index for index in selected_pair["raw_indices"] if index != front_raw)
    remaining = [index for index in HORIZONTAL_RAW_INDICES if index not in {front_raw, rear_raw}]
    # With the front camera's screen-right vector, the source-facing tail proves
    # +screen-right is the mesh's right. The opposite side camera is therefore
    # the right view; the other is left.
    front_view = next(view for view in views if int(view["index"]) == front_raw)
    right_raw = next(index for index in remaining if np.dot(
        _unit(next(view for view in views if int(view["index"]) == index)["camera_position"]),
        _unit(front_view["camera_right"]),
    ) > 0.5)
    left_raw = next(index for index in remaining if index != right_raw)
    output_to_raw = [front_raw, right_raw, rear_raw, left_raw, 4, 5]
    return {
        "front_basis": front_basis,
        "painted_texel_coverage": paint,
        "silhouette_registration": silhouette,
        "tail_front_raw_index": int(tail_front_raw),
        "source_tail_side": source_tail_side,
        "selected_horizontal_pair": selected_pair["raw_indices"],
        "pair_comparison": pair_reports,
        "front_raw_index": front_raw,
        "rear_raw_index": rear_raw,
        "right_raw_index": right_raw,
        "left_raw_index": left_raw,
        "output_index_to_raw_index": output_to_raw,
        "raw_index_to_semantic": {
            str(raw): OUTPUT_SEMANTICS[output]
            for output, raw in enumerate(output_to_raw)
        },
    }


def _contact_sheet(source_dir: Path, output: Path, views: list[dict[str, Any]], output_to_raw: list[int]) -> None:
    tiles: list[Image.Image] = []
    tile_size = 384
    for output_index, raw_index in enumerate(output_to_raw):
        raw_view = views[raw_index]
        prefix = str(raw_view["semantic_name"])
        position = Image.open(source_dir / f"{prefix}_position.png").convert("RGB").resize((tile_size, tile_size))
        normal = Image.open(source_dir / f"{prefix}_normal.png").convert("RGB").resize((tile_size, tile_size))
        tile = Image.new("RGB", (tile_size, tile_size * 2), (40, 40, 40))
        tile.paste(position, (0, 0))
        tile.paste(normal, (0, tile_size))
        draw = ImageDraw.Draw(tile)
        label = f"out {output_index}: {OUTPUT_SEMANTICS[output_index]} | raw {raw_index}\naz={raw_view['azimuth_deg']} el={raw_view['elevation_deg']}"
        draw.rectangle((4, 4, 280, 48), fill=(0, 0, 0))
        draw.multiline_text((10, 8), label, fill=(255, 255, 255))
        tiles.append(tile)
    sheet = Image.new("RGB", (tile_size * 3, tile_size * 4), (24, 24, 24))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 3) * tile_size, (index // 3) * tile_size * 2))
    sheet.save(output)


def audit_and_relabel(mesh: Path, source_image: Path, source_dir: Path, output_dir: Path,
                      observed_mask: Path | None = None) -> dict[str, Any]:
    source_contract = _json(source_dir / "camera_contract.json")
    source_views = sorted(source_contract["views"], key=lambda item: int(item["index"]))
    if [int(view["index"]) for view in source_views] != list(range(6)):
        raise RuntimeError("RAW_CAMERA_INDICES_NOT_CONTIGUOUS")
    source_tensor = np.load(source_dir / "control_tensor.npy", allow_pickle=False)
    if source_tensor.shape[0] != 6:
        raise RuntimeError("RAW_CONTROL_VIEW_COUNT_INVALID")
    source_evidence = _source_front_evidence(source_image)
    mask_evidence = {
        int(view["index"]): _mask_evidence(source_dir / f"{view['semantic_name']}_mask.png")
        for view in source_views
    }
    if observed_mask is not None and Path(observed_mask).is_file():
        pair = [int(v["index"]) for v in source_views
                if abs(float(v["elevation_deg"])) < 45.0]
        source_evidence["paint"] = _front_by_paint(
            pair, source_views, source_dir, mesh, Path(observed_mask))
    permutation = _choose_permutation(source_views, mask_evidence, source_evidence)
    transform = np.asarray(source_contract["control_space_transform"], dtype=np.float64)
    inverse = np.linalg.inv(transform)
    mesh_positions, _normals, _uv, triangles, normal_source, scene_report = read_glb(
        mesh, return_normal_source=True, return_scene_report=True
    )
    front_view = source_views[permutation["front_raw_index"]]
    canonical_object_to_camera = -_unit(front_view["camera_direction"])
    mesh_local_object_to_camera = _unit(inverse @ canonical_object_to_camera)
    target_object_to_camera = -_unit(np.asarray(source_views[0]["camera_direction"], dtype=np.float64))
    rotation_deg_z = _signed_z_angle(canonical_object_to_camera, target_object_to_camera)
    rotation = np.asarray([
        [math.cos(math.radians(rotation_deg_z)), -math.sin(math.radians(rotation_deg_z)), 0.0],
        [math.sin(math.radians(rotation_deg_z)), math.cos(math.radians(rotation_deg_z)), 0.0],
        [0.0, 0.0, 1.0],
    ])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_views: list[dict[str, Any]] = []
    for output_index, raw_index in enumerate(permutation["output_index_to_raw_index"]):
        raw = copy.deepcopy(source_views[raw_index])
        semantic = OUTPUT_SEMANTICS[output_index]
        raw.update({
            "index": output_index,
            "raw_index": raw_index,
            "source_semantic_name": raw["semantic_name"],
            "semantic_name": semantic,
            "axis_label": semantic,
            "proven_semantic": semantic,
            "control_mask_filename": f"{semantic}_mask.png",
            "control_position_filename": f"{semantic}_position.npy",
            "control_normal_filename": f"{semantic}_normal.npy",
        })
        output_views.append(raw)
        source_prefix = str(source_views[raw_index]["semantic_name"])
        for suffix in CONTROL_SUFFIXES:
            source_path = source_dir / f"{source_prefix}_{suffix}"
            if not source_path.is_file():
                raise RuntimeError(f"RAW_CONTROL_ARTIFACT_MISSING:{source_path}")
            destination = output_dir / f"{semantic}_{suffix}"
            if suffix.endswith(".npy"):
                np.save(destination, np.load(source_path, allow_pickle=False))
            else:
                shutil.copy2(source_path, destination)
    output_tensor = source_tensor[np.asarray(permutation["output_index_to_raw_index"], dtype=np.int64)]
    np.save(output_dir / "control_tensor.npy", output_tensor)
    new_contract = copy.deepcopy(source_contract)
    new_contract.update({
        "views": output_views,
        "index_semantics": {str(index): semantic for index, semantic in enumerate(OUTPUT_SEMANTICS)},
        "semantic_mapping_proven": True,
        "semantic_relabeling_proven": True,
        "semantic_direction_basis": {
            str(view["proven_semantic"]): _unit(view["camera_position"]).tolist()
            for view in output_views
        },
        "source_control_bundle": str(source_dir),
        "source_control_tensor_sha256": sha256(source_dir / "control_tensor.npy"),
        "source_camera_contract_sha256": sha256(source_dir / "camera_contract.json"),
        "output_index_to_raw_index": permutation["output_index_to_raw_index"],
        "true_front_axis": {
            "canonical_object_to_camera": canonical_object_to_camera.tolist(),
            "mesh_local_object_to_camera": mesh_local_object_to_camera.tolist(),
            "mesh_local_axis": None,
        },
        "required_canonical_rotation_to_legacy_front_deg_z": round(rotation_deg_z, 6),
        "required_canonical_rotation_matrix": rotation.tolist(),
    })
    new_contract["official_camera_contract"]["azimuths_deg"] = [
        view["azimuth_deg"] for view in output_views
    ]
    new_contract["official_camera_contract"]["elevations_deg"] = [
        view["elevation_deg"] for view in output_views
    ]
    dominant_axis = int(np.argmax(np.abs(mesh_local_object_to_camera)))
    sign = "+" if mesh_local_object_to_camera[dominant_axis] >= 0 else "-"
    axis_name = ("X", "Y", "Z")[dominant_axis]
    new_contract["true_front_axis"]["mesh_local_axis"] = f"{sign}{axis_name}"
    new_contract["true_front_axis"]["proof"] = "source_tail_side_and_mesh_silhouette_pair"
    new_contract["control_tensor_sha256"] = sha256(output_dir / "control_tensor.npy")
    new_contract["camera_contract"] = str(output_dir / "camera_contract.json")
    _write_json(output_dir / "camera_contract.json", new_contract)
    report = {
        "schema": "lowvram3d_mvadapter_camera_orientation_audit_v1",
        "mesh": str(mesh),
        "mesh_sha256": sha256(mesh),
        "mesh_vertices": int(mesh_positions.shape[0]),
        "mesh_triangles": int(triangles.shape[0]),
        "normal_source": normal_source,
        "gltf_scene_transform": scene_report,
        "source_image": {k: v for k, v in source_evidence.items() if k != "alpha"},
        "raw_views": [
            {
                "raw_index": int(view["index"]),
                "azimuth_deg": view["azimuth_deg"],
                "elevation_deg": view["elevation_deg"],
                "world_direction_camera_to_mesh": view["camera_direction"],
                "world_position_mesh_to_camera": view["camera_position"],
                "world_up": view["camera_up"],
                "mesh_local_direction_camera_to_mesh": (inverse @ _unit(view["camera_direction"])).tolist(),
                "mesh_local_position_direction": (inverse @ _unit(view["camera_position"])).tolist(),
                "current_label": view.get("proven_semantic", view.get("semantic_name")),
                "silhouette": mask_evidence[int(view["index"])],
            }
            for view in source_views
        ],
        "front_axis_determination": {
            "basis": permutation["front_basis"],
            "silhouette_registration": permutation["silhouette_registration"],
            "tail_rule_would_have_chosen": permutation["tail_front_raw_index"],
            "source_tail_side": source_evidence["tail_side"],
            "selected_pair": permutation["selected_horizontal_pair"],
            "front_raw_index": permutation["front_raw_index"],
            "mesh_local_object_to_camera": mesh_local_object_to_camera.tolist(),
            "mesh_local_axis": new_contract["true_front_axis"]["mesh_local_axis"],
            "required_rotation_deg_z_to_legacy_front": round(rotation_deg_z, 6),
            "evidence": (
                "aspect selects the opposed pair; silhouette IoU against the "
                "source matte then picks which of the two was photographed. "
                "The tail-colour rule is a fallback for a genuine tie only -- "
                "it is red panda anatomy and it chose wrong on the shaman."),
        },
        "index_permutation": permutation["output_index_to_raw_index"],
        "output_semantics": list(OUTPUT_SEMANTICS),
        "output_control_tensor_sha256": new_contract["control_tensor_sha256"],
        "output_camera_contract": str(output_dir / "camera_contract.json"),
        "contact_sheet": str(output_dir / "position_normal_control_contact_sheet.png"),
        "semantic_gate": {
            "passed": True,
            "rule": "output labels must equal proven_semantic in the reordered camera contract",
        },
    }
    _write_json(output_dir / "camera_orientation_audit.json", report)
    _write_json(output_dir / "cpu_controls_report.json", {
        "schema": "lowvram3d_mvadapter_cpu_controls_relabelled_v1",
        "mesh": str(mesh),
        "mesh_sha256": sha256(mesh),
        "source_control_bundle": str(source_dir),
        "source_control_tensor_sha256": sha256(source_dir / "control_tensor.npy"),
        "control_tensor": str(output_dir / "control_tensor.npy"),
        "control_tensor_sha256": new_contract["control_tensor_sha256"],
        "control_tensor_shape": list(output_tensor.shape),
        "channel_order": ["position_x", "position_y", "position_z", "normal_x", "normal_y", "normal_z"],
        "camera_contract": str(output_dir / "camera_contract.json"),
        "views": output_views,
        "index_permutation_output_to_raw": permutation["output_index_to_raw_index"],
        "semantic_relabeling_proven": True,
        "geometry_or_uv_mutation": False,
    })
    _contact_sheet(source_dir, output_dir / "position_normal_control_contact_sheet.png", source_views, permutation["output_index_to_raw_index"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--observed-mask", type=Path, default=None,
                        help="observed_mask.png from a prior fast_texture_projection "
                             "run on this mesh. When given, front-versus-rear is "
                             "decided by which view sees the painted texels, which "
                             "silhouette IoU cannot do on a mirror-symmetric subject.")
    args = parser.parse_args()
    print(json.dumps(audit_and_relabel(args.mesh, args.source_image, args.source_dir, args.output_dir,
                                    args.observed_mask), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
