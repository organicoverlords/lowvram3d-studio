"""Map which source camera owns which texel in a Hunyuan3D-Paint bake.

The painted fennec came out covered in hard-edged brown/cream plates. Three
hypotheses were live: too few texels per face (averaging), multiview
appearance inconsistency (the generated views disagree), or projection/view
assignment (individually fine views blended onto the mesh badly).

Only the third predicts *hard* boundaries, and reading the vendor's bake found
three separate discontinuities that would produce them:

  1. cos_image[cos_image < cos_thres] = 0        -- a cut, not a falloff
  2. project_cos_map = weight * cos ** bake_exp  -- bake_exp 4, near winner-take-all
  3. fast_bake_texture drops a whole view when 99% of its footprint is already
     painted, in fixed camera order -- all-or-nothing, and order-dependent

So this replays the exact same projection and blend with the generated views
replaced by flat debug colours, one per camera. Nothing else changes. Any
structure in the output is attributable to geometry and blending alone, because
the inputs carry no structure at all.

Deliberately no diffusion, no light remover, no UNet: this loads the renderer
only, so it can run while a generation job owns the GPU.

Outputs, all under the subject's compare directory:
  provenance_blend.png   the weights as the pipeline actually mixes them
  provenance_argmax.png  the winning camera per texel, flat colours
  provenance.json        per-view coverage, and which views got dropped
"""

import argparse
import json
import pathlib
import sys


# Distinct and far apart in hue so a boundary between any two is unmistakable,
# and so a blended edge is visibly a gradient rather than a fourth colour.
VIEW_COLOURS = [
    (255, 60, 60),     # azim 0    elev 0    front, weight 1.0
    (60, 255, 60),     # azim 90   elev 0    right, weight 0.1
    (60, 120, 255),    # azim 180  elev 0    back,  weight 0.5
    (255, 230, 60),    # azim 270  elev 0    left,  weight 0.1
    (255, 60, 255),    # azim 0    elev 90   top,   weight 0.05
    (60, 255, 255),    # azim 180  elev -90  bottom, weight 0.05
]
VIEW_NAMES = ["front", "right", "back", "left", "top", "bottom"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--render-size", type=int, default=512)
    ap.add_argument("--texture-size", type=int, default=1024)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image

    torch.backends.cudnn.enabled = False

    from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # These must match Hunyuan3DTexGenConfig exactly, so they are read off the
    # vendor class rather than retyped -- a vendor update cannot then silently
    # desynchronise the diagnostic from the thing it is diagnosing.
    from hy3dgen.texgen.pipelines import Hunyuan3DTexGenConfig
    cfg = Hunyuan3DTexGenConfig.__new__(Hunyuan3DTexGenConfig)
    Hunyuan3DTexGenConfig.__init__(cfg, "", "", "hunyuan3d-paint-v2-0-turbo")
    azims = list(cfg.candidate_camera_azims)
    elevs = list(cfg.candidate_camera_elevs)
    weights = list(cfg.candidate_view_weights)
    bake_exp = cfg.bake_exp

    render = MeshRender(default_resolution=args.render_size,
                        texture_size=args.texture_size)

    # load_mesh takes the mesh itself, not a path -- the pipeline hands it a
    # trimesh object it already has in hand.
    import trimesh
    scene = trimesh.load(args.mesh, process=False)
    render.load_mesh(scene.to_geometry() if hasattr(scene, "geometry") else scene)

    flat = np.zeros((args.render_size, args.render_size, 3), dtype=np.uint8)

    projected, cos_maps, per_view = [], [], []
    for i, (elev, azim, weight) in enumerate(zip(elevs, azims, weights)):
        flat[:] = VIEW_COLOURS[i]
        texture, cos_map, _boundary = render.back_project(
            Image.fromarray(flat), elev, azim)
        cos_map = weight * (cos_map ** bake_exp)
        projected.append(texture)
        cos_maps.append(cos_map)
        per_view.append({
            "index": i,
            "name": VIEW_NAMES[i],
            "elev": elev,
            "azim": azim,
            "weight": weight,
            "texels_covered": int((cos_map > 0).sum().item()),
        })

    # fast_bake_texture, replayed rather than called, so the 99%-skip decision
    # can be recorded instead of happening invisibly.
    total = float(args.texture_size * args.texture_size)
    merged = torch.zeros(render.texture_size + (3,), device=render.device)
    trust = torch.zeros(render.texture_size + (1,), device=render.device)
    argmax_score = torch.zeros(render.texture_size + (1,), device=render.device)
    argmax_view = torch.full(render.texture_size + (1,), -1,
                             dtype=torch.int16, device=render.device)

    for i, (texture, cos_map) in enumerate(zip(projected, cos_maps)):
        view_sum = (cos_map > 0).sum()
        painted_sum = ((cos_map > 0) * (trust > 0)).sum()
        ratio = float(painted_sum / view_sum) if view_sum > 0 else 1.0
        per_view[i]["already_painted_fraction"] = round(ratio, 4)
        if view_sum > 0 and ratio > 0.99:
            per_view[i]["dropped_by_99pct_rule"] = True
            continue
        per_view[i]["dropped_by_99pct_rule"] = False
        merged += texture * cos_map
        trust += cos_map

        # Ownership is tracked over the same views the blend actually used, so
        # a dropped view cannot own a texel it never contributed to.
        wins = cos_map > argmax_score
        argmax_score = torch.where(wins, cos_map, argmax_score)
        argmax_view = torch.where(wins, torch.full_like(argmax_view, i), argmax_view)

    merged = merged / torch.clamp(trust, min=1e-8)

    blend = (merged.clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(blend).save(out_dir / "provenance_blend.png")

    owner = argmax_view[..., 0].cpu().numpy()
    hard = np.zeros((args.texture_size, args.texture_size, 3), dtype=np.uint8)
    for i, colour in enumerate(VIEW_COLOURS):
        hard[owner == i] = colour
    Image.fromarray(hard).save(out_dir / "provenance_argmax.png")

    for entry in per_view:
        won = int((owner == entry["index"]).sum())
        entry["texels_won"] = won
        entry["texels_won_pct"] = round(100.0 * won / total, 2)

    receipt = {
        "schema": "lowvram3d_paint_provenance_v1",
        "mesh": args.mesh,
        "render_size": args.render_size,
        "texture_size": args.texture_size,
        "bake_exp": bake_exp,
        "unowned_texels_pct": round(100.0 * float((owner < 0).sum()) / total, 2),
        "views": per_view,
    }
    (out_dir / "provenance.json").write_text(json.dumps(receipt, indent=2))
    json.dump(receipt, sys.stdout, indent=2)
    print()


main()
