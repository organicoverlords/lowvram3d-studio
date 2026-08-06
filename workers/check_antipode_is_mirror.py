"""Is the derived antipode really the antipode? Test the one thing that must hold.

Under orthographic projection the silhouette from u and from -u is the same set
of points in 3D. In IMAGE space the two differ by a horizontal mirror, because
looking from the far side reverses image x. So for the derivation to be right:

    mask(yaw+180, -pitch, -roll)  ==  fliplr( mask(yaw, pitch, roll) )

exactly, up to the independent bbox re-fit each pose performs. If it holds, the
antipode is exact and the IoU gap between the two poses is entirely explained by
the source silhouette being laterally asymmetric. If it fails, the derivation or
the rasteriser is wrong and every conclusion resting on it is void.
"""
import sys
import numpy as np
import cv2
import trimesh
from PIL import Image

sys.path.insert(0, r"C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803\workers")
from fast_texture_projection import rotation, fit_to_mask, _barycentric_grid

E = r"C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803\evidence\compare\panda2"

scene = trimesh.load(E + r"\panda_mt_uv.glb", process=False)
mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
vertices = np.asarray(mesh.vertices, np.float64)
triangles = np.asarray(mesh.faces, np.int64)
centred = vertices - vertices.mean(axis=0)

alpha = np.asarray(Image.open(E + r"\panda2_2048_alpha.png").convert("L"))
probe = 192
small = cv2.resize((alpha > 127).astype(np.uint8), (probe, probe),
                   interpolation=cv2.INTER_NEAREST) > 0

stride = max(1, triangles.shape[0] // 60000)
faces = triangles[::stride]
bary = _barycentric_grid(4)


def raster(yaw, pitch, roll):
    matrix = rotation(yaw, pitch, roll)
    rotated = centred @ matrix.T
    scale, offset = fit_to_mask(rotated, small)
    xy = rotated[:, :2] * scale + offset
    corners = xy[faces]
    wa, wb = bary[None, :, 0, None], bary[None, :, 1, None]
    points = (corners[:, 0][:, None, :] * (1.0 - wa - wb)
              + corners[:, 1][:, None, :] * wa
              + corners[:, 2][:, None, :] * wb)
    xs = np.clip(points[..., 0].astype(np.int32), 0, probe - 1).ravel()
    ys = np.clip(points[..., 1].astype(np.int32), 0, probe - 1).ravel()
    hit = np.zeros((probe, probe), bool)
    hit[ys, xs] = True
    return hit


def iou(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


FITTED = (-1.667, 13.333, 0.0)  # from front_texture/projection_receipt.json
ANTIPODE = (FITTED[0] + 180.0, -FITTED[1], -FITTED[2])

fitted = raster(*FITTED)
antipode = raster(*ANTIPODE)

print(f"fitted   {FITTED}   IoU vs source {iou(fitted, small):.4f}")
print(f"antipode {ANTIPODE}   IoU vs source {iou(antipode, small):.4f}")
print()
print(f"antipode vs fliplr(fitted)   IoU {iou(antipode, np.fliplr(fitted)):.4f}   <-- must be ~1.0")
print(f"antipode vs fitted           IoU {iou(antipode, fitted):.4f}")
print()
print(f"source silhouette vs its own mirror  IoU {iou(small, np.fliplr(small)):.4f}"
      "   <-- how symmetric the source is")

Image.fromarray((np.concatenate([fitted, np.fliplr(fitted), antipode, small],
                                axis=1) * 255).astype(np.uint8)).save(
    r"C:\Users\Lauri\AppData\Local\Temp\claude\C--Users-Lauri-Desktop"
    r"\bef7e8c6-36b0-437d-85a9-2492519bc896\scratchpad\antipode_masks.png")
print("\nwrote antipode_masks.png: fitted | fliplr(fitted) | antipode | source")

print("\n--- all four sign combinations ---")
y0, p0, r0 = FITTED
for label, pose in (
    ("fitted            (y,   p,  r)", (y0, p0, r0)),
    ("spun 180          (y+180, p,  r)", (y0 + 180, p0, r0)),
    ("spun 180 + mirror (y+180,-p, -r)", (y0 + 180, -p0, -r0)),
    ("mirror only       (y,  -p, -r)", (y0, -p0, -r0)),
):
    m = raster(*pose)
    print(f"{label}  IoU vs source {iou(m, small):.4f}")
