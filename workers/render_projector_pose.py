"""Render the mesh in the PROJECTOR's own rotation convention at both poses.

Every previous attempt to answer "which way does yaw -1.667 face" compared a
number from one camera convention against a number from another. This renders
both candidate poses using fast_texture_projection.rotation itself, so there is
nothing to translate: what you see is what the projector paints from.
"""
import sys
import numpy as np
import trimesh
from PIL import Image, ImageDraw

sys.path.insert(0, r"C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803\workers")
from fast_texture_projection import rotation

E = r"C:\Users\Lauri\Desktop\lowvram3d-scene-smoke-20260803\evidence\compare\panda2"
OUT = (r"C:\Users\Lauri\AppData\Local\Temp\claude\C--Users-Lauri-Desktop"
       r"\bef7e8c6-36b0-437d-85a9-2492519bc896\scratchpad")

scene = trimesh.load(E + r"\panda_mt_uv.glb", process=False)
mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
vertices = np.asarray(mesh.vertices, np.float64)
vertices -= vertices.mean(axis=0)
vertices /= max(float(np.abs(vertices).max()), 1e-9)
faces = np.asarray(mesh.faces, np.int64)
normals = np.asarray(mesh.vertex_normals, np.float64)

SIZE = 512


def clay(yaw, pitch, roll):
    """Shaded z-buffered render under the projector's rotation matrix."""
    matrix = rotation(yaw, pitch, roll)
    v = vertices @ matrix.T
    n = normals @ matrix.T

    span = float(np.abs(v[:, :2]).max()) or 1.0
    scale = (SIZE * 0.44) / span
    screen = np.empty((len(v), 2))
    screen[:, 0] = v[:, 0] * scale + SIZE * 0.5
    screen[:, 1] = v[:, 1] * scale + SIZE * 0.5

    image = np.ones((SIZE, SIZE), np.float64)
    zbuf = np.full((SIZE, SIZE), -np.inf)
    # Directed, not abs: a surface turned away from the camera must not shade
    # like one facing it. Using abs() here is a bug this project has shipped.
    shade = np.clip(n[:, 2], 0.0, 1.0) * 0.65 + 0.30
    tri = screen[faces]
    depth = v[faces][:, :, 2]
    tri_shade = shade[faces].mean(axis=1)

    for index in np.argsort(depth.mean(axis=1)):
        t = tri[index]
        x0, y0 = np.floor(t.min(axis=0)).astype(int)
        x1, y1 = np.ceil(t.max(axis=0)).astype(int) + 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, SIZE), min(y1, SIZE)
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        px, py = xs + 0.5, ys + 0.5
        (ax, ay), (bx, by), (cx, cy) = t
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if abs(area) < 1e-12:
            continue
        w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / area
        w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / area
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        z = w0 * depth[index, 0] + w1 * depth[index, 1] + w2 * depth[index, 2]
        window = zbuf[y0:y1, x0:x1]
        write = inside & (z > window)
        if not write.any():
            continue
        window[write] = z[write]
        image[y0:y1, x0:x1][write] = tri_shade[index]
    return (np.clip(image, 0, 1) * 255).astype("uint8")


POSES = [("FITTED  yaw -1.667 pitch 13.333  (IoU 0.915)", (-1.667, 13.333, 0.0)),
         ("ANTIPODE yaw 178.333 pitch -13.333 (IoU 0.657)", (178.333, -13.333, 0.0))]

source = Image.open(E + r"\panda2_512.png").convert("RGBA")
flat = Image.new("RGB", source.size, (255, 255, 255))
flat.paste(source, mask=source.split()[3])

bar = 26
canvas = Image.new("RGB", (SIZE * 3, SIZE + bar), (20, 20, 24))
draw = ImageDraw.Draw(canvas)
canvas.paste(flat.resize((SIZE, SIZE), Image.LANCZOS), (0, bar))
draw.text((6, 7), "SOURCE", fill=(255, 255, 120))
for i, (label, pose) in enumerate(POSES, start=1):
    canvas.paste(Image.fromarray(clay(*pose)).convert("RGB"), (i * SIZE, bar))
    draw.text((i * SIZE + 6, 7), label, fill=(255, 255, 120))
canvas.save(OUT + r"\projector_poses.png")
print("wrote projector_poses.png")
