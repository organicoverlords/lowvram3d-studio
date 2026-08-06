"""Did the photograph land on the geometry it depicts, or 180 degrees from it?

Every front-axis heuristic this project has tried answers a question adjacent to
the one that matters:

    tail colour            -> which side is the tail on in the SOURCE IMAGE
    silhouette IoU         -> which camera's OUTLINE matches the source
    painted-texel coverage -> which camera the PROJECTION painted from

The last is the strongest of the three and it still cannot catch a 180 degree
error, because it is circular: it reports where paint landed, and if the
projection solved the front backwards then the paint is on the back and the
audit confidently agrees. On the red panda it did exactly that, 0.8364 against
0.0016, while the photographic face sat on the back of the character's head.

The question none of them ask is whether the photograph landed on geometry that
looks like the photograph. This asks it, by rendering the photo-textured mesh
through a camera's own `triangle_ids` and comparing that to the same camera's
normal render. A face has high-frequency geometry; a hood does not. So where the
photograph carries most of its detail, the surface underneath should carry
detail too, and a 180 degree error breaks that correspondence.

    py workers/check_photo_landed_on_detail.py \
       --controls evidence/compare/panda2/controls_512_audited \
       --mesh evidence/compare/panda2/front_texture/textured.glb \
       --atlas evidence/compare/panda2/front_texture/basecolor.png \
       --out evidence/compare/panda2/photo_vs_geometry.png

This is a gate you read, not one you trust blindly: it emits the correlation and
the sheet, and the sheet is the evidence. See the retraction at the top of
docs/JANUS-six-view-defect-20260806.md for why that distinction is not
pedantic here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

#: THE CORRELATION DOES NOT WORK AS A GATE. Measured on the two assets whose
#: answer is known:
#:
#:     panda (front axis WRONG)   index 0: 0.0809   true opposite: 0.2788
#:     whale (front axis RIGHT)   index 0: 0.1240   true opposite: 0.3027
#:
#: The two fronts are 0.08 and 0.12 -- no separation -- and on both assets the
#: unobserved hemisphere scores HIGHER than the observed one, which is backwards.
#: Invented camouflage is high-frequency everywhere, so it correlates with fur
#: geometry better than a photograph does. Any threshold drawn through these four
#: numbers is fitted to noise.
#:
#: It is left here, computed and printed, precisely so nobody re-derives it and
#: believes it. There is no boolean verdict in this worker's output. The sheet is
#: the evidence, a person reads it, and that is currently the only thing that has
#: ever caught a 180 degree front-axis error in this route.
CORRELATION_IS_NOT_A_GATE = True


def detail_energy(array):
    """High-frequency energy per pixel, as a stand-in for 'there is shape here'."""
    import numpy as np

    if array.ndim == 3:
        array = array.mean(axis=2)
    gy, gx = np.gradient(array.astype(np.float64))
    return np.abs(gx) + np.abs(gy)


def compare(controls: Path, mesh: Path, atlas: Path, out: Path,
            worker: Path) -> dict:
    import numpy as np
    from PIL import Image, ImageDraw

    contract = json.loads((controls / "camera_contract.json").read_text(encoding="utf-8"))
    views = sorted(contract["views"], key=lambda v: int(v["index"]))
    base = np.asarray(views[0]["camera_direction"], np.float64)
    opposite = int(min(range(1, 6), key=lambda i: float(
        np.dot(base, np.asarray(views[i]["camera_direction"], np.float64)))))

    rows = []
    for index in (0, opposite):
        name = str(views[index]["proven_semantic"])
        rendered = out.parent / f"_photo_through_{name}.png"
        subprocess.run(
            [sys.executable, str(worker), "--mesh", str(mesh), "--atlas", str(atlas),
             "--controls", str(controls), "--view", name, "--out", str(rendered)],
            check=True, capture_output=True)
        photo = np.asarray(Image.open(rendered).convert("RGB"), np.float64)
        normal = np.load(controls / f"{name}_normal.npy")
        surface = np.linalg.norm(normal, axis=2) > 0.1

        photo_detail = detail_energy(photo)[surface]
        geom_detail = detail_energy(normal)[surface]
        if photo_detail.std() < 1e-9 or geom_detail.std() < 1e-9:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(photo_detail, geom_detail)[0, 1])
        rows.append({"index": index, "name": name, "render": str(rendered),
                     "detail_correlation": round(correlation, 4)})

    size, bar = 340, 28
    sheet = Image.new("RGB", (size * 2, (size + bar) * 2), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for row_index, row in enumerate(rows):
        y = row_index * (size + bar)
        label = (f"index {row['index']} ({row['name']})   "
                 f"photo/geometry detail correlation {row['detail_correlation']}")
        draw.rectangle([0, y, size * 2, y + bar], fill=(12, 12, 14))
        draw.text((8, y + 8), label, fill=(255, 255, 120))
        photo = Image.open(row["render"]).convert("RGB").resize((size, size), Image.LANCZOS)
        nrm = Image.open(controls / f"{row['name']}_normal.png").convert("RGB").resize(
            (size, size), Image.LANCZOS)
        sheet.paste(photo, (0, y + bar))
        sheet.paste(nrm, (size, y + bar))
        draw.text((8, y + bar + 6), "PHOTO ATLAS", fill=(255, 255, 255))
        draw.text((size + 8, y + bar + 6), "GEOMETRY (normals)", fill=(255, 255, 255))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)

    return {
        "schema": "lowvram3d_photo_landed_on_detail_v2",
        "controls": str(controls),
        "true_opposite_index": opposite,
        "views": rows,
        "sheet": str(out),
        "verdict": "HUMAN_REVIEW_REQUIRED",
        "note": ("detail_correlation is DIAGNOSTIC ONLY and does not separate a "
                 "correct front axis from a 180-degree-wrong one: panda(wrong)=0.081 "
                 "vs whale(right)=0.124. Read the sheet. If the photographic detail "
                 "sits on featureless geometry, the axis is wrong."),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", required=True, type=Path)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--atlas", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--worker", type=Path,
                        default=Path(__file__).with_name("render_view_from_controls.py"))
    args = parser.parse_args(argv)

    report = compare(args.controls, args.mesh, args.atlas, args.out, args.worker)
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
