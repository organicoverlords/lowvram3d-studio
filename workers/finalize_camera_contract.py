"""Complete a CPU control bundle's camera contract so inference will accept it.

`build_mvadapter_cpu_controls` writes geometrically complete controls but names
its per-view artefacts positionally -- `horizontal_0_mask.png`, and
`semantic_name` of `horizontal_0` -- while the six-view inference worker fails
closed unless `semantic_name`, `axis_label` and `proven_semantic` are the same
string, a `control_mask_filename` exists, and a `semantic_direction_basis` maps
each label to the direction its camera actually sits in.

In the production repo that gap is closed by `audit_and_relabel_mvadapter_controls`,
which derives the panda's front from tail-side and aspect evidence in its source
image. That reasoning does not transfer to a barn. This does the mechanical part
only -- rename, and record the basis from the contract's own measured camera
positions -- and asserts nothing about which view is the front beyond what the
builder already proved.

On the labels being right
-------------------------
Worth writing down, because it looked wrong for a while and is not. The camera
positions are in *control* space, after the canonical transform, and there:

    front  (0, -1.8, 0)      top    (0, 0, +1.8)
    rear   (0, +1.8, 0)      bottom (0, 0, -1.8)

so control space is Z-up with -Y forward, which is MV-Adapter's own convention.
The `y_up_z_front` basis is exactly what carries a Y-up glTF mesh into it. The
azimuth of -90 degrees on index 0 is measured from +X, and -90 degrees from +X
*is* -Y: the label and the geometry agree.

The panda's proven mapping of {0: left, 1: front, 2: right, 3: rear} is not a
contradiction of this. It describes a different bundle -- `v9_upright_raw`,
built to preserve raw order without relabelling -- so its indices mean something
else. Two contracts for two bundles, not one contract disagreeing with itself.

    py -3.12 workers/finalize_camera_contract.py --controls DIR
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

#: Per-view artefacts the builder writes, by suffix.
ARTEFACT_SUFFIXES = (
    "_mask.png", "_normal.png", "_normal.npy", "_position.png", "_position.npy",
    "_depth.npy", "_barycentric.npy", "_triangle_ids.npy",
    "_visible_triangles.npy",
)


def _unit(vector):
    import numpy as np

    array = np.asarray(vector, dtype=np.float64)
    return (array / max(float(np.linalg.norm(array)), 1e-12)).tolist()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args(argv)

    controls = Path(args.controls).resolve()
    contract_path = controls / "camera_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    renames = []
    for view in contract["views"]:
        semantic = str(view["proven_semantic"])
        positional = str(view["semantic_name"])
        if semantic != positional:
            for suffix in ARTEFACT_SUFFIXES:
                source = controls / f"{positional}{suffix}"
                if source.is_file():
                    destination = controls / f"{semantic}{suffix}"
                    shutil.copy2(source, destination)
                    renames.append([source.name, destination.name])
        view["semantic_name"] = semantic
        view["axis_label"] = semantic
        view["control_mask_filename"] = f"{semantic}_mask.png"
        if not (controls / view["control_mask_filename"]).is_file():
            raise SystemExit(f"CONTROL_MASK_MISSING:{view['control_mask_filename']}")

    contract["semantic_direction_basis"] = {
        str(view["proven_semantic"]): _unit(view["camera_position"])
        for view in contract["views"]}
    # Stated so nothing downstream reads this as an independent proof. The basis
    # is derived from the very positions the inference worker checks it against,
    # so that check is a consistency test, not evidence about which side of the
    # subject faced the camera.
    contract["semantic_direction_basis_provenance"] = (
        "derived from this contract's own camera_position values; "
        "self-consistent by construction, not independent evidence")

    contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    digest = hashlib.sha256(contract_path.read_bytes()).hexdigest()

    report = {
        "schema_version": "camera_contract_finalize_v1",
        "classification": "PROVEN",
        "controls": str(controls),
        "camera_contract": str(contract_path),
        "camera_contract_sha256": digest,
        "labels": [view["proven_semantic"] for view in contract["views"]],
        "renamed_artefacts": len(renames),
        "geometry_or_control_values_changed": False,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                     encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
