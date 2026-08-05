"""Attach a proven raw-index -> semantic permutation to a control bundle's camera contract.

The raw bundle names its horizontal cameras ``horizontal_0..3`` and the builder's fixture
guesses which is which.  That guess was wrong for this asset, so the permutation is supplied
externally and applied here.

Raw order is preserved: view ``i`` of the control tensor stays view ``i``, and every record
keeps an explicit ``control_file_prefix`` pointing at the untouched raw arrays.  Only the
labels change, so nothing downstream can infer a semantic from a list position.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from build_mvadapter_cpu_controls import CANONICAL_INVERSE

CONTROL_SUFFIXES = {
    "control_position_filename": "_position.npy",
    "control_normal_filename": "_normal.npy",
    "control_depth_filename": "_depth.npy",
    "control_mask_filename": "_mask.png",
    "control_triangle_id_filename": "_triangle_ids.npy",
    "control_barycentric_filename": "_barycentric.npy",
}
OPPOSED_PAIRS = (("front", "rear", "front_rear_direction_dot"),
                 ("left", "right", "left_right_direction_dot"),
                 ("top", "bottom", "top_bottom_direction_dot"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    return array / max(float(np.linalg.norm(array)), 1e-12)


def apply_permutation(bundle: Path, raw_to_semantic: dict[int, str], provenance: dict) -> dict:
    contract = json.loads((bundle / "camera_contract.json").read_text(encoding="utf-8"))
    views = sorted(contract["views"], key=lambda item: int(item["index"]))
    if [int(view["index"]) for view in views] != list(range(6)):
        raise RuntimeError("CAMERA_PERMUTATION_RAW_INDEX_INVALID")
    if sorted(raw_to_semantic) != list(range(6)) or len(set(raw_to_semantic.values())) != 6:
        raise RuntimeError("CAMERA_PERMUTATION_NOT_A_BIJECTION")

    previous = {int(view["index"]): view.get("proven_semantic") for view in views}
    by_label: dict[str, dict] = {}
    for view in views:
        index = int(view["index"])
        semantic = raw_to_semantic[index]
        prefix = str(view["semantic_name"])
        for field, suffix in CONTROL_SUFFIXES.items():
            candidate = bundle / f"{prefix}{suffix}"
            if not candidate.is_file():
                raise RuntimeError(f"CAMERA_PERMUTATION_CONTROL_MISSING:{candidate.name}")
            view[field] = candidate.name
        view["control_file_prefix"] = prefix
        view["raw_index"] = index
        view["raw_semantic_name"] = prefix
        view["superseded_semantic"] = previous[index]
        view["semantic_name"] = semantic
        view["axis_label"] = semantic
        view["proven_semantic"] = semantic
        view["semantic_source"] = provenance["semantic_source"]
        # Control space is the canonical frame the rasteriser worked in; the mesh-local
        # direction is what a reader needs to reason about the asset itself.
        view["camera_direction_control_space"] = list(view["camera_direction"])
        view["camera_direction_mesh_local"] = (
            CANONICAL_INVERSE @ np.asarray(view["camera_direction"], dtype=np.float64)).tolist()
        view["camera_up_mesh_local"] = (
            CANONICAL_INVERSE @ np.asarray(view["camera_up"], dtype=np.float64)).tolist()
        by_label[semantic] = view

    missing = [label for _a, _b, _k in OPPOSED_PAIRS for label in (_a, _b) if label not in by_label]
    if missing:
        raise RuntimeError(f"CAMERA_PERMUTATION_LABEL_MISSING:{missing}")

    opposition = {}
    for first, second, key in OPPOSED_PAIRS:
        dot = float(np.dot(unit(by_label[first]["camera_direction"]),
                           unit(by_label[second]["camera_direction"])))
        if dot > -0.999:
            raise RuntimeError(f"CAMERA_PERMUTATION_NOT_OPPOSED:{first}:{second}:{dot}")
        contract[key] = dot
        opposition[key] = dot

    contract["views"] = views
    contract["index_semantics"] = {str(index): raw_to_semantic[index] for index in range(6)}
    contract["raw_to_semantic"] = {str(index): raw_to_semantic[index] for index in range(6)}
    contract["semantic_to_raw"] = {label: index for index, label in raw_to_semantic.items()}
    contract["semantic_direction_basis"] = {
        label: unit(view["camera_position"]).tolist() for label, view in by_label.items()}
    contract["raw_order_preserved"] = True
    contract["control_arrays_rewritten"] = False
    contract["semantic_mapping_proven"] = True
    contract["semantic_mapping_source"] = provenance["semantic_source"]
    contract["semantic_mapping_evidence"] = provenance["evidence"]
    contract["superseded_index_semantics"] = {str(k): v for k, v in previous.items()}
    contract["superseded_classification"] = "PANDA_CAMERA_SEMANTICS_PREVIOUS_CONTRACT_REJECTED"
    contract["classification"] = "PANDA_CAMERA_SEMANTICS_PROVEN_BY_USER_REVIEW_AND_CAMERA_VECTORS"
    contract["opposition_checks"] = opposition
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--raw-to-semantic", required=True,
                        help='JSON object, e.g. {"0":"left","1":"rear",...}')
    parser.add_argument("--semantic-source", default="USER_VISION_REVIEW_20260803")
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    bundle = Path(args.bundle)
    mapping = {int(k): str(v) for k, v in json.loads(args.raw_to_semantic).items()}
    provenance = {"semantic_source": args.semantic_source, "evidence": list(args.evidence)}
    contract = apply_permutation(bundle, mapping, provenance)

    destination = bundle / "camera_contract.json"
    backup = bundle / "camera_contract_pre_permutation.json"
    if not backup.exists():
        backup.write_text(destination.read_text(encoding="utf-8"), encoding="utf-8")
    destination.write_text(json.dumps(contract, indent=2), encoding="utf-8")

    report = {
        "schema": "camera_permutation_applied_v1",
        "classification": contract["classification"],
        "superseded_classification": contract["superseded_classification"],
        "bundle": str(bundle),
        "camera_contract": str(destination),
        "camera_contract_sha256": sha256_file(destination),
        "camera_contract_pre_permutation": str(backup),
        "raw_to_semantic": contract["raw_to_semantic"],
        "semantic_to_raw": contract["semantic_to_raw"],
        "superseded_index_semantics": contract["superseded_index_semantics"],
        "opposition_checks": contract["opposition_checks"],
        "raw_order_preserved": True,
        "control_arrays_rewritten": False,
        "semantic_mapping_source": args.semantic_source,
        "views": [
            {
                "raw_index": int(view["raw_index"]),
                "semantic_label": view["semantic_name"],
                "superseded_semantic": view["superseded_semantic"],
                "azimuth_deg": view["azimuth_deg"],
                "elevation_deg": view["elevation_deg"],
                "camera_direction_control_space": view["camera_direction_control_space"],
                "camera_direction_mesh_local": view["camera_direction_mesh_local"],
                "camera_up_control_space": view["camera_up"],
                "camera_up_mesh_local": view["camera_up_mesh_local"],
                "control_file_prefix": view["control_file_prefix"],
                **{field: view[field] for field in CONTROL_SUFFIXES},
            }
            for view in sorted(contract["views"], key=lambda item: int(item["raw_index"]))
        ],
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"CAMERA_PERMUTATION_APPLIED {contract['classification']} "
          f"map={contract['raw_to_semantic']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
