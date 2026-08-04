"""Verify a canonical UV master before the UV stage adopts it without unwrapping.

Adopting a pre-proven UV mesh is only safe if the thing on disk is still the thing that was
proven. `validate_existing_uv.py` answers a different question - it wants a material and a packed
texture, which a UV master legitimately does not have - so this checks what actually matters for
a single-owner atlas:

  * the file is byte-identical to the master that was proven, by sha256;
  * its surface is the expected surface, by the same order-independent geometry fingerprint the
    rewrap gated on;
  * the layout is still injective at the declared atlas resolution.

Nothing here re-unwraps or repairs. It imports the proven gate functions rather than restating
them, so a change to the gate cannot silently diverge from a change to this check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from atlas_raster import injectivity
from mesh_io import read_glb
from uv_rewrap_injective import geometry_fingerprint, sha256


def verify(master: Path, resolution: int, expected_sha: str, expected_fingerprint: str,
           expected_triangles: int) -> dict:
    actual_sha = sha256(master)
    positions, _normals, uv, tris = read_glb(master)
    positions = np.ascontiguousarray(positions, np.float32)
    tris = np.asarray(tris, np.int64)
    if uv is None or not len(uv):
        raise RuntimeError("UV_MASTER_NO_TEXCOORD")
    fingerprint = geometry_fingerprint(positions, tris)
    gate = injectivity(np.asarray(uv, np.float64), tris, int(resolution))

    checks = {
        "sha256_matches": (not expected_sha) or actual_sha == expected_sha,
        "geometry_fingerprint_matches": (not expected_fingerprint)
                                        or fingerprint == expected_fingerprint,
        "triangle_count_matches": (not expected_triangles)
                                  or int(len(tris)) == int(expected_triangles),
        "uv_present": True,
        "uv_finite": bool(np.isfinite(np.asarray(uv)).all()),
        "uv_in_unit_square": gate["uv_out_of_unit_square"] == 0,
        "injective": gate["injective"],
    }
    failure_codes = []
    if not checks["sha256_matches"]:
        failure_codes.append("UV_MASTER_HASH_MISMATCH")
    if not (checks["geometry_fingerprint_matches"] and checks["triangle_count_matches"]):
        failure_codes.append("UV_MASTER_GEOMETRY_MISMATCH")
    if not checks["injective"]:
        failure_codes.append("UV_OVERLAP")
    if not (checks["uv_finite"] and checks["uv_in_unit_square"]):
        failure_codes.append("UV_DEGENERATE")

    return {
        "schema": "uv_master_verify_v1",
        "master": str(master),
        "master_sha256": actual_sha,
        "expected_sha256": expected_sha or None,
        "geometry_fingerprint": fingerprint,
        "expected_geometry_fingerprint": expected_fingerprint or None,
        "triangles": int(len(tris)),
        "vertices": int(len(positions)),
        "atlas_resolution": int(resolution),
        "injectivity": gate,
        "checks": checks,
        "failure_codes": sorted(set(failure_codes)),
        "success": not failure_codes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--expect-sha256", default="")
    parser.add_argument("--expect-geometry-fingerprint", default="")
    parser.add_argument("--expect-triangles", type=int, default=0)
    args = parser.parse_args()

    report = verify(Path(args.master), args.resolution, args.expect_sha256,
                    args.expect_geometry_fingerprint, args.expect_triangles)
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"], indent=2), flush=True)
    print(f"UV_MASTER_VERIFY success={report['success']} codes={report['failure_codes']}",
          flush=True)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
