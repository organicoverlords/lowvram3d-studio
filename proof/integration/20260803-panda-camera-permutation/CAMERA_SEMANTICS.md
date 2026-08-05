# Panda camera semantics — corrected permutation (2026-08-03)

Classification: **`PANDA_CAMERA_SEMANTICS_PROVEN_BY_USER_REVIEW_AND_CAMERA_VECTORS`**
Supersedes: **`PANDA_CAMERA_SEMANTICS_PREVIOUS_CONTRACT_REJECTED`**

Historical receipts are retained. Nothing was deleted or silently renamed; the pre-permutation
contract is kept beside the corrected one as `camera_contract_pre_permutation.json`.

## Mapping

| raw index | corrected semantic | legacy contract (v6, by direction) | v8 builder fixture guess |
| --- | --- | --- | --- |
| 0 | left | left | front |
| 1 | **rear** | **front** | right |
| 2 | right | right | rear |
| 3 | **front** | **rear** | left |
| 4 | top | top | top |
| 5 | bottom | bottom | bottom |

The legacy contract had **front and rear swapped**; the side and vertical pairs were already
correct. The v8 builder's own fixture guess was a different, also-wrong rotation of the
horizontal ring — it is superseded by the same decision.

```
semantic_to_raw = {"front": 3, "right": 2, "rear": 1, "left": 0, "top": 4, "bottom": 5}
raw_to_semantic = {0: "left", 1: "rear", 2: "right", 3: "front", 4: "top", 5: "bottom"}
```

## Evidence

User vision review of `contact_sheet_a_raw_index.png`:

* raw3 presents the muzzle, eyes, ears and face toward the camera;
* raw1 is dominated by the tail and rear equipment;
* raw0 and raw2 are opposite lateral profiles;
* raw4 shows upper surfaces and the long body axis, raw5 the underside.

Face-like texture appearing in raw4/raw5 does **not** make either a front camera. It comes
from head surfaces still being visible at steep elevation, and from view-conditioned texture
leakage in the earlier generated atlas.

The automatic bilateral-symmetry probe (`prove_camera_semantics.py`) put the facing axis on
the raw1/raw3 pair with a 0.096 margin, which agrees with the corrected mapping on the
**axis**. It could not resolve the sign, which is why the run stopped for user review rather
than guessing.

## Camera-vector opposition checks

| pair | dot |
| --- | --- |
| front (raw3) · rear (raw1) | -1.0 |
| left (raw0) · right (raw2) | -1.0 |
| top (raw4) · bottom (raw5) | -0.99999994 |

## Downstream effect on the old 384x20 run

`MVADAPTER_NUMERICALLY_PROVEN_OLD_CONTROLS_UNPROMOTED`. Its numerical gate stands. Its visual
QA claim is reduced to `NOT_FINAL` because its controls were built from a mesh that still
contained the bar, and its filenames used the swapped front/rear mapping. It is not to be
described as final visual QA passed.
