"""Asset profiles for Pipeline V2.

A profile is the set of decisions that used to be hardcoded per asset: how many LODs, how big the
atlas, whether a rig is needed, whether thin features must survive decimation, whether props get
separated. Every one of these was a manual choice during the shaman run, and every one of them is
a place a later stage silently did the wrong thing.

Detection is deliberately conservative. It reads the source silhouette - aspect ratio, how much of
the bounding box the subject fills, how much of its area sits in thin structures - and returns a
profile plus the evidence behind it. When the evidence is weak it says so rather than guessing;
`Auto` then falls back to the safest profile that preserves the most (thin features on, props
separated), because over-preserving costs triangles while under-preserving destroys the asset.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field


@dataclass(frozen=True)
class AssetProfile:
    name: str
    lod_triangle_targets: tuple[int, ...]
    texture_resolution: int
    rig_required: bool
    preserve_thin_features: bool
    separate_props: bool
    expected_upright_axis: str
    # Longest-axis-to-shortest ratio beyond which the generated mesh is treated as collapsed.
    max_axis_ratio: float
    # Fraction of the height above which debris is stripped when detached and small.
    debris_height_min: float
    uv_max_charts: int
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


PROFILES: dict[str, AssetProfile] = {
    "humanoid": AssetProfile(
        name="humanoid",
        lod_triangle_targets=(220_000, 80_000, 35_000, 12_000),
        texture_resolution=4096,
        rig_required=True,
        preserve_thin_features=False,
        separate_props=False,
        expected_upright_axis="y",
        max_axis_ratio=6.0,
        debris_height_min=0.70,
        uv_max_charts=1250,
    ),
    "humanoid_complex_accessories": AssetProfile(
        name="humanoid_complex_accessories",
        lod_triangle_targets=(220_000, 80_000, 35_000, 12_000),
        texture_resolution=4096,
        rig_required=True,
        preserve_thin_features=True,
        separate_props=True,
        expected_upright_axis="y",
        max_axis_ratio=6.0,
        debris_height_min=0.70,
        uv_max_charts=3000,
        notes="Hanging cords, pendants and a held prop. Thin features must survive decimation and "
              "detached ornaments must not be mistaken for debris.",
    ),
    "quadruped": AssetProfile(
        name="quadruped",
        lod_triangle_targets=(180_000, 70_000, 30_000, 10_000),
        texture_resolution=4096,
        rig_required=True,
        preserve_thin_features=False,
        separate_props=False,
        expected_upright_axis="y",
        max_axis_ratio=5.0,
        debris_height_min=0.80,
        uv_max_charts=1250,
    ),
    "flying_creature": AssetProfile(
        name="flying_creature",
        lod_triangle_targets=(180_000, 70_000, 30_000, 10_000),
        texture_resolution=4096,
        rig_required=True,
        preserve_thin_features=True,
        separate_props=False,
        expected_upright_axis="y",
        max_axis_ratio=8.0,
        debris_height_min=0.80,
        uv_max_charts=2000,
        notes="Wings and feathers are thin by design; a flatness check must not read them as collapse.",
    ),
    "static_prop": AssetProfile(
        name="static_prop",
        lod_triangle_targets=(120_000, 45_000, 18_000, 6_000),
        texture_resolution=2048,
        rig_required=False,
        preserve_thin_features=False,
        separate_props=False,
        expected_upright_axis="y",
        max_axis_ratio=12.0,
        debris_height_min=0.75,
        uv_max_charts=800,
    ),
    "vehicle": AssetProfile(
        name="vehicle",
        lod_triangle_targets=(200_000, 80_000, 30_000, 10_000),
        texture_resolution=4096,
        rig_required=False,
        preserve_thin_features=False,
        separate_props=True,
        expected_upright_axis="y",
        max_axis_ratio=10.0,
        debris_height_min=0.85,
        uv_max_charts=1500,
    ),
    "building": AssetProfile(
        name="building",
        lod_triangle_targets=(250_000, 90_000, 35_000, 12_000),
        texture_resolution=4096,
        rig_required=False,
        preserve_thin_features=False,
        separate_props=False,
        expected_upright_axis="y",
        max_axis_ratio=15.0,
        debris_height_min=0.90,
        uv_max_charts=2000,
    ),
    "environment_piece": AssetProfile(
        name="environment_piece",
        lod_triangle_targets=(150_000, 60_000, 25_000, 8_000),
        texture_resolution=2048,
        rig_required=False,
        preserve_thin_features=True,
        separate_props=False,
        expected_upright_axis="y",
        max_axis_ratio=20.0,
        debris_height_min=0.85,
        uv_max_charts=1500,
    ),
}

SAFEST_PROFILE = "humanoid_complex_accessories"


def foreground_mask(image) -> "object":
    """Boolean foreground mask from an image that may or may not carry alpha.

    Keys the background from the border rather than from a fixed luminance threshold. A threshold
    works only when the background happens to be near-white; on anything else it selects the whole
    frame, which is how profile detection came back with a bounding-box fill of 0.999 and picked the
    wrong profile for a subject made almost entirely of thin hanging parts.
    """
    import cv2
    import numpy as np

    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, 3] > 127

    colour = image[:, :, :3] if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    colour = colour.astype(np.float32)
    border = np.concatenate([colour[0], colour[-1], colour[:, 0], colour[:, -1]])
    background = np.median(border, axis=0)
    distance = np.linalg.norm(colour - background, axis=2)
    mask = (distance > 28.0).astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        # Keep the subject and anything of comparable scale (detached ornaments, held props),
        # but drop speckle so the thin-area measure is not dominated by matting noise.
        threshold = stats[largest, cv2.CC_STAT_AREA] * 0.002
        keep = np.zeros(count, bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= threshold
        mask = keep[labels]
    return mask.astype(bool)


@dataclass
class ProfileDetection:
    profile: str
    confidence: float
    evidence: dict = field(default_factory=dict)
    fell_back: bool = False


def detect_profile(mask) -> ProfileDetection:
    """Pick a profile from a boolean foreground mask of the source image.

    Uses three cheap, explainable signals: the subject's aspect ratio, how much of its bounding box
    it fills, and how much of its area sits in structures thin enough to be lost by decimation.
    """
    import numpy as np
    import cv2

    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return ProfileDetection(SAFEST_PROFILE, 0.0, {"error": "empty mask"}, fell_back=True)

    height = float(ys.max() - ys.min() + 1)
    width = float(xs.max() - xs.min() + 1)
    aspect = height / max(width, 1.0)
    fill = float(mask.sum()) / max(height * width, 1.0)

    # Erosion removes anything thinner than the kernel; what disappears was thin. The kernel is a
    # fraction of the subject's height rather than a fixed pixel count, so the measure means the
    # same thing at 512px and at 4096px - a fixed kernel makes the same asset look thin or solid
    # purely by how large the source image happens to be.
    kernel = max(3, int(round(height * 0.022)) | 1)
    solid = cv2.erode(mask.astype(np.uint8), np.ones((kernel, kernel), np.uint8), iterations=1)
    thin_fraction = 1.0 - float(solid.sum()) / max(float(mask.sum()), 1.0)

    evidence = {
        "aspect_height_over_width": round(aspect, 4),
        "bounding_box_fill": round(fill, 4),
        "thin_area_fraction": round(thin_fraction, 4),
        "erosion_kernel_px": kernel,
    }

    if aspect >= 1.15 and thin_fraction >= 0.28:
        return ProfileDetection("humanoid_complex_accessories", 0.75, evidence)
    if aspect >= 1.15 and fill >= 0.34:
        return ProfileDetection("humanoid", 0.70, evidence)
    if aspect < 0.85 and fill >= 0.40:
        return ProfileDetection("quadruped", 0.55, evidence)
    if aspect < 0.85 and thin_fraction >= 0.35:
        return ProfileDetection("flying_creature", 0.50, evidence)
    if 0.85 <= aspect < 1.15 and fill >= 0.5:
        return ProfileDetection("static_prop", 0.45, evidence)
    # Nothing matched with any confidence. Preserving too much is recoverable; destroying thin
    # geometry is not, so fall back to the profile that preserves the most.
    return ProfileDetection(SAFEST_PROFILE, 0.20, evidence, fell_back=True)


def resolve_profile(name: str) -> AssetProfile:
    key = (name or "").strip().lower()
    if key in ("", "auto"):
        raise ValueError("resolve_profile requires a concrete profile; run detect_profile first")
    if key not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; known: {', '.join(sorted(PROFILES))}")
    return PROFILES[key]
