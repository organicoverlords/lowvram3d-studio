"""Pipeline V2: a generic, resumable, self-correcting asset pipeline.

Replaces per-asset orchestration scripts. Everything specific to an asset lives in a manifest;
everything specific to a *kind* of asset lives in a profile; the stages themselves are generic and
reuse the workers that are already proven.

Three properties drive the design, and all three came from watching the shaman run fail:

* **Fail closed.** A stage passes only when its gate says so. A gate that could not run - a detector
  that timed out, a report that was never written - is a failure, never a pass. The shaman shipped a
  UV atlas whose overlap detector timed out and reported zeroes, and those zeroes read as clean.
* **Never overwrite the last proven candidate.** Each stage writes into `candidate/` and is promoted
  to `proven/` only after its gate passes. A failed retry cannot destroy a good earlier result.
* **Bounded self-repair.** Each failure code maps to one repair recipe that adjusts parameters and
  re-runs the stage, at most twice. After that the stage stops and asks for a human. The alternative
  - retrying until something passes - is how a pipeline learns to produce technically valid rubbish.

Resume is by input hash: a stage whose receipt records the same input hashes and a passing verdict
is skipped entirely.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lowvram3d.asset_profiles import (  # noqa: E402
    PROFILES, detect_profile, foreground_mask, resolve_profile,
)

STAGES = [
    "INGEST", "GENERATE", "GEOMETRY_QA", "CLEAN", "LOD", "UV", "BAKE",
    "TEXTURE", "TEXTURE_QA", "PARTS", "RIG_READINESS", "RIG", "EXPORT",
]

# Failure code -> bounded repair recipe. Each entry names the stage to re-run and the parameter
# overrides to apply. Applied at most MAX_RETRIES times per stage, then the stage stops.
REPAIR_POLICY: dict[str, dict] = {
    "UV_ROW_ORIENTATION_MISMATCH": {
        "stage": "TEXTURE", "overrides": {"convert_atlas_orientation": True},
        "rationale": "atlas is stored in the projector's inverted row convention; convert it",
    },
    "FLAT_NEUTRAL_ATLAS_REGIONS": {
        "stage": "TEXTURE", "overrides": {"repaint_priors": True, "donor_neighbours": 16},
        "rationale": "component-local prior collapsed to one colour; repaint from nearest observed donors",
    },
    "UNFINISHED_SYNTHESIS": {
        "stage": "TEXTURE", "overrides": {"detail_strength": 0.9},
        "rationale": "synthesized regions lack surface detail; re-inject cavity/AO high frequencies",
    },
    "PLASTIC_ROUGHNESS": {
        "stage": "TEXTURE", "overrides": {"roughness_floor": 0.45, "metallic_max": 0.60},
        "rationale": "specular sheen washes out detail; raise the roughness floor and cut metallic",
    },
    "MATERIAL_ID_NOISE": {
        "stage": "BAKE", "overrides": {"weld_components": True},
        "rationale": "connectivity computed on split vertices; weld by position before component ids",
    },
    "FLOATING_DEBRIS": {
        "stage": "CLEAN", "overrides": {"strip_debris": True, "debris_height_min": 0.50},
        "rationale": "shards remain after stripping; widen the height band and strip again",
    },
    "UV_OVERLAP": {
        "stage": "UV", "overrides": {"route": "xatlas", "prune_in_place": True},
        "rationale": "re-unwrapping reshuffles charts; prune offending faces without re-unwrapping",
    },
    "UV_DEGENERATE": {
        "stage": "UV", "overrides": {"route": "xatlas", "prune_in_place": True},
        "rationale": "same prune pass removes sub-epsilon triangles without rescaling charts",
    },
    "CAMERA_LABEL_MISMATCH": {
        "stage": "TEXTURE_QA", "overrides": {"invert_camera_yaw": True},
        "rationale": "glTF -Z imports as Blender +Y; flip the yaw convention before re-rendering",
    },
    "BACKGROUND_CONTAMINATION": {
        "stage": "TEXTURE", "overrides": {"alpha_min": 0.6},
        "rationale": "matting leaked; require a stronger alpha before a texel may be projected",
    },
    "EMPTY_ACTIVE_POINT_SET": {
        "stage": "GENERATE", "overrides": {"steps": 2},
        "rationale": "one bounded step-count retry for an empty decoder surface",
    },
    "EXPECTED_REDUCTION_DIM_NON_ZERO": {
        "stage": "GENERATE", "overrides": {"steps": 2},
        "rationale": "one bounded step-count retry for an empty decoder surface",
    },
    "BAD_ORIENTATION": {
        "stage": "CLEAN", "overrides": {"reorient_upright": True},
        "rationale": "mesh is lying down or collapsed; re-run the upright reorientation",
    },
    "REAR_MIRRORS_FRONT": {
        "stage": "TEXTURE", "overrides": {"bar_mirrored_views": True},
        "rationale": "a mirrored view contributed real pixels; bar non-real views from projection",
    },
}
MAX_RETRIES = 2


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_inputs(paths) -> dict:
    result = {}
    for path in paths:
        p = Path(path)
        result[str(p)] = sha256(p) if p.is_file() else None
    return result


@dataclass
class StageResult:
    status: str                       # passed | failed | skipped
    outputs: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)
    failure_codes: list = field(default_factory=list)
    detail: str = ""


class Pipeline:
    def __init__(self, manifest: dict, root: Path, python: str, blender: str, verbose: bool = True):
        self.manifest = manifest
        self.root = root
        self.python = python
        self.blender = blender
        self.verbose = verbose
        self.state_dir = root / "state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.profile = resolve_profile(manifest["profile"])

    # -- infrastructure ---------------------------------------------------------------------
    def log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def stage_dir(self, stage: str) -> Path:
        path = self.state_dir / stage
        (path / "candidate").mkdir(parents=True, exist_ok=True)
        (path / "proven").mkdir(parents=True, exist_ok=True)
        return path

    def receipt_path(self, stage: str) -> Path:
        return self.state_dir / stage / "receipt.json"

    def read_receipt(self, stage: str) -> dict | None:
        path = self.receipt_path(stage)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def write_receipt(self, stage: str, payload: dict) -> None:
        self.receipt_path(stage).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run(self, command: list, cwd: Path | None = None, env_extra: dict | None = None,
            timeout: float | None = None) -> tuple[int, str]:
        env = dict(os.environ)
        # Keep numerical results and all quality gates unchanged while allowing native numerical
        # backends used by a child stage to use the available CPU.  The exact UV overlap census is
        # Python control flow and will not become magically parallel from these variables, but
        # Blender, image preparation, and any BLAS/OpenMP kernels can benefit on future attempts.
        # PIPELINE_CPU_THREADS is an opt-in override for machines where the scheduler should use a
        # different count; it never changes projection, packing, or acceptance semantics.
        cpu_threads = os.environ.get("PIPELINE_CPU_THREADS") or str(os.cpu_count() or 1)
        env.setdefault("OMP_NUM_THREADS", cpu_threads)
        env.setdefault("OPENBLAS_NUM_THREADS", cpu_threads)
        env.setdefault("MKL_NUM_THREADS", cpu_threads)
        env.setdefault("NUMEXPR_NUM_THREADS", cpu_threads)
        env.setdefault("BLIS_NUM_THREADS", cpu_threads)
        env.setdefault("OMP_DYNAMIC", "FALSE")
        env.setdefault("OMP_PROC_BIND", "spread")
        env.setdefault("OMP_PLACES", "cores")
        repo_paths = [
            str(REPO_ROOT / "blender"),
            str(REPO_ROOT / "src"),
            str(REPO_ROOT / "workers"),
            str(REPO_ROOT),
        ]
        existing_pythonpath = env.get("PYTHONPATH", "")
        # A caller that needs an extra import root (a generator with its own vendored package,
        # say) means "as well as", never "instead of": clobbering PYTHONPATH here would take the
        # repo's own workers off the path and fail on an import rather than on the asset.
        extra_pythonpath = (env_extra or {}).get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            repo_paths
            + ([extra_pythonpath] if extra_pythonpath else [])
            + ([existing_pythonpath] if existing_pythonpath else [])
        )
        if env_extra:
            env.update({k: v for k, v in env_extra.items() if k != "PYTHONPATH"})
        try:
            process = subprocess.run([str(c) for c in command], cwd=str(cwd or REPO_ROOT),
                                     env=env, capture_output=True, text=True,
                                     timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return 124, output + f"\nPROCESS_TIMEOUT_SECONDS={timeout}"
        output = (process.stdout or "") + (process.stderr or "")
        return process.returncode, output

    def promote(self, stage: str, files: dict) -> dict:
        """Copy candidate outputs into proven/ and return their hashes."""
        proven = self.stage_dir(stage) / "proven"
        hashes = {}
        for name, path in files.items():
            source = Path(path)
            if not source.exists():
                continue
            destination = proven / source.name
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            hashes[name] = {"path": str(destination), "sha256": sha256(destination),
                            "bytes": destination.stat().st_size}
        return hashes

    # -- stage driver -----------------------------------------------------------------------
    def execute(self, stage: str, inputs: list, runner) -> dict:
        previous = self.read_receipt(stage)
        current_hashes = hash_inputs(inputs)
        if previous and previous.get("status") == "passed" and previous.get("input_hashes") == current_hashes:
            self.log(f"[{stage}] skipped - inputs unchanged and previously passed")
            return previous

        overrides: dict = {}
        attempts = []
        for attempt in range(MAX_RETRIES + 1):
            started = time.time()
            result: StageResult = runner(overrides)
            record = {
                "attempt": attempt,
                "status": result.status,
                "overrides": dict(overrides),
                "gates": result.gates,
                "failure_codes": result.failure_codes,
                "detail": result.detail,
                "seconds": round(time.time() - started, 1),
            }
            attempts.append(record)
            if result.status == "passed":
                receipt = {
                    "stage": stage, "status": "passed",
                    "profile": self.profile.name,
                    "asset_id": self.manifest["asset_id"],
                    "input_hashes": current_hashes,
                    "outputs": self.promote(stage, result.outputs),
                    "gates": result.gates,
                    "attempts": attempts,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                self.write_receipt(stage, receipt)
                self.log(f"[{stage}] passed on attempt {attempt}")
                return receipt

            repairs = [REPAIR_POLICY[c] for c in result.failure_codes if c in REPAIR_POLICY]
            if stage == "GENERATE" and attempt >= 1:
                repairs = []
            if attempt >= MAX_RETRIES or not repairs:
                receipt = {
                    "stage": stage, "status": "failed",
                    "profile": self.profile.name,
                    "asset_id": self.manifest["asset_id"],
                    "input_hashes": current_hashes,
                    "gates": result.gates,
                    "failure_codes": result.failure_codes,
                    "attempts": attempts,
                    "needs_human": True,
                    "detail": result.detail or "no bounded repair recipe for these codes"
                              if not repairs else "retry budget exhausted",
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                self.write_receipt(stage, receipt)
                self.log(f"[{stage}] FAILED codes={result.failure_codes} - stopping, human needed")
                return receipt

            for repair in repairs:
                overrides.update(repair["overrides"])
            self.log(f"[{stage}] attempt {attempt} failed {result.failure_codes}; "
                     f"applying repair -> {overrides}")
        raise AssertionError("unreachable")


def build_manifest(image: Path, profile_name: str, output_root: Path, asset_id: str | None) -> dict:
    import cv2
    import numpy as np

    raw = cv2.imread(str(image), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"could not read source image {image}")
    mask = foreground_mask(raw)

    detection = None
    if (profile_name or "auto").lower() == "auto":
        detection = detect_profile(mask)
        profile_name = detection.profile

    profile = resolve_profile(profile_name)
    identifier = asset_id or image.stem.lower().replace(" ", "_")
    return {
        "manifest_version": 2,
        "asset_id": identifier,
        "source": {"path": str(image), "sha256": sha256(image), "bytes": image.stat().st_size},
        "profile": profile.name,
        "profile_detection": None if detection is None else {
            "confidence": detection.confidence, "evidence": detection.evidence,
            "fell_back": detection.fell_back,
        },
        "profile_settings": profile.to_dict(),
        "generator": "mini_turbo",
        "geometry": {
            "lod_triangle_targets": list(profile.lod_triangle_targets),
            "preserve_thin_features": profile.preserve_thin_features,
            "max_axis_ratio": profile.max_axis_ratio,
            "debris_height_min": profile.debris_height_min,
        },
        "texture": {"resolution": profile.texture_resolution, "uv_max_charts": profile.uv_max_charts},
        "rig": {"required": profile.rig_required, "separate_props": profile.separate_props},
        "output_root": str(output_root / identifier),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline V2 asset runner")
    parser.add_argument("--image", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--asset-id", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--blender", default=r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe")
    parser.add_argument("--from-stage", default="INGEST")
    parser.add_argument("--to-stage", default="TEXTURE_QA")
    parser.add_argument("--existing-master", default="", help="skip GENERATE and adopt this GLB")
    parser.add_argument("--write-manifest-only", action="store_true")
    args = parser.parse_args()

    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    else:
        if not args.image:
            raise SystemExit("--image or --manifest is required")
        output_root = Path(args.output_root or (REPO_ROOT / "pipeline-v2-runs"))
        manifest = build_manifest(Path(args.image), args.profile, output_root, args.asset_id or None)

    root = Path(manifest["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"MANIFEST {manifest_path}", flush=True)
    print(f"ASSET {manifest['asset_id']} profile={manifest['profile']}"
          + (f" (detected, confidence {manifest['profile_detection']['confidence']})"
             if manifest.get("profile_detection") else ""), flush=True)
    if args.write_manifest_only:
        return

    from pipeline_v2_stages import register_stages  # noqa: E402  local import keeps this file readable

    pipeline = Pipeline(manifest, root, args.python, args.blender)
    stages = register_stages(pipeline, manifest, existing_master=args.existing_master)

    start = STAGES.index(args.from_stage.upper())
    stop = STAGES.index(args.to_stage.upper())
    summary = {}
    for stage in STAGES[start: stop + 1]:
        if stage not in stages:
            summary[stage] = {"status": "not_implemented"}
            print(f"[{stage}] not implemented in this build - skipping", flush=True)
            continue
        receipt = stages[stage]()
        summary[stage] = {"status": receipt["status"],
                          "failure_codes": receipt.get("failure_codes", [])}
        if receipt["status"] == "failed":
            break

    (root / "pipeline_summary.json").write_text(json.dumps(
        {"asset_id": manifest["asset_id"], "profile": manifest["profile"], "stages": summary},
        indent=2), encoding="utf-8")
    failed = [s for s, v in summary.items() if v["status"] == "failed"]
    print(f"PIPELINE_RESULT asset={manifest['asset_id']} "
          f"{'FAILED at ' + failed[0] if failed else 'COMPLETED'} "
          f"stages={json.dumps(summary)}", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
