"""One entrypoint for both scene pipelines, with a shared result contract.

The two pipelines answer different questions and must be graded on different
tests. Photometric reproduces a specific photograph and matches the source view
*by construction*, so scoring it against that view proves nothing -- which is
exactly the mistake that let a flat textured shell pass as a scene. Structural
builds real geometry and will never match the source image pixel-wise, so
grading it that way is a category error in the other direction.

Both therefore emit `pipeline_result_v1` with an explicit `graded_on` list, so
comparisons stay honest and a metric that does not apply is absent rather than
quietly zero.

    py -3.12 -m lowvram3d.pipelines --list
    .../envs/image-world-moge/Scripts/python.exe -m lowvram3d.pipelines \\
        --pipeline photometric.splats --image in.png --scene-id barn
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

WORK_ROOT = Path(r"C:\AI\ScenePipelineSmoke")


@dataclass(frozen=True)
class Pipeline:
    name: str
    summary: str
    bet: str
    graded_on: tuple[str, ...]
    not_applicable: tuple[str, ...]
    run: Callable[..., dict[str, Any]] = field(repr=False, default=None)


def _photometric_mesh(image: Path, scene_id: str, work: Path,
                      max_triangles: int, **_: Any) -> dict[str, Any]:
    from .moge_reconstruct import reconstruct

    return reconstruct(image, work / f"{scene_id}_mesh.glb",
                       work / f"{scene_id}_mesh_receipt.json",
                       max_triangles=max_triangles)


def _photometric_splats(image: Path, scene_id: str, work: Path,
                        max_splats: int, **_: Any) -> dict[str, Any]:
    from .splat_export import export

    return export(image, work / f"{scene_id}.ply",
                  work / f"{scene_id}_splat_receipt.json",
                  max_splats=max_splats)


def _structural(image: Path, scene_id: str, work: Path, **_: Any) -> dict[str, Any]:
    """Structural is driven by the resumable stage runner, not a single call."""
    return {
        "classification": "REQUIRES_STAGE_RUNNER",
        "entrypoint": "lowvram3d.image_to_scene_pipeline",
        "reason": "structural composition is resumable and stage-accounted; run it "
                  "through the pipeline entrypoint rather than this dispatcher",
        "known_limitation": "semantic analysis is a stub, so every region collapses "
                            "to one visual_shell and the builders emit scaled cubes",
    }


PIPELINES: dict[str, Pipeline] = {
    "photometric.mesh": Pipeline(
        name="photometric.mesh",
        summary="MoGe depth mesh, depth-edge culled, UV textured",
        bet="reproduce this photograph in 3D",
        graded_on=("source_view_similarity", "offaxis_stability", "unobserved_fraction"),
        not_applicable=("navigable_fraction", "actor_semantic_variety"),
        run=_photometric_mesh,
    ),
    "photometric.splats": Pipeline(
        name="photometric.splats",
        summary="MoGe point map as INRIA 3DGS splats; no connectivity, no smearing",
        bet="reproduce this photograph in 3D",
        graded_on=("source_view_similarity", "offaxis_stability", "unobserved_fraction"),
        not_applicable=("navigable_fraction", "actor_semantic_variety"),
        run=_photometric_splats,
    ),
    "structural": Pipeline(
        name="structural",
        summary="semantic regions to real authored actors with collision and navmesh",
        bet="build a real scene the photograph describes",
        graded_on=("offaxis_stability", "navigable_fraction", "actor_semantic_variety"),
        not_applicable=("source_view_similarity", "unobserved_fraction"),
        run=_structural,
    ),
}


def run(pipeline_name: str, image: Path, scene_id: str,
        work_root: Path = WORK_ROOT, max_triangles: int = 1_500_000,
        max_splats: int = 2_000_000) -> dict[str, Any]:
    if pipeline_name not in PIPELINES:
        raise KeyError(f"unknown pipeline {pipeline_name!r}; "
                       f"choose from {sorted(PIPELINES)}")
    pipeline = PIPELINES[pipeline_name]
    work = work_root / scene_id
    work.mkdir(parents=True, exist_ok=True)

    detail = pipeline.run(image=image, scene_id=scene_id, work=work,
                          max_triangles=max_triangles, max_splats=max_splats)

    return {
        "schema_version": "pipeline_result_v1",
        "pipeline": pipeline.name,
        "bet": pipeline.bet,
        "summary": pipeline.summary,
        # Absent rather than zero: a metric that does not apply must not be
        # comparable to one that does.
        "graded_on": list(pipeline.graded_on),
        "not_applicable": list(pipeline.not_applicable),
        "scene_id": scene_id,
        "image": str(image),
        "work_dir": str(work),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "detail": detail,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--pipeline")
    parser.add_argument("--image")
    parser.add_argument("--scene-id")
    parser.add_argument("--work-root", default=str(WORK_ROOT))
    parser.add_argument("--max-triangles", type=int, default=1_500_000)
    parser.add_argument("--max-splats", type=int, default=2_000_000)
    parser.add_argument("--receipt")
    args = parser.parse_args(argv)

    if args.list:
        for pipeline in PIPELINES.values():
            print(f"{pipeline.name:22s} {pipeline.summary}")
            print(f"{'':22s} bet: {pipeline.bet}")
            print(f"{'':22s} graded on: {', '.join(pipeline.graded_on)}")
        return 0

    if not (args.pipeline and args.image and args.scene_id):
        parser.error("--pipeline, --image and --scene-id are required")

    result = run(args.pipeline, Path(args.image), args.scene_id,
                 Path(args.work_root), args.max_triangles, args.max_splats)
    if args.receipt:
        path = Path(args.receipt)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
