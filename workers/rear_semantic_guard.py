"""Localized rear semantic provenance and face-like artifact gate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lowvram3d.texture_provenance import Lineage, load_npz, validate_no_forbidden_lineage


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a=cv2.resize(a,(128,128),interpolation=cv2.INTER_AREA).astype(np.float32); b=cv2.resize(b,(128,128),interpolation=cv2.INTER_AREA).astype(np.float32); a=(a-a.mean())/(a.std()+1e-6); b=(b-b.mean())/(b.std()+1e-6); return float(np.mean(a*b))


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--glb",required=True); p.add_argument("--basecolor",required=True); p.add_argument("--rear-visible-ids",required=True); p.add_argument("--rear-head-ids",required=True); p.add_argument("--atlas-provenance",required=True); p.add_argument("--source-face-crop",default=""); p.add_argument("--rear-crop",default=""); p.add_argument("--output-dir",required=True); p.add_argument("--report",required=True); args=p.parse_args()
    prov=load_npz(args.atlas_provenance); visible=np.load(args.rear_visible_ids); head=np.load(args.rear_head_ids).astype(np.int64); head_mask=np.isin(visible,head); flat=prov.get("triangle_id")
    if flat is not None:
        rear_atlas=np.isin(flat,head); forbidden=np.uint16(Lineage.ORIGINAL_FACE|Lineage.FACE_REFINEMENT|Lineage.GENERATED_FRONT); lineage_mask=rear_atlas&(prov["lineage"]&forbidden!=0)
    else:
        rear_atlas=np.zeros_like(prov["lineage"],bool); lineage_mask=rear_atlas
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    cv2.imwrite(str(out/"rear_head_atlas_mask.png"),(rear_atlas.astype(np.uint8)*255)); cv2.imwrite(str(out/"rear_lineage_overlay.png"),(lineage_mask.astype(np.uint8)*255)); cv2.imwrite(str(out/"rear_source_class_overlay.png"),prov.get("source_class",np.zeros_like(prov["lineage"],np.uint8)))
    similarity=None
    if args.source_face_crop and args.rear_crop and Path(args.source_face_crop).exists() and Path(args.rear_crop).exists(): similarity=correlation(cv2.imread(args.source_face_crop,cv2.IMREAD_GRAYSCALE),cv2.imread(args.rear_crop,cv2.IMREAD_GRAYSCALE))
    lineage_ok=not bool(lineage_mask.any()); visual_block=similarity is not None and similarity>0.75
    report={"schema":"rear_semantic_guard_v1","blocking_code":"REAR_FACE_LIKE_ARTIFACT" if (not lineage_ok or visual_block) else None,"back_visible_face_lineage_texels":int(lineage_mask.sum()),"rear_head_triangle_count":int(len(head)),"localized_front_correlation":similarity,"manual_full_resolution_review_required":True,"passed_lineage":lineage_ok,"passed_heuristic":not visual_block,"decision":"REJECTED" if (not lineage_ok or visual_block) else "PENDING_MANUAL_APPROVAL","glb":args.glb}
    Path(args.report).parent.mkdir(parents=True,exist_ok=True); Path(args.report).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(f"REAR_GUARD lineage={lineage_mask.sum()} similarity={similarity} decision={report['decision']}",flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
