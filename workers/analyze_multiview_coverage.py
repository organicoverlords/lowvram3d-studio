"""Analyze how candidate cameras expose currently synthesized atlas surface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lowvram3d.texture_provenance import Lineage, load_npz


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--mesh",required=True); p.add_argument("--atlas-provenance",required=True); p.add_argument("--triangle-provenance",required=True); p.add_argument("--protected-mask",required=True); p.add_argument("--views",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--report",required=True); args=p.parse_args()
    atlas=load_npz(args.atlas_provenance); tri=load_npz(args.triangle_provenance); protected=cv2.imread(args.protected_mask,cv2.IMREAD_GRAYSCALE)>0; views=json.loads(Path(args.views).read_text(encoding="utf-8")); views=views.get("views",views) if isinstance(views,dict) else views; out=Path(args.output_dir); (out/"coverage_overlays").mkdir(parents=True,exist_ok=True)
    lineage=np.asarray(atlas["lineage"]); reports=[]
    for item in views:
        name=item.get("view_name",item.get("name")); evidence=item.get("evidence")
        if not evidence or not Path(evidence).exists():
            reports.append({"view_name":name,"status":"EVIDENCE_MISSING","expected_new_semantic_coverage_percent":0.0}); continue
        with np.load(evidence) as e:
            ids=e["triangle_id"]; visible=ids>=0; triangle_lineage=np.asarray(tri["lineage"]); visible_lineage=triangle_lineage[np.clip(ids,0,triangle_lineage.shape[0]-1)]
            forbidden=(visible_lineage&np.uint16(Lineage.ORIGINAL_FACE|Lineage.FACE_REFINEMENT))!=0; protected_overlap=visible & (ids>=0) & np.isin(ids,np.flatnonzero(tri["lineage"]&np.uint16(Lineage.ORIGINAL_FACE)!=0))
            prior=(visible_lineage&np.uint16(Lineage.COMPONENT_PRIOR|Lineage.GLOBAL_PRIOR))!=0; donor=(visible_lineage&np.uint16(Lineage.DONOR_TRANSFER))!=0; direct=(visible_lineage&np.uint16(Lineage.ORIGINAL_FACE|Lineage.ORIGINAL_NONFACE))!=0
            overlay=np.zeros((*ids.shape,3),np.uint8); overlay[direct]=(0,200,0); overlay[donor]=(0,220,220); overlay[prior&(visible_lineage&np.uint16(Lineage.GLOBAL_PRIOR)==0)]=(0,140,255); overlay[prior&(visible_lineage&np.uint16(Lineage.GLOBAL_PRIOR)!=0)]=(0,0,255); overlay[forbidden]=(180,0,180); cv2.imwrite(str(out/"coverage_overlays"/f"{name}.png"),cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
            reports.append({"view_name":name,"status":"ANALYZED","visible_triangles":int(np.unique(ids[visible]).size),"visible_pixels":int(visible.sum()),"original_source_coverage_percent":round(float(direct[visible].mean()*100),3),"donor_coverage_percent":round(float(donor[visible].mean()*100),3),"component_prior_coverage_percent":round(float((prior&(visible_lineage&np.uint16(Lineage.GLOBAL_PRIOR)==0))[visible].mean()*100),3),"global_prior_coverage_percent":round(float((visible_lineage&np.uint16(Lineage.GLOBAL_PRIOR)!=0)[visible].mean()*100),3),"forbidden_face_lineage_exposure":int(forbidden.sum()),"protected_face_overlap":int(protected_overlap.sum()),"expected_new_semantic_coverage_percent":round(float((prior[visible]).mean()*100),3)})
    np.savez_compressed(out/"coverage_analysis.npz", **{r["view_name"]:np.asarray([r.get("expected_new_semantic_coverage_percent",0.0)],np.float32) for r in reports})
    report={"schema":"coverage_analysis_v1","mesh":args.mesh,"views":reports,"status":"PROVEN" if reports else "NO_EVIDENCE"}; Path(args.report).parent.mkdir(parents=True,exist_ok=True); Path(args.report).write_text(json.dumps(report,indent=2),encoding="utf-8"); (out/"coverage_analysis.json").write_text(json.dumps(report,indent=2),encoding="utf-8"); print(f"COVERAGE_ANALYSIS views={len(reports)}",flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
