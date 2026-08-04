"""Deterministic next-best semantic view planner."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--coverage-analysis",required=True); p.add_argument("--mesh-hash",required=True); p.add_argument("--uv-hash",required=True); p.add_argument("--provenance-hash",required=True); p.add_argument("--output",required=True); p.add_argument("--min-useful-coverage",type=float,default=5.0); args=p.parse_args()
    data=json.loads(Path(args.coverage_analysis).read_text(encoding="utf-8")); candidates=[]
    for row in data.get("views",[]):
        if row.get("status")!="ANALYZED": continue
        useful=float(row.get("expected_new_semantic_coverage_percent",0)); donor=float(row.get("donor_coverage_percent",0)); prior=float(row.get("component_prior_coverage_percent",0))+float(row.get("global_prior_coverage_percent",0)); overlap=float(row.get("protected_face_overlap",0)); risk=float(row.get("forbidden_face_lineage_exposure",0)); redundancy=float(row.get("original_source_coverage_percent",0)); score=(useful+0.7*donor+0.3*prior)-5*overlap-4*risk-1.5*redundancy; candidates.append((score,row))
    candidates.sort(key=lambda x:(-x[0],x[1]["view_name"])); out=Path(args.output)
    if not candidates or candidates[0][0] <= args.min_useful_coverage:
        out.write_text(json.dumps({"status":"NO_VALUABLE_MULTIVIEW_REQUEST","candidates":[]},indent=2),encoding="utf-8"); print("NO_VALUABLE_MULTIVIEW_REQUEST",flush=True); return 0
    score,row=candidates[0]; camera=row.get("camera",{"azimuth_degrees":180,"elevation_degrees":0,"orthographic_scale":0}); cam_hash=hashlib.sha256(json.dumps(camera,sort_keys=True).encode()).hexdigest(); request={"status":"VIEW_REQUEST_READY","request_id":f"shaman-{row['view_name']}-001","view_name":row["view_name"],"camera":camera,"camera_hash":cam_hash,"mesh_hash":args.mesh_hash,"uv_hash":args.uv_hash,"provenance_hash":args.provenance_hash,"target_semantic_class":f"GENERATED_{row['view_name'].upper().replace('-','_')}","expected_new_coverage_percent":float(row.get("expected_new_semantic_coverage_percent",0)),"planner_score":float(score),"protected_regions":["front_face"],"forbidden_content":["eyes","beak","front_face_plate","front_mask"],"conditioning_inputs":[]}; out.write_text(json.dumps(request,indent=2),encoding="utf-8"); print(f"VIEW_PLAN {request['view_name']} score={score:.4f}",flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
