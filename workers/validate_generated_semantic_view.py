"""Validate a generated semantic view before it can enter fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--request",required=True); p.add_argument("--result",required=True); p.add_argument("--output-dir",required=True); p.add_argument("--report",required=True); args=p.parse_args(); req=json.loads(Path(args.request).read_text(encoding="utf-8")); res=json.loads(Path(args.result).read_text(encoding="utf-8")); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); reasons=[]
    if res.get("status")!="GENERATED_VIEW_ACCEPTED": reasons.append("GENERATED_VIEW_REJECTED_REGISTRATION")
    if res.get("camera_hash")!=req.get("camera_hash"): reasons.append("GENERATED_VIEW_REJECTED_CAMERA")
    image=cv2.imread(str(res.get("image","")),cv2.IMREAD_UNCHANGED) if res.get("image") else None
    if image is None: reasons.append("GENERATED_VIEW_REJECTED_BACKGROUND")
    else:
        alpha=image[:,:,3] if image.ndim==3 and image.shape[2]>=4 else np.full(image.shape[:2],255,np.uint8); foreground=float((alpha>16).mean()); contamination=float((alpha==0).mean());
        if foreground<0.01 or contamination<0.01: reasons.append("GENERATED_VIEW_REJECTED_BACKGROUND")
    status="GENERATED_VIEW_ACCEPTED" if not reasons else reasons[0]; report={"schema":"generated_view_validation_v1","status":status,"reasons":reasons,"request_id":req.get("request_id"),"camera_hash_match":res.get("camera_hash")==req.get("camera_hash"),"fusion_eligible":status=="GENERATED_VIEW_ACCEPTED"}; Path(args.report).parent.mkdir(parents=True,exist_ok=True); Path(args.report).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(status,flush=True); return 0 if status=="GENERATED_VIEW_ACCEPTED" else 2


if __name__=="__main__": raise SystemExit(main())
