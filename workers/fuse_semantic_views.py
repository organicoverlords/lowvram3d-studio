"""Fuse accepted semantic views while preserving protected atlas bytes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from lowvram3d.texture_provenance import Lineage, load_npz, save_npz
from semantic_multiband_blend import blend


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--basecolor",required=True); p.add_argument("--view-manifest",required=True); p.add_argument("--protected-mask",required=True); p.add_argument("--atlas-provenance",required=True); p.add_argument("--output-atlas",required=True); p.add_argument("--output-provenance",required=True); p.add_argument("--report",required=True); args=p.parse_args()
    base=cv2.cvtColor(cv2.imread(args.basecolor,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB).astype(np.float32)/255.; protected=cv2.imread(args.protected_mask,cv2.IMREAD_GRAYSCALE)>0; prov=load_npz(args.atlas_provenance); manifest=json.loads(Path(args.view_manifest).read_text(encoding="utf-8")); contributors=[]; out=base.copy()
    for item in manifest.get("views",[]):
        if item.get("status")!="GENERATED_VIEW_ACCEPTED": continue
        image=cv2.cvtColor(cv2.imread(item["atlas"],cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB).astype(np.float32)/255.; mask=cv2.imread(item["mask"],cv2.IMREAD_GRAYSCALE)>0; out=blend(out,image,mask,protected); contributors.append(item.get("view_name",item.get("request_id","unknown")))
    cv2.imwrite(args.output_atlas,cv2.cvtColor((out*255).astype(np.uint8),cv2.COLOR_RGB2BGR)); Path(args.output_provenance).parent.mkdir(parents=True,exist_ok=True); save_npz(args.output_provenance,prov); report={"schema":"multiview_fusion_v1","accepted_views":contributors,"protected_unchanged":bool(np.array_equal(out[protected],base[protected])),"frequency_policy":"authoritative high frequency, narrow medium seam, low frequency leveling"}; Path(args.report).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(f"MULTIVIEW_FUSE accepted={len(contributors)}",flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
