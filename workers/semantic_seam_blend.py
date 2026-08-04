"""Localized low/medium-frequency seam blend with protected pixels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--basecolor",required=True); p.add_argument("--replacement",required=True); p.add_argument("--region-mask",required=True); p.add_argument("--protected-mask",required=True); p.add_argument("--output",required=True); p.add_argument("--report",required=True); args=p.parse_args()
    base=cv2.cvtColor(cv2.imread(args.basecolor,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB).astype(np.float32)/255.; replacement=cv2.cvtColor(cv2.imread(args.replacement,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB).astype(np.float32)/255.; region=cv2.imread(args.region_mask,cv2.IMREAD_GRAYSCALE)>0; protected=cv2.imread(args.protected_mask,cv2.IMREAD_GRAYSCALE)>0
    if base.shape!=replacement.shape or base.shape[:2]!=region.shape or protected.shape!=region.shape: raise RuntimeError("SEAM_BLEND_DIMENSION_MISMATCH")
    ring=cv2.dilate(region.astype(np.uint8),np.ones((5,5),np.uint8),iterations=1)>0; ring &= ~region; ring &= ~protected
    low_base=cv2.GaussianBlur(base,(0,0),8); low_rep=cv2.GaussianBlur(replacement,(0,0),8); medium_base=base-low_base; medium_rep=replacement-low_rep
    out=base.copy(); out[region]=low_rep[region]+medium_rep[region]*.35; alpha=np.zeros(region.shape,np.float32); alpha[ring]=.5; out[ring]=low_base[ring]*(1-alpha[ring,None])+low_rep[ring]*alpha[ring,None]+medium_base[ring]*.5
    out[protected]=base[protected]; cv2.imwrite(args.output,cv2.cvtColor(np.clip(out*255,0,255).astype(np.uint8),cv2.COLOR_RGB2BGR))
    report={"schema":"semantic_seam_blend_v1","region_texels":int(region.sum()),"ring_texels":int(ring.sum()),"protected_unchanged":bool(np.array_equal(out[protected],base[protected])),"global_blur":False}
    Path(args.report).parent.mkdir(parents=True,exist_ok=True); Path(args.report).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(f"SEAM_BLEND region={region.sum()} ring={ring.sum()}",flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
