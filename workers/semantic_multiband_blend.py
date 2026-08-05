"""Localized frequency-aware blend primitive."""
from __future__ import annotations

import argparse
import cv2
import numpy as np


def blend(base, incoming, mask, protected):
    low_a=cv2.GaussianBlur(base,(0,0),8); low_b=cv2.GaussianBlur(incoming,(0,0),8); med_a=base-low_a; med_b=incoming-low_b
    out=base.copy(); out[mask]=low_b[mask]+med_b[mask]*.65; ring=cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8))>0; ring&=~mask; ring&=~protected; alpha=.35; out[ring]=(low_a[ring]*(1-alpha)+low_b[ring]*alpha)+(med_a[ring]*(1-alpha)+med_b[ring]*alpha)*.5; out[protected]=base[protected]; return np.clip(out,0,1)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--base",required=True); p.add_argument("--incoming",required=True); p.add_argument("--mask",required=True); p.add_argument("--protected",required=True); p.add_argument("--output",required=True); args=p.parse_args()
    base=cv2.cvtColor(cv2.imread(args.base,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB).astype(np.float32)/255.; incoming=cv2.cvtColor(cv2.imread(args.incoming,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB).astype(np.float32)/255.; mask=cv2.imread(args.mask,cv2.IMREAD_GRAYSCALE)>0; protected=cv2.imread(args.protected,cv2.IMREAD_GRAYSCALE)>0; out=blend(base,incoming,mask,protected); cv2.imwrite(args.output,cv2.cvtColor((out*255).astype(np.uint8),cv2.COLOR_RGB2BGR)); print(f"MULTIBAND_BLEND mask={mask.sum()} protected_unchanged={np.array_equal(out[protected],base[protected])}",flush=True)


if __name__=="__main__": main()
