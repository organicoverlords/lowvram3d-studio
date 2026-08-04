"""Materialize honest V3 provenance for an existing atlas without repainting it.

This adapter is intentionally conservative: legacy synthesized texels are marked as
component-prior only when the prior stage explicitly exposed that tier, and the report
keeps V3 completeness false when transitive donor ancestry is unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from lowvram3d.texture_provenance import Lineage, SourceClass, create_empty_atlas_provenance, create_empty_triangle_provenance, rasterize_triangle_lineage_to_atlas, save_npz, summarize_provenance
from mesh_io import read_glb


def owner_map(uv, tris, size):
    owner=np.full((size,size),-1,np.int32); px=uv*(size-1)
    for tid,tri in enumerate(tris):
        a=px[tri]; x0,y0=np.maximum(np.floor(a.min(0)).astype(int),0); x1,y1=np.minimum(np.ceil(a.max(0)).astype(int),size-1)
        if x1<x0 or y1<y0: continue
        xs,ys=np.meshgrid(np.arange(x0,x1+1),np.arange(y0,y1+1)); (ax,ay),(bx,by),(cx,cy)=a; den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(den)<1e-12: continue
        fx,fy=xs+.5,ys+.5; w0=((by-cy)*(fx-cx)+(cx-bx)*(fy-cy))/den; w1=((cy-ay)*(fx-cx)+(ax-cx)*(fy-cy))/den; w2=1-w0-w1; inside=(w0>=-1e-4)&(w1>=-1e-4)&(w2>=-1e-4); owner[ys[inside],xs[inside]]=tid
    return owner


def main():
    p=argparse.ArgumentParser(); p.add_argument("--mesh",required=True); p.add_argument("--basecolor",required=True); p.add_argument("--coverage",required=True); p.add_argument("--protected-mask",required=True); p.add_argument("--face-report",required=True); p.add_argument("--output-dir",required=True); args=p.parse_args()
    _pos,_n,uv,tris=read_glb(Path(args.mesh)); uv=uv.astype(np.float64); tris=tris.astype(np.int64); base=cv2.imread(args.basecolor,cv2.IMREAD_COLOR); cov=cv2.imread(args.coverage,cv2.IMREAD_GRAYSCALE); protected=cv2.imread(args.protected_mask,cv2.IMREAD_GRAYSCALE)>0; size=base.shape[0]; owner=owner_map(uv,tris,size); island=owner>=0; observed=island&(cov>=255); face_report=json.loads(Path(args.face_report).read_text(encoding="utf-8")); face_ids=np.asarray(face_report.get("selected_face_triangle_ids",[]),np.int64); tri=create_empty_triangle_provenance(len(tris)); tri["lineage"][face_ids]|=np.uint16(Lineage.ORIGINAL_FACE|Lineage.FACE_REFINEMENT); tri["source_class"][face_ids]=np.uint8(SourceClass.FACE_REFINEMENT); tri_obs=np.unique(owner[observed]); tri_obs=tri_obs[tri_obs>=0]; nonface=np.setdiff1d(tri_obs,face_ids); tri["lineage"][nonface]|=np.uint16(Lineage.ORIGINAL_NONFACE); tri["source_class"][nonface]=np.uint8(SourceClass.ORIGINAL_NONFACE); synth=np.unique(owner[island&~observed&~protected]); synth=synth[synth>=0]; tri["lineage"][synth]|=np.uint16(Lineage.COMPONENT_PRIOR); tri["source_class"][synth]=np.uint8(SourceClass.COMPONENT_PRIOR); atlas=rasterize_triangle_lineage_to_atlas(tri,owner); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); save_npz(out/"triangle_provenance.npz",tri); save_npz(out/"atlas_provenance.npz",atlas); summary={"schema":"texture_provenance_v3","triangle_summary":summarize_provenance(tri),"atlas_summary":summarize_provenance(atlas),"legacy_transitive_donor_ancestry_available":False,"TEXTURE_PROVENANCE_V3_PROVEN":False,"TRANSITIVE_DONOR_LINEAGE_PROVEN":False,"REAR_FACE_LINEAGE_ZERO_PROVEN":False,"FRONT_PROTECTED_TEXTURE_UNCHANGED":True,"reason":"existing one-view atlas does not retain donor ancestry through repaint/detail fill"}; (out/"texture_provenance_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8"); print("CURRENT_PROVENANCE_MATERIALIZED V3=WAITING",flush=True)


if __name__=="__main__": main()
