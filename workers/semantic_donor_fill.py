"""Lineage-safe low-frequency donor fill for unobserved atlas regions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from mesh_io import read_glb, triangle_components
from lowvram3d.texture_provenance import Lineage, load_npz, save_npz


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh", required=True); p.add_argument("--basecolor", required=True); p.add_argument("--coverage", required=True)
    p.add_argument("--triangle-provenance", required=True); p.add_argument("--target-triangles", required=True)
    p.add_argument("--output-atlas", required=True); p.add_argument("--output-provenance", required=True); p.add_argument("--report", required=True)
    args = p.parse_args()
    positions, _n, uv, tris = read_glb(Path(args.mesh)); positions=positions.astype(np.float64); uv=uv.astype(np.float64); tris=tris.astype(np.int64)
    base = cv2.cvtColor(cv2.imread(args.basecolor, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    cov = cv2.imread(args.coverage, cv2.IMREAD_GRAYSCALE); owner = np.full(cov.shape, -1, np.int32)
    px=uv*(cov.shape[0]-1)
    for tid, tri in enumerate(tris):
        a=px[tri]; x0,y0=np.maximum(np.floor(a.min(0)).astype(int),0); x1,y1=np.minimum(np.ceil(a.max(0)).astype(int),cov.shape[0]-1)
        if x1<x0 or y1<y0: continue
        xs,ys=np.meshgrid(np.arange(x0,x1+1),np.arange(y0,y1+1)); (ax,ay),(bx,by),(cx,cy)=a; den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(den)<1e-12: continue
        fx,fy=xs+.5,ys+.5; w0=((by-cy)*(fx-cx)+(cx-bx)*(fy-cy))/den; w1=((cy-ay)*(fx-cx)+(ax-cx)*(fy-cy))/den; w2=1-w0-w1; inside=(w0>=-1e-4)&(w1>=-1e-4)&(w2>=-1e-4); owner[ys[inside],xs[inside]]=tid
    prov=load_npz(args.triangle_provenance); target=np.load(args.target_triangles).astype(bool)
    component,_=triangle_components(positions,tris,4e-4); centroids=positions[tris].mean(1); e1=positions[tris[:,1]]-positions[tris[:,0]]; e2=positions[tris[:,2]]-positions[tris[:,0]]; normals=np.cross(e1,e2); normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
    observed=cov>=255; counts=np.zeros(len(tris),np.int64); sums=np.zeros((len(tris),3),np.float64); valid=(owner>=0)&observed; np.add.at(counts,owner[valid],1); np.add.at(sums,owner[valid],base[valid]); has=counts>0
    forbidden=np.uint16(Lineage.ORIGINAL_FACE|Lineage.FACE_REFINEMENT|Lineage.GENERATED_FRONT); donor_ok=has & ((prov["lineage"]&forbidden)==0)
    smooth=cv2.GaussianBlur(base,(0,0),10); centres=uv[tris].mean(1); ui=np.clip((centres[:,0]*(base.shape[1]-1)).astype(int),0,base.shape[1]-1); vi=np.clip((centres[:,1]*(base.shape[0]-1)).astype(int),0,base.shape[0]-1); colour=smooth[vi,ui]
    target_colour=np.zeros((len(tris),3),np.float32); donor_ids=np.full(len(tris),-1,np.int64); tiers=np.full(len(tris),"unresolved",dtype=object); safe=np.median(colour[donor_ok],axis=0) if donor_ok.any() else np.array([.28,.28,.28],np.float32)
    for tid in np.flatnonzero(target):
        ids=np.flatnonzero(donor_ok&(component==component[tid]));
        if not len(ids): ids=np.flatnonzero(donor_ok)
        if not len(ids): target_colour[tid]=safe; tiers[tid]="component_prior"; continue
        d,ix=cKDTree(centroids[ids]).query(centroids[tid],k=min(24,len(ids))); d=np.atleast_1d(d); ix=np.atleast_1d(ix); chosen=ids[ix]; dot=normals[chosen]@normals[tid]; ok=dot>=.15
        if not ok.any(): ok=np.ones(len(chosen),bool)
        chosen,d=chosen[ok],d[ok]; w=1/np.maximum(d,1e-6); w/=w.sum(); target_colour[tid]=(colour[chosen]*w[:,None]).sum(0); donor_ids[tid]=chosen[np.argmax(w)]; tiers[tid]="safe_donor"
    output=base.copy(); mask=(owner>=0)&np.isin(owner,np.flatnonzero(target)); output[mask]=target_colour[owner[mask]]; cv2.imwrite(args.output_atlas,cv2.cvtColor(np.clip(output*255,0,255).astype(np.uint8),cv2.COLOR_RGB2BGR))
    atlas=load_npz(args.triangle_provenance) if False else None
    report={"schema":"semantic_donor_fill_v1","target_triangles":int(target.sum()),"target_texels":int(mask.sum()),"forbidden_lineage":int(forbidden),"face_lineage_donors":0,"tiers":{str(k):int(v) for k,v in zip(*np.unique(tiers[target],return_counts=True))}}
    Path(args.report).parent.mkdir(parents=True,exist_ok=True); Path(args.report).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"SEMANTIC_DONOR_FILL target={target.sum()} texels={mask.sum()}",flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
