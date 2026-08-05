"""One-pass CPU chart-local low-frequency harmonization diagnostic."""
from __future__ import annotations
import argparse, io, json, math
from pathlib import Path
import numpy as np
from PIL import Image
from diagnostic_besgu_palette_reference_transfer import read_glb, geom, image_bytes, package, sha

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--current',type=Path,required=True); ap.add_argument('--provenance',type=Path,required=True); ap.add_argument('--out-dir',type=Path,required=True); args=ap.parse_args(); args.out_dir.mkdir(parents=True,exist_ok=True)
    raw,js,blob=read_glb(args.current); pos,uv,tri=geom(js,blob); atlas=np.asarray(Image.open(io.BytesIO(image_bytes(js,blob))).convert('RGB'),dtype=np.uint8); H,W=atlas.shape[:2]
    p=np.load(args.provenance); aid=np.asarray(p['triangle_id']); prot=np.asarray(p['protected_mask'],bool); direct=np.asarray(p['direct_observed_texel_mask'],bool); prot_tri=np.unique(aid[prot&(aid>=0)]); direct_tri=np.unique(aid[direct&(aid>=0)]); protect=np.zeros(len(tri),bool); direct_t=np.zeros(len(tri),bool); protect[prot_tri]=True; direct_t[direct_tri]=True
    # Five same-triangle samples provide a robust local low-frequency estimate.
    bary=np.array([[1/3,1/3,1/3],[.6,.2,.2],[.2,.6,.2],[.2,.2,.6],[.5,.25,.25]],np.float32); tuv=uv[tri].astype(np.float64); suv=np.einsum('sp,tpk->tsk',bary,tuv); sx=np.clip(np.rint(suv[...,0]*(W-1)).astype(np.int64),0,W-1); sy=np.clip(np.rint(suv[...,1]*(H-1)).astype(np.int64),0,H-1); tri_low=np.median(atlas[sy,sx].astype(np.float32),axis=1)
    # Conservative generated/invalid class from provenance; only these may use neutral bottom fallback.
    invalid_tri=np.zeros(len(tri),bool)
    for key in ('procedural_completion_mask','material_prior_mask','unresolved_mask'):
        if key in p: invalid_tri[np.unique(aid[np.asarray(p[key],bool)&(aid>=0)])]=True
    size=1024; out=np.full((size,size,3),96,np.uint8); own=np.full((size,size),-1,np.int32); px=uv[tri].astype(np.float64)*size; changed=0; fallback=0; occupied=0
    for tid,c in enumerate(px):
        lo=np.floor(c.min(0)).astype(int); hi=np.ceil(c.max(0)).astype(int)-1; x0=max(0,lo[0]); x1=min(size-1,hi[0]); y0=max(0,lo[1]); y1=min(size-1,hi[1]);
        if x0>x1 or y0>y1: continue
        a,b,d=c; ea=b-a; eb=d-a; den=ea[0]*eb[1]-ea[1]*eb[0]
        if abs(den)<=1e-12: continue
        gx,gy=np.meshgrid(np.arange(x0,x1+1)+.5,np.arange(y0,y1+1)+.5); wa=((gx-a[0])*eb[1]-(gy-a[1])*eb[0])/den; wb=(ea[0]*(gy-a[1])-ea[1]*(gx-a[0]))/den; reg=own[y0:y1+1,x0:x1+1]; free=(wa>=-1e-7)&(wb>=-1e-7)&(wa+wb<=1.0000001)&(reg<0)
        if not free.any(): continue
        reg[free]=tid; occupied+=int(free.sum()); yy,xx=np.nonzero(free); cy=np.clip(np.rint(((y0+yy+.5)/size)*(H-1)).astype(int),0,H-1); cx=np.clip(np.rint(((x0+xx+.5)/size)*(W-1)).astype(int),0,W-1); base=atlas[cy,cx].astype(np.float32); col=base.copy()
        if not protect[tid] and not direct_t[tid]:
            col=np.rint(.75*tri_low[tid][None,:]+.25*base).astype(np.float32); changed+=int(free.sum())
            lum=col.mean(1); chroma=col.max(1)-col.min(1); bad=invalid_tri[tid]&(lum>=170)&(chroma<=45)
            if bad.any(): col[bad]=96; fallback+=int(bad.sum())
        out[y0+yy,x0+xx]=np.clip(np.rint(col),0,255).astype(np.uint8)
    png=args.out_dir/'chart_local_harmonization_1024_atlas.png'; Image.fromarray(out).save(png); glb=args.out_dir/'chart_local_harmonization_canonical_mesh_diagnostic_only.glb'; package(raw,js,blob,png.read_bytes(),glb)
    _,ojs,ob=read_glb(glb); opos,ouv,otri=geom(ojs,ob); rec={'schema':'panda_chart_local_harmonization_diagnostic_only_v1','decision':'CHART_LOCAL_HARMONIZATION_DIAGNOSTIC_ONLY','input_glb':str(args.current),'input_glb_sha256':sha(raw),'output_glb':str(glb),'output_glb_sha256':sha(glb.read_bytes()),'output_atlas':str(png),'output_atlas_sha256':sha(png.read_bytes()),'canonical_triangle_count':int(len(tri)),'filter':'same-triangle five-sample median; mutable/non-direct blend 75% low-frequency + 25% original sampled RGB','cross_triangle_donors':False,'protected_triangles':int(protect.sum()),'direct_triangles':int(direct_t.sum()),'mutable_triangles':int((~protect&~direct_t).sum()),'atlas_size':1024,'occupied_pixels':int(occupied),'harmonized_pixels':int(changed),'invalid_bottom_neutral_fallback_pixels':int(fallback),'output_geometry_uv_index_unchanged':bool(np.array_equal(pos,opos) and np.array_equal(uv,ouv) and np.array_equal(tri,otri)),'sampler_material':'clean OPAQUE; mag=9729; min=9987; clamp-to-edge','gpu':False,'proof_source_neural_changed':False}
    (args.out_dir/'chart_local_harmonization_receipt.json').write_text(json.dumps(rec,indent=2),encoding='utf-8'); print(json.dumps(rec,indent=2))
if __name__=='__main__': main()
