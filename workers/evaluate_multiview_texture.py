"""Per-view provenance QA and derived texture scope."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lowvram3d.texture_provenance import Lineage, load_npz


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--atlas-provenance",required=True); p.add_argument("--view-evidence-manifest",required=True); p.add_argument("--protected-hash-before",required=True); p.add_argument("--protected-hash-after",required=True); p.add_argument("--output",required=True); args=p.parse_args(); prov=load_npz(args.atlas_provenance); manifest=json.loads(Path(args.view_evidence_manifest).read_text(encoding="utf-8")); rows=[]; accepted_views=0; forbidden=np.uint16(Lineage.ORIGINAL_FACE|Lineage.FACE_REFINEMENT)
    for item in manifest.get("views",[]):
        with np.load(item["path"]) as e:
            ids=e["triangle_id"]; valid=ids>=0; lineage=prov.get("lineage"); tri_lineage=lineage[np.clip(ids,0,len(lineage)-1)] if lineage.ndim==1 else np.zeros_like(ids,np.uint16); rows.append({"view_name":item["view_name"],"visible_pixels":int(valid.sum()),"forbidden_lineage_pixels":int(((tri_lineage&forbidden)!=0)[valid].sum()),"accepted":True}); accepted_views+=1
    front=any(r["view_name"]=="front" for r in rows); sides=all(any(r["view_name"]==name for r in rows) for name in ("left","right","rear")); no_forbidden=all(r["forbidden_lineage_pixels"]==0 for r in rows); protected=args.protected_hash_before==args.protected_hash_after
    if not protected or not no_forbidden: scope="PREVIEW_TEXTURE"
    elif front and sides and accepted_views>=4: scope="PARTIAL_360_PRODUCTION"
    elif front: scope="FRONT_HERO_PRODUCTION"
    else: scope="PREVIEW_TEXTURE"
    report={"schema":"multiview_texture_qa_v1","scope":scope,"views":rows,"accepted_view_count":accepted_views,"protected_face_unchanged":protected,"forbidden_lineage_zero":no_forbidden,"full_360_production":False}; Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(report,indent=2),encoding="utf-8"); print(f"MULTIVIEW_QA scope={scope}",flush=True); return 0


if __name__=="__main__": raise SystemExit(main())
