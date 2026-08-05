"""Backend-neutral V4 launcher; never silently falls back to donor synthesis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from lowvram3d.view_generation_contract import BACKENDS, ViewResult, write_result


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--request",required=True); p.add_argument("--backend",required=True,choices=sorted(BACKENDS)); p.add_argument("--output-dir",required=True); p.add_argument("--portable-request",default=""); args=p.parse_args(); request=json.loads(Path(args.request).read_text(encoding="utf-8")); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    # This repository intentionally does not bind V4 to a neural generator. Existing
    # backends can be plugged in later; a missing backend produces a portable request.
    portable=Path(args.portable_request or out/"portable_view_request.json"); portable.write_text(json.dumps({"schema":"portable_semantic_view_request_v1","request":request,"backend":args.backend,"status":"VIEW_BACKEND_NOT_AVAILABLE"},indent=2),encoding="utf-8")
    result={"request_id":request.get("request_id"),"backend":args.backend,"status":"VIEW_BACKEND_NOT_AVAILABLE","portable_request":str(portable),"source_class":request.get("target_semantic_class"),"camera_hash":request.get("camera_hash")}
    (out/"view_result.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print("VIEW_BACKEND_NOT_AVAILABLE",flush=True); return 2


if __name__=="__main__": raise SystemExit(main())
