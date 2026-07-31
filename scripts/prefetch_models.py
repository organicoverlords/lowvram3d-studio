from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--skip", action="store_true")
    parser.add_argument("--include-triposr", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    # Windows only creates symlinks under Developer Mode or elevation. Without this the cache
    # raises WinError 1314 while placing an already-downloaded blob into the snapshot directory,
    # failing the model after its bytes are on disk. Copying costs disk, never a redownload.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if args.skip:
        print(json.dumps({"status": "skipped", "cache": str(cache), "models": []}, indent=2))
        return

    from huggingface_hub import snapshot_download

    required = [
        {"repo": "stabilityai/stable-diffusion-2-1-base", "patterns": None, "revision": None},
        {"repo": "huanngzh/mv-adapter", "patterns": ["mvadapter_tg2mv_sd21.safetensors", "*.json", "README*", "LICENSE*"], "revision": None},
        {
            "repo": "ZhengPeng7/BiRefNet",
            "patterns": ["*.safetensors", "*.json", "*.py", "README*", "LICENSE*", "requirements.txt"],
            "revision": "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
        },
    ]
    optional = [{"repo": "stabilityai/TripoSR", "patterns": None, "revision": None}] if args.include_triposr else []
    results: list[dict[str, object]] = []
    required_failed = False
    for spec in required + optional:
        repo = str(spec["repo"])
        patterns = spec.get("patterns")
        revision = spec.get("revision")
        is_required = spec in required
        try:
            path = snapshot_download(
                repo_id=repo,
                allow_patterns=patterns,
                revision=revision,
                local_files_only=args.verify_only,
            )
            results.append({
                "repo": repo,
                "required": is_required,
                "status": "verified" if args.verify_only else "downloaded",
                "path": path,
                "revision": revision,
            })
        except Exception as exc:
            results.append({"repo": repo, "required": is_required, "status": "failed", "error": str(exc)})
            required_failed = required_failed or is_required

    payload = {
        "status": "failed" if required_failed else "passed",
        "cache": str(cache),
        "models": results,
    }
    print(json.dumps(payload, indent=2))
    if required_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
