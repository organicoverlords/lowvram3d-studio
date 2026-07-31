from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from lowvram3d.comfyui_client import ComfyUIClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-url", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--input-image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--timeout", type=int, default=2400)
    args = parser.parse_args()

    client = ComfyUIClient(args.comfy_url)
    if not client.health():
        raise RuntimeError(f"ComfyUI is not reachable at {args.comfy_url}")
    image_name = client.upload_image(Path(args.input_image))
    output_dir = Path(args.output).parent / "comfy_outputs"
    files = client.run_api_workflow(
        Path(args.workflow),
        {
            "INPUT_IMAGE": image_name,
            "INPUT_IMAGE_PATH": str(Path(args.input_image).resolve()),
            "PROMPT": args.prompt,
            "OUTPUT_DIR": str(output_dir.resolve()),
        },
        output_dir,
        timeout_seconds=args.timeout,
    )
    mesh = next((path for path in files if path.suffix.lower() in {".glb", ".gltf", ".obj", ".ply"}), None)
    if mesh is None:
        raise RuntimeError(f"Mini Turbo workflow returned no mesh: {[str(path) for path in files]}")
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(mesh, target)


if __name__ == "__main__":
    main()
