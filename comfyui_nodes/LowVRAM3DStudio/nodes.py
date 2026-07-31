from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path


class LowVRAM3DGameReady:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image_path": ("STRING", {"default": "C:/assets/concept.png"}),
            "prompt": ("STRING", {"multiline": True, "default": "game-ready detailed asset"}),
            "worker_url": ("STRING", {"default": "http://127.0.0.1:8400"}),
            "rig_kind": (["auto", "humanoid", "creature", "mechanical"],),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("final_glb_path", "receipt_path")
    FUNCTION = "run"
    CATEGORY = "LowVRAM3DStudio"

    def run(self, image_path: str, prompt: str, worker_url: str, rig_kind: str):
        source = Path(image_path)
        if not source.is_file():
            raise FileNotFoundError(source)
        payload = {
            "image_base64": base64.b64encode(source.read_bytes()).decode("ascii"),
            "image_filename": source.name,
            "image_mime_type": "image/png",
            "prompt": prompt,
            "name": source.stem,
            "project_id": "comfyui",
            "card_id": source.stem,
            "rig_kind": rig_kind,
        }
        request = urllib.request.Request(
            worker_url.rstrip("/") + "/v1/full",
            data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=14400) as response:
            data = response.read()
            receipt = response.headers.get("X-LowVRAM3D-Receipt", "")
        output = source.parent / f"{source.stem}_game_ready.glb"
        output.write_bytes(data)
        return (str(output), receipt)


NODE_CLASS_MAPPINGS = {"LowVRAM3DGameReady": LowVRAM3DGameReady}
NODE_DISPLAY_NAME_MAPPINGS = {"LowVRAM3DGameReady": "LowVRAM Picture → Game Ready 3D"}
