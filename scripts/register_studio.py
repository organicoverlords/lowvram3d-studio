from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def request_json(url: str, method: str = "GET", body: dict[str, Any] | None = None) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        raw = response.read()
        return json.loads(raw.decode("utf-8")) if raw else None


def wait(url: str, seconds: int = 120) -> None:
    deadline = time.time() + seconds
    last = ""
    while time.time() < deadline:
        try:
            request_json(url + "/api/settings")
            return
        except Exception as exc:
            last = str(exc)
            time.sleep(1)
    raise RuntimeError(f"3D Gen Studio did not become ready: {last}")


def provider(id_value: str, name: str, api_type: str, url: str, body: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": id_value,
        "name": name,
        "type": api_type,
        "url": url,
        "headers": {},
        "body": body,
    }


def register(studio_url: str, worker_url: str, comfy_path: str, comfy_url: str, create_project: bool) -> dict[str, Any]:
    wait(studio_url)
    current = request_json(studio_url + "/api/settings")
    custom = [entry for entry in current.get("apis", {}).get("custom", []) if not str(entry.get("id", "")).startswith("lowvram3d_")]
    custom.extend([
        provider(
            "lowvram3d_full", "LowVRAM One-Click — Auto Class", "mesh-generation", worker_url + "/v1/full",
            {
                "image_base64": "{imageBase64}", "image_mime_type": "{imageMimeType}",
                "image_filename": "{imageFilename}", "prompt": "{prompt}", "name": "{name}",
                "project_id": "{projectId}", "card_id": "{cardId}",
                "asset_type": "auto", "quality_preset": "gameplay",
                "separate_movable_parts": True, "texture_resolution": 2048,
                "lod_enabled": True, "remove_hidden_geometry": False,
                "experimental_semantic_split": False, "background_removal": True,
                "animation_preset": "dance", "resume_failed_job": True,
            },
        ),
        provider(
            "lowvram3d_generate", "LowVRAM Local Generate", "mesh-generation", worker_url + "/v1/generate",
            {
                "image_base64": "{imageBase64}", "image_mime_type": "{imageMimeType}",
                "image_filename": "{imageFilename}", "prompt": "{prompt}", "name": "{name}",
                "project_id": "{projectId}", "card_id": "{cardId}"
            },
        ),
        provider(
            "lowvram3d_texture", "LowVRAM Local Texture (3 lanes)", "mesh-texturing", worker_url + "/v1/texture",
            {
                "mesh_base64": "{meshBase64}", "mesh_mime_type": "{meshMimeType}",
                "mesh_filename": "{meshFilename}", "prompt": "{prompt}", "name": "{name}",
                "project_id": "{projectId}", "card_id": "{cardId}"
            },
        ),
        provider(
            "lowvram3d_rig", "LowVRAM Local Rig + Game Ready", "mesh-rigging", worker_url + "/v1/rig",
            {
                "mesh_base64": "{meshBase64}", "mesh_mime_type": "{meshMimeType}",
                "mesh_filename": "{meshFilename}", "prompt": "{prompt}", "name": "{name}",
                "project_id": "{projectId}", "card_id": "{cardId}", "rig_kind": "auto"
            },
        ),
    ])
    full_names = {
        "avatar": "Photoreal Human Avatar + Dance",
        "character": "Character",
        "creature": "Creature",
        "vehicle": "Vehicle",
        "prop": "Prop",
        "building": "Building",
        "room": "Room / Interior",
        "scene": "Scene / Environment",
        "level": "Level / World Chunk",
    }
    for asset_type, display in full_names.items():
        custom.append(
            provider(
                f"lowvram3d_full_{asset_type}",
                f"LowVRAM One-Click — {display}",
                "mesh-generation",
                worker_url + "/v1/full",
                {
                    "image_base64": "{imageBase64}",
                    "image_mime_type": "{imageMimeType}",
                    "image_filename": "{imageFilename}",
                    "prompt": "{prompt}",
                    "name": "{name}",
                    "project_id": "{projectId}",
                    "card_id": "{cardId}",
                    "asset_type": asset_type,
                    "quality_preset": "gameplay",
                    "separate_movable_parts": asset_type in {"vehicle", "prop", "building", "room"},
                    "texture_resolution": 2048,
                    "lod_enabled": True,
                    "remove_hidden_geometry": False,
                    "experimental_semantic_split": False,
                    "background_removal": asset_type not in {"building", "room", "scene", "level"},
                    "animation_preset": "dance" if asset_type == "avatar" else "auto",
                    "resume_failed_job": True,
                },
            )
        )

    postprocess_names = {
        "avatar": "Human Avatar",
        "character": "Character",
        "creature": "Creature",
        "vehicle": "Vehicle",
        "prop": "Prop",
        "building": "Building",
        "room": "Room / Interior",
        "scene": "Scene / Environment",
        "level": "Level / World Chunk",
    }
    for asset_type, display in postprocess_names.items():
        custom.append(
            provider(
                f"lowvram3d_post_{asset_type}",
                f"LowVRAM Post-Process — {display}",
                "mesh-texturing",
                worker_url + "/v1/postprocess",
                {
                    "mesh_base64": "{meshBase64}",
                    "mesh_mime_type": "{meshMimeType}",
                    "mesh_filename": "{meshFilename}",
                    "prompt": "{prompt}",
                    "name": "{name}",
                    "project_id": "{projectId}",
                    "card_id": "{cardId}",
                    "asset_type": asset_type,
                    "quality_preset": "gameplay",
                    "separate_movable_parts": asset_type in {"vehicle", "prop", "building", "room"},
                    "texture_resolution": 2048,
                    "lod_enabled": True,
                    "remove_hidden_geometry": False,
                    "experimental_semantic_split": False,
                    "animation_preset": "dance" if asset_type == "avatar" else "auto",
                    "resume_failed_job": True,
                },
            )
        )
    patch = {
        "initialSetupComplete": True,
        "apis": {
            "comfyui": {
                "path": comfy_path,
                "url": comfy_url.rsplit(":", 1)[0],
                "port": comfy_url.rsplit(":", 1)[1] if ":" in comfy_url.rsplit("/", 1)[-1] else "8188",
            },
            "meshtools": {"url": "http://127.0.0.1", "port": "8200", "autoStart": False},
            "rigtools": {"url": "http://127.0.0.1", "port": "8300", "autoStart": False},
            "custom": custom,
        },
        "mcp": {"enabled": True, "token": ""},
    }
    settings = request_json(studio_url + "/api/settings", "POST", patch)
    result: dict[str, Any] = {"providers": [p["id"] for p in custom if str(p.get("id", "")).startswith("lowvram3d_")]}
    if create_project:
        project = request_json(studio_url + "/api/projects", "POST", {
            "name": "LowVRAM Picture to Game Asset",
            "description": "Imported high-poly mesh → analyse → class-specific split → optimize → UV → PBR bake → validate",
            "type": "graph",
        })
        project_id = int(project["id"])
        specs = [
            ("Source High-Poly Mesh", "Mesh", 0, 0),
            ("Analyse + Split", "Mesh", 280, 0),
            ("Retopo + UV", "Mesh", 560, 0),
            ("PBR Bake + Validate", "Mesh", 840, 0),
        ]
        nodes = []
        for name, node_type, x, y in specs:
            nodes.append(request_json(studio_url + "/api/graph/nodes", "POST", {
                "projectId": project_id, "nodeTypeName": node_type, "name": name,
                "xPos": x, "yPos": y, "status": "idle", "progress": 0,
                "metadata": {"lowvram3d": True},
            }))
        for source, target in zip(nodes, nodes[1:]):
            request_json(studio_url + "/api/graph/connections", "POST", {
                "projectId": project_id, "sourceNodeId": source["id"], "targetNodeId": target["id"]
            })
        result.update({"project_id": project_id, "project_name": project.get("name"), "node_ids": [n["id"] for n in nodes]})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studio-url", default="http://127.0.0.1:8311")
    parser.add_argument("--worker-url", default="http://127.0.0.1:8400")
    parser.add_argument("--comfy-path", default="")
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--create-project", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    result = register(args.studio_url.rstrip("/"), args.worker_url.rstrip("/"), args.comfy_path, args.comfy_url, args.create_project)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
