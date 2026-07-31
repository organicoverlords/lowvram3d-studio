from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests


class StudioClient:
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def wait_ready(self, seconds: int = 90) -> None:
        deadline = time.time() + seconds
        last_error = "not started"
        while time.time() < deadline:
            try:
                response = requests.get(f"{self.base_url}/api/settings", timeout=3)
                if response.ok:
                    return
                last_error = f"HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = str(exc)
            time.sleep(2)
        raise RuntimeError(f"3D Gen Studio not ready: {last_error}")

    def get_settings(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/api/settings", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def update_settings(self, patch: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/api/settings", json=patch, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def create_project(self, name: str, project_type: str = "graph") -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/projects",
            json={"name": name, "type": project_type},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def import_asset(self, file_path: Path, asset_type: str, thumbnail: Path | None = None) -> dict[str, Any]:
        handles = []
        try:
            main = file_path.open("rb")
            handles.append(main)
            files: list[tuple[str, tuple[str, Any, str]]] = [
                ("files", (file_path.name, main, self._mime(file_path)))
            ]
            if thumbnail and thumbnail.is_file():
                thumb = thumbnail.open("rb")
                handles.append(thumb)
                files.append(("thumbnail:0", (thumbnail.name, thumb, self._mime(thumbnail))))
            response = requests.post(
                f"{self.base_url}/api/assets/library/import?assetType={asset_type}",
                files=files,
                timeout=max(self.timeout, 180),
            )
            response.raise_for_status()
            payload = response.json()
            imported = payload.get("imported", [])
            if not imported:
                raise RuntimeError(f"Studio imported no asset: {payload}")
            return imported[0]
        finally:
            for handle in handles:
                handle.close()

    def link_asset(self, project_id: int, asset_id: int, cascade: bool = True) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/projects/{project_id}/assets",
            json={"assetId": asset_id, "cascadeChildren": cascade},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def create_node(
        self,
        project_id: int,
        name: str,
        node_type: str,
        x: float,
        y: float,
        asset_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/graph/nodes",
            json={
                "projectId": project_id,
                "nodeTypeName": node_type,
                "name": name,
                "xPos": x,
                "yPos": y,
                "assetId": asset_id,
                "metadata": metadata or {},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def connect(self, project_id: int, source_id: int, target_id: int) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/api/graph/connections",
            json={"projectId": project_id, "sourceNodeId": source_id, "targetNodeId": target_id},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _mime(path: Path) -> str:
        return {
            ".glb": "model/gltf-binary",
            ".gltf": "model/gltf+json",
            ".fbx": "application/octet-stream",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".json": "application/json",
        }.get(path.suffix.lower(), "application/octet-stream")
