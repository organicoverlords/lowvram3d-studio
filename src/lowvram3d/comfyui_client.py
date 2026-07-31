from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests


class ComfyUIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def health(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/system_stats", timeout=3).ok
        except requests.RequestException:
            return False

    def object_info(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/object_info", timeout=60)
        response.raise_for_status()
        return response.json()

    def upload_image(self, path: Path) -> str:
        with path.open("rb") as handle:
            response = requests.post(
                f"{self.base_url}/upload/image",
                files={"image": (path.name, handle, "application/octet-stream")},
                data={"type": "input", "overwrite": "true"},
                timeout=120,
            )
        response.raise_for_status()
        return response.json().get("name", path.name)

    def load_api_workflow(self, workflow_path: Path, replacements: dict[str, Any]) -> dict[str, Any]:
        workflow = json.loads(workflow_path.read_text(encoding="utf-8-sig"))
        if "nodes" in workflow and "links" in workflow:
            workflow = self.ui_to_api(workflow, self.object_info())
        return self._replace(copy.deepcopy(workflow), replacements)

    def run_api_workflow(
        self,
        workflow_path: Path,
        replacements: dict[str, Any],
        output_dir: Path,
        timeout_seconds: int = 1800,
    ) -> list[Path]:
        prompt = self.load_api_workflow(workflow_path, replacements)
        response = requests.post(
            f"{self.base_url}/prompt",
            json={"prompt": prompt, "client_id": self.client_id},
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("node_errors"):
            raise RuntimeError(f"ComfyUI rejected workflow: {body['node_errors']}")
        prompt_id = body["prompt_id"]
        deadline = time.time() + timeout_seconds
        history: dict[str, Any] | None = None
        while time.time() < deadline:
            poll = requests.get(f"{self.base_url}/history/{prompt_id}", timeout=30)
            poll.raise_for_status()
            payload = poll.json()
            if prompt_id in payload:
                history = payload[prompt_id]
                break
            time.sleep(2)
        if history is None:
            raise TimeoutError(f"ComfyUI workflow timed out: {prompt_id}")
        if history.get("status", {}).get("status_str") == "error":
            raise RuntimeError(f"ComfyUI execution failed: {history.get('status')}")
        outputs = self._collect_files(history, output_dir)
        if not outputs:
            raise RuntimeError("ComfyUI completed but returned no files")
        return outputs

    def ui_to_api(self, workflow: dict[str, Any], object_info: dict[str, Any]) -> dict[str, Any]:
        nodes = {int(node["id"]): node for node in workflow.get("nodes", [])}
        links = {int(link[0]): link for link in workflow.get("links", [])}
        target = self._shape_save_node(nodes)
        keep = self._ancestor_ids(target, nodes, links)
        api: dict[str, Any] = {}
        for node_id in sorted(keep):
            node = nodes[node_id]
            class_type = node["type"]
            schema = object_info.get(class_type)
            if not schema:
                raise RuntimeError(f"Installed ComfyUI has no node type: {class_type}")
            linked = {entry["name"]: entry.get("link") for entry in node.get("inputs", []) if entry.get("link") is not None}
            widget_values = iter(node.get("widgets_values", []))
            inputs: dict[str, Any] = {}
            ordered = list(schema.get("input", {}).get("required", {}).keys()) + list(schema.get("input", {}).get("optional", {}).keys())
            for name in ordered:
                if name in linked:
                    link = links[int(linked[name])]
                    inputs[name] = [str(link[1]), int(link[2])]
                    continue
                try:
                    inputs[name] = next(widget_values)
                except StopIteration:
                    pass
            if class_type == "LoadImage":
                inputs["image"] = "${INPUT_IMAGE}"
            if "Save 3D Mesh" in class_type:
                for key in list(inputs):
                    if "path" in key.lower() or "filename" in key.lower():
                        inputs[key] = "${OUTPUT_DIR}/mini_turbo_mesh.glb"
            api[str(node_id)] = {"class_type": class_type, "inputs": inputs, "_meta": {"title": node.get("title") or class_type}}
        return api

    @staticmethod
    def _shape_save_node(nodes: dict[int, dict[str, Any]]) -> int:
        candidates = []
        for node_id, node in nodes.items():
            if "Save 3D Mesh" not in str(node.get("type", "")):
                continue
            text = " ".join(str(v) for v in node.get("widgets_values", [])).lower()
            score = 10 if "shape" in text else 0
            candidates.append((score, node_id))
        if not candidates:
            raise RuntimeError("UI workflow contains no Save 3D Mesh node")
        return max(candidates)[1]

    @staticmethod
    def _ancestor_ids(target: int, nodes: dict[int, dict[str, Any]], links: dict[int, list[Any]]) -> set[int]:
        keep: set[int] = set()
        stack = [target]
        while stack:
            node_id = stack.pop()
            if node_id in keep:
                continue
            keep.add(node_id)
            for entry in nodes[node_id].get("inputs", []):
                link_id = entry.get("link")
                if link_id is not None and int(link_id) in links:
                    stack.append(int(links[int(link_id)][1]))
        return keep

    def _collect_files(self, history: dict[str, Any], output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: list[Path] = []
        for node_output in history.get("outputs", {}).values():
            for values in node_output.values():
                if not isinstance(values, list):
                    continue
                for item in values:
                    if isinstance(item, str):
                        source = Path(item)
                        if source.is_file():
                            target = output_dir / source.name
                            target.write_bytes(source.read_bytes())
                            saved.append(target)
                        continue
                    if not isinstance(item, dict) or not item.get("filename"):
                        continue
                    query = urlencode({
                        "filename": item["filename"],
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                    })
                    response = requests.get(f"{self.base_url}/view?{query}", timeout=180)
                    response.raise_for_status()
                    target = output_dir / Path(item["filename"]).name
                    target.write_bytes(response.content)
                    saved.append(target)
        return saved

    def _replace(self, value: Any, replacements: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {key: self._replace(item, replacements) for key, item in value.items()}
        if isinstance(value, list):
            return [self._replace(item, replacements) for item in value]
        if isinstance(value, str):
            result: Any = value
            for key, replacement in replacements.items():
                token = "${" + key + "}"
                if result == token:
                    return replacement
                result = result.replace(token, str(replacement))
            return result
        return value
