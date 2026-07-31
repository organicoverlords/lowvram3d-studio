from __future__ import annotations

import json
import re
import time
from pathlib import Path


def safe_key(project_id: str, card_id: str) -> str:
    value = f"{project_id}_{card_id}" or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


class ContextStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, project_id: str, card_id: str, data: dict) -> None:
        payload = {**data, "project_id": project_id, "card_id": card_id, "updated_at": time.time()}
        (self.root / f"{safe_key(project_id, card_id)}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def get_exact(self, project_id: str, card_id: str) -> dict:
        exact = self.root / f"{safe_key(project_id, card_id)}.json"
        if not exact.is_file():
            return {}
        try:
            return json.loads(exact.read_text(encoding="utf-8-sig"))
        except Exception:
            return {}

    def get(self, project_id: str, card_id: str) -> dict:
        exact_data = self.get_exact(project_id, card_id)
        if exact_data:
            return exact_data
        # 3D Gen Studio may use a new card id for texturing. Recover the latest
        # source image for the same project instead of losing appearance context.
        candidates: list[dict] = []
        for path in self.root.glob(f"{safe_key(project_id, '')}*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                if str(data.get("project_id", "")) == str(project_id):
                    candidates.append(data)
            except Exception:
                continue
        return max(candidates, key=lambda item: float(item.get("updated_at", 0)), default={})
