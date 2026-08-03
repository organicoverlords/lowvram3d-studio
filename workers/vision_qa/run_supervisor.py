#!/usr/bin/env python3
"""Run or validate a local OpenAI-compatible visual supervisor.

This worker is deliberately server-agnostic. It can target a local vLLM, SGLang,
llama.cpp, Ollama-compatible proxy, or other OpenAI-compatible endpoint. It has no
model dependency itself and defaults to dry-run. Network access is never required for
contract validation.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from lowvram3d.vision_qa.contracts import ContractError, ModelDecision, VisionQaPacket
from lowvram3d.vision_qa.policy import evaluate_decisions
from lowvram3d.vision_qa.prompting import SYSTEM_PROMPT, build_user_prompt, select_image_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="qwen3.5-2b-supervisor")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-image-bytes", type=int, default=4_000_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--decision", type=Path, help="Validate an existing decision instead of calling a model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        packet = VisionQaPacket.from_dict(json.loads(args.packet.read_text(encoding="utf-8")))
        if args.decision:
            raw_decision = json.loads(args.decision.read_text(encoding="utf-8"))
            decision = ModelDecision.from_dict(raw_decision)
            outcome = evaluate_decisions(packet, decision)
            return write_result(args.output, packet, decision, outcome.to_dict(), "VALIDATED_EXISTING_DECISION")

        payload = build_payload(packet, args)
        if args.dry_run:
            result = {
                "schema": "vision_qa_supervisor_run_v1",
                "classification": "DRY_RUN_NO_MODEL_CALLED",
                "packet_id": packet.packet_id,
                "model": args.model,
                "endpoint": args.endpoint,
                "payload": payload,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            return 0

        raw = post_json(args.endpoint, payload, args.timeout)
        decision_payload = extract_decision(raw)
        decision_payload["model_id"] = args.model
        decision = ModelDecision.from_dict(decision_payload)
        outcome = evaluate_decisions(packet, decision)
        return write_result(args.output, packet, decision, outcome.to_dict(), "MODEL_CALLED")
    except (ContractError, OSError, ValueError, KeyError, urllib.error.URLError) as exc:
        error = {
            "schema": "vision_qa_supervisor_run_v1",
            "classification": "FAILED_CLOSED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(error, indent=2), encoding="utf-8")
        print(json.dumps(error), file=sys.stderr)
        return 2


def build_payload(packet: VisionQaPacket, args: argparse.Namespace) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": build_user_prompt(packet)}]
    root = args.packet.parent
    for artifact in select_image_artifacts(packet.artifacts, max_images=args.max_images):
        path = Path(artifact.path)
        if not path.is_absolute():
            path = root / path
        if not path.is_file() or path.stat().st_size > args.max_image_bytes:
            continue
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{data}"},
        })
    return {
        "model": args.model,
        "temperature": 0.0,
        "max_tokens": 1400,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    }


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def extract_decision(response: dict[str, Any]) -> dict[str, Any]:
    text = response["choices"][0]["message"]["content"]
    if isinstance(text, list):
        text = "".join(str(item.get("text", "")) for item in text if isinstance(item, dict))
    text = str(text).strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("model response must be one JSON object")
    return data


def write_result(output: Path, packet: VisionQaPacket, decision: ModelDecision, outcome: dict, classification: str) -> int:
    result = {
        "schema": "vision_qa_supervisor_run_v1",
        "classification": classification,
        "packet_id": packet.packet_id,
        "decision": decision.to_dict(),
        "control_outcome": outcome,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if outcome["status"] not in {"INVALID_DECISION"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
