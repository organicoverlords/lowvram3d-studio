"""Run one bounded GPU texture-image job through a local ComfyUI API workflow.

This worker is intentionally model-agnostic.  The workflow must be exported from
ComfyUI with ``File -> Export (API)`` and a small binding JSON maps semantic
inputs (source/depth/normal/mask/prompt/view/resolution/seed/output prefix) to
node input fields.  It is suitable for source delighting, geometry-conditioned
side/rear generation, and masked inpainting without coupling the production
pipeline to one changing custom-node graph.

Safety properties for the GTX 1660 SUPER 6 GB lane:
- exactly one GPU job at a time through an exclusive lock;
- free-VRAM preflight and immutable attempt receipts;
- 512 first attempt, one lower-resolution retry only on OOM;
- timeout interrupt, queue deletion and model unload;
- no FlashAttention requirement and no geometry/UV mutation;
- all failed outputs remain diagnostic and are never promoted automatically.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OOM_MARKERS = (
    "out of memory",
    "cuda outofmemoryerror",
    "torch.cuda.outofmemoryerror",
    "allocation on device",
    "cudnn_status_alloc_failed",
)


@dataclass(frozen=True)
class Binding:
    node: str
    input: str


class ComfyError(RuntimeError):
    pass


class ComfyOOM(ComfyError):
    pass


class ComfyTimeout(ComfyError):
    pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def request_json(
    base_url: str,
    route: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{route}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ComfyError(f"HTTP_{exc.code}:{route}:{detail[-1500:]}") from exc
    except urllib.error.URLError as exc:
        raise ComfyError(f"COMFYUI_UNREACHABLE:{route}:{exc}") from exc
    if not payload:
        return {}
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ComfyError(f"INVALID_JSON:{route}:{payload[:300]!r}") from exc


def encode_multipart(fields: dict[str, str], file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = f"----lowvram3d-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for key, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode(),
                str(value).encode(),
                b"\r\n",
            ]
        )
    chunks.extend(
        [
            f"--{boundary}\r\n".encode(),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{path.name}"\r\n'
            ).encode(),
            b"Content-Type: application/octet-stream\r\n\r\n",
            path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def upload_image(base_url: str, path: Path, subfolder: str) -> str:
    if not path.is_file():
        raise ComfyError(f"INPUT_IMAGE_MISSING:{path}")
    body, content_type = encode_multipart(
        {"type": "input", "overwrite": "true", "subfolder": subfolder}, "image", path
    )
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/upload/image",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read())
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        raise ComfyError(f"IMAGE_UPLOAD_FAILED:{path}:{exc}") from exc
    name = result.get("name") or result.get("filename")
    returned_subfolder = result.get("subfolder", subfolder)
    if not name:
        raise ComfyError(f"IMAGE_UPLOAD_RESPONSE_INVALID:{result}")
    return f"{returned_subfolder}/{name}" if returned_subfolder else str(name)


def read_bindings(config: dict[str, Any]) -> dict[str, Binding]:
    result: dict[str, Binding] = {}
    raw = config.get("bindings", {})
    if not isinstance(raw, dict):
        raise ComfyError("BINDINGS_MUST_BE_OBJECT")
    for name, value in raw.items():
        if not isinstance(value, dict) or "node" not in value or "input" not in value:
            raise ComfyError(f"INVALID_BINDING:{name}")
        result[name] = Binding(node=str(value["node"]), input=str(value["input"]))
    return result


def apply_binding(workflow: dict[str, Any], binding: Binding, value: Any) -> None:
    node = workflow.get(binding.node)
    if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
        raise ComfyError(f"BOUND_NODE_MISSING:{binding.node}")
    if binding.input not in node["inputs"]:
        raise ComfyError(f"BOUND_INPUT_MISSING:{binding.node}.{binding.input}")
    node["inputs"][binding.input] = value


def nvidia_smi_memory() -> dict[str, int | str | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
        row = result.stdout.strip().splitlines()[0]
        name, total, free = [part.strip() for part in row.rsplit(",", 2)]
        return {"name": name, "total_mb": int(total), "free_mb": int(free)}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {"name": None, "total_mb": None, "free_mb": None}


def comfy_system_stats(base_url: str) -> dict[str, Any]:
    try:
        return request_json(base_url, "/system_stats", timeout=15)
    except ComfyError:
        return {}


def error_text(history_entry: dict[str, Any]) -> str:
    parts: list[str] = []
    status = history_entry.get("status")
    if status is not None:
        parts.append(json.dumps(status, ensure_ascii=False))
    for message in history_entry.get("messages", []) or []:
        parts.append(json.dumps(message, ensure_ascii=False))
    return "\n".join(parts)


def classify_error(text: str) -> type[ComfyError]:
    lowered = text.lower()
    return ComfyOOM if any(marker in lowered for marker in OOM_MARKERS) else ComfyError


def queue_prompt(base_url: str, workflow: dict[str, Any], client_id: str) -> str:
    result = request_json(
        base_url,
        "/prompt",
        method="POST",
        body={"prompt": workflow, "client_id": client_id},
        timeout=60,
    )
    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise ComfyError(f"PROMPT_REJECTED:{result}")
    return str(prompt_id)


def interrupt(base_url: str, prompt_id: str) -> None:
    for route, body in (
        ("/interrupt", {}),
        ("/queue", {"delete": [prompt_id]}),
    ):
        try:
            request_json(base_url, route, method="POST", body=body, timeout=15)
        except ComfyError:
            pass


def free_models(base_url: str) -> None:
    try:
        request_json(
            base_url,
            "/free",
            method="POST",
            body={"unload_models": True, "free_memory": True},
            timeout=60,
        )
    except ComfyError:
        pass


def wait_for_history(base_url: str, prompt_id: str, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        history = request_json(base_url, f"/history/{prompt_id}", timeout=30)
        entry = history.get(prompt_id)
        if isinstance(entry, dict):
            text = error_text(entry)
            status = entry.get("status", {})
            status_text = str(status).lower()
            if "error" in status_text or "execution_error" in text.lower():
                raise classify_error(text)(f"COMFY_EXECUTION_FAILED:{text[-3000:]}")
            if entry.get("outputs"):
                return entry
        time.sleep(1.0)
    interrupt(base_url, prompt_id)
    raise ComfyTimeout(f"COMFY_TIMEOUT:{prompt_id}:{timeout_seconds}s")


def download_outputs(
    base_url: str,
    history_entry: dict[str, Any],
    output_dir: Path,
    output_nodes: list[str] | None,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(output_nodes or [])
    written: list[Path] = []
    for node_id, node_output in (history_entry.get("outputs") or {}).items():
        if selected and str(node_id) not in selected:
            continue
        for image in node_output.get("images", []) or []:
            query = urllib.parse.urlencode(
                {
                    "filename": image["filename"],
                    "subfolder": image.get("subfolder", ""),
                    "type": image.get("type", "output"),
                }
            )
            request = urllib.request.Request(f"{base_url.rstrip('/')}/view?{query}")
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = response.read()
            suffix = Path(image["filename"]).suffix or ".png"
            destination = output_dir / f"node_{node_id}_{len(written):02d}{suffix}"
            destination.write_bytes(payload)
            if destination.stat().st_size <= 64:
                raise ComfyError(f"OUTPUT_TOO_SMALL:{destination}")
            written.append(destination)
    if not written:
        raise ComfyError("COMFY_OUTPUT_IMAGE_MISSING")
    return written


@contextmanager
def exclusive_gpu_lock(lock_path: Path, stale_seconds: int = 1800):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists() and time.time() - lock_path.stat().st_mtime > stale_seconds:
        lock_path.unlink(missing_ok=True)
    try:
        descriptor = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ComfyError(f"GPU_JOB_ALREADY_RUNNING:{lock_path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()} started={time.time()}\n".encode())
        os.close(descriptor)
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def build_attempt_workflow(
    workflow_template: dict[str, Any],
    bindings: dict[str, Binding],
    values: dict[str, Any],
) -> dict[str, Any]:
    workflow = copy.deepcopy(workflow_template)
    for name, value in values.items():
        binding = bindings.get(name)
        if binding is not None and value is not None:
            apply_binding(workflow, binding, value)
    return workflow


def run_attempt(
    *,
    base_url: str,
    workflow_template: dict[str, Any],
    bindings: dict[str, Binding],
    uploaded: dict[str, str],
    output_nodes: list[str] | None,
    output_dir: Path,
    view: str,
    prompt: str,
    negative_prompt: str,
    seed: int,
    mesh_path: str,
    view_index: int,
    resolution: int,
    timeout_seconds: float,
    minimum_free_mb: int,
) -> dict[str, Any]:
    memory_before = nvidia_smi_memory()
    free_mb = memory_before.get("free_mb")
    if isinstance(free_mb, int) and free_mb < minimum_free_mb:
        raise ComfyError(f"GPU_HEADROOM_INSUFFICIENT:{free_mb}<{minimum_free_mb}MB")
    values = {
        **uploaded,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "view": view,
        "view_name": view,
        "width": resolution,
        "height": resolution,
        "resolution": resolution,
        "seed": seed,
        "mesh": mesh_path,
        "view_index": view_index,
        "output_prefix": f"lowvram3d/{view}/{seed}",
    }
    workflow = build_attempt_workflow(workflow_template, bindings, values)
    client_id = str(uuid.uuid4())
    started = time.monotonic()
    prompt_id = queue_prompt(base_url, workflow, client_id)
    try:
        history = wait_for_history(base_url, prompt_id, timeout_seconds)
        outputs = download_outputs(base_url, history, output_dir, output_nodes)
    except Exception:
        interrupt(base_url, prompt_id)
        raise
    finally:
        free_models(base_url)
    return {
        "success": True,
        "prompt_id": prompt_id,
        "resolution": resolution,
        "duration_seconds": round(time.monotonic() - started, 3),
        "memory_before": memory_before,
        "memory_after": nvidia_smi_memory(),
        "outputs": [
            {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in outputs
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--depth", default="")
    parser.add_argument("--normal", default="")
    parser.add_argument("--mask", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mesh", default="")
    parser.add_argument("--view-index", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fallback-resolution", type=int, default=384)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--minimum-free-mb", type=int, default=1200)
    parser.add_argument("--report", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    workflow_path = Path(config["workflow_api"])
    if not workflow_path.is_absolute():
        workflow_path = (config_path.parent / workflow_path).resolve()
    workflow_template = json.loads(workflow_path.read_text(encoding="utf-8"))
    bindings = read_bindings(config)
    base_url = str(config.get("base_url", "http://127.0.0.1:8188"))
    output_nodes = [str(value) for value in config.get("output_nodes", [])] or None
    output_dir = Path(args.output_dir)
    report_path = Path(args.report) if args.report else output_dir / "gpu_job_receipt.json"
    lock_path = Path(config.get("gpu_lock", Path(tempfile.gettempdir()) / "lowvram3d-gpu.lock"))

    required_bindings = {"source", "width", "height", "seed"}
    missing = sorted(required_bindings - set(bindings))
    if missing:
        raise ComfyError(f"REQUIRED_BINDINGS_MISSING:{','.join(missing)}")
    if args.dry_run:
        build_attempt_workflow(
            workflow_template,
            bindings,
            {
                "source": "dry-run.png",
                "depth": "dry-depth.png" if args.depth else None,
                "normal": "dry-normal.png" if args.normal else None,
                "mask": "dry-mask.png" if args.mask else None,
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "view": args.view,
                "view_name": args.view,
                "width": args.resolution,
                "height": args.resolution,
                "resolution": args.resolution,
                "seed": args.seed,
                "output_prefix": "lowvram3d/dry-run",
            },
        )
        print(json.dumps({"success": True, "dry_run": True}, indent=2))
        return 0

    inputs = {
        "source": Path(args.source),
        "depth": Path(args.depth) if args.depth else None,
        "normal": Path(args.normal) if args.normal else None,
        "mask": Path(args.mask) if args.mask else None,
    }
    for name, path in inputs.items():
        if path is not None and not path.is_file():
            raise ComfyError(f"INPUT_MISSING:{name}:{path}")

    receipt: dict[str, Any] = {
        "schema": "lowvram3d_comfyui_gpu_texture_job_v1",
        "success": False,
        "view": args.view,
        "config": str(config_path),
        "workflow_api": str(workflow_path),
        "workflow_sha256": sha256(workflow_path),
        "input_sha256": {name: sha256(path) for name, path in inputs.items() if path is not None},
        "policy": {
            "one_gpu_job_at_a_time": True,
            "initial_resolution": args.resolution,
            "fallback_resolution": args.fallback_resolution,
            "oom_retries": 1,
            "timeout_seconds": args.timeout,
            "minimum_free_mb": args.minimum_free_mb,
            "unload_models_after_attempt": True,
            "geometry_or_uv_mutation": False,
        },
        "system_stats": comfy_system_stats(base_url),
        "attempts": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)

    with exclusive_gpu_lock(lock_path):
        subfolder = f"lowvram3d/{uuid.uuid4().hex}"
        uploaded = {
            name: upload_image(base_url, path, subfolder)
            for name, path in inputs.items()
            if path is not None
        }
        resolutions = [args.resolution]
        if args.fallback_resolution > 0 and args.fallback_resolution != args.resolution:
            resolutions.append(args.fallback_resolution)
        for attempt_index, resolution in enumerate(resolutions):
            attempt_dir = output_dir / f"attempt_{attempt_index + 1}_{resolution}"
            try:
                attempt = run_attempt(
                    base_url=base_url,
                    workflow_template=workflow_template,
                    bindings=bindings,
                    uploaded=uploaded,
                    output_nodes=output_nodes,
                    output_dir=attempt_dir,
                    view=args.view,
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    seed=args.seed,
                    mesh_path=args.mesh,
                    view_index=args.view_index,
                    resolution=resolution,
                    timeout_seconds=args.timeout,
                    minimum_free_mb=args.minimum_free_mb,
                )
                receipt["attempts"].append(attempt)
                receipt["success"] = True
                receipt["selected_attempt"] = attempt_index + 1
                break
            except ComfyOOM as exc:
                receipt["attempts"].append(
                    {"success": False, "resolution": resolution, "classification": "OOM", "error": str(exc)}
                )
                if attempt_index + 1 >= len(resolutions):
                    break
            except ComfyTimeout as exc:
                receipt["attempts"].append(
                    {"success": False, "resolution": resolution, "classification": "TIMEOUT", "error": str(exc)}
                )
                break
            except ComfyError as exc:
                receipt["attempts"].append(
                    {"success": False, "resolution": resolution, "classification": "ERROR", "error": str(exc)}
                )
                break

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2), flush=True)
    return 0 if receipt["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
