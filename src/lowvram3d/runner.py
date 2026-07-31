from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import threading
import time
from pathlib import Path

import psutil

from .contracts import StageReceipt, now_ms
from .provenance import (
    PROVENANCE_SCHEMA,
    arm_artifact_provenance,
    artifact_provenance_is_valid,
    fingerprint_command_inputs,
    stage_command_fingerprint,
    write_artifact_provenance,
)


GPU_HEAVY_STAGE_TOKENS = ("subject_preprocess", "birefnet", "mini_turbo", "mv_adapter", "proxy_geometry", "texture_proxy", "triposr")
_GPU_STAGE_LOCK = threading.Lock()


class StageFailure(RuntimeError):
    def __init__(self, message: str, receipt: StageReceipt):
        super().__init__(message)
        self.receipt = receipt


def total_gpu_memory_used_mb() -> int | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return max(values) if values else None
    except Exception:
        return None


def total_gpu_memory_mb() -> int | None:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        return max(values) if values else None
    except Exception:
        return None


# Memory the card is allowed to keep in reserve before a stage is considered doomed.
GPU_RESERVE_MB = 256


def gpu_budget(ceiling_mb: int, baseline_mb: int | None) -> tuple[int, int | None]:
    """Return (delta_budget, hard_cap) for a stage's GPU accounting.

    nvidia-smi reports whole-card usage, and on Windows WDDM per-process attribution is
    unavailable (used_memory comes back as N/A). Enforcing a fixed absolute ceiling therefore
    charges this stage for the desktop, the browser and any other resident CUDA context: a
    ~1.1 GB baseline on a 6 GB card silently removes a fifth of the budget and kills stages
    for memory they never allocated. Instead, budget the stage's own growth over the baseline
    captured when it started, and keep an absolute stop just below the physical card size so a
    genuine runaway is still caught before the driver thrashes.
    """
    total = total_gpu_memory_mb()
    if not total:
        # No card information: fall back to the configured absolute ceiling.
        return ceiling_mb, None
    hard_cap = total - GPU_RESERVE_MB
    if baseline_mb is None:
        return ceiling_mb, hard_cap
    # The stage may grow into whatever the card actually has free. Deriving this from the
    # configured ceiling instead would charge the stage for the baseline a second time.
    return max(total - baseline_mb - GPU_RESERVE_MB, 512), hard_cap


def terminate_tree(process: subprocess.Popen[bytes] | subprocess.Popen[str]) -> None:
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        _, alive = psutil.wait_procs(children, timeout=3)
        for child in alive:
            child.kill()
        parent.terminate()
        try:
            parent.wait(timeout=3)
        except psutil.TimeoutExpired:
            parent.kill()
    except (psutil.NoSuchProcess, ProcessLookupError):
        return


def _json_artifact_is_valid(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        return True
    if payload.get("success") is False:
        return False
    status = str(payload.get("status", "")).lower()
    return status not in {"failed", "error", "invalid"}


def _image_artifact_is_valid(path: Path) -> bool:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        return width > 0 and height > 0


def _glb_artifact_is_valid(path: Path) -> bool:
    data = path.read_bytes()
    if len(data) < 20:
        return False
    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2 or declared_length != len(data):
        return False
    offset = 12
    chunk_count = 0
    saw_json = False
    while offset < len(data):
        if offset + 8 > len(data):
            return False
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        if chunk_length < 0 or offset + chunk_length > len(data):
            return False
        payload = data[offset : offset + chunk_length]
        offset += chunk_length
        chunk_count += 1
        if chunk_count == 1:
            if chunk_type != 0x4E4F534A:  # JSON
                return False
            try:
                document = json.loads(payload.rstrip(b" \x00").decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return False
            if not isinstance(document, dict) or document.get("asset", {}).get("version") != "2.0":
                return False
            saw_json = True
        elif chunk_type not in {0x004E4942}:  # BIN
            return False
    return offset == len(data) and saw_json


def artifact_is_valid(path: Path, *, check_provenance: bool = True) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    suffix = path.suffix.lower()
    try:
        if suffix in {".json", ".gltf"}:
            valid = _json_artifact_is_valid(path)
        elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            valid = _image_artifact_is_valid(path)
        elif suffix == ".glb":
            valid = _glb_artifact_is_valid(path)
        else:
            valid = True
        if not valid:
            return False
        return not check_provenance or artifact_provenance_is_valid(path)
    except (OSError, ValueError, json.JSONDecodeError, struct.error):
        return False


def ensure_artifacts(paths: dict[str, str], *, check_provenance: bool = True) -> list[str]:
    return [name for name, raw in paths.items() if not artifact_is_valid(Path(raw), check_provenance=check_provenance)]


def read_tail(path: Path, limit: int = 6000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return text[-limit:]
    except OSError:
        return ""


def classify_failure(text: str, violations: list[str], exit_code: int | None, missing: list[str]) -> str:
    value = text.lower()
    if any("gpu memory ceiling" in item.lower() for item in violations):
        return "vram_ceiling"
    if any("timeout" in item.lower() for item in violations):
        return "timeout"
    if "cuda out of memory" in value or "outofmemoryerror" in value or "cublas_status_alloc_failed" in value:
        return "cuda_oom"
    if any(token in value for token in ("no kernel image", "invalid device function", "cuda error 209", "sm_75 support")):
        return "cuda_unsupported"
    if any(token in value for token in ("modulenotfounderror", "no module named", "cannot import name", "could not find torch")):
        return "dependency_missing"
    if any(token in value for token in ("rpc failed", "early eof", "connection reset", "connection aborted", "temporary failure", "timed out")):
        return "network"
    if missing:
        return "missing_artifact"
    if exit_code not in (None, 0):
        return "nonzero_exit"
    return "unknown"


def _run_stage_impl(
    stage: str,
    command: list[str],
    cwd: Path,
    logs_dir: Path,
    required_artifacts: dict[str, str],
    vram_ceiling_mb: int,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> StageReceipt:
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / f"{stage}.stdout.log"
    stderr_path = logs_dir / f"{stage}.stderr.log"
    input_fingerprints = fingerprint_command_inputs(command, cwd, required_artifacts)
    command_fingerprint = stage_command_fingerprint(command, cwd, env, required_artifacts, input_fingerprints)
    receipt = StageReceipt(
        stage=stage,
        status="running",
        started_at=now_ms(),
        command=command,
        provenance_schema=PROVENANCE_SCHEMA,
        command_fingerprint=command_fingerprint,
        input_fingerprints=input_fingerprints,
    )
    receipt.notes.extend([
        f"stdout={stdout_path}",
        f"stderr={stderr_path}",
        f"provenance_schema={PROVENANCE_SCHEMA}",
    ])
    arm_artifact_provenance(required_artifacts)
    peak_vram = 0
    peak_ram = 0
    stop_monitor = threading.Event()
    violation: list[str] = []
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            process = subprocess.Popen(command, cwd=str(cwd), env=merged_env, stdout=stdout, stderr=stderr, text=True)
        except OSError as exc:
            receipt.finished_at = now_ms()
            receipt.status = "failed"
            receipt.failure_class = "process_launch"
            receipt.error = f"Could not start stage executable: {exc}"
            raise StageFailure(receipt.error, receipt) from exc

        baseline_vram = total_gpu_memory_used_mb()
        delta_budget, hard_cap = gpu_budget(vram_ceiling_mb, baseline_vram)
        if baseline_vram is not None:
            receipt.notes.append(
                f"gpu_baseline_mb={baseline_vram} delta_budget_mb={delta_budget} hard_cap_mb={hard_cap}"
            )

        def monitor() -> None:
            nonlocal peak_vram, peak_ram
            while not stop_monitor.is_set() and process.poll() is None:
                used = total_gpu_memory_used_mb()
                if used is not None:
                    peak_vram = max(peak_vram, used)
                    grown = used - baseline_vram if baseline_vram is not None else None
                    if grown is not None and grown > delta_budget:
                        violation.append(
                            f"GPU memory grew {grown} MB above the {baseline_vram} MB baseline, "
                            f"over this stage's {delta_budget} MB budget (total {used} MB)"
                        )
                        terminate_tree(process)
                        return
                    if hard_cap is not None and used > hard_cap:
                        violation.append(f"GPU memory nearly exhausted: {used} > {hard_cap} MB")
                        terminate_tree(process)
                        return
                    if hard_cap is None and used > vram_ceiling_mb:
                        violation.append(f"GPU memory ceiling exceeded: {used} > {vram_ceiling_mb} MB")
                        terminate_tree(process)
                        return
                try:
                    parent = psutil.Process(process.pid)
                    rss = parent.memory_info().rss + sum(
                        child.memory_info().rss for child in parent.children(recursive=True)
                    )
                    peak_ram = max(peak_ram, int(rss / 1024 / 1024))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                time.sleep(1)

        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            violation.append(f"Stage timeout after {timeout_seconds}s")
            terminate_tree(process)
            exit_code = -9
        finally:
            stop_monitor.set()
            watcher.join(timeout=3)

    receipt.finished_at = now_ms()
    receipt.exit_code = exit_code
    receipt.peak_vram_mb = peak_vram or None
    receipt.peak_ram_mb = peak_ram or None
    receipt.artifacts = required_artifacts
    missing = ensure_artifacts(required_artifacts, check_provenance=False)
    if violation or exit_code != 0 or missing:
        log_tail = "\n".join(part for part in (read_tail(stdout_path), read_tail(stderr_path)) if part)
        receipt.status = "failed"
        receipt.failure_class = classify_failure(log_tail, violation, exit_code, missing)
        pieces = violation + ([f"Exit code {exit_code}"] if exit_code else [])
        if missing:
            pieces.append("Missing artifacts: " + ", ".join(missing))
        receipt.error = "; ".join(pieces) or "Unknown stage failure"
        if log_tail:
            receipt.notes.append("log_tail=" + log_tail[-2000:])
        raise StageFailure(receipt.error, receipt)

    try:
        artifact_fingerprints, provenance_files = write_artifact_provenance(
            stage=stage,
            command=command,
            cwd=cwd,
            env=env,
            artifact_paths=required_artifacts,
            input_fingerprints=input_fingerprints,
            command_fingerprint=command_fingerprint,
        )
        receipt.artifact_fingerprints = artifact_fingerprints
        receipt.provenance_files = provenance_files
        invalid_provenance = ensure_artifacts(required_artifacts)
        if invalid_provenance:
            raise OSError("Provenance verification failed for: " + ", ".join(invalid_provenance))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        receipt.status = "failed"
        receipt.failure_class = "provenance_write"
        receipt.error = f"Could not seal stage artifacts with provenance: {exc}"
        raise StageFailure(receipt.error, receipt) from exc

    receipt.status = "passed"
    return receipt


def is_gpu_heavy_stage(stage: str) -> bool:
    value = stage.lower()
    return any(token in value for token in GPU_HEAVY_STAGE_TOKENS)


def run_stage(
    stage: str,
    command: list[str],
    cwd: Path,
    logs_dir: Path,
    required_artifacts: dict[str, str],
    vram_ceiling_mb: int,
    env: dict[str, str] | None = None,
    timeout_seconds: int | None = None,
) -> StageReceipt:
    if not is_gpu_heavy_stage(stage):
        return _run_stage_impl(
            stage, command, cwd, logs_dir, required_artifacts, vram_ceiling_mb, env, timeout_seconds
        )
    wait_started = now_ms()
    with _GPU_STAGE_LOCK:
        waited = now_ms() - wait_started
        try:
            receipt = _run_stage_impl(
                stage, command, cwd, logs_dir, required_artifacts, vram_ceiling_mb, env, timeout_seconds
            )
            receipt.notes.append(f"gpu_lock_wait_ms={waited}")
            return receipt
        except StageFailure as exc:
            exc.receipt.notes.append(f"gpu_lock_wait_ms={waited}")
            raise
