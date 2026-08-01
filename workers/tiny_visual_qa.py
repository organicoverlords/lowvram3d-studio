"""A micro visual judge for localized before/after repair crops.

Scope is deliberately narrow. This reviews three small crops - the canonical source, the current
baseline and a candidate repair - and answers one multiple-choice question. It does not inspect
pipeline stages, does not drive Blender, and does not decide anything on its own: the verdict is
handed to `lowvram3d.visual_qa_policy`, which owns every interpretation and gating rule.

The model is never fetched implicitly. A production run must point at an already-installed local
model directory; downloading happens only under `scripts/install-tiny-visual-qa.ps1`, which passes
--allow-download explicitly.

Run it through `scripts/run-tiny-visual-qa.ps1` so the optional environment is used and the main
pipeline environments stay untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for candidate in (REPO_ROOT / "src", REPO_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from lowvram3d.visual_qa_policy import (  # noqa: E402
    MODE_OFF,
    MODES,
    VISUAL_MODEL_UNAVAILABLE,
    VISUAL_TIMEOUT,
    ManifestError,
    build_result,
    gate_outcome,
    parse_choice,
    unavailable_result,
    validate_manifest,
)

DEFAULT_MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"
# The contract allows up to 512 px. On CPU, three 512 px crops with Idefics3 sub-image splitting
# blow past the 20 s budget during prefill alone, so crops are capped lower and splitting is off.
# That keeps one comparison request comfortably inside the timeout.
MAX_CROP_PIXELS = 384
DISABLE_IMAGE_SPLITTING = True
MAX_NEW_TOKENS = 48
HARD_TIMEOUT_SECONDS = 20.0

QUESTION = """You are inspecting a single small repaired detail of a 3D character.

Image 1 is the original concept art.
Image 2 is the CURRENT model before the repair.
Image 3 is the CANDIDATE model after the repair.

Feature under review: {feature_name}
Intended result: {expected_description}
Constraints: {constraints}

Choose exactly one option and reply with only its letter.
A = the candidate faithfully preserves the intended feature
B = the candidate is visibly too large, generic, distorted, or damages the original design
C = the evidence is insufficient

ANSWER:"""


class JudgeUnavailable(RuntimeError):
    """The model could not be loaded or reached."""


def peak_memory() -> dict:
    """Peak process working set and, when CUDA is in use, peak VRAM. No extra dependencies."""
    report = {"peak_rss_mib": None, "peak_vram_mib": None}
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_info = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        if get_info is None:  # older Windows exposes it from psapi.dll
            get_info = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        get_info.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get_info.restype = wintypes.BOOL
        if get_info(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            report["peak_rss_mib"] = round(counters.PeakWorkingSetSize / 1048576.0, 1)
    except Exception:  # pragma: no cover - platform dependent
        pass
    try:
        import torch

        if torch.cuda.is_available():
            report["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 1048576.0, 1)
    except Exception:  # pragma: no cover - optional
        pass
    return report


class TinyVisualJudge:
    """Loads the VLM once and reuses it for many candidate checks."""

    def __init__(self, model_id: str = DEFAULT_MODEL, model_dir: str | None = None,
                 device: str = "auto", allow_download: bool = False) -> None:
        self.model_id = model_id
        self.model_dir = model_dir
        self.requested_device = device
        self.allow_download = allow_download
        self.device = "cpu"
        self.load_seconds = 0.0
        self._model = None
        self._processor = None
        self._torch = None
        self._letter_ids: dict[str, list[int]] = {}
        self._poisoned = False

    # ---------------------------------------------------------------- loading

    def _resolve_source(self) -> str:
        if self.model_dir:
            path = Path(self.model_dir)
            if not path.exists():
                raise JudgeUnavailable(f"model directory does not exist: {path}")
            return str(path)
        if not self.allow_download:
            raise JudgeUnavailable(
                "no local model directory was given and downloading is disabled; "
                "run scripts/install-tiny-visual-qa.ps1 first"
            )
        return self.model_id

    def load(self) -> None:
        if self._model is not None:
            return
        if not self.allow_download:
            # Belt and braces: even a cached-but-stale repo must not trigger a network fetch.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        started = time.monotonic()
        try:
            import torch
            import transformers
            from transformers import AutoProcessor

            # transformers >=5 renamed the vision2seq auto class.
            auto_model = getattr(transformers, "AutoModelForImageTextToText", None) or getattr(
                transformers, "AutoModelForVision2Seq", None
            )
            if auto_model is None:
                raise ImportError("no image-text-to-text auto class in transformers")
        except Exception as exc:  # pragma: no cover - environment dependent
            raise JudgeUnavailable(f"visual QA environment is not installed: {exc}") from exc

        source = self._resolve_source()
        wants_cuda = self.requested_device in ("auto", "cuda") and torch.cuda.is_available()
        self.device = "cuda" if wants_cuda else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        try:
            self._processor = AutoProcessor.from_pretrained(source)
            self._model = auto_model.from_pretrained(source, dtype=dtype)
            self._model.to(self.device)
            self._model.eval()
        except Exception as exc:
            raise JudgeUnavailable(f"could not load {source}: {exc}") from exc

        self._torch = torch
        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        for letter in ("A", "B", "C"):
            ids = set()
            for variant in (letter, f" {letter}"):
                encoded = tokenizer.encode(variant, add_special_tokens=False)
                if encoded:
                    ids.add(encoded[0])
            self._letter_ids[letter] = sorted(ids)
        self.load_seconds = time.monotonic() - started

    # ---------------------------------------------------------------- inference

    @staticmethod
    def _load_crop(path: str):
        from PIL import Image

        image = Image.open(path).convert("RGB")
        longest = max(image.size)
        if longest > MAX_CROP_PIXELS:
            scale = MAX_CROP_PIXELS / float(longest)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.LANCZOS,
            )
        return image

    def _build_prompt(self, manifest: dict) -> str:
        constraints = manifest.get("constraints") or []
        return QUESTION.format(
            feature_name=manifest["feature_name"],
            expected_description=manifest.get("expected_description", "(unspecified)"),
            constraints="; ".join(str(c) for c in constraints) if constraints else "(none)",
        )

    def _infer(self, manifest: dict, prompt: str) -> tuple[str, dict]:
        torch = self._torch
        images = [
            self._load_crop(manifest["source_crop"]),
            self._load_crop(manifest["before_crop"]),
            self._load_crop(manifest["candidate_crop"]),
        ]
        messages = [{
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}],
        }]
        text = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        processor_kwargs = {}
        if DISABLE_IMAGE_SPLITTING:
            processor_kwargs["do_image_splitting"] = False
        try:
            inputs = self._processor(text=text, images=images, return_tensors="pt",
                                     **processor_kwargs)
        except TypeError:
            inputs = self._processor(text=text, images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        trimmed = generated.sequences[0][inputs["input_ids"].shape[1]:]
        response = self._processor.decode(trimmed, skip_special_tokens=True).strip()

        # Confidence comes from the first generated token's distribution over A/B/C rather than
        # from the model narrating a number, which a 256M model does not do reliably.
        probabilities = {}
        if generated.scores:
            logits = generated.scores[0][0].float()
            softmax = torch.softmax(logits, dim=-1)
            for letter, ids in self._letter_ids.items():
                probabilities[letter] = float(max(softmax[i].item() for i in ids)) if ids else 0.0
        return response, probabilities

    def judge(self, manifest: dict, threshold_source: str = "logits"):
        """One structured comparison request, with a single retry on malformed output."""
        manifest = validate_manifest(manifest)
        if self._poisoned:
            return unavailable_result(self.model_id, VISUAL_TIMEOUT, device=self.device), {}
        self.load()
        prompt = self._build_prompt(manifest)

        attempts = []
        for attempt in range(2):
            box: dict = {}

            def run():
                try:
                    box["result"] = self._infer(manifest, prompt)
                except Exception as exc:  # pragma: no cover - runtime dependent
                    box["error"] = exc

            started = time.monotonic()
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            thread.join(HARD_TIMEOUT_SECONDS)
            elapsed = time.monotonic() - started

            if thread.is_alive():
                # The worker cannot be killed safely, so refuse to reuse this judge.
                self._poisoned = True
                return unavailable_result(
                    self.model_id, VISUAL_TIMEOUT, device=self.device,
                    load_seconds=self.load_seconds, inference_seconds=elapsed,
                ), {"prompt": prompt, "attempts": attempts, "device": self.device}

            if "error" in box:
                raise JudgeUnavailable(str(box["error"]))

            response, probabilities = box["result"]
            attempts.append({"response": response, "probabilities": probabilities,
                             "seconds": round(elapsed, 3)})
            choice = parse_choice(response)
            if choice is not None:
                total = sum(probabilities.values()) or 0.0
                confidence = (probabilities.get(choice, 0.0) / total) if total > 0 else 0.0
                result = build_result(
                    choice, confidence, self.model_id, raw_response=response, device=self.device,
                    load_seconds=self.load_seconds, inference_seconds=elapsed,
                )
                return result, {"prompt": prompt, "attempts": attempts,
                                "probabilities": probabilities, "device": self.device}

        # Both attempts malformed.
        result = build_result(
            None, 0.0, self.model_id,
            raw_response=attempts[-1]["response"] if attempts else "",
            load_seconds=self.load_seconds,
            inference_seconds=attempts[-1]["seconds"] if attempts else 0.0,
        )
        return result, {"prompt": prompt, "attempts": attempts, "device": self.device}


_JUDGE: TinyVisualJudge | None = None


def get_judge(**kwargs) -> TinyVisualJudge:
    """Module-level singleton so a batch of candidate checks pays the load cost once."""
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = TinyVisualJudge(**kwargs)
    return _JUDGE


def write_receipt(receipt_dir: Path, manifest: dict, result, extras: dict, outcome: dict) -> Path:
    """Persist the exact crops, prompt, raw response and parsed receipt."""
    receipt_dir.mkdir(parents=True, exist_ok=True)
    crops = receipt_dir / "crops"
    crops.mkdir(exist_ok=True)
    stored = {}
    for key in ("source_crop", "before_crop", "candidate_crop"):
        source = Path(manifest[key])
        if source.exists():
            target = crops / f"{key}{source.suffix or '.png'}"
            shutil.copy2(source, target)
            stored[key] = str(target)

    (receipt_dir / "prompt.txt").write_text(extras.get("prompt", ""), encoding="utf-8")
    raw = "\n\n".join(
        f"--- attempt {i + 1} ({a.get('seconds')}s) ---\n{a.get('response', '')}"
        for i, a in enumerate(extras.get("attempts", []))
    )
    (receipt_dir / "raw_response.txt").write_text(raw, encoding="utf-8")

    receipt = {
        **result.to_contract(),
        "choice": result.choice,
        "device": extras.get("device", "unknown"),
        "letter_probabilities": extras.get("probabilities", {}),
        "gate_outcome": outcome,
        "manifest": manifest,
        "stored_crops": stored,
    }
    path = receipt_dir / "visual_qa_receipt.json"
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny visual QA judge for localized repairs")
    parser.add_argument("--manifest", required=True, action="append",
                        help="path to an input manifest JSON (repeatable)")
    parser.add_argument("--receipt-dir", required=True)
    parser.add_argument("--mode", choices=MODES, default="auto")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-dir", default=os.environ.get("LOWVRAM3D_TINY_VQA_MODEL_DIR"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-download", action="store_true",
                        help="only used by the installer; never set this in a production run")
    parser.add_argument("--hard-gates-passed", choices=("true", "false"), default="true")
    parser.add_argument("--repeat", type=int, default=1,
                        help="run each manifest N times to measure output stability")
    args = parser.parse_args()

    if args.mode == MODE_OFF:
        print(json.dumps({"status": "unavailable", "mode": "off",
                          "reason": "visual QA disabled"}))
        return 0

    hard_gates = args.hard_gates_passed == "true"
    receipt_root = Path(args.receipt_dir)
    judge = get_judge(model_id=args.model, model_dir=args.model_dir,
                      device=args.device, allow_download=args.allow_download)

    exit_code = 0
    for index, manifest_path in enumerate(args.manifest):
        try:
            manifest = validate_manifest(json.loads(Path(manifest_path).read_text("utf-8")))
        except (ManifestError, json.JSONDecodeError, OSError) as exc:
            print(f"MANIFEST_INVALID {manifest_path}: {exc}", file=sys.stderr)
            return 2

        for repeat in range(max(1, args.repeat)):
            try:
                result, extras = judge.judge(manifest)
            except JudgeUnavailable as exc:
                print(f"JUDGE_UNAVAILABLE {exc}", file=sys.stderr)
                result = unavailable_result(args.model, VISUAL_MODEL_UNAVAILABLE)
                extras = {"prompt": "", "attempts": [], "device": "unknown"}

            outcome = gate_outcome(result, args.mode, hard_gates_passed=hard_gates)
            stem = Path(manifest_path).stem
            suffix = f"-run{repeat + 1}" if args.repeat > 1 else ""
            path = write_receipt(receipt_root / f"{stem}{suffix}", manifest,
                                 result, extras, outcome)

            contract = result.to_contract()
            print(json.dumps({
                "manifest": manifest_path,
                "run": repeat + 1,
                **contract,
                "device": extras.get("device", "unknown"),
                "promote": outcome["promote"],
                "blocking": outcome["blocking"],
                "receipt": str(path),
            }))
            if outcome["blocking"]:
                exit_code = 1

    print(json.dumps({"benchmark": {
        "model": args.model,
        "device": judge.device,
        "cold_load_seconds": round(judge.load_seconds, 3),
        **peak_memory(),
    }}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
