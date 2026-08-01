"""Decision policy for the tiny visual-QA judge.

This module is deliberately free of torch, transformers and image libraries so the policy can be
tested in the ordinary environment. `workers/tiny_visual_qa.py` supplies the model output; every
interpretation of that output happens here.

The judge answers one multiple-choice question:

    A - candidate faithfully preserves the intended feature
    B - candidate is visibly too large, generic, distorted or damages the original design
    C - evidence is insufficient

Anything except a valid, high-confidence A is non-promotable. Visual acceptance is only ever an
additional hurdle: it can help promote a candidate, never rescue one that failed a hard gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

PROMOTION_CONFIDENCE = 0.80

MODE_OFF = "off"
MODE_ADVISORY = "advisory_only"
MODE_AUTO = "auto"
MODE_REQUIRED = "required"
MODES = (MODE_OFF, MODE_ADVISORY, MODE_AUTO, MODE_REQUIRED)

# The default. Benchmarked 2026-08-02: TINY_VISUAL_MODEL_DISCRIMINATION_NOT_PROVEN. Neither
# SmolVLM-256M nor SmolVLM-500M rejected the oversized staff repair, and the 256M model answered
# "insufficient evidence" even for an unrelated crop. Until a model demonstrably discriminates,
# its opinion is advisory: it may add weight to a rejection, never carry a promotion, and never
# block the base pipeline.
DEFAULT_MODE = MODE_ADVISORY

STATUS_PASSED = "passed"
STATUS_REJECTED = "rejected"
STATUS_UNCERTAIN = "uncertain"
STATUS_UNAVAILABLE = "unavailable"

DECISION_ACCEPT = "accept"
DECISION_REJECT = "reject"
DECISION_UNCERTAIN = "uncertain"

VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH = "VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH"
VISUAL_GENERIC_REPAIR = "VISUAL_GENERIC_REPAIR"
VISUAL_COLLATERAL_DAMAGE = "VISUAL_COLLATERAL_DAMAGE"
VISUAL_INSUFFICIENT_EVIDENCE = "VISUAL_INSUFFICIENT_EVIDENCE"
VISUAL_LOW_CONFIDENCE = "VISUAL_LOW_CONFIDENCE"
VISUAL_MALFORMED_OUTPUT = "VISUAL_MALFORMED_OUTPUT"
VISUAL_MODEL_UNAVAILABLE = "VISUAL_MODEL_UNAVAILABLE"
VISUAL_TIMEOUT = "VISUAL_TIMEOUT"
VISUAL_OK = "VISUAL_OK"

_GENERIC_HINTS = ("generic", "donut", "doughnut", "perfect circle", "machine", "uniform")
_OVERSIZE_HINTS = ("too large", "larger", "oversized", "bigger", "enlarged", "wider")
_DAMAGE_HINTS = ("damage", "damaged", "destroy", "destroyed", "missing", "broken", "erased")

REQUIRED_CHECK_KEYS = (
    "feature_matches_source",
    "original_shape_preserved",
    "collateral_damage_visible",
    "candidate_looks_generic_or_oversized",
)

MANIFEST_KEYS = (
    "source_crop",
    "before_crop",
    "candidate_crop",
    "feature_name",
    "expected_description",
    "constraints",
)


class ManifestError(ValueError):
    """Raised when an input manifest does not satisfy the documented contract."""


@dataclass
class VisualQAResult:
    status: str
    decision: str
    confidence: float
    checks: dict
    reason_codes: list
    model: str
    device: str = "unknown"
    load_seconds: float = 0.0
    inference_seconds: float = 0.0
    raw_response: str = ""
    choice: str = ""

    def to_contract(self) -> dict:
        """The documented output contract, in a stable key order."""
        return {
            "status": self.status,
            "decision": self.decision,
            "confidence": round(float(self.confidence), 4),
            "checks": {key: bool(self.checks.get(key, False)) for key in REQUIRED_CHECK_KEYS},
            "reason_codes": list(self.reason_codes),
            "model": self.model,
            "device": self.device,
            "load_seconds": round(float(self.load_seconds), 3),
            "inference_seconds": round(float(self.inference_seconds), 3),
        }


def validate_manifest(manifest: Any) -> dict:
    """Reject a manifest that does not carry every field the judge depends on."""
    if not isinstance(manifest, dict):
        raise ManifestError("manifest must be a JSON object")
    missing = [key for key in MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ManifestError(f"manifest is missing required keys: {', '.join(sorted(missing))}")
    for key in ("source_crop", "before_crop", "candidate_crop", "feature_name"):
        value = manifest[key]
        if not isinstance(value, str) or not value.strip():
            raise ManifestError(f"manifest key '{key}' must be a non-empty string")
    if not isinstance(manifest["constraints"], (list, tuple)):
        raise ManifestError("manifest key 'constraints' must be a list")
    return dict(manifest)


def parse_choice(text: str) -> str | None:
    """Extract the multiple-choice letter, or None when the output is malformed.

    A tiny model rambles, so accept an explicit `ANSWER: X`, a bare leading letter, or a first
    standalone A/B/C. Anything else is malformed and must not be guessed at.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    body = text.strip()

    labelled = re.search(r"\bANSWER\s*[:\-]?\s*\(?([ABC])\b", body, re.IGNORECASE)
    if labelled:
        return labelled.group(1).upper()

    leading = re.match(r"^\(?([ABC])\b", body, re.IGNORECASE)
    if leading:
        return leading.group(1).upper()

    standalone = re.search(r"(?<![A-Za-z])([ABC])(?![A-Za-z])", body)
    if standalone:
        return standalone.group(1).upper()
    return None


def _hits(text: str, hints) -> bool:
    lowered = (text or "").lower()
    return any(hint in lowered for hint in hints)


def build_result(
    choice: str | None,
    confidence: float,
    model: str,
    raw_response: str = "",
    device: str = "unknown",
    load_seconds: float = 0.0,
    inference_seconds: float = 0.0,
    threshold: float = PROMOTION_CONFIDENCE,
) -> VisualQAResult:
    """Turn a parsed model answer into the output contract."""
    confidence = max(0.0, min(1.0, float(confidence)))
    common = dict(
        model=model,
        device=device,
        raw_response=raw_response,
        load_seconds=load_seconds,
        inference_seconds=inference_seconds,
        choice=choice or "",
    )

    if choice is None:
        return VisualQAResult(
            status=STATUS_UNCERTAIN,
            decision=DECISION_UNCERTAIN,
            confidence=0.0,
            checks=_checks(False, False, False, False),
            reason_codes=[VISUAL_MALFORMED_OUTPUT],
            **common,
        )

    if choice == "B":
        collateral = _hits(raw_response, _DAMAGE_HINTS)
        codes = []
        if _hits(raw_response, _GENERIC_HINTS):
            codes.append(VISUAL_GENERIC_REPAIR)
        # Scale/shape mismatch is the default reading of B: it is the option's primary meaning,
        # so a B answer always carries it even when the model volunteers no wording.
        if not codes or _hits(raw_response, _OVERSIZE_HINTS):
            codes.insert(0, VISUAL_FEATURE_SCALE_OR_SHAPE_MISMATCH)
        if collateral:
            codes.append(VISUAL_COLLATERAL_DAMAGE)
        return VisualQAResult(
            status=STATUS_REJECTED,
            decision=DECISION_REJECT,
            confidence=confidence,
            checks=_checks(False, False, collateral, True),
            reason_codes=list(dict.fromkeys(codes)),
            **common,
        )

    if choice == "C":
        return VisualQAResult(
            status=STATUS_UNCERTAIN,
            decision=DECISION_UNCERTAIN,
            confidence=confidence,
            checks=_checks(False, False, False, False),
            reason_codes=[VISUAL_INSUFFICIENT_EVIDENCE],
            **common,
        )

    # choice == "A"
    if confidence < threshold:
        return VisualQAResult(
            status=STATUS_UNCERTAIN,
            decision=DECISION_UNCERTAIN,
            confidence=confidence,
            checks=_checks(True, True, False, False),
            reason_codes=[VISUAL_LOW_CONFIDENCE],
            **common,
        )
    return VisualQAResult(
        status=STATUS_PASSED,
        decision=DECISION_ACCEPT,
        confidence=confidence,
        checks=_checks(True, True, False, False),
        reason_codes=[VISUAL_OK],
        **common,
    )


def _checks(feature: bool, shape: bool, collateral: bool, generic: bool) -> dict:
    return {
        "feature_matches_source": feature,
        "original_shape_preserved": shape,
        "collateral_damage_visible": collateral,
        "candidate_looks_generic_or_oversized": generic,
    }


def unavailable_result(model: str, reason: str = VISUAL_MODEL_UNAVAILABLE,
                       device: str = "unknown",
                       load_seconds: float = 0.0,
                       inference_seconds: float = 0.0) -> VisualQAResult:
    """The judge could not run at all: no model, or it exceeded the hard timeout."""
    return VisualQAResult(
        status=STATUS_UNAVAILABLE,
        decision=DECISION_UNCERTAIN,
        confidence=0.0,
        checks=_checks(False, False, False, False),
        reason_codes=[reason],
        model=model,
        device=device,
        load_seconds=load_seconds,
        inference_seconds=inference_seconds,
    )


def gate_outcome(result: VisualQAResult | None, mode: str,
                 hard_gates_passed: bool = True) -> dict:
    """Combine the judge with the hard gates.

    Rules that do not bend:
      - visual acceptance never overrides a failed topology/hash/spatial-delta/export gate;
      - only ACCEPT at or above the confidence threshold can help promote;
      - in `auto`, an unavailable, timed-out or uncertain judge preserves the baseline and
        continues rather than failing the run;
      - in `required`, anything short of acceptance blocks.
    """
    if mode not in MODES:
        raise ValueError(f"unknown visual QA mode: {mode}")

    if mode == MODE_OFF or result is None:
        return {
            "mode": mode,
            "promote": bool(hard_gates_passed),
            "blocking": False,
            "reason": "visual QA disabled" if mode == MODE_OFF else "visual QA not run",
        }

    accepted = (
        result.status == STATUS_PASSED
        and result.decision == DECISION_ACCEPT
        and result.confidence >= PROMOTION_CONFIDENCE
    )

    if not hard_gates_passed:
        return {
            "mode": mode,
            "promote": False,
            "blocking": True,
            "reason": "hard gates failed; visual acceptance cannot override them",
        }

    if result.status == STATUS_REJECTED:
        # A rejection is always allowed to discard a candidate, in advisory mode too: the model
        # saying "this looks wrong" is only ever used to withhold promotion, never to grant it.
        return {
            "mode": mode,
            "promote": False,
            "blocking": True,
            "reason": "visual judge rejected the candidate: "
                      + ", ".join(result.reason_codes),
        }

    if accepted:
        if mode == MODE_ADVISORY:
            # Advisory acceptance is corroboration, not authority. It cannot be the only
            # positive evidence, so it never promotes on its own.
            return {
                "mode": mode,
                "promote": False,
                "blocking": False,
                "reason": "visual judge accepted, but advisory_only is never sufficient alone; "
                          "promotion needs the deterministic gate",
                "advisory_agreement": True,
            }
        return {"mode": mode, "promote": True, "blocking": False,
                "reason": "visual judge accepted the candidate"}

    # unavailable / uncertain / low confidence
    if mode in (MODE_AUTO, MODE_ADVISORY):
        return {
            "mode": mode,
            "promote": False,
            "blocking": False,
            "reason": "visual judge inconclusive; preserving the baseline and continuing",
        }
    return {
        "mode": mode,
        "promote": False,
        "blocking": True,
        "reason": "visual judge inconclusive and mode is 'required'",
    }
