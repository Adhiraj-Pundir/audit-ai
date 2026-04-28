import hashlib
import json
import os
import re
from pathlib import Path

from .inference import get_logits
from .mitigation import redact_surname

SURNAME_PATTERNS = [
    r"Credit profile of (?:Mr\.|Ms\.|Mrs\.)\s+\w+\s+(\w+)",
    r"(?:Mr\.|Ms\.|Mrs\.)\s+\w+\s+(\w+)",
]

REFERENCE_SURNAMES = {
    "dalit": "Khatik",
    "muslim": "Rahman",
    "upper_caste": "Sharma",
}

_CACHE_PATH = Path(__file__).parent / "data" / "cached_scores.json"
_score_cache: dict | None = None


def _load_cache() -> dict:
    global _score_cache
    if _score_cache is None:
        if _CACHE_PATH.exists():
            with open(_CACHE_PATH) as f:
                _score_cache = json.load(f)
        else:
            _score_cache = {}
    return _score_cache


def _cache_key(text: str, mitigation: str) -> str:
    return hashlib.sha256(f"{mitigation}:{text}".encode()).hexdigest()[:16]


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_surname(text: str) -> str | None:
    for pat in SURNAME_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def _replace_surname(text: str, old: str, new: str) -> str:
    return re.sub(r"\b" + re.escape(old) + r"\b", new, text)


# ── Main scoring logic ─────────────────────────────────────────────────────────

def score_application(application_text: str, mitigation: str = "none") -> dict:
    # CACHE_ONLY is read per-request so `gcloud run services update --update-env-vars
    # CACHE_ONLY=1` takes effect on the next request without redeploying the image.
    if os.environ.get("CACHE_ONLY") == "1":
        cache = _load_cache()
        key = _cache_key(application_text, mitigation)
        if key in cache:
            return cache[key]
        # Key not in cache — return zeroed-out stub so the API never 500s
        return _stub_response(mitigation)

    if mitigation == "redaction":
        surname = extract_surname(application_text)
        if surname:
            application_text = redact_surname(application_text, surname)
        result = get_logits(application_text)
        return {
            "decision": "APPROVED" if result["margin"] > 0 else "REJECTED",
            "approved_logit": result["approved_logit"],
            "rejected_logit": result["rejected_logit"],
            "margin": result["margin"],
            "bias_flag": False,
            "bias_flag_reason": None,
            "counterfactual_probe": None,
            "mitigation_applied": "redaction",
        }

    # mitigation=none — score then run 3 counterfactual probes
    base = get_logits(application_text)
    surname = extract_surname(application_text)

    probe_results = []
    if surname:
        for axis, ref_name in REFERENCE_SURNAMES.items():
            swapped = _replace_surname(application_text, surname, ref_name)
            r = get_logits(swapped)
            probe_results.append(
                {
                    "axis": axis,
                    "surname": ref_name,
                    "margin": r["margin"],
                    "delta": base["margin"] - r["margin"],
                }
            )

    max_delta = max((abs(p["delta"]) for p in probe_results), default=0.0)
    max_p = (
        max(probe_results, key=lambda p: abs(p["delta"]))
        if probe_results
        else None
    )
    bias_flagged = max_delta > 0.5

    bias_reason = None
    if bias_flagged and max_p:
        bias_reason = (
            f"Surname swap to '{max_p['surname']}' would change the logit margin by "
            f"{abs(max_p['delta']):.2f}, exceeding 0.5 threshold. "
            "This decision is demographically sensitive — recommend human review."
        )

    return {
        "decision": "APPROVED" if base["margin"] > 0 else "REJECTED",
        "approved_logit": base["approved_logit"],
        "rejected_logit": base["rejected_logit"],
        "margin": base["margin"],
        "bias_flag": bias_flagged,
        "bias_flag_reason": bias_reason,
        "counterfactual_probe": {
            "tested_surnames": [p["surname"] for p in probe_results],
            "max_delta": round(max_delta, 4),
            "max_delta_surname": max_p["surname"] if max_p else None,
        },
        "mitigation_applied": "none",
    }


def _stub_response(mitigation: str) -> dict:
    return {
        "decision": "REJECTED",
        "approved_logit": 0.0,
        "rejected_logit": 0.0,
        "margin": 0.0,
        "bias_flag": False,
        "bias_flag_reason": None,
        "counterfactual_probe": None,
        "mitigation_applied": mitigation,
    }
