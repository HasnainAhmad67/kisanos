"""KisanOS Vision Agent  (agents/vision/agent.py)

analyze_image(image_bytes, growth_stage=None, force_dummy=False, assessment_id=None) -> dict
    Returns the team's MANDATORY JSON contract (see contract.py / schema.json).

Flow:  quality gate -> (free multimodal API: Gemini or Groq | dummy fallback)
       -> safety sanitiser -> contract layer -> final validation.
The agent only DESCRIBES visible symptoms. It never confirms a disease and
never gives chemical advice. Confidence is capped at Medium.

run_analysis() is the internal step that produces the rich internal dict; the
backend should only ever call analyze_image().
"""
import re

from .quality import check_quality
from .dummy import dummy_result
from . import llm
from . import contract

ALLOWED_CLASSES = {
    "healthy_looking", "yellowing", "rust_like_pustules",
    "spots_or_blotches", "visible_insects", "drying", "unclear",
}
SOURCES = {  # S-numbers match the Vision Agent research report
    "healthy_looking": "S1", "yellowing": "S1", "rust_like_pustules": "S1, S2",
    "spots_or_blotches": "S1, S2", "visible_insects": "S1", "drying": "S1", "unclear": None,
}
LIMITATIONS = "Photo shows visible signs only; cannot confirm cause."
FORBIDDEN = re.compile(
    r"\b(pesticide|fungicide|insecticide|herbicide|chemical|spray|sprayed|dose|dosage|ml per|kg per|urea|dap|"
    r"imidacloprid|propiconazole|tebuconazole)\b", re.I)


def _base(status):
    return {
        "agent": "vision", "status": status, "mode": None, "provider": None,
        "image_quality": None, "crop_detected": None, "visible_findings": [],
        "confidence": "Low", "confidence_reason": "", "expert_referral": False,
        "limitations": LIMITATIONS, "message_for_farmer": "", "_flags": [],
    }


def _flag(out, name):
    if name not in out["_flags"]:
        out["_flags"].append(name)


def _clean_text(t, fallback="", out=None):
    t = str(t or "").strip()
    if FORBIDDEN.search(t):
        if out is not None:
            _flag(out, contract.FLAG_UNSAFE_REMOVED)
        return fallback
    return t


def _sanitize(raw: dict, out: dict) -> dict:
    """Enforce the safety rules on whatever the model returned."""
    out.setdefault("_flags", [])
    crop = str(raw.get("crop_detected", "")).lower()
    if crop != "wheat":
        out["status"], out["crop_detected"] = "unsupported", _clean_text(crop, "unknown", out) or "unknown"
        out["expert_referral"] = False
        out["message_for_farmer"] = "This does not look like a wheat photo. KisanOS supports wheat only."
        return out

    out["crop_detected"] = "wheat"
    findings = []
    for f in raw.get("visible_findings", []) or []:
        cls = f.get("class") if isinstance(f, dict) else None
        if cls not in ALLOWED_CLASSES:
            _flag(out, contract.FLAG_CLASS_DROPPED)
            continue
        findings.append({"class": cls, "detail": _clean_text(f.get("detail"), "visible in photo", out),
                         "source": SOURCES[cls]})
    if not findings:
        findings = [{"class": "unclear", "detail": "no clear symptom described", "source": None}]
    out["visible_findings"] = findings

    conf = raw.get("confidence")
    if conf not in ("Low", "Medium"):
        _flag(out, contract.FLAG_CONF_CAPPED)
    out["confidence"] = conf if conf in ("Low", "Medium") else "Low"   # never High (VIS-06)
    out["confidence_reason"] = _clean_text(raw.get("confidence_reason"), "Photo-only analysis.", out)

    classes = {f["class"] for f in findings}
    out["expert_referral"] = bool(
        "rust_like_pustules" in classes or "unclear" in classes or out["confidence"] == "Low")

    if "rust_like_pustules" in classes:
        msg = "Rust-like dots are visible. Please show this to an agriculture expert soon."
    elif out["expert_referral"]:
        msg = "The photo is not clear enough to be sure. Please ask an agriculture expert."
    else:
        msg = "Some visible signs were noted. Please check the field and ask an expert if unsure."
    out["message_for_farmer"] = msg
    out["status"] = "ok"
    return out


def run_analysis(image_bytes: bytes, growth_stage=None, force_dummy: bool = False) -> dict:
    """Internal step: quality gate -> model/dummy -> sanitiser. Returns the internal dict."""
    quality = check_quality(image_bytes)
    out = _base("needs_better_photo")
    out["image_quality"] = quality
    if not quality["passed"]:
        out["confidence_reason"] = "Photo quality too low for analysis."
        out["message_for_farmer"] = " ".join(quality["tips"])
        return out                                      # VIS-03 / VIS-09

    note = ""
    if not force_dummy and llm.llm_available():
        try:
            which = llm.provider()
            raw, out["mode"] = llm.analyze_with_llm(image_bytes, growth_stage), "llm"
            out["provider"] = which
        except Exception as e:                          # network/API/JSON problem
            raw, out["mode"] = dummy_result(), "dummy_fallback"
            note = f"Model call failed ({type(e).__name__}); showing example output."
    else:
        raw, out["mode"] = dummy_result(), "dummy"

    out = _sanitize(raw, out)
    if out["mode"] == "dummy_fallback":
        out["confidence_reason"] = note
    return out


def analyze_image(image_bytes: bytes, growth_stage=None, force_dummy: bool = False,
                  assessment_id=None) -> dict:
    """PUBLIC entry point. Always returns the mandatory team JSON, never raises."""
    try:
        result = contract.build_contract(run_analysis(image_bytes, growth_stage, force_dummy),
                                         assessment_id)
        problems = contract.validate_contract(result)
        if problems:                                    # last line of defence
            raise ValueError("; ".join(problems))
        return result
    except Exception as e:                              # safe fallback for the backend
        return contract.error_contract(assessment_id, e)
