"""Final team-wide output contract for the KisanOS Vision Agent.

build_contract()  : internal analysis dict  -> mandatory JSON dict
error_contract()  : safe 'error' output when anything goes wrong
validate_contract(): pure-Python mirror of the backend checks (no extra packages)

The top-level keys, their order and their types are FIXED by the team lead.
Do not add, rename or remove any key.
"""
import re
import uuid
from datetime import datetime, timezone

AGENT_ID = "vision"
VERSION = "1.0"

STATUSES = ("complete", "partial", "unavailable", "error")
EVIDENCE_BANDS = ("low", "medium", "high", "not_calibrated")
SOURCE_STATUSES = ("official", "supporting", "secondary", "unverified")
PROVIDERS = ("open-meteo", "gemini", "groq", "amis", "self-hosted")

REQUIRED_KEYS = (
    "agent_id", "assessment_id", "status", "summary", "observations",
    "possible_causes", "checks", "evidence_band", "evidence_reason", "sources",
    "provider_or_model", "version", "created_at", "safety_flags",
)
SOURCE_KEYS = ("title", "url", "publisher", "retrieved_at", "source_status")

# Same word list the agent sanitiser uses (rule 4: no chemical / pesticide / dose).
FORBIDDEN = re.compile(
    r"\b(pesticide|fungicide|insecticide|herbicide|chemical|spray|sprayed|dose|dosage|ml per|kg per|"
    r"urea|dap|imidacloprid|propiconazole|tebuconazole)\b", re.I)
_ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# Safety-flag vocabulary. Names deliberately avoid the forbidden keywords above,
# so the backend keyword check can never trip on a flag name.
FLAG_UNSAFE_REMOVED = "unsafe_advice_removed"
FLAG_CONF_CAPPED = "confidence_capped"
FLAG_CLASS_DROPPED = "unrecognised_label_dropped"
FLAG_EXPERT = "expert_referral_recommended"
FLAG_NON_WHEAT = "non_wheat_image"
FLAG_QUALITY = "photo_quality_failed"
FLAG_DUMMY = "example_output_not_real_analysis"
FLAG_EXCEPTION = "agent_exception"

# ---------------------------------------------------------------- sources
# S-numbers come from the Vision Agent research report.
# TODO(Ghulam): fill S1 / S2 from the research report. Entries left as None are
# skipped, so `sources` can never contain an incomplete object.
# Example of a filled entry:
#   "S1": {"title": "...", "url": "https://...", "publisher": "...",
#          "retrieved_at": "2026-10-03T12:00:00Z", "source_status": "supporting"}
SOURCE_REGISTRY = {"S1": None, "S2": None}

MODEL_SOURCES = {
    "gemini": {"title": "Google Gemini API (multimodal photo analysis)",
               "url": "https://ai.google.dev/gemini-api/docs", "publisher": "Google",
               "source_status": "unverified"},
    "groq": {"title": "Groq API (multimodal photo analysis)",
             "url": "https://console.groq.com/docs", "publisher": "Groq",
             "source_status": "unverified"},
}

# ---------------------------------------------------------------- class tables
LABELS = {
    "healthy_looking": "No visible symptoms",
    "yellowing": "Yellowing",
    "rust_like_pustules": "Rust-like pustules",
    "spots_or_blotches": "Spots or blotches",
    "visible_insects": "Visible insects",
    "drying": "Drying",
    "unclear": "Unclear",
}
# Hedged, non-chemical, never a diagnosis. Align wording with the research report.
CAUSES = {
    "healthy_looking": [],
    "yellowing": ["Water stress (too little or too much moisture)", "Nutrient-related stress",
                  "Early foliar disease"],
    "rust_like_pustules": ["Possible rust-type fungal disease (cannot be confirmed from a photo)"],
    "spots_or_blotches": ["Possible fungal leaf spot or blotch disease",
                          "Physical damage or environmental stress"],
    "visible_insects": ["Insect feeding or infestation (species not identified from a photo)"],
    "drying": ["Water or heat stress", "Advanced disease or pest damage", "Natural leaf ageing"],
    "unclear": ["Not enough visible detail to suggest a cause"],
}
CHECKS = {
    "healthy_looking": ["Re-check the field after a few days and take a new photo if symptoms appear"],
    "yellowing": ["Check whether older or younger leaves are yellow, and whether it is uniform or in patches",
                  "Check recent irrigation and soil moisture"],
    "rust_like_pustules": ["Rub a leaf with a finger and see if orange-brown powder comes off",
                           "Check how many plants in the field show the dots"],
    "spots_or_blotches": ["Check whether spots are on lower or upper leaves and if they are spreading",
                          "Look for the same spots on several plants"],
    "visible_insects": ["Check the underside of leaves and the plant base for insects",
                        "Estimate how many plants are affected"],
    "drying": ["Check whether drying starts at the leaf tips or covers the whole leaf",
               "Check soil moisture and the last irrigation date"],
    "unclear": ["Take a closer, sharper photo of the affected leaf in daylight"],
}
EXPERT_CHECK = "Show the affected leaf to a local agriculture expert for confirmation"


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uuid(value=None) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError):
        return str(uuid.uuid4())


def _dedupe(items):
    seen, out = set(), []
    for i in items:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _skeleton(assessment_id, status, summary, band, reason, provider, flags):
    return {
        "agent_id": AGENT_ID,
        "assessment_id": _uuid(assessment_id),
        "status": status,
        "summary": summary,
        "observations": [],
        "possible_causes": [],
        "checks": [],
        "evidence_band": band,
        "evidence_reason": reason,
        "sources": [],
        "provider_or_model": provider,
        "version": VERSION,
        "created_at": now_iso(),
        "safety_flags": list(flags),
    }


def _sources_for(findings, provider):
    out = []
    if provider in MODEL_SOURCES:
        m = MODEL_SOURCES[provider]
        out.append({"title": m["title"], "url": m["url"], "publisher": m["publisher"],
                    "retrieved_at": now_iso(), "source_status": m["source_status"]})
    ids = []
    for f in findings:
        ids += [s.strip() for s in (f.get("source") or "").split(",") if s.strip()]
    for sid in _dedupe(ids):
        entry = SOURCE_REGISTRY.get(sid)
        if entry:
            out.append({"title": entry["title"], "url": entry["url"], "publisher": entry["publisher"],
                        "retrieved_at": entry.get("retrieved_at") or now_iso(),
                        "source_status": entry["source_status"]})
    return out


def build_contract(internal: dict, assessment_id=None) -> dict:
    """Map the agent's internal result onto the mandatory JSON."""
    flags = list(internal.get("_flags", []))
    status_in = internal.get("status")
    mode = internal.get("mode")
    quality = internal.get("image_quality") or {}
    provider = internal.get("provider") if internal.get("provider") in PROVIDERS else None

    # --- photo failed the quality gate: nothing was analysed
    if status_in == "needs_better_photo":
        issues = quality.get("issues", [])
        c = _skeleton(
            assessment_id, "unavailable",
            "The photo did not pass the image-quality check, so no analysis was performed. "
            "A clearer photo is needed; the Crop Agent should rely on text symptoms.",
            "not_calibrated",
            "No analysis performed: photo failed quality check (" + (", ".join(issues) or "unknown") + ").",
            "self-hosted", flags + [FLAG_QUALITY])
        c["observations"] = [f"Photo issue: {i.replace('_', ' ')}" for i in issues]
        c["checks"] = list(quality.get("tips", [])) + \
            ["Describe the symptoms in text so another agent can still help"]
        return c

    # --- not a wheat photo
    if status_in == "unsupported":
        p = provider or "self-hosted"
        c = _skeleton(
            assessment_id, "unavailable",
            "The photo does not appear to show a wheat crop. KisanOS supports wheat only, "
            "so no symptoms were assessed.",
            "not_calibrated", "Photo not recognised as wheat; no symptom analysis was done.",
            p, flags + [FLAG_NON_WHEAT])
        c["observations"] = [f"Crop detected in photo: {internal.get('crop_detected') or 'unknown'}"]
        c["checks"] = ["Send a clear, close photo of a wheat leaf or plant"]
        c["sources"] = _sources_for([], p)
        return c

    # --- normal analysis
    findings = internal.get("visible_findings", [])
    classes = [f["class"] for f in findings]
    names = _dedupe([LABELS[x].lower() for x in classes])
    is_dummy = mode in ("dummy", "dummy_fallback")
    conf = internal.get("confidence")
    band = "not_calibrated" if is_dummy else ("medium" if conf == "Medium" else "low")
    p = "self-hosted" if is_dummy else (provider or "self-hosted")

    summary = (f"Photo analysis of a wheat crop. Visible signs noted: {', '.join(names)}. "
               f"{internal.get('limitations', '')}")
    if internal.get("expert_referral"):
        summary += " Expert review is recommended."
    if is_dummy:
        summary = "EXAMPLE OUTPUT ONLY, not a real analysis. " + summary
    reason = internal.get("confidence_reason") or "Photo-only analysis."
    if not is_dummy:
        reason += " Confidence is the model's own estimate and is capped at medium."

    all_flags = flags + ([FLAG_EXPERT] if internal.get("expert_referral") else []) + \
        ([FLAG_DUMMY] if is_dummy else [])
    c = _skeleton(assessment_id, "partial" if is_dummy else "complete",
                  summary.strip(), band, reason, p, all_flags)
    c["observations"] = [f"{LABELS[f['class']]}: {f['detail']}" for f in findings]
    c["possible_causes"] = _dedupe([x for cl in classes for x in CAUSES[cl]])
    c["checks"] = _dedupe([x for cl in classes for x in CHECKS[cl]] +
                          ([EXPERT_CHECK] if internal.get("expert_referral") else []))
    c["sources"] = _sources_for(findings, p)
    return c


def error_contract(assessment_id=None, exc=None) -> dict:
    kind = type(exc).__name__ if exc is not None else "unknown"
    c = _skeleton(assessment_id, "error",
                  "The Vision Agent could not complete the analysis because of an internal error.",
                  "not_calibrated", f"Internal error ({kind}); no analysis was returned.",
                  "self-hosted", [FLAG_EXCEPTION])
    c["checks"] = ["Try again with a new photo", "Describe the symptoms in text so another agent can help"]
    return c


def _strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def _is_iso(s) -> bool:
    if not isinstance(s, str) or not _ISO_Z.match(s):
        return False
    try:
        datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except ValueError:
        return False


def validate_contract(d) -> list:
    """Return a list of problems (empty list = valid). Mirrors the backend checks."""
    errs = []
    if not isinstance(d, dict):
        return ["output is not an object"]
    if tuple(d.keys()) != REQUIRED_KEYS:
        missing = [k for k in REQUIRED_KEYS if k not in d]
        extra = [k for k in d if k not in REQUIRED_KEYS]
        errs.append(f"keys differ from contract (missing={missing}, extra={extra}, or wrong order)")
    if d.get("agent_id") != AGENT_ID:
        errs.append("agent_id must be 'vision'")
    try:
        uuid.UUID(str(d.get("assessment_id")))
    except ValueError:
        errs.append("assessment_id is not a UUID")
    if d.get("status") not in STATUSES:
        errs.append("invalid status")
    if d.get("evidence_band") not in EVIDENCE_BANDS:
        errs.append("invalid evidence_band")
    if d.get("provider_or_model") not in PROVIDERS:
        errs.append("invalid provider_or_model")
    for k in ("summary", "evidence_reason", "version"):
        if not isinstance(d.get(k), str) or not d.get(k):
            errs.append(f"{k} must be a non-empty string")
    for k in ("observations", "possible_causes", "checks", "safety_flags"):
        v = d.get(k)
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            errs.append(f"{k} must be a list of strings")
    if not _is_iso(d.get("created_at")):
        errs.append("created_at is not ISO-8601 UTC (YYYY-MM-DDTHH:MM:SSZ)")
    srcs = d.get("sources")
    if not isinstance(srcs, list):
        errs.append("sources must be a list")
    else:
        for i, s in enumerate(srcs):
            if not isinstance(s, dict) or tuple(s.keys()) != SOURCE_KEYS:
                errs.append(f"sources[{i}] has wrong keys"); continue
            if not all(isinstance(s[k], str) and s[k] for k in SOURCE_KEYS):
                errs.append(f"sources[{i}] has an empty or non-string field")
            if not str(s["url"]).startswith(("http://", "https://")):
                errs.append(f"sources[{i}].url is not a URL")
            if not _is_iso(s["retrieved_at"]):
                errs.append(f"sources[{i}].retrieved_at is not ISO-8601")
            if s["source_status"] not in SOURCE_STATUSES:
                errs.append(f"sources[{i}].source_status is invalid")
    for s in _strings(d):
        if FORBIDDEN.search(s):
            errs.append("forbidden chemical/pesticide/dose keyword in output")
            break
    return errs
