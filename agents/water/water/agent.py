"""KisanOS Water Agent  (agents/water/agent.py)

analyze_water(crop, area, last_irrigation_days, ...) -> dict   in the MANDATORY team JSON format.

Flow:  validate inputs -> rule engine -> build the fixed JSON -> (optional) AI_ENHANCED -> safety check.
The irrigation status is decided by fixed rules, never by an LLM. The agent only gives an advisory
estimate and safe field checks: no irrigation amounts, no chemicals. It never raises: any problem
becomes a valid JSON with status "error" or "unavailable".

The mandatory format has no field for the irrigation verdict, so it is written at the start of
"summary" as  "Irrigation status: <value>."  Use get_irrigation_status(result) to read it, and
AI_ENHANCED.priority_level (high / medium / low) when the enhanced section is requested.
"""
import re
import uuid
from datetime import datetime, timezone

try:
    from . import ai_enhance
    from .schema import AgentOutput, find_unsafe, validate_output
except ImportError:                     # running directly from the agent folder
    import ai_enhance
    from schema import AgentOutput, find_unsafe, validate_output

VERSION = "1.0"
AGENT_ID = "water"
SOURCES_RETRIEVED_AT = "2026-10-03T00:00:00Z"        # date the sources were looked up

# --------------------------------------------------------------------------------------
# Sources (W1-W4 match the Water Agent research report order)
# --------------------------------------------------------------------------------------
SOURCES = {
    "W1": {"title": "Irrigation scheduling (wheat)", "url": "https://agri.sindh.gov.pk/irrigation",
           "publisher": "Government of Sindh, Agriculture Department", "source_status": "official"},
    "W2": {"title": "Wheat irrigation at crown root, tillering, jointing/booting and milking stages (Punjab Agricultural University study)",
           "url": "https://epubs.icar.org.in/index.php/IJAgS/article/download/101461/39839/340822",
           "publisher": "ICAR ePubs (Indian Journal of Agricultural Sciences)", "source_status": "supporting"},
    "W3": {"title": "Wheat yield response to irrigation levels (Pakistan Journal of Agricultural Sciences)",
           "url": "https://www.pakjas.com.pk/papers/63.pdf",
           "publisher": "Pakistan Journal of Agricultural Sciences", "source_status": "supporting"},
    "W4": {"title": "Open-Meteo forecast API documentation (daily precipitation and FAO ET0)",
           "url": "https://open-meteo.com/en/docs", "publisher": "Open-Meteo", "source_status": "supporting"},
}

# --------------------------------------------------------------------------------------
# Rule engine. basis: source | derived | tunable_default | farmer_input
#   source          - stated by W1-W3
#   derived         - KisanOS value derived from the spacing of source-recommended irrigations
#   tunable_default - engineering starting value, NOT from a source: validate with an agronomist
# --------------------------------------------------------------------------------------
# (name, label, first_day, last_day, critical_for_water, source ids) - days after sowing, indicative
STAGES = [
    ("establishment",         "Establishment (before crown root stage)", 0,   19,  False, "W1"),
    ("crown_root_initiation", "Crown root initiation",                  20,  30,  True,  "W1, W2"),
    ("tillering",             "Tillering",                              31,  55,  True,  "W1, W2"),
    ("jointing_booting",      "Jointing / booting",                     56,  79,  True,  "W2, W3"),
    ("flowering_heading",     "Heading / flowering",                    80,  99,  True,  "W1, W3"),
    ("milking_dough",         "Milking / dough (grain filling)",        100, 120, True,  "W1, W2"),
    ("ripening_maturity",     "Ripening / maturity",                    121, 400, False, "W2"),
]
STAGE_NAMES = [s[0] for s in STAGES]
STAGE_INFO = {s[0]: {"name": s[0], "label": s[1], "critical": s[4], "source": s[5]} for s in STAGES}
_ALIASES = {
    "establishment": ["establishment", "germination", "emergence", "seedling", "sowing"],
    "crown_root_initiation": ["crown_root_initiation", "crown_root", "crown_root_stage", "cri"],
    "tillering": ["tillering", "tiller"],
    "jointing_booting": ["jointing_booting", "jointing", "late_jointing", "booting", "stem_elongation"],
    "flowering_heading": ["flowering_heading", "heading_flowering", "flowering", "heading", "earing", "anthesis"],
    "milking_dough": ["milking_dough", "milking", "milk", "dough", "grain_filling", "grain_fill"],
    "ripening_maturity": ["ripening_maturity", "ripening", "maturity", "harvest"],
}
ALIASES = {a: n for n, lst in _ALIASES.items() for a in lst}

THRESHOLDS = {"critical": (14, 21), "other": (21, 28)}   # (soon, recommended) days since last irrigation - DERIVED
RAIN_SIGNIFICANT_MM = 10.0     # forecast rain, next 3 days  - tunable default
ET0_HIGH = 5.0                 # mm/day                       - tunable default
TEMP_HIGH_C = 35.0             # average daily maximum, deg C - tunable default
STATUS = ["not_needed_now", "may_be_needed_soon", "check_recommended"]
SOIL_VALUES = ("dry", "moist", "wet")
WHEAT_NAMES = {"wheat", "gandum", "gandam", "gundum", "گندم"}

ACTIONS = {
    "check_recommended": [
        "Check soil moisture at root depth today (dig a little soil and squeeze it in your hand).",
        "If the soil is dry, plan irrigation at the next available canal or tubewell turn.",
        "Avoid standing water; do not flood the field.",
        "Monitor the crop for 48 hours and re-check.",
    ],
    "may_be_needed_soon": [
        "Check soil moisture at root depth in the next 1-2 days.",
        "Be ready to irrigate at the next available turn if the soil is dry.",
        "Monitor the crop for 48 hours.",
    ],
    "not_needed_now": [
        "No irrigation action is needed today; re-check soil moisture before the next planned irrigation.",
        "Keep monitoring the crop and the weather forecast.",
    ],
}
RAIN_ACTION = "Re-check soil moisture after the rain before irrigating."
EXPERT_ACTION = "Ask a local agriculture officer or expert to confirm the growth stage and irrigation need."

DEMO_INPUT = {      # PRD demo: wheat, Bahawalpur, last irrigation 5 days ago (stage/weather values are examples)
    "crop": "wheat", "area": "Bahawalpur", "last_irrigation_days": 5, "growth_stage": "flowering",
    "weather": {"rain_next_3d_mm": 0.0, "avg_max_temp_c": 34.0, "et0_mm_per_day": 5.5},
}


def normalize_stage(text):
    if text is None:
        return None
    return ALIASES.get(str(text).strip().lower().replace("-", "_").replace(" ", "_"))


def stage_from_das(das):
    if das is None:
        return None
    for name, _l, lo, hi, _c, _s in STAGES:
        if lo <= das <= hi:
            return name
    return None


def _num(x):
    if isinstance(x, bool) or x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _ev(factor, detail, source, basis):
    return {"factor": factor, "detail": detail, "source": source, "basis": basis}


def assess(stage_name, last_days, weather=None, soil=None):
    """Return (status, evidence, flags). stage_name may be None (unknown)."""
    weather = weather or {}
    ev, flags = [], {"rain_deferred": False, "heat_raised": False, "soon_reached": False}
    if stage_name == "ripening_maturity":
        ev.append(_ev("growth_stage", "The crop is near maturity; recommended irrigation schedules usually end around the dough stage.", "W1, W2", "source"))
        return STATUS[0], ev, flags

    info = STAGE_INFO.get(stage_name)
    critical = True if info is None else info["critical"]
    soft, hard = THRESHOLDS["critical" if critical else "other"]
    if info is None:
        ev.append(_ev("growth_stage", "Growth stage is unknown, so the stricter (critical-stage) limits are used.", "W1, W2", "tunable_default"))
    elif critical:
        ev.append(_ev("growth_stage", f"{info['label']} is a critical stage for water in wheat.", info["source"], "source"))
    else:
        ev.append(_ev("growth_stage", f"{info['label']}: the first irrigation is usually given 3-4 weeks after sowing.", info["source"], "source"))

    level = 2 if last_days >= hard else 1 if last_days >= soft else 0
    flags["soon_reached"] = last_days >= soft
    ev.append(_ev("days_since_irrigation", f"Last irrigation was {last_days} days ago; a check is usually due after about {soft}-{hard} days at this stage.", "W1, W2", "derived"))

    et0, tmax, rain = _num(weather.get("et0_mm_per_day")), _num(weather.get("avg_max_temp_c")), _num(weather.get("rain_next_3d_mm"))
    if critical and ((et0 is not None and et0 >= ET0_HIGH) or (tmax is not None and tmax >= TEMP_HIGH_C)):
        level += 1
        flags["heat_raised"] = True
        ev.append(_ev("weather_heat", "Hot, dry weather increases crop water use (high evapotranspiration or temperature).", "W4", "tunable_default"))
    if soil == "dry":
        level += 1
        ev.append(_ev("soil_moisture", "Farmer reports the soil is dry.", None, "farmer_input"))
    if rain is not None:
        if rain >= RAIN_SIGNIFICANT_MM and level > 0:
            level -= 1
            flags["rain_deferred"] = True
            ev.append(_ev("rain_forecast", f"Significant rain is forecast in the next 3 days ({rain:g} mm), so irrigation can usually wait.", "W4", "tunable_default"))
        else:
            ev.append(_ev("rain_forecast", f"Little or no significant rain is forecast in the next 3 days ({rain:g} mm).", "W4", "farmer_input"))
    level = min(2, max(0, level))
    if soil == "wet":
        level = 0
        ev.append(_ev("soil_moisture", "Farmer reports the soil is wet, so irrigation is not needed now.", None, "farmer_input"))
    elif soil == "moist":
        ev.append(_ev("soil_moisture", "Farmer reports the soil is moist.", None, "farmer_input"))
    return STATUS[level], ev, flags


# --------------------------------------------------------------------------------------
# JSON building helpers
# --------------------------------------------------------------------------------------
def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sources_for(evidence):
    ids = set()
    for e in evidence:
        for sid in str(e.get("source") or "").replace(" ", "").split(","):
            if sid in SOURCES:
                ids.add(sid)
    return [{**SOURCES[i], "retrieved_at": SOURCES_RETRIEVED_AT} for i in sorted(ids)]


def _envelope(status, summary, observations=None, causes=None, checks=None, band="not_calibrated",
              reason="", sources=None, flags=None, provider="self-hosted", enhanced=None, assessment_id=None):
    data = {"agent_id": AGENT_ID, "assessment_id": assessment_id or str(uuid.uuid4()), "status": status,
            "summary": summary, "observations": observations or [], "possible_causes": causes or [],
            "checks": checks or [], "evidence_band": band, "evidence_reason": reason,
            "sources": sources or [], "provider_or_model": provider, "version": VERSION,
            "created_at": _now(), "safety_flags": flags or []}
    if enhanced:
        data["AI_ENHANCED"] = enhanced
    return AgentOutput(**data).model_dump(exclude_none=True)


def _to_int(x, lo, hi):
    """Return (value, ok). None -> (None, True)."""
    if x is None:
        return None, True
    if isinstance(x, bool):
        return None, False
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None, False
    return (int(v), True) if lo <= v <= hi and v == int(v) else (None, False)


def _clean_area(area):
    a = re.sub(r"[^\w\s\-,.]", "", str(area or ""), flags=re.U).strip()[:60]
    return "" if find_unsafe(a) else a


def get_irrigation_status(result: dict):
    """Read the verdict (not_needed_now | may_be_needed_soon | check_recommended) from the summary."""
    m = re.match(r"Irrigation status: (\w+)\.", str((result or {}).get("summary", "")))
    return m.group(1) if m and m.group(1) in STATUS else None


# --------------------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------------------
def analyze_water(crop, area, last_irrigation_days, growth_stage=None, days_after_sowing=None,
                  weather=None, soil_moisture=None, enhance: bool = False, use_llm: bool = True,
                  assessment_id=None) -> dict:
    """weather (optional, from the Weather Agent): {"rain_next_3d_mm", "avg_max_temp_c", "et0_mm_per_day"}
       soil_moisture (optional, farmer-reported): "dry" | "moist" | "wet"
       enhance=True adds the optional AI_ENHANCED section (Groq if GROQ_API_KEY is set, else fixed
       Urdu templates); use_llm=False forces the templates."""
    try:
        uuid.UUID(str(assessment_id))
        aid = str(assessment_id)
    except (ValueError, AttributeError, TypeError):
        aid = str(uuid.uuid4())                                # missing / invalid id -> make a fresh one
    try:
        result = _analyze(crop, area, last_irrigation_days, growth_stage, days_after_sowing,
                          weather, soil_moisture, enhance, use_llm, aid)
        problems = validate_output(result)                     # same checks the backend will run
        if problems:
            raise ValueError("; ".join(problems[:3]))
        return result
    except Exception as e:                                     # never crash the orchestrator
        flag = "unsafe_content_blocked" if "banned keyword" in str(e) else "agent_internal_error"
        return _envelope("error", "The Water Agent could not produce an assessment.",
                         checks=["Please check the inputs and try again, or ask a local agriculture expert."],
                         reason="Internal error; no assessment was made.", flags=[flag], assessment_id=aid)


def _analyze(crop, area, last_irrigation_days, growth_stage, days_after_sowing, weather,
             soil_moisture, enhance, use_llm, assessment_id):
    aid = assessment_id if assessment_id else str(uuid.uuid4())

    # ---- 1. crop and required inputs ----------------------------------------------------
    if str(crop or "").strip().lower() not in WHEAT_NAMES:
        return _envelope("unavailable", "Water Agent supports wheat only; no irrigation assessment was made.",
                         checks=["Select wheat, or ask a local agriculture expert about other crops."],
                         reason="Unsupported crop.", assessment_id=aid)
    missing = []
    if not str(area or "").strip():
        missing.append("area")
    if last_irrigation_days is None:
        missing.append("last irrigation (days ago)")
    if missing:
        return _envelope("unavailable", "Required input is missing, so no irrigation assessment was made.",
                         checks=["Please provide: " + ", ".join(missing) + "."],
                         reason="Required input missing.", assessment_id=aid)
    days, ok_days = _to_int(last_irrigation_days, 0, 150)
    das, ok_das = _to_int(days_after_sowing, 0, 200)
    if not ok_days or not ok_das:
        return _envelope("error", "An input value is invalid, so no irrigation assessment was made.",
                         checks=["Enter days since last irrigation as a whole number (0-150) and days after sowing as a whole number (0-200)."],
                         reason="Invalid input.", assessment_id=aid)

    # ---- 2. optional inputs ---------------------------------------------------------------
    soil = str(soil_moisture).strip().lower() if soil_moisture else None
    low_reasons, flags = [], []
    if soil is not None and soil not in SOIL_VALUES:
        soil = None
        low_reasons.append("soil report not recognised (use dry, moist or wet)")
    weather = weather if isinstance(weather, dict) else None

    # ---- 3. growth stage ------------------------------------------------------------------
    s_text = normalize_stage(growth_stage) if growth_stage else None
    if growth_stage and s_text is None:
        low_reasons.append("growth stage text not recognised")
    s_das = stage_from_das(das)
    stage, conflict = None, False
    if s_text:
        stage = s_text
        if s_das and abs(STAGE_NAMES.index(s_das) - STAGE_NAMES.index(s_text)) >= 2:
            conflict = True
            low_reasons.append("growth stage and days after sowing disagree (the stage given was used)")
    elif s_das:
        stage = s_das
    if stage is None:
        low_reasons.append("growth stage unknown")
    info = STAGE_INFO.get(stage)

    # ---- 4. rules ---------------------------------------------------------------------------
    status, evidence, rflags = assess(stage, days, weather, soil)
    if not weather:
        low_reasons.append("no weather data")
    if soil == "wet" and days >= 21:
        low_reasons.append("soil report conflicts with the long irrigation gap")

    # ---- 5. fixed-format fields ---------------------------------------------------------
    stage_txt = f" at the {info['label']} stage" if info else " (growth stage unknown)"
    a = _clean_area(area)
    drivers = []
    if rflags["heat_raised"]:
        drivers.append("hot, dry weather raises crop water use")
    if rflags["rain_deferred"]:
        drivers.append("significant rain is forecast")
    if soil in ("dry", "wet"):
        drivers.append(f"soil reported {soil}")
    summary = (f"Irrigation status: {status}. Wheat{' in ' + a if a else ''}{stage_txt}. "
               f"Last irrigation was {days} days ago" + ("; " + "; ".join(drivers) if drivers else "") + ". "
               "Advisory estimate from rules; soil moisture was not measured.")
    observations = [e["detail"] for e in evidence]
    causes = []
    if status != "not_needed_now":
        if rflags["soon_reached"]:
            causes.append("Time since the last irrigation has reached the usual limit for this stage.")
        if rflags["heat_raised"]:
            causes.append("Hot, dry weather increases crop water use.")
        if soil == "dry":
            causes.append("The soil is reported to be dry.")
        if info and info["critical"]:
            causes.append("Wheat is at a growth stage that is sensitive to water shortage.")
    if not causes:
        causes = ["No water-stress factor was found in the given inputs."]
    checks = list(ACTIONS[status]) + ([RAIN_ACTION] if rflags["rain_deferred"] and status != "not_needed_now" else [])
    if stage is None or conflict:
        checks.append(EXPERT_ACTION)
        flags.append("expert_referral_recommended")

    base = ("Day limits are derived from source-recommended irrigation spacing and are not field-calibrated"
            + ("; heat and rain limits are tunable defaults" if weather else "") + ".")
    if low_reasons:
        band, out_status = "low", "partial"
        reason = "Reduced because: " + "; ".join(low_reasons) + ". " + base
    else:
        band, out_status = "medium", "complete"
        reason = "Growth stage, irrigation gap and weather were all provided. " + base

    # ---- 6. optional AI_ENHANCED ----------------------------------------------------------
    enhanced, provider = None, "self-hosted"
    if enhance:
        facts = {"irrigation_status": status, "growth_stage": info["label"] if info else "unknown",
                 "critical_stage": info["critical"] if info else None, "days_since_last_irrigation": days,
                 "hot_dry_weather": rflags["heat_raised"], "rain_deferred": rflags["rain_deferred"],
                 "soil_moisture_reported": soil}
        enhanced, provider, eflags = ai_enhance.build_ai_enhanced(facts, use_llm=use_llm)
        flags += eflags

    return _envelope(out_status, summary, observations, causes, checks, band, reason,
                     _sources_for(evidence), flags, provider, enhanced, aid)
