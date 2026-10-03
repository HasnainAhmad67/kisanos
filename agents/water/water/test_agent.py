"""Run:  python test_agent.py                       (from this agent folder)
        python -m agents.water.test_agent            (from the kisanos/ folder)
        pytest agents/water"""
import json
import os
import uuid

try:
    from . import ai_enhance
    from .agent import DEMO_INPUT, analyze_water, get_irrigation_status
    from .schema import find_unsafe, validate_output
except ImportError:                     # running directly from the agent folder
    import ai_enhance
    from agent import DEMO_INPUT, analyze_water, get_irrigation_status
    from schema import find_unsafe, validate_output

KEYS = ["agent_id", "assessment_id", "status", "summary", "observations", "possible_causes", "checks",
        "evidence_band", "evidence_reason", "sources", "provider_or_model", "version", "created_at", "safety_flags"]
W = {"rain_next_3d_mm": 0.0, "avg_max_temp_c": 24.0, "et0_mm_per_day": 2.5}      # cool, dry, low ET0


def _run(**kw):
    base = dict(crop="wheat", area="Bahawalpur", last_irrigation_days=5, growth_stage="tillering", weather=W)
    base.update(kw)
    r = analyze_water(**base)
    assert validate_output(r) == [], validate_output(r)          # every output passes the backend checks
    return r


# ---------------------------------------------------------------- mandatory format ----
def test_demo_scenario_and_exact_format():
    r = analyze_water(**DEMO_INPUT)
    assert validate_output(r) == []
    assert list(r.keys()) == KEYS                                  # exact fields, exact order, no extras
    assert r["agent_id"] == "water" and r["version"] == "1.0" and r["provider_or_model"] == "self-hosted"
    assert r["status"] == "complete" and r["evidence_band"] == "medium"
    assert get_irrigation_status(r) == "may_be_needed_soon"
    assert any("soil moisture" in c.lower() for c in r["checks"]) and any("48 hours" in c for c in r["checks"])
    assert r["created_at"].endswith("Z") and uuid.UUID(r["assessment_id"])
    assert all(isinstance(r[k], list) for k in ("observations", "possible_causes", "checks", "sources", "safety_flags"))


def test_enums_and_no_high_band_across_many_inputs():
    seen = set()
    for crop in ("wheat", "rice", None):
        for area in ("Multan", ""):
            for days in (None, 0, 10, 20, 40, -1, "x"):
                for stage in (None, "tillering", "maturity", "weird"):
                    for weather in (None, W, {"rain_next_3d_mm": 30}, 5):
                        r = analyze_water(crop, area, days, growth_stage=stage, weather=weather)
                        assert validate_output(r) == []
                        assert r["status"] in ("complete", "partial", "unavailable", "error")
                        assert r["evidence_band"] in ("low", "medium", "not_calibrated")   # never 'high'
                        seen.add(r["status"])
    assert seen == {"complete", "partial", "unavailable", "error"}


def test_status_and_band_mapping():
    assert _run()["status"] == "complete" and _run()["evidence_band"] == "medium"
    p = _run(weather=None)
    assert p["status"] == "partial" and p["evidence_band"] == "low" and "no weather data" in p["evidence_reason"]
    u = analyze_water("rice", "Multan", 3)
    assert u["status"] == "unavailable" and u["evidence_band"] == "not_calibrated" and u["sources"] == []
    assert analyze_water("wheat", "", 3)["status"] == "unavailable"
    assert analyze_water("wheat", "Multan", None)["status"] == "unavailable"
    for bad in (-1, 999, "abc", True, 2.5):
        assert analyze_water("wheat", "Multan", bad)["status"] == "error", bad
    assert analyze_water("wheat", "Multan", 5, days_after_sowing=-3)["status"] == "error"
    assert analyze_water("gandum", "Multan", "7", growth_stage="tillering", weather=W)["status"] == "complete"


def test_ids_timestamps_and_custom_assessment_id():
    assert _run()["assessment_id"] != _run()["assessment_id"]
    mine = str(uuid.uuid4())
    assert analyze_water(**DEMO_INPUT, assessment_id=mine)["assessment_id"] == mine
    r = analyze_water(**DEMO_INPUT, assessment_id="not-a-uuid")      # bad id -> fresh id, still valid
    assert validate_output(r) == [] and r["status"] == "complete"


def test_sources_format():
    r = analyze_water(**DEMO_INPUT)
    assert len(r["sources"]) >= 3
    for s in r["sources"]:
        assert set(s) == {"title", "url", "publisher", "retrieved_at", "source_status"}
        assert s["url"].startswith("https://") and s["source_status"] in ("official", "supporting", "secondary", "unverified")


def test_never_raises_on_junk_input():
    for kw in (dict(crop=None, area=None, last_irrigation_days=None), dict(crop=5, area=[], last_irrigation_days={}),
               dict(crop="wheat", area="x", last_irrigation_days=3, weather=123, growth_stage=5, soil_moisture=[]),
               dict(crop="wheat", area="x", last_irrigation_days=3, weather={"rain_next_3d_mm": "abc", "et0_mm_per_day": [1]})):
        r = analyze_water(**kw)
        assert validate_output(r) == []


def test_area_text_cannot_inject_banned_words():
    r = analyze_water("wheat", "Bahawalpur; spray urea 50 kg", 5, growth_stage="tillering", weather=W, enhance=True, use_llm=False)
    assert validate_output(r) == [] and "urea" not in json.dumps(r).lower() and "spray" not in json.dumps(r).lower()


# ------------------------------------------------------------------------ rules ----
def test_short_gap_cool_weather_not_needed_and_long_gap_recommended():
    assert get_irrigation_status(_run()) == "not_needed_now"
    assert get_irrigation_status(_run(last_irrigation_days=15)) == "may_be_needed_soon"
    assert get_irrigation_status(_run(last_irrigation_days=25)) == "check_recommended"


def test_establishment_uses_3_to_4_week_rule():
    f = lambda d: get_irrigation_status(_run(growth_stage="germination", last_irrigation_days=d))
    assert (f(15), f(22), f(29)) == ("not_needed_now", "may_be_needed_soon", "check_recommended")


def test_rain_heat_dry_wet_and_maturity():
    r = _run(last_irrigation_days=25, weather=dict(W, rain_next_3d_mm=15.0))
    assert get_irrigation_status(r) == "may_be_needed_soon" and any("after the rain" in c for c in r["checks"])
    hot = dict(W, et0_mm_per_day=6.0)
    assert get_irrigation_status(_run(weather=hot)) == "may_be_needed_soon"
    assert get_irrigation_status(_run(weather=hot, soil_moisture="dry")) == "check_recommended"
    w = _run(last_irrigation_days=30, soil_moisture="wet")
    assert get_irrigation_status(w) == "not_needed_now" and w["status"] == "partial"
    assert get_irrigation_status(_run(growth_stage="maturity", last_irrigation_days=40)) == "not_needed_now"


def test_stage_from_days_conflict_and_unknown():
    assert "Crown root" in _run(growth_stage=None, days_after_sowing=25)["summary"]
    u = _run(growth_stage=None)
    assert u["status"] == "partial" and "expert_referral_recommended" in u["safety_flags"] and "unknown" in u["summary"]
    c = _run(growth_stage="maturity", days_after_sowing=30)
    assert c["status"] == "partial" and "expert_referral_recommended" in c["safety_flags"]


# --------------------------------------------------------------------- AI_ENHANCED ----
def test_ai_enhanced_optional_section():
    assert "AI_ENHANCED" not in analyze_water(**DEMO_INPUT)
    r = analyze_water(**DEMO_INPUT, enhance=True, use_llm=False)
    e = r["AI_ENHANCED"]
    assert set(e) == {"urdu_summary", "roman_urdu", "audio_script_urdu", "emoji_visual", "farmer_explanation", "priority_level"}
    assert e["priority_level"] == "medium" and r["provider_or_model"] == "self-hosted" and r["safety_flags"] == []
    assert get_irrigation_status(r) == "may_be_needed_soon"
    assert _run(last_irrigation_days=25, enhance=True, use_llm=False)["AI_ENHANCED"]["priority_level"] == "high"
    assert _run(enhance=True, use_llm=False)["AI_ENHANCED"]["priority_level"] == "low"
    assert "AI_ENHANCED" not in analyze_water("rice", "x", 3, enhance=True)            # nothing to enhance
    assert "ai_enhancement_unavailable" in analyze_water(**DEMO_INPUT, enhance=True)["safety_flags"] \
        or os.environ.get("GROQ_API_KEY")                                              # no key -> templates + flag


class _Fake:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.calls = content, exc, []
        outer = self
        class _C:
            def create(self, **kw):
                outer.calls.append(kw)
                if outer.exc:
                    raise outer.exc
                m = type("M", (), {"content": outer.content})()
                return type("R", (), {"choices": [type("Ch", (), {"message": m})()]})()
        self.chat = type("Chat", (), {"completions": _C()})()


def _with_fake(fake, fn):
    c, a = ai_enhance._client, ai_enhance.llm_available
    ai_enhance._client, ai_enhance.llm_available = (lambda: fake), (lambda: True)
    try:
        return fn()
    finally:
        ai_enhance._client, ai_enhance.llm_available = c, a


_GOOD = json.dumps({"urdu_summary": "جلد پانی کی ضرورت ہو سکتی ہے، براہ کرم کھیت کی نمی دیکھیں۔",
                    "roman_urdu": "Jald pani ki zaroorat ho sakti hai, barah-e-karam khet ki nami dekhein.",
                    "audio_script_urdu": "کسان بھائی، جلد پانی کی ضرورت ہو سکتی ہے۔ کھیت کی نمی دیکھیں۔",
                    "farmer_explanation": "اہم مرحلے میں پانی کی کمی پیداوار کو متاثر کر سکتی ہے۔"}, ensure_ascii=False)


def test_groq_gpt_oss_120b_path():
    fake = _Fake("```json\n" + _GOOD + "\n```")
    r = _with_fake(fake, lambda: analyze_water(**DEMO_INPUT, enhance=True))
    assert validate_output(r) == []
    assert r["provider_or_model"] == "groq" and r["AI_ENHANCED"]["urdu_summary"].startswith("جلد پانی")
    assert r["AI_ENHANCED"]["priority_level"] == "medium" and get_irrigation_status(r) == "may_be_needed_soon"
    call = fake.calls[0]
    assert call["model"] == "openai/gpt-oss-120b"
    sent = json.loads(call["messages"][1]["content"])
    assert sent["irrigation_status"] == "may_be_needed_soon" and "Bahawalpur" not in json.dumps(sent)


def test_groq_unsafe_or_broken_reply_falls_back_to_templates():
    bad = [json.dumps({**json.loads(_GOOD), "urdu_summary": "کھاد ڈالیں اور سپرے کریں۔"}, ensure_ascii=False),
           json.dumps({**json.loads(_GOOD), "roman_urdu": "Spray karein aur 50 mm pani dein."}),
           json.dumps({**json.loads(_GOOD), "roman_urdu": "Pani ki zaroorat nahi hai."}),
           json.dumps({**json.loads(_GOOD), "farmer_explanation": "english only text here"}),
           '{"urdu_summary": "x"}']
    for content in bad:
        r = _with_fake(_Fake(content), lambda: analyze_water(**DEMO_INPUT, enhance=True))
        assert validate_output(r) == [] and r["provider_or_model"] == "self-hosted"
        assert "ai_output_rejected" in r["safety_flags"], content
    for fake in (_Fake(exc=TimeoutError("slow")), _Fake("not json")):
        r = _with_fake(fake, lambda: analyze_water(**DEMO_INPUT, enhance=True))
        assert r["status"] == "complete" and "ai_enhancement_unavailable" in r["safety_flags"]
        assert r["AI_ENHANCED"]["priority_level"] == "medium"


# ------------------------------------------------------ the validator itself works ----
def test_validator_catches_format_violations():
    good = analyze_water(**DEMO_INPUT)
    def broke(**ch): return validate_output({**good, **ch})
    assert broke(status="done") and broke(evidence_band="very_high") and broke(created_at="03/10/2026")
    assert broke(assessment_id="123") and broke(version=1.0) and broke(observations="one")
    assert broke(extra_field="x") and broke(summary="Spray the field")
    assert broke(sources=[{"title": "t", "url": "ftp://x", "publisher": "p", "retrieved_at": "2026-10-03T00:00:00Z", "source_status": "official"}])
    assert broke(sources=[{"title": "t"}])
    missing = dict(good); missing.pop("checks")
    assert validate_output(missing)
    assert find_unsafe("apply urea") and find_unsafe("کھاد ڈالیں") and not find_unsafe("check soil moisture")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
