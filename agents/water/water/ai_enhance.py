"""OPTIONAL AI_ENHANCED section (Urdu / Roman Urdu / audio script) for the Water Agent.

Groq model: openai/gpt-oss-120b.  Needs: pip install groq, and GROQ_API_KEY in your .env.
The model only WORDS the farmer text. The priority level and emoji come from the rule result,
and every model reply is checked before use. If anything fails, fixed Urdu templates are used,
so the agent still returns a valid section (core agent output is never affected)."""
import json
import os
import re

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:                       # python-dotenv is optional
    pass

try:
    from .schema import QUANTITY, find_unsafe
except ImportError:                     # running directly from the agent folder
    from schema import QUANTITY, find_unsafe

MODEL = os.environ.get("WATER_MODEL", "openai/gpt-oss-120b")

PRIORITY = {"check_recommended": "high", "may_be_needed_soon": "medium", "not_needed_now": "low"}
EMOJI = {"not_needed_now": "🌾✅", "may_be_needed_soon": "🌾💧", "check_recommended": "🌾💧⚠️"}

_UR = {
    "not_needed_now": "ابھی پانی لگانا ضروری نظر نہیں آتا۔ فصل پر نظر رکھیں اور اگلے پانی سے پہلے زمین کی نمی دیکھ لیں۔",
    "may_be_needed_soon": "جلد پانی کی ضرورت ہو سکتی ہے۔ اپنے کھیت کی نمی چیک کریں اور اگلی نہر یا ٹیوب ویل کی باری کا منصوبہ بنائیں۔",
    "check_recommended": "اب پانی کی جانچ کرنا بہتر ہے۔ آج زمین کی نمی دیکھیں؛ اگر زمین سوکھی ہو تو اگلی دستیاب باری پر پانی لگانے کا منصوبہ بنائیں۔",
}
_UR_RAIN = " بارش کی توقع ہے، اس لیے بارش کے بعد دوبارہ دیکھیں۔"
_RO = {
    "not_needed_now": "Abhi pani lagana zaroori nazar nahi aata. Fasal par nazar rakhein aur agle pani se pehle zameen ki nami check karein.",
    "may_be_needed_soon": "Jald pani ki zaroorat ho sakti hai. Apne khet ki nami check karein aur agli nehar ya tubewell ki baari ka plan banayein.",
    "check_recommended": "Ab pani ka check karna behtar hai. Aaj zameen ki nami check karein; agar zameen sookhi ho to agli baari par pani lagane ka plan banayein.",
}
_RO_RAIN = " Baarish ki umeed hai, is liye baarish ke baad dobara check karein."
_WHY = {
    "not_needed_now": "ضرورت سے زیادہ پانی بھی نقصان دے سکتا ہے، اس لیے نمی دیکھ کر فیصلہ کریں۔",
    "may_be_needed_soon": "گندم کے اہم مرحلوں میں پانی کی کمی پیداوار کو نقصان پہنچا سکتی ہے، اس لیے وقت پر نمی دیکھنا اہم ہے۔",
    "check_recommended": "گندم کے اہم مرحلوں میں پانی کی کمی پیداوار کو نقصان پہنچا سکتی ہے، اس لیے آج ہی نمی کی جانچ اہم ہے۔",
}

SYSTEM = """You are the Water Agent of KisanOS, a decision-support app for wheat farmers in Pakistan.
You receive a JSON summary produced by fixed irrigation rules. Write short farmer-friendly texts.

Rules:
- Use ONLY facts in the summary. Do NOT change the recommendation.
- Do NOT give irrigation amounts, depths or durations, and do NOT mention pesticides, fertilizer, doses, chemicals or product names.
- Do NOT diagnose any disease or pest. This is an advisory estimate; never say irrigation is certain.
- Each text: 1-2 short sentences.

Reply with ONLY a JSON object, no other text:
{"urdu_summary": "<Urdu script>", "roman_urdu": "<Urdu in English letters>",
 "audio_script_urdu": "<Urdu script, easy to read aloud, starts by addressing the farmer>",
 "farmer_explanation": "<Urdu script, why this matters>"}"""

_ARABIC = re.compile(r"[\u0600-\u06FF]")
_NO_NEED = re.compile(r"(ضرورت نہیں|ضروری نہیں|zaroorat nahi|zaroori nahi|no need)", re.I)
_DO_NOW = re.compile(r"(ابھی پانی لگائیں|abhi pani lagayein)", re.I)


def llm_available() -> bool:
    if not os.environ.get("GROQ_API_KEY"):
        return False
    try:
        import groq  # noqa: F401
        return True
    except ImportError:
        return False


def _client():
    from groq import Groq
    return Groq(timeout=20.0, max_retries=1)


def _call_llm(facts: dict) -> dict:
    client = _client()
    kwargs = dict(
        model=MODEL, temperature=0.2, max_completion_tokens=2000, reasoning_effort="low",
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}],
    )
    try:
        resp = client.chat.completions.create(**kwargs)
    except TypeError:                   # older SDK without reasoning_effort
        kwargs.pop("reasoning_effort")
        resp = client.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    return json.loads(text)


def _valid(d, status) -> bool:
    keys = ("urdu_summary", "roman_urdu", "audio_script_urdu", "farmer_explanation")
    if not isinstance(d, dict) or any(not isinstance(d.get(k), str) for k in keys):
        return False
    for k in keys:
        t = d[k].strip()
        if not 5 <= len(t) <= 350 or find_unsafe(t) or QUANTITY.search(t):
            return False
    if not all(_ARABIC.search(d[k]) for k in ("urdu_summary", "audio_script_urdu", "farmer_explanation")):
        return False
    ro = d["roman_urdu"]
    if sum(c.isascii() for c in ro) / len(ro) < 0.95:
        return False
    joined = " ".join(d[k] for k in keys)
    if status != "not_needed_now" and _NO_NEED.search(joined):
        return False
    if status == "not_needed_now" and _DO_NOW.search(joined):
        return False
    return True


def _template(facts: dict) -> dict:
    s, rain = facts["irrigation_status"], bool(facts.get("rain_deferred")) and facts["irrigation_status"] != "not_needed_now"
    ur = _UR[s] + (_UR_RAIN if rain else "")
    return {"urdu_summary": ur, "roman_urdu": _RO[s] + (_RO_RAIN if rain else ""),
            "audio_script_urdu": "کسان بھائی، " + ur, "farmer_explanation": _WHY[s]}


def build_ai_enhanced(facts: dict, use_llm: bool = True):
    """facts: irrigation_status, growth_stage, critical_stage, days_since_last_irrigation,
    hot_dry_weather, rain_deferred, soil_moisture_reported.
    Returns (ai_enhanced_dict, provider, flags)."""
    s, flags, provider, texts = facts["irrigation_status"], [], "self-hosted", None
    if use_llm and llm_available():
        try:
            raw = _call_llm(facts)
            if _valid(raw, s):
                texts, provider = {k: raw[k].strip() for k in
                                   ("urdu_summary", "roman_urdu", "audio_script_urdu", "farmer_explanation")}, "groq"
            else:
                flags.append("ai_output_rejected")
        except Exception:
            flags.append("ai_enhancement_unavailable")
    elif use_llm:
        flags.append("ai_enhancement_unavailable")      # model wanted but no key / no groq package
    if texts is None:
        texts = _template(facts)
    emoji = EMOJI[s] + ("☀️" if facts.get("hot_dry_weather") else "")
    return {**texts, "emoji_visual": emoji, "priority_level": PRIORITY[s]}, provider, flags
