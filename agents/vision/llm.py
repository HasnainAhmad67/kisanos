"""Multimodal-model call using a FREE API: Google Gemini or Groq.

No extra packages needed (uses only the standard library + Pillow, which the
quality gate already needs).

Pick a provider by setting ONE of these environment variables:
    GEMINI_API_KEY   -> Google Gemini  (free key: https://aistudio.google.com/apikey)
    GROQ_API_KEY     -> Groq           (free key: https://console.groq.com/keys)
If both are set, Gemini is used. Force one with LLM_PROVIDER=gemini|groq.

Optional model overrides (free-tier model names change over time):
    GEMINI_MODEL   (default gemini-2.5-flash)
    GROQ_MODEL     (default meta-llama/llama-4-scout-17b-16e-instruct, vision-capable)

Returns the model's raw JSON dict; agent.py sanitises it afterwards, so
nothing returned here is trusted.
"""
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request

from PIL import Image, ImageOps

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
TIMEOUT_S = 60
MAX_SIDE = 1280          # downscale before upload (Groq limits base64 images to ~4 MB)

PROMPT = """You are the Vision Agent of KisanOS, a decision-support app for wheat farmers in Pakistan.
Look at the photo and describe ONLY what is visibly there. Do NOT diagnose a disease,
do NOT name insect species, nutrient deficiencies, pesticides, doses or fertilizer rates.

Allowed visible classes (use these exact names):
healthy_looking, yellowing, rust_like_pustules, spots_or_blotches, visible_insects, drying, unclear

Rules:
- If the photo is not wheat, set crop_detected to "not_wheat" and visible_findings to [].
- For yellowing, say whether older or younger leaves, and uniform or patchy, if you can see it.
- For rust_like_pustules, mention the colour (yellow-orange, reddish-brown, black) if visible.
- 'healthy_looking' means only 'no visible symptoms', never 'the crop is healthy'.
- confidence is "Low" or "Medium" only. Give a one-line reason.
{stage_line}
Reply with ONLY a JSON object, no other text:
{{"crop_detected": "wheat" or "not_wheat",
  "visible_findings": [{{"class": "...", "detail": "..."}}],
  "confidence": "Low" or "Medium",
  "confidence_reason": "..."}}"""


def provider():
    """Return 'gemini', 'groq' or None, based on environment variables."""
    forced = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if forced == "gemini":
        return "gemini" if os.environ.get("GEMINI_API_KEY") else None
    if forced == "groq":
        return "groq" if os.environ.get("GROQ_API_KEY") else None
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("GROQ_API_KEY"):
        return "groq"
    return None


def llm_available() -> bool:
    return provider() is not None


def _prepare_jpeg(image_bytes: bytes) -> str:
    """Fix rotation, shrink, re-encode as JPEG -> base64 string."""
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _post_json(url: str, payload: dict, headers: dict) -> dict:
    """POST JSON; retry once on rate-limit / overload (429, 503)."""
    body = json.dumps(payload).encode()
    for attempt in (1, 2):
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json", "User-Agent": "kisanos-vision/1.0", **headers})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt == 1:
                time.sleep(3)
                continue
            detail = e.read().decode(errors="replace")[:200]
            raise RuntimeError(f"HTTP {e.code} from model API: {detail}") from e


def _call_gemini(b64: str, prompt: str) -> str:
    data = _post_json(
        GEMINI_URL.format(model=GEMINI_MODEL),
        {"contents": [{"parts": [
            {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
            {"text": prompt}]}],
         "generationConfig": {"temperature": 0, "maxOutputTokens": 800,
                              "responseMimeType": "application/json"}},
        {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def _call_groq(b64: str, prompt: str) -> str:
    data = _post_json(
        GROQ_URL,
        {"model": GROQ_MODEL, "temperature": 0, "max_tokens": 600,
         "response_format": {"type": "json_object"},
         "messages": [{"role": "user", "content": [
             {"type": "text", "text": prompt},
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]},
        {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"})
    return data["choices"][0]["message"]["content"]


def _parse_json(text: str) -> dict:
    """Tolerate ```json fences or extra words around the JSON object."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


def analyze_with_llm(image_bytes: bytes, growth_stage=None) -> dict:
    which = provider()
    if which is None:
        raise RuntimeError("No GEMINI_API_KEY or GROQ_API_KEY set")
    stage_line = f"The farmer says the growth stage is: {growth_stage}." if growth_stage else ""
    prompt = PROMPT.format(stage_line=stage_line)
    b64 = _prepare_jpeg(image_bytes)
    text = _call_gemini(b64, prompt) if which == "gemini" else _call_groq(b64, prompt)
    return _parse_json(text)
