"""Run:  python -m agents.vision.test_agent   (from the kisanos/ folder)  or  pytest agents/vision"""
import io
import numpy as np
from PIL import Image, ImageFilter

from .agent import analyze_image, run_analysis
from .contract import validate_contract, REQUIRED_KEYS
from .quality import check_quality


def _png(arr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype("uint8")).save(buf, "PNG")
    return buf.getvalue()


def _leafy(w=1000, h=800, seed=1):
    """Synthetic 'green leaf field': green base + strong texture (sharp)."""
    rng = np.random.default_rng(seed)
    base = np.zeros((h, w, 3))
    base[..., 0], base[..., 1], base[..., 2] = 70, 140, 50
    noise = rng.normal(0, 28, (h, w, 1))
    return np.clip(base + noise, 0, 255)


def test_good_photo_passes_and_dummy_output_valid():
    r = run_analysis(_png(_leafy()), force_dummy=True)
    assert r["status"] == "ok" and r["image_quality"]["passed"]
    assert r["confidence"] in ("Low", "Medium")           # never High
    assert r["expert_referral"] is True                   # dummy has rust-like pustules
    assert r["limitations"].startswith("Photo shows visible signs only")


def test_blurry_photo_rejected():
    img = Image.fromarray(_leafy().astype("uint8")).filter(ImageFilter.GaussianBlur(12))
    buf = io.BytesIO(); img.save(buf, "PNG")
    r = run_analysis(buf.getvalue(), force_dummy=True)
    assert r["status"] == "needs_better_photo" and "blurry" in r["image_quality"]["issues"]


def test_dark_photo_rejected():
    r = run_analysis(_png(_leafy() * 0.1), force_dummy=True)
    assert r["status"] == "needs_better_photo" and "too_dark" in r["image_quality"]["issues"]


def test_non_plant_photo_rejected():
    rng = np.random.default_rng(3)
    grey = np.clip(rng.normal(128, 40, (800, 1000, 1)).repeat(3, axis=2), 0, 255)
    r = run_analysis(_png(grey), force_dummy=True)
    assert "no_plant" in r["image_quality"]["issues"]


def test_tiny_and_corrupt_files_rejected():
    assert "low_resolution" in check_quality(_png(_leafy(200, 150)))["issues"]
    assert check_quality(b"not an image")["issues"] == ["bad_file"]


def test_sanitizer_blocks_unsafe_model_output():
    from .agent import _sanitize, _base
    raw = {"crop_detected": "wheat", "confidence": "High",
           "confidence_reason": "Spray propiconazole 200 ml per acre",
           "visible_findings": [{"class": "yellowing", "detail": "use urea 50 kg"},
                                {"class": "made_up_disease", "detail": "x"}]}
    r = _sanitize(raw, _base("ok"))
    assert r["confidence"] == "Low"                       # High is not allowed
    assert "propiconazole" not in r["confidence_reason"].lower()
    assert all(f["class"] != "made_up_disease" for f in r["visible_findings"])
    assert "urea" not in r["visible_findings"][0]["detail"].lower()


def test_not_wheat_is_unsupported():
    from .agent import _sanitize, _base
    r = _sanitize({"crop_detected": "not_wheat"}, _base("ok"))
    assert r["status"] == "unsupported" and r["visible_findings"] == []


def _fake_post(reply_builder):
    """Return a replacement for llm._post_json that records the request."""
    calls = []
    def fake(url, payload, headers):
        calls.append((url, payload, headers))
        return reply_builder
    return fake, calls


MODEL_JSON = ('{"crop_detected": "wheat", "visible_findings": [{"class": "yellowing", '
              '"detail": "older leaves"}], "confidence": "Medium", "confidence_reason": "clear photo"}')


def test_gemini_path_with_mocked_api():
    import os
    from . import llm
    os.environ.pop("GROQ_API_KEY", None); os.environ.pop("LLM_PROVIDER", None)
    os.environ["GEMINI_API_KEY"] = "test-key"
    reply = {"candidates": [{"content": {"parts": [{"text": "```json\n" + MODEL_JSON + "\n```"}]}}]}
    fake, calls = _fake_post(reply)
    real, llm._post_json = llm._post_json, fake
    try:
        r = run_analysis(_png(_leafy()))
    finally:
        llm._post_json = real; os.environ.pop("GEMINI_API_KEY")
    assert r["status"] == "ok" and r["mode"] == "llm"
    assert r["visible_findings"][0]["class"] == "yellowing"
    url, payload, headers = calls[0]
    assert "generativelanguage.googleapis.com" in url and headers["x-goog-api-key"] == "test-key"
    assert payload["contents"][0]["parts"][0]["inline_data"]["mime_type"] == "image/jpeg"


def test_groq_path_with_mocked_api():
    import os
    from . import llm
    os.environ.pop("GEMINI_API_KEY", None); os.environ.pop("LLM_PROVIDER", None)
    os.environ["GROQ_API_KEY"] = "test-key"
    reply = {"choices": [{"message": {"content": "Here you go: " + MODEL_JSON}}]}
    fake, calls = _fake_post(reply)
    real, llm._post_json = llm._post_json, fake
    try:
        r = run_analysis(_png(_leafy()))
    finally:
        llm._post_json = real; os.environ.pop("GROQ_API_KEY")
    assert r["status"] == "ok" and r["mode"] == "llm"
    url, payload, headers = calls[0]
    assert "api.groq.com" in url and headers["Authorization"] == "Bearer test-key"
    assert payload["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_api_failure_falls_back_to_dummy():
    import os
    from . import llm
    os.environ["GROQ_API_KEY"] = "test-key"
    def boom(url, payload, headers):
        raise RuntimeError("HTTP 429 from model API")
    real, llm._post_json = llm._post_json, boom
    try:
        r = run_analysis(_png(_leafy()))
    finally:
        llm._post_json = real; os.environ.pop("GROQ_API_KEY")
    assert r["status"] == "ok" and r["mode"] == "dummy_fallback"
    assert "Model call failed" in r["confidence_reason"]


def test_no_key_means_dummy_mode():
    import os
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "LLM_PROVIDER"):
        os.environ.pop(k, None)
    assert run_analysis(_png(_leafy()))["mode"] == "dummy"



# ---------------- final team-contract tests (mandatory JSON) ----------------
def _with_mock_llm(model_json, provider="gemini"):
    """Run analyze_image with a mocked Gemini/Groq reply; returns the contract dict."""
    import os
    from . import llm
    for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "LLM_PROVIDER"):
        os.environ.pop(k, None)
    if provider == "gemini":
        os.environ["GEMINI_API_KEY"] = "k"
        reply = {"candidates": [{"content": {"parts": [{"text": model_json}]}}]}
    else:
        os.environ["GROQ_API_KEY"] = "k"
        reply = {"choices": [{"message": {"content": model_json}}]}
    real, llm._post_json = llm._post_json, (lambda u, p, h: reply)
    try:
        return analyze_image(_png(_leafy()))
    finally:
        llm._post_json = real
        for k in ("GEMINI_API_KEY", "GROQ_API_KEY"):
            os.environ.pop(k, None)


def test_contract_llm_complete():
    r = _with_mock_llm(MODEL_JSON)
    assert validate_contract(r) == [], validate_contract(r)
    assert list(r.keys()) == list(REQUIRED_KEYS)                 # exact keys, exact order
    assert r["agent_id"] == "vision" and r["status"] == "complete"
    assert r["evidence_band"] == "medium" and r["provider_or_model"] == "gemini"
    assert r["observations"] and r["possible_causes"] and r["checks"]
    assert r["sources"] and r["sources"][0]["source_status"] == "unverified"
    assert r["version"] == "1.0" and r["created_at"].endswith("Z")


def test_contract_groq_provider():
    r = _with_mock_llm(MODEL_JSON, "groq")
    assert validate_contract(r) == [] and r["provider_or_model"] == "groq"


def test_contract_dummy_is_partial_not_calibrated():
    r = analyze_image(_png(_leafy()), force_dummy=True)
    assert validate_contract(r) == []
    assert r["status"] == "partial" and r["evidence_band"] == "not_calibrated"
    assert r["provider_or_model"] == "self-hosted"
    assert "example_output_not_real_analysis" in r["safety_flags"]
    assert r["summary"].startswith("EXAMPLE OUTPUT ONLY")


def test_contract_api_failure_is_partial():
    import os
    from . import llm
    os.environ["GROQ_API_KEY"] = "k"
    def boom(u, p, h):
        raise RuntimeError("HTTP 429")
    real, llm._post_json = llm._post_json, boom
    try:
        r = analyze_image(_png(_leafy()))
    finally:
        llm._post_json = real; os.environ.pop("GROQ_API_KEY")
    assert validate_contract(r) == [] and r["status"] == "partial"
    assert r["evidence_band"] == "not_calibrated"


def test_contract_bad_photo_is_unavailable():
    r = analyze_image(_png(_leafy() * 0.1), force_dummy=True)
    assert validate_contract(r) == []
    assert r["status"] == "unavailable" and r["evidence_band"] == "not_calibrated"
    assert "photo_quality_failed" in r["safety_flags"] and r["sources"] == []
    assert any("too dark" in o for o in r["observations"]) and r["checks"]


def test_contract_corrupt_file_is_unavailable():
    r = analyze_image(b"not an image")
    assert validate_contract(r) == [] and r["status"] == "unavailable"


def test_contract_non_wheat_is_unavailable():
    r = _with_mock_llm('{"crop_detected": "not_wheat", "visible_findings": []}')
    assert validate_contract(r) == [] and r["status"] == "unavailable"
    assert "non_wheat_image" in r["safety_flags"]


def test_contract_blocks_unsafe_text_and_flags_it():
    unsafe = ('{"crop_detected": "wheat", "confidence": "High", '
              '"confidence_reason": "Spray propiconazole 200 ml per acre", '
              '"visible_findings": [{"class": "yellowing", "detail": "use urea 50 kg"}, '
              '{"class": "made_up_disease", "detail": "x"}]}')
    r = _with_mock_llm(unsafe)
    assert validate_contract(r) == []                          # no forbidden keyword anywhere
    assert r["evidence_band"] == "low"                         # High never passes through
    for f in ("unsafe_advice_removed", "confidence_capped", "unrecognised_label_dropped"):
        assert f in r["safety_flags"], f


def test_contract_never_raises_returns_error_status():
    r = analyze_image(None)                                    # garbage in
    assert validate_contract(r) == [] and r["status"] == "error"
    assert r["safety_flags"] == ["agent_exception"]


def test_assessment_id_passthrough_and_fallback():
    uid = "123e4567-e89b-12d3-a456-426614174000"
    assert analyze_image(_png(_leafy()), force_dummy=True, assessment_id=uid)["assessment_id"] == uid
    r = analyze_image(_png(_leafy()), force_dummy=True, assessment_id="not-a-uuid")
    assert validate_contract(r) == [] and r["assessment_id"] != "not-a-uuid"


def test_validator_catches_contract_breaks():
    good = analyze_image(_png(_leafy()), force_dummy=True)
    assert validate_contract({**good, "status": "ok"})
    assert validate_contract({**good, "evidence_band": "High"})
    assert validate_contract({**good, "created_at": "2026-10-03 12:00:00"})
    assert validate_contract({**good, "observations": "text"})
    assert validate_contract({**good, "extra": 1})
    bad = dict(good); bad.pop("safety_flags")
    assert validate_contract(bad)
    assert validate_contract({**good, "checks": ["apply fungicide"]})
    assert validate_contract({**good, "sources": [{"title": "t"}]})


def test_schema_json_matches_output():
    import json, os
    try:
        import jsonschema
    except ImportError:
        print("  (jsonschema not installed - skipped)"); return
    schema = json.load(open(os.path.join(os.path.dirname(__file__), "schema.json")))
    jsonschema.validate(analyze_image(_png(_leafy()), force_dummy=True), schema)
    jsonschema.validate(_with_mock_llm(MODEL_JSON), schema)


def test_pydantic_model_matches_output():
    try:
        from .schema import VisionOutput
    except ImportError:
        print("  (pydantic not installed - skipped)"); return
    VisionOutput(**analyze_image(_png(_leafy()), force_dummy=True))
    VisionOutput(**_with_mock_llm(MODEL_JSON))


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t(); print("PASS", t.__name__)
    print(f"{len(tests)} tests passed")
