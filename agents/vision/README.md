# Vision Agent (Member 3)  -  kisanos/agents/vision/

Looks at a wheat photo and reports ONLY what is visible. It never confirms a disease
and never gives chemical advice. Confidence is capped at Medium.

## Pipeline
1. Quality gate (quality.py): blur, brightness, resolution, plant present, file size.
   If it fails -> status "needs_better_photo" with farmer tips (Crop Agent then falls back to text symptoms).
2. Analysis: free multimodal API (llm.py: Gemini or Groq) when GEMINI_API_KEY or GROQ_API_KEY is set,
   otherwise dummy output (dummy.py). Also falls back to dummy if the API call fails.
3. Safety sanitiser (agent.py): allowed classes only, no High confidence, no pesticide/dose words,
   rust-like or unclear or Low confidence -> expert_referral = true, non-wheat -> "unsupported".

## Output = the team's MANDATORY JSON (contract.py, schema.json, schema.py)
`analyze_image()` always returns exactly these 14 keys (never raises; on any failure it returns status "error"):
agent_id, assessment_id, status, summary, observations, possible_causes, checks, evidence_band,
evidence_reason, sources, provider_or_model, version, created_at, safety_flags.

| Situation | status | evidence_band | provider_or_model | safety_flags (examples) |
|---|---|---|---|---|
| Real model analysis | complete | low / medium (never high) | gemini / groq | expert_referral_recommended, confidence_capped |
| Dummy mode or model call failed | partial | not_calibrated | self-hosted | example_output_not_real_analysis |
| Photo failed quality gate | unavailable | not_calibrated | self-hosted | photo_quality_failed |
| Not a wheat photo | unavailable | not_calibrated | gemini / groq | non_wheat_image |
| Unexpected exception | error | not_calibrated | self-hosted | agent_exception |

Blocked chemical/pesticide/dose text is removed and recorded as `unsafe_advice_removed`.
The old internal fields (expert_referral, message_for_farmer, image_quality, ...) are folded into
`checks`, `observations`, `summary` and `safety_flags`; no extra top-level keys are added.

## Use
    from agents.vision import analyze_image
    result = analyze_image(open("leaf.jpg", "rb").read(), growth_stage="tillering",
                           assessment_id="<uuid from the orchestrator, optional>")

Dummy mode (no API key, for backend testing):  analyze_image(bytes, force_dummy=True)

## Setup
    pip install -r agents/vision/requirements.txt     # numpy + Pillow; pydantic only for schema.py
    # Choose ONE free provider (no credit card needed):
    export GEMINI_API_KEY=...     # free key: https://aistudio.google.com/apikey
    export GROQ_API_KEY=...       # free key: https://console.groq.com/keys
    # Optional:
    export LLM_PROVIDER=groq              # force one if both keys are set (default: gemini)
    export GEMINI_MODEL=gemini-2.5-flash  # default
    export GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct   # default (vision model)

Free-tier notes: limits are per minute / per day and model names change, so if you get an
HTTP 404 or 429 check the provider's model list and rate-limit page. Google may use free-tier
prompts to improve its products, so do not send private farm photos you cannot share.

## Test
    python -m agents.vision.test_agent      (run from the kisanos/ folder)

## FastAPI hook (for the backend/Tech Lead)
    from fastapi import FastAPI, UploadFile, Form
    from agents.vision import analyze_image
    app = FastAPI()

    @app.post("/vision")
    async def vision(photo: UploadFile, growth_stage: str | None = Form(None)):
        return analyze_image(await photo.read(), growth_stage)

## Notes
- schema.json is now the FINAL team contract v1.0 (no longer a draft).
- TODO: fill SOURCE_REGISTRY (S1, S2) in contract.py from the research report; until then only the model entry appears in `sources`.
- Quality thresholds in quality.py are starting values; tune them on real photos.
- The tests use synthetic images. The real-model mode (llm.py) was tested only with mocked API replies,
  not a live Gemini/Groq key - run it once with your key and 20-30 real wheat photos before the demo.
