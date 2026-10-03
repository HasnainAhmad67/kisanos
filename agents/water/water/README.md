# Water Agent  -  kisanos/agents/water/

Tells a wheat farmer whether irrigation may be needed soon, from days since the last irrigation,
growth stage and the Weather Agent's forecast. Returns the team's MANDATORY JSON format (unchanged).

The irrigation status comes from fixed rules inside agent.py - never from an LLM. The agent gives an
advisory estimate and safe field checks only: no irrigation amounts, no chemicals, no doses.

## Files
    agent.py          main logic + rule engine -> returns the mandatory JSON
    schema.py         Pydantic models + validate_output() = the same checks the backend runs
    ai_enhance.py     OPTIONAL AI_ENHANCED section (Groq openai/gpt-oss-120b, Urdu / Roman Urdu / audio)
    test_agent.py     local tests (15)
    requirements.txt  pydantic (required), groq + python-dotenv (optional)

## Run
    pip install -r requirements.txt
    python -c "from agent import analyze_water, DEMO_INPUT; print(analyze_water(**DEMO_INPUT))"
    # from the kisanos/ folder:  from agents.water.agent import analyze_water

    result = analyze_water("wheat", "Bahawalpur", 5, growth_stage="flowering",
                           weather={"rain_next_3d_mm": 0, "avg_max_temp_c": 34, "et0_mm_per_day": 5.5})

Inputs: crop, area, last_irrigation_days (required); growth_stage OR days_after_sowing, weather
(Weather Agent: rain_next_3d_mm, avg_max_temp_c, et0_mm_per_day), soil_moisture ("dry"/"moist"/"wet"),
enhance, use_llm, assessment_id (all optional).

## Test
    python test_agent.py                      (from this folder)
    python -m agents.water.test_agent         (from the kisanos/ folder)

## .env (never push it to GitHub)
    GROQ_API_KEY=your_groq_key_here           # only needed for AI_ENHANCED with the real model
    WATER_MODEL=openai/gpt-oss-120b           # optional, this is the default
No key? analyze_water(..., enhance=True) still works: fixed Urdu templates are used.

## How the mandatory fields are filled
- agent_id "water" | assessment_id uuid4 (or the one you pass) | version "1.0" | created_at UTC ISO-8601 with Z
- status: complete (stage, irrigation gap and weather all given) | partial (something missing/conflicting,
  e.g. no weather, unknown stage) | unavailable (not wheat, or area / last irrigation missing) | error (invalid numbers or internal error)
- evidence_band: medium (complete) | low (partial) | not_calibrated (unavailable / error). Never "high":
  the day limits are derived from sources and are not field-calibrated. evidence_reason says why.
- summary: starts with "Irrigation status: not_needed_now | may_be_needed_soon | check_recommended."
  The mandatory format has no separate field for this verdict. Read it with get_irrigation_status(result).
- observations = facts used; possible_causes = factors behind the result; checks = safe farmer checks.
- sources: official / supporting entries, each with title, url, publisher, retrieved_at, source_status.
- provider_or_model: "self-hosted" (rules) or "groq" (only when the Groq text was actually used).
- safety_flags: expert_referral_recommended | ai_output_rejected | ai_enhancement_unavailable |
  unsafe_content_blocked | agent_internal_error. (Flag names avoid the banned keywords on purpose.)

## Safety
- Banned words (pesticide, spray, dose, fertilizer, urea, ... also Urdu / Roman Urdu) are checked on every
  string of the output. If anything slips through, the agent returns status "error" with a safe message.
- The model sees only a small structured summary (no free text typed by the farmer). Its reply is rejected
  if it has banned words, irrigation amounts/durations, or contradicts the rule result -> templates are used.
- The agent never raises; bad input becomes a valid JSON.

## Notes / limits
- Stage names and "critical stage" come from the sources. The day limits (14/21 days critical stages,
  21/28 otherwise) are DERIVED from the spacing of recommended irrigations; the rain (10 mm), ET0 (5 mm/day)
  and temperature (35 C) limits are tunable defaults. Validate them with an agronomist.
- Sources are Sindh (W1), Indian Punjab (W2), Faisalabad (W3) and Open-Meteo docs (W4); none is Bahawalpur-specific.
  Open each link and confirm the titles before you submit. retrieved_at is the date they were looked up.
- Rules assume irrigated (canal / tubewell) wheat. Rain-fed wheat and soil type are out of scope.
- The live Groq call (ai_enhance.py) was tested only with a fake client, not the real API. Run it once with your key.
- Urdu / Roman Urdu templates should be read by a native speaker before the demo.
