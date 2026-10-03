"""Fixed example output. Lets the backend/orchestrator be tested before the
real agent is wired up (Phase 1/2), and is the offline fallback for the demo."""


def dummy_result() -> dict:
    return {
        "crop_detected": "wheat",
        "visible_findings": [
            {"class": "yellowing", "detail": "older leaves, fairly uniform"},
            {"class": "rust_like_pustules", "detail": "small orange-brown dots on leaf"},
        ],
        "confidence": "Medium",
        "confidence_reason": "Example output only (dummy mode).",
    }
