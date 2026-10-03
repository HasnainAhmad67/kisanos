"""Pydantic models for the KisanOS mandatory agent JSON + the same checks the backend will run.

The format is FIXED by the team instructions - do not add, rename or retype fields.
The only extra key allowed is the optional AI_ENHANCED section.
"""
import re
import uuid
from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# Words that must never appear anywhere in the output (rule 4). Checked on every string.
_UNSAFE_LATIN = re.compile(
    r"\b(pesticides?|fungicides?|insecticides?|herbicides?|chemicals?|spray(?:ed|ing)?|doses?|dosage|"
    r"fertili[sz]ers?|urea|dap|npk|potash|nitrogen|ml per|kg per|imidacloprid|propiconazole|tebuconazole|"
    r"khaad|khad|dawai|dawa|keera\s?maar)\b", re.I)
_UNSAFE_URDU = re.compile(r"(کیڑے\s?مار|سپرے|کھاد|زرعی دوا|دوائی|ڈوز)")
# Invented irrigation amounts / durations are not allowed either (PRD: evidence + checks, no prescriptions).
QUANTITY = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mm|cm|inch|inches|litres?|liters?|gallons?|cusecs?|acre[- ]?inch)\b"
    r"|\b(?:irrigat\w*|water\w*|flood\w*)\s+(?:for\s+)?\d+\s*(?:hours?|hrs?|minutes?)\b", re.I)


def find_unsafe(text: str):
    """Return the first banned keyword found in text, or None."""
    m = _UNSAFE_LATIN.search(text) or _UNSAFE_URDU.search(text)
    return m.group(0) if m else None


def _iso(v: str) -> str:
    if not ISO_Z.match(v):
        raise ValueError("must look like 2026-10-03T12:00:00Z")
    datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ")
    return v


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: str
    url: str
    publisher: str
    retrieved_at: str
    source_status: Literal["official", "supporting", "secondary", "unverified"]

    @field_validator("url")
    @classmethod
    def _url(cls, v):
        if not v.startswith(("http://", "https://")):
            raise ValueError("url must start with http:// or https://")
        return v

    @field_validator("retrieved_at")
    @classmethod
    def _ts(cls, v):
        return _iso(v)


class AIEnhanced(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    urdu_summary: str
    roman_urdu: str
    audio_script_urdu: str
    emoji_visual: str
    farmer_explanation: str
    priority_level: Literal["high", "medium", "low"]


class AgentOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    agent_id: Literal["weather", "water", "vision", "crop", "market"]
    assessment_id: str
    status: Literal["complete", "partial", "unavailable", "error"]
    summary: str
    observations: List[str]
    possible_causes: List[str]
    checks: List[str]
    evidence_band: Literal["low", "medium", "high", "not_calibrated"]
    evidence_reason: str
    sources: List[Source]
    provider_or_model: str
    version: str
    created_at: str
    safety_flags: List[str]
    AI_ENHANCED: Optional[AIEnhanced] = None          # optional section

    @field_validator("assessment_id")
    @classmethod
    def _uuid(cls, v):
        uuid.UUID(v)
        return v

    @field_validator("created_at")
    @classmethod
    def _ts(cls, v):
        return _iso(v)

    @field_validator("summary")
    @classmethod
    def _summary(cls, v):
        if not v.strip():
            raise ValueError("summary must not be empty")
        return v


def _strings(obj, path=""):
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _strings(v, f"{path}[{i}]")


def validate_output(data: dict) -> list:
    """Backend-style check. Returns a list of problems ([] = passes)."""
    problems = []
    try:
        AgentOutput.model_validate(data)
    except ValidationError as e:
        problems += [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
    for path, s in _strings(data):
        bad = find_unsafe(s)
        if bad:
            problems.append(f"{path}: banned keyword '{bad}'")
    return problems
