"""Pydantic models for the final team contract (same pattern as agents/weather/schema.py).

Needs:  pip install pydantic>=2
agent.py does NOT import this file at runtime (it validates with contract.validate_contract,
which needs no extra package). Use this for the backend / local checks:

    from agents.vision.schema import VisionOutput
    VisionOutput(**analyze_image(photo_bytes))     # raises ValidationError if the contract is broken
"""
from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

ISO_Z = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
UUID_RE = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1)
    url: str = Field(pattern=r"^https?://")
    publisher: str = Field(min_length=1)
    retrieved_at: str = Field(pattern=ISO_Z)
    source_status: Literal["official", "supporting", "secondary", "unverified"]


class VisionOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_id: Literal["vision"]
    assessment_id: str = Field(pattern=UUID_RE)
    status: Literal["complete", "partial", "unavailable", "error"]
    summary: str = Field(min_length=1)
    observations: List[str]
    possible_causes: List[str]
    checks: List[str]
    evidence_band: Literal["low", "medium", "high", "not_calibrated"]
    evidence_reason: str = Field(min_length=1)
    sources: List[Source]
    provider_or_model: Literal["open-meteo", "gemini", "groq", "amis", "self-hosted"]
    version: str = Field(min_length=1)
    created_at: str = Field(pattern=ISO_Z)
    safety_flags: List[str]
