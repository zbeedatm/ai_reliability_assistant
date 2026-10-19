from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


class TraceStep(BaseModel):
    step: str
    input: str | None = None
    output: str
    data_sources: list[str] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    intent: Literal["prioritize", "machine_specific", "site_compare", "fault_filter", "out_of_scope"]
    capabilities: list[str]
    entities: dict[str, list[str]] = Field(default_factory=dict)
    reasoning: str = ""


class AssistantResponse(BaseModel):
    answer: str
    reasoning: str
    data_used: list[str] = Field(default_factory=list)
    machines_referenced: list[str] = Field(default_factory=list)
    limitations_or_assumptions: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"
    capabilities_used: list[str] = Field(default_factory=list)
    retrieved_records: list[dict] = Field(default_factory=list)
    trace: list[TraceStep] = Field(default_factory=list)
