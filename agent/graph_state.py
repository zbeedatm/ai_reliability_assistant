from __future__ import annotations
from typing import Annotated, TypedDict
from operator import add
from .models import TraceStep, PlannerOutput, AssistantResponse


class GraphState(TypedDict):
    question: str
    plan: PlannerOutput | None

    # Annotated[list, add] = each node appends its slice; LangGraph merges atomically.
    # This is what makes parallel branches safe: reliability_analyzer and
    # maintenance_analyzer both write context_blocks in the same superstep without
    # overwriting each other.
    context_blocks: Annotated[list[str], add]
    trace: Annotated[list[TraceStep], add]
    retrieved_profiles: Annotated[list[dict], add]
    capabilities_used: Annotated[list[str], add]

    # Single-writer fields (last-write-wins, no reducer needed)
    response: AssistantResponse | None
    not_found_machine: str | None
    planner_parse_failed: bool
