from __future__ import annotations
import json
import re
from google import genai

_MODEL = "gemini-2.5-flash"
from ..models import AssistantResponse, TraceStep

_SYNTHESIS_SYSTEM = """You are a reliability engineering assistant. Your task is to answer the user's question using ONLY the data provided below.

Rules:
1. Do NOT invent machine IDs, site names, alerts, work orders, or recommendations not present in the provided data.
2. If the data is insufficient or contradictory, say so clearly and set confidence to "low".
3. For out-of-scope questions (e.g., cybersecurity, financial data), set confidence to "low" and explain this is outside the available data.
4. Always cite specific machine IDs, alert IDs, and work order IDs when making claims.
5. Return ONLY valid JSON matching the schema below — no markdown, no extra text.

Output JSON schema:
{
  "answer": "string — clear, direct answer to the question",
  "reasoning": "string — step-by-step explanation citing specific data points",
  "data_used": ["machines", "alerts", "work_orders", "maintenance_history"],
  "machines_referenced": ["M-XXX"],
  "limitations_or_assumptions": ["string"],
  "confidence": "high | medium | low",
  "capabilities_used": ["list of capability names used"]
}"""


def synthesize(
    question: str,
    context_blocks: list[str],
    trace_so_far: list[TraceStep],
    capabilities_used: list[str],
    retrieved_profiles: list[dict],
    client: genai.Client,
) -> AssistantResponse:
    context_text = "\n\n---\n\n".join(context_blocks)
    capabilities_str = ", ".join(capabilities_used) if capabilities_used else "none"
    trace_str = json.dumps([t.model_dump() for t in trace_so_far], indent=2)

    prompt = f"""{_SYNTHESIS_SYSTEM}

        User question: {question}

        Capabilities already run: {capabilities_str}

        Orchestration trace so far:
        {trace_str}

        Retrieved data:
        {context_text}

        Now produce the JSON answer:"""

    response = client.models.generate_content(model=_MODEL, contents=prompt)
    raw = response.text.strip()

    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback_response(question, raw, trace_so_far, capabilities_used)

    synth_trace = TraceStep(
        step="answer_synthesizer",
        input=f"question + {len(context_blocks)} context blocks",
        output=parsed.get("answer", "")[:200],
        data_sources=parsed.get("data_used", []),
    )

    full_trace = list(trace_so_far) + [synth_trace]

    # Narrow retrieved_profiles down to just the machines the LLM actually cited,
    # so retrieved_records matches what the answer talks about instead of dumping
    # everything that was fetched (e.g. all 10k profiles on the `prioritize` path).
    # Example: retrieved_profiles has M-006, M-104, M-201 (all fetched this run);
    # parsed["machines_referenced"] == ["M-104"] (the answer only discusses M-104)
    # -> relevant_records ends up holding just the M-104 profile.
    referenced_ids = set(parsed.get("machines_referenced", []))
    relevant_records = [
        p for p in retrieved_profiles
        if p and p.get("machine", {}).get("machineId") in referenced_ids
    ]

    return AssistantResponse(
        answer=parsed.get("answer", ""),
        reasoning=parsed.get("reasoning", ""),
        data_used=parsed.get("data_used", []),
        machines_referenced=parsed.get("machines_referenced", []),
        limitations_or_assumptions=parsed.get("limitations_or_assumptions", []),
        confidence=parsed.get("confidence", "medium"),
        capabilities_used=parsed.get("capabilities_used", capabilities_used),
        retrieved_records=relevant_records,
        trace=full_trace,
    )


def _fallback_response(
    question: str, raw: str, trace: list[TraceStep], capabilities_used: list[str]
) -> AssistantResponse:
    fallback_trace = list(trace) + [
        TraceStep(
            step="answer_synthesizer",
            input=question,
            output="JSON parse failed — returning fallback response",
        )
    ]
    return AssistantResponse(
        answer="I was unable to produce a structured answer for this question. Please rephrase or try again.",
        reasoning=f"Raw LLM output could not be parsed as JSON: {raw[:300]}",
        data_used=[],
        machines_referenced=[],
        limitations_or_assumptions=["LLM output was not valid JSON."],
        confidence="low",
        capabilities_used=capabilities_used,
        retrieved_records=[],
        trace=fallback_trace,
    )
