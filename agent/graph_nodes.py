"""
LangGraph node factories and router functions.

Each node is a pure function (GraphState → partial state dict).
Factories inject dependencies via closure so nodes remain plain callables
(no LangGraph-specific boilerplate inside the function body).

Router functions return a node name (str) or list of node names (parallel fan-out).
"""
from __future__ import annotations
import json
import re
from google import genai

_MODEL = "gemini-2.5-flash"

from .graph_state import GraphState
from .models import TraceStep, PlannerOutput
from .capabilities.data_retriever import DataRetriever
from .capabilities import reliability_analyzer as ra
from .capabilities import maintenance_analyzer as ma
from .capabilities import site_comparator as sc
from .capabilities.answer_synthesizer import synthesize
from vector_store import _machine_doc_text

# --------------------------------------------------------------------------- #
# Planner                                                                      #
# --------------------------------------------------------------------------- #

_PLANNER_SYSTEM = """You are the planning layer of an industrial reliability assistant.

Given a user question, classify the intent and decide which capabilities to invoke.

Available intents:
- "prioritize": user wants to know which machines to work on first (week planning, highest risk)
- "machine_specific": user asks about a specific machine (M-XXX)
- "site_compare": user wants to compare two or more sites
- "fault_filter": user asks about a category of fault/condition (heat, vibration, bearing, etc.)
- "out_of_scope": question cannot be answered with machine reliability data

Return ONLY valid JSON — no markdown, no extra text:
{
  "intent": "one of the 5 intents above",
  "capabilities": ["list of capabilities needed"],
  "entities": {
    "machines": ["M-XXX if mentioned"],
    "sites": ["Site X if mentioned"],
    "fault_types": ["heat/vibration/bearing etc if mentioned"]
  },
  "reasoning": "one sentence explaining the decision"
}"""


def make_planner_node(client: genai.Client):
    def planner_node(state: GraphState) -> dict:
        """
        Always the entry node (START -> planner). Classifies the question into one of
        5 intents and extracts entities (machine IDs, sites, fault types) via a single
        Gemini call. The resulting `plan` drives every downstream routing decision in
        route_after_planner. On a JSON parse failure, fails safe: routes as
        `out_of_scope` (short-circuits straight to synthesizer, no capability calls)
        rather than guessing a data-heavy intent like `fault_filter`. `planner_parse_failed`
        is set so synthesizer_node can give an honest "couldn't interpret this" message
        instead of the generic out-of-scope refusal.
        """
        question = state["question"]
        response = client.models.generate_content(model=_MODEL, contents=f"{_PLANNER_SYSTEM}\n\nUser question: {question}\n\nJSON:")
        # strips a leading fence, optionally followed by the json language tag (e.g. ```json\n).
        raw = re.sub(r"^```(?:json)?\s*", "", response.text.strip())
        # strips a trailing fence at the end of the response.
        raw = re.sub(r"\s*```$", "", raw)
        planner_parse_failed = False
        try:
            plan = PlannerOutput(**json.loads(raw))
        except Exception:
            planner_parse_failed = True
            plan = PlannerOutput(
                intent="out_of_scope",
                capabilities=[],
                entities={},
                reasoning="Planner fallback — could not parse structured plan.",
            )
        return {
            "plan": plan,
            "planner_parse_failed": planner_parse_failed,
            "trace": [TraceStep(
                step="planner",
                input=question,
                output=f"intent={plan.intent}, entities={plan.entities}. {plan.reasoning}",
            )],
        }
    return planner_node


# --------------------------------------------------------------------------- #
# Routers (conditional edge functions)                                         #
# --------------------------------------------------------------------------- #

def route_after_planner(state: GraphState) -> str:
    """
    Routes to the first capability node for the detected intent.
    Returning a string names the next node; LangGraph follows the path_map.
    """
    intent = state["plan"].intent
    return {
        "out_of_scope":     "synthesizer",
        "machine_specific": "retrieve_machine",
        "prioritize":       "retrieve_all",
        "site_compare":     "site_comparator",
        "fault_filter":     "semantic_search",
    }.get(intent, "semantic_search")


def route_after_retrieve_machine(state: GraphState) -> str:
    """Short-circuits to synthesizer when a machine ID is not found."""
    return "synthesizer" if state.get("not_found_machine") else "maintenance_analyzer"


# --------------------------------------------------------------------------- #
# Data retrieval nodes                                                         #
# --------------------------------------------------------------------------- #

def make_retrieve_machine_node(retriever: DataRetriever):
    def retrieve_machine_node(state: GraphState) -> dict:
        """
        machine_specific path. Resolves machine IDs from the planner's entities, falling
        back to a regex scan of the raw question (e.g. "M-104") if the planner missed
        them. Validates each ID against the dataset before doing anything else — if any
        ID doesn't exist, sets `not_found_machine` so route_after_retrieve_machine
        short-circuits straight to synthesizer instead of wasting an analyzer call and
        an LLM synthesis call on data that isn't there.
        On success, builds both structured profiles (for maintenance_analyzer) and text
        context_blocks (for synthesizer's grounding context).
        """
        plan = state["plan"]
        machine_ids = plan.entities.get("machines") or [
            m.upper() for m in re.findall(r"[Mm]-\d+", state["question"])
        ]
        for mid in machine_ids:
            #  validation stops a request before it ever reaches the LLM, so there's nothing to hallucinate about.
            if not retriever.validate_machine(mid):
                return {
                    "not_found_machine": mid,
                    "trace": [TraceStep(
                        step="data_retriever",
                        input=mid,
                        output=f"Machine {mid} not found in dataset.",
                        data_sources=["machines"],
                    )],
                    "capabilities_used": ["data_retriever"],
                }

        profiles = [retriever.get_machine_profile(mid) for mid in machine_ids]
        return {
            "retrieved_profiles": profiles,
            "context_blocks": [_machine_doc_text(p) for p in profiles],
            "trace": [TraceStep(
                step="data_retriever",
                input=f"machine lookup: {machine_ids}",
                output=f"Found profiles for: {machine_ids}",
                data_sources=["machines", "alerts", "work_orders", "maintenance_history"],
            )],
            "capabilities_used": ["data_retriever"],
        }
    return retrieve_machine_node


def make_retrieve_all_node(retriever: DataRetriever):
    def retrieve_all_node(state: GraphState) -> dict:
        """
        prioritize path. Fetches every machine profile in the dataset — this is the
        fan-out point in orchestrator.py: its output (`retrieved_profiles`) feeds BOTH
        reliability_analyzer and maintenance_analyzer in the same superstep, which run
        concurrently. That's why this node doesn't build `context_blocks` itself; each
        analyzer builds its own slice from `retrieved_profiles` and they merge via the
        `Annotated[list, add]` reducer (see graph_state.py).
        """
        profiles = retriever.get_all_machines()
        return {
            "retrieved_profiles": profiles,
            "trace": [TraceStep(
                step="data_retriever",
                input="all machines",
                output=f"Retrieved {len(profiles)} machine profiles",
                data_sources=["machines", "alerts", "work_orders", "maintenance_history"],
            )],
            "capabilities_used": ["data_retriever"],
        }
    return retrieve_all_node


def make_semantic_search_node(retriever: DataRetriever):
    def semantic_search_node(state: GraphState) -> dict:
        """
        fault_filter path. Runs a vector similarity search over the embedded machine
        documents to find candidates matching a fault description (e.g. "bearing
        issues"), then re-fetches each hit's full structured profile from the
        retriever. That re-fetch matters: the embedding text is for ranking relevance,
        not for scoring — reliability_analyzer downstream needs typed fields
        (healthStatus, criticality, alert status), not a flattened text blob.
        Feeds only into reliability_analyzer (no maintenance_analyzer in this path).
        """
        question = state["question"]
        results = retriever.semantic_search(question, n=15)
        # See QA - Q: semantic_search fetches by ID a second time
        profiles = [
            p for r in results
            if (p := retriever.get_machine_profile(r["metadata"]["machine_id"]))
        ]
        return {
            "retrieved_profiles": profiles,
            "context_blocks": [_machine_doc_text(p) for p in profiles],
            "trace": [retriever.build_retrieval_trace(question, results)],
            "capabilities_used": ["data_retriever"],
        }
    return semantic_search_node


# --------------------------------------------------------------------------- #
# Analyzer nodes (stateless — read retrieved_profiles from state)             #
# These two run IN PARALLEL for the "prioritize" intent.                      #
# LangGraph fans out from retrieve_all to both, then joins at synthesizer.    #
# --------------------------------------------------------------------------- #

def reliability_analyzer_node(state: GraphState) -> dict:
    """
    Deterministic (no LLM) risk-scoring node — ranks the machines already sitting in
    `retrieved_profiles` by a weighted formula (criticality, health, alert severity,
    overdue/urgent work orders). Reused across two paths: runs in parallel with
    maintenance_analyzer for `prioritize`, and runs alone after semantic_search for
    `fault_filter`. Deterministic on purpose — the same inputs must always produce the
    same ranking so a maintenance planner can trust it.
    """
    scored = ra.rank_machines_by_priority(state["retrieved_profiles"])
    return {
        "context_blocks": [f"Machine priority ranking:\n{ra.build_risk_summary(scored, top_n=10)}"],
        "trace": [ra.build_trace_step(scored)],
        "capabilities_used": ["reliability_analyzer"],
    }


def maintenance_analyzer_node(state: GraphState) -> dict:
    """
    Deterministic (no LLM) node — summarizes work-order and maintenance-history context
    for whatever is in `retrieved_profiles`. Reused across two paths: runs in parallel
    with reliability_analyzer for `prioritize`, and runs alone after retrieve_machine
    for `machine_specific`.
    """
    profiles = state["retrieved_profiles"]
    return {
        "context_blocks": [f"Maintenance context:\n{ma.summarize_maintenance_context(profiles)}"],
        "trace": [ma.build_trace_step(profiles)],
        "capabilities_used": ["maintenance_analyzer"],
    }


# --------------------------------------------------------------------------- #
# Site comparator node                                                         #
# --------------------------------------------------------------------------- #

def make_site_comparator_node(retriever: DataRetriever):
    def site_comparator_node(state: GraphState) -> dict:
        """
        site_compare path. Resolves two site names — from planner entities, falling
        back to a regex scan of the question, then to a hardcoded default if still
        short. Pulls each site's machine profiles and combines a site-level comparison
        (sc.compare_sites) with a per-site risk ranking, calling
        reliability_analyzer's scoring function directly rather than routing through
        the graph node (this path never touches the reliability_analyzer graph node).
        The only node besides synthesizer that reports two capabilities used.
        """
        sites = state["plan"].entities.get("sites") or []
        if len(sites) < 2:
            found = re.findall(r"Site\s+[A-Za-z]", state["question"])
            sites = list(dict.fromkeys(s.title() for s in found))
        if len(sites) < 2:
            sites = ["Site A", "Site B"]

        site_a, site_b = sites[0], sites[1]
        profiles_a = retriever.get_site_profiles(site_a)
        profiles_b = retriever.get_site_profiles(site_b)

        comparison = sc.compare_sites(profiles_a, profiles_b)
        scored_a = ra.rank_machines_by_priority(profiles_a)
        scored_b = ra.rank_machines_by_priority(profiles_b)

        return {
            "retrieved_profiles": [p for p in profiles_a + profiles_b if p],
            "context_blocks": [
                f"Site comparison:\n{sc.format_comparison(site_a, site_b, comparison)}",
                f"{site_a} machines:\n{ra.build_risk_summary(scored_a, top_n=5)}\n"
                f"{site_b} machines:\n{ra.build_risk_summary(scored_b, top_n=5)}",
            ],
            "trace": [
                sc.build_trace_step(site_a, site_b, comparison),
                TraceStep(
                    step="reliability_analyzer",
                    input=f"per-site risk ranking for {site_a} and {site_b}",
                    output=f"Ranked {len(profiles_a)} + {len(profiles_b)} machines",
                    data_sources=["machines", "alerts"],
                ),
            ],
            "capabilities_used": ["site_comparator", "reliability_analyzer"],
        }
    return site_comparator_node


# --------------------------------------------------------------------------- #
# Synthesizer node (always the terminal node)                                 #
# --------------------------------------------------------------------------- #

def make_synthesizer_node(client: genai.Client):
    def synthesizer_node(state: GraphState) -> dict:
        """
        Terminal node — every path in the graph ends here (see orchestrator.py, all
        edges converge on synthesizer -> END). Four mutually exclusive branches:
          1. `not_found_machine` is set        -> canned "machine not found" response.
          2. `planner_parse_failed` is set      -> canned "couldn't interpret" response.
          3. `plan.intent == "out_of_scope"`    -> canned refusal response.
          4. otherwise                          -> real Gemini synthesis over the
             `context_blocks` collected from whichever nodes ran upstream.
        Branches 1-3 skip the LLM call entirely and are built deterministically in
        code: there's no reliability data to reason over, so a model call would either
        hallucinate or just restate the same conclusion — paying for a call that adds
        no value. Branch 2 is deliberately distinct from branch 3: both route through
        `plan.intent == "out_of_scope"` (the planner sets that intent on a parse
        failure so routing fails safe — see planner_node), but conflating their
        user-facing messages would be dishonest. A parse failure says nothing about
        whether the question is actually out of scope; telling the user "this is
        outside the assistant's scope" when the real problem was a malformed LLM
        response mislabels the failure both for the user and in the trace. Branch 4 is
        the only path where the model actually writes the answer text, and it's
        grounded only in what upstream nodes retrieved (the model never sees data
        outside `context_blocks`).
        """
        question = state["question"]
        plan = state.get("plan")

        # Machine not found
        if state.get("not_found_machine"):
            mid = state["not_found_machine"]
            from .models import AssistantResponse
            response = AssistantResponse(
                answer=f"Machine {mid} does not exist in the dataset. Please verify the machine ID.",
                reasoning=f"No record found for '{mid}' in machines.json.",
                data_used=["machines"],
                machines_referenced=[],
                limitations_or_assumptions=[f"Machine {mid} is not present in the dataset."],
                confidence="low",
                capabilities_used=list(state.get("capabilities_used", [])),
                retrieved_records=[],
                trace=list(state.get("trace", [])) + [TraceStep(
                    step="answer_synthesizer", input=question,
                    output=f"Machine {mid} not found — short-circuit response.",
                )],
            )
            return {"response": response}

        # Planner parse failure — fail safe with an honest message, not the
        # generic out-of-scope refusal (the question may well be in-scope; the
        # planner just couldn't produce structured output for it).
        if state.get("planner_parse_failed"):
            from .models import AssistantResponse
            response = AssistantResponse(
                answer=(
                    "I couldn't confidently interpret that question — please try "
                    "rephrasing it."
                ),
                reasoning="Planner output could not be parsed as valid structured data; "
                          "failing safe rather than guessing an intent.",
                data_used=[],
                machines_referenced=[],
                limitations_or_assumptions=["Planner response was not valid structured output."],
                confidence="low",
                capabilities_used=[],
                retrieved_records=[],
                trace=list(state.get("trace", [])) + [TraceStep(
                    step="answer_synthesizer", input=question,
                    output="Planner parse failure — no capabilities invoked.",
                )],
            )
            return {"response": response}

        # Out of scope
        if plan and plan.intent == "out_of_scope":
            from .models import AssistantResponse
            response = AssistantResponse(
                answer=(
                    "This question is outside the scope of the reliability assistant. "
                    "I can only answer questions about machine health, alerts, work orders, "
                    "and maintenance history based on the provided industrial dataset."
                ),
                reasoning=plan.reasoning or "Question does not relate to machine reliability data.",
                data_used=[],
                machines_referenced=[],
                limitations_or_assumptions=["No reliability data supports this question."],
                confidence="low",
                capabilities_used=[],
                retrieved_records=[],
                trace=list(state.get("trace", [])) + [TraceStep(
                    step="answer_synthesizer", input=question,
                    output="Out of scope — no capabilities invoked.",
                )],
            )
            return {"response": response}

        # Normal Gemini synthesis
        response = synthesize(
            question=question,
            context_blocks=list(state.get("context_blocks", [])),
            trace_so_far=list(state.get("trace", [])),
            capabilities_used=list(state.get("capabilities_used", [])),
            retrieved_profiles=list(state.get("retrieved_profiles", [])),
            client=client,
        )
        return {"response": response}

    return synthesizer_node
