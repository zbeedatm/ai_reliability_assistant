"""
LangGraph orchestrator.

Why LangGraph instead of plain Python functions
------------------------------------------------
The previous orchestrator was a sequential if/elif block in a single run() method.
That works for a linear pipeline, but has three limitations this graph addresses:

1. PARALLEL EXECUTION
   For the "prioritize" intent, reliability_analyzer and maintenance_analyzer are
   independent — they both read the same list of machine profiles and produce
   separate context blocks. The LangGraph graph fans out from retrieve_all to BOTH
   nodes simultaneously, cutting wall-clock time roughly in half for that path.
   The Annotated[list, add] state reducers ensure their outputs merge correctly
   without race conditions.

2. EXPLICIT TOPOLOGY
   The routing logic is a declared graph (nodes + edges) rather than a method body.
   You can call `graph.get_graph().draw_ascii()` to visualize the exact flow,
   and adding a new capability is one add_node + one add_edge rather than editing
   branching logic scattered through run().

3. CLEAN CONDITIONAL FAN-IN
   LangGraph knows that synthesizer sits downstream of both parallel branches and
   waits for all ACTIVE predecessors in the current superstep before firing.
   The "machine_specific" path activates only maintenance_analyzer; synthesizer
   fires after it alone — no manual join code needed.

Graph topology
--------------
                       ┌─────────────────────────────────────────┐
                       │              START                      │
                       └──────────────┬──────────────────────────┘
                                      ▼
                                  [ planner ]
                                      │
               ┌──────────┬───────────┼────────────┬─────────────┐
               ▼          ▼           ▼            ▼             ▼
         synthesizer* retrieve_   retrieve_    site_        semantic_
         (out-of-      machine      all       comparator     search
          scope)          │          │             │              │
                          │      ┌───┴───┐         │              │
                          ▼      ▼       ▼         │              │
                    maint. reliability  maint.     │         reliability
                    analyzer  analyzer  analyzer   │          analyzer
                          │      │       │         │              │
                          └──────┴───────┴─────────┴──────────────┘
                                         │
                                         ▼
                                    [ synthesizer ]
                                         │
                                       END

The "prioritize" path is the parallel branch:
  retrieve_all ──► reliability_analyzer ─┐
               └──► maintenance_analyzer ─┴──► synthesizer
Both analyzer nodes run concurrently; synthesizer waits for both.

* Same edge, two triggers: this short-circuit fires both for genuine
  out-of-scope questions AND for planner JSON-parse failures (planner_node
  sets intent="out_of_scope" on parse failure to fail safe with zero
  capability calls). synthesizer_node still gives each case a distinct,
  honest user-facing message via the separate planner_parse_failed flag —
  see the synthesizer_node docstring below.
"""
from __future__ import annotations
import os
from google import genai
from langgraph.graph import StateGraph, START, END

from .graph_state import GraphState
from .graph_nodes import (
    make_planner_node,
    make_retrieve_machine_node,
    make_retrieve_all_node,
    make_semantic_search_node,
    reliability_analyzer_node,
    maintenance_analyzer_node,
    make_site_comparator_node,
    make_synthesizer_node,
    route_after_planner,
    route_after_retrieve_machine,
)
from data_loader import DataLoader
from vector_store import VectorStore
from .capabilities.data_retriever import DataRetriever
from .models import AssistantResponse


class Orchestrator:
    def __init__(self, data_loader: DataLoader, vector_store: VectorStore, api_key: str | None = None):
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        client = genai.Client(api_key=key)

        retriever = DataRetriever(data_loader, vector_store)
        self._graph = self._build_graph(client, retriever)

    def _build_graph(self, client: genai.Client, retriever):
        builder = StateGraph(GraphState)

        # ── Nodes ────────────────────────────────────────────────────────────
        builder.add_node("planner",              make_planner_node(client))
        builder.add_node("retrieve_machine",     make_retrieve_machine_node(retriever))
        builder.add_node("retrieve_all",         make_retrieve_all_node(retriever))
        builder.add_node("semantic_search",      make_semantic_search_node(retriever))
        builder.add_node("reliability_analyzer", reliability_analyzer_node)
        builder.add_node("maintenance_analyzer", maintenance_analyzer_node)
        builder.add_node("site_comparator",      make_site_comparator_node(retriever))
        builder.add_node("synthesizer",          make_synthesizer_node(client))

        # ── Entry ────────────────────────────────────────────────────────────
        builder.add_edge(START, "planner")

        # ── Planner → intent-based routing ───────────────────────────────────
        builder.add_conditional_edges(
            "planner",
            route_after_planner,
            {
                "retrieve_machine": "retrieve_machine",
                "retrieve_all":     "retrieve_all",
                "semantic_search":  "semantic_search",
                "site_comparator":  "site_comparator",
                "synthesizer":      "synthesizer",   # out_of_scope short-circuit
            },
        )

        # ── machine_specific path ────────────────────────────────────────────
        # retrieve_machine validates existence; routes to synthesizer on not-found
        # NOTE: this path never touches semantic_search. The user gave (or the planner
        # extracted) an exact machine ID, so it's a direct lookup, not a similarity
        # problem — running it through embedding + nearest-neighbor search would be
        # slower AND less precise (it could return a similar-but-wrong machine instead
        # of the exact one asked about). semantic_search exists only for fault_filter,
        # where the query is a natural-language description with no exact key to look
        # up ("machines with bearing issues" has no machine_id to query on).
        builder.add_conditional_edges(
            "retrieve_machine",
            route_after_retrieve_machine,
            {
                "maintenance_analyzer": "maintenance_analyzer",
                "synthesizer":          "synthesizer",
            },
        )
        builder.add_edge("maintenance_analyzer", "synthesizer")

        # ── prioritize path — PARALLEL FAN-OUT ──────────────────────────────
        # retrieve_all activates BOTH analyzers in the same superstep.
        # LangGraph runs them concurrently; synthesizer waits for both to finish.
        # The Annotated[list, add] reducer on context_blocks / trace merges
        # both nodes' outputs atomically before synthesizer reads from state.
        builder.add_edge("retrieve_all", "reliability_analyzer")
        builder.add_edge("retrieve_all", "maintenance_analyzer")
        builder.add_edge("reliability_analyzer", "synthesizer")
        # maintenance_analyzer → synthesizer already defined above

        # ── site_compare path ────────────────────────────────────────────────
        builder.add_edge("site_comparator", "synthesizer")

        # ── fault_filter path ────────────────────────────────────────────────
        # NOTE: unlike "prioritize", this only fans out to reliability_analyzer —
        # there is no semantic_search → maintenance_analyzer edge. A fault_filter
        # question ("machines with bearing issues") gets a risk ranking but no
        # maintenance/work-order summary in context_blocks. Not clearly intentional;
        # flag before assuming maintenance context is always present in synthesis.
        builder.add_edge("semantic_search", "reliability_analyzer")
        # reliability_analyzer → synthesizer already defined above

        # ── Terminal ─────────────────────────────────────────────────────────
        builder.add_edge("synthesizer", END)

        return builder.compile()

    def run(self, question: str) -> AssistantResponse:
        initial_state: GraphState = {
            "question": question,
            "plan": None,
            "context_blocks": [],
            "trace": [],
            "retrieved_profiles": [],
            "capabilities_used": [],
            "response": None,
            "not_found_machine": None,
            "planner_parse_failed": False,
        }
        final_state = self._graph.invoke(initial_state)
        return final_state["response"]
