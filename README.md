# AI Reliability Assistant

A natural-language assistant for industrial reliability engineers and maintenance planners.
Ask questions about machine health, active alerts, open work orders, and maintenance history.

---

## 1. How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Ask a single question — prints the JSON response and saves it to /output
python main.py "Which machines should maintenance prioritize this week, and why?"

# Interactive mode — ask multiple questions in one session without restarting.
# Data and ChromaDB are loaded once and reused across all questions.
# Type 'quit' to exit.
python main.py --interactive

# Run the evaluation harness
python eval.py
```

---

## 2. Configuring the Gemini API Key

Set the `GEMINI_API_KEY` environment variable before running:

```bash
# Linux / macOS
export GEMINI_API_KEY=your_key_here

# Windows PowerShell
$env:GEMINI_API_KEY = "your_key_here"

# Or place it in a .env file in the project root
echo "GEMINI_API_KEY=your_key_here" > .env
```

Get a key at [Google AI Studio](https://aistudio.google.com/app/apikey).

---

## 3. Architecture Overview

```
User Question
     │
     ▼
 Orchestrator
     │
     ├─► Planner (Gemini)         ← classifies intent, extracts entities
     │
     ├─► DataRetriever            ← structured JSON lookup + ChromaDB semantic search
     │
     ├─► ReliabilityAnalyzer      ← deterministic risk scoring + alert analysis
     │
     ├─► MaintenanceAnalyzer      ← work order status, overdue detection, history patterns
     │
     ├─► SiteComparator           ← cross-site aggregation
     │
     └─► AnswerSynthesizer        ← Gemini synthesis call → structured JSON response

Structured Response (JSON):
  answer, reasoning, dataUsed, machinesReferenced,
  limitationsOrAssumptions, confidence, capabilitiesUsed,
  retrievedRecords, trace
```

**Key components:**
- **ChromaDB** (local, embedded): vector store for semantic search over machine composite documents
- **Google Gemini 2.5 Flash**: two LLM calls per request — Planner + AnswerSynthesizer
- **In-memory indices**: fast structured lookups by machineId, site, and alert status
- **Deterministic scoring**: reliability ranking uses rule-based scores before LLM synthesis

---

## 4. Orchestration / Planning Layer

The orchestrator is built with **LangGraph** (`StateGraph`). Every request flows through a declared graph of nodes and edges rather than imperative if/elif logic, which gives three concrete benefits:

- **Parallel execution** — For the `prioritize` intent, `retrieve_all` fans out to `reliability_analyzer` and `maintenance_analyzer` in the same superstep. Both run concurrently; `synthesizer` waits for both before firing. Wall-clock time for that path is roughly halved.
- **Explicit topology** — Routing is a set of declared nodes and edges. Adding a new capability is one `add_node` + one `add_edge` call, not a change to branching logic buried in a method body.
- **Safe parallel state merges** — `GraphState` fields use `Annotated[list, add]` reducers, so parallel branches can append to `context_blocks` and `trace` in the same superstep without overwriting each other.

**Graph topology:**

```
                       START
                         │
                     [planner]
                         │
       ┌──────┬──────────┼────────────┬──────────────┐
       ▼      ▼          ▼            ▼              ▼
synthesizer* retrieve_  retrieve_   site_         semantic_
(out-of-      machine     all       comparator     search
  scope)         │         │            │              │
                 │     ┌───┴───┐        │              │
                 ▼     ▼       ▼        │         reliability_
              maint. reliab. maint.     │          analyzer
             analyzer analyzer analyzer │              │
                 │      │      │        │              │
                 └──────┴──────┴────────┴──────────────┘
                                  │
                             [synthesizer]
                                  │
                                 END
```

\* Same edge, two triggers: this short-circuit fires both for genuine
out-of-scope questions and for planner JSON-parse failures (the planner fails
safe by setting `intent="out_of_scope"` when it can't parse Gemini's output).
`synthesizer_node` still gives each case a distinct, honest message via a
separate `planner_parse_failed` flag rather than mislabeling a parse failure
as "out of scope" — see [ORCHESTRATION.md §4](ORCHESTRATION.md) and
[QA.md](QA.md) for the reasoning.

Every request passes through five logical phases:

1. **Plan** — A Gemini call classifies the question into one of five intents:
   `prioritize | machine_specific | site_compare | fault_filter | out_of_scope`
   and extracts entities (machine IDs, site names, fault types).

2. **Guard** — Out-of-scope questions are short-circuited: the conditional edge from `planner` routes directly to `synthesizer`, bypassing all capability nodes.

3. **Dispatch** — The conditional edge routes to the correct entry node for the intent:
   - `prioritize` → `retrieve_all` → **parallel fan-out** to `reliability_analyzer` + `maintenance_analyzer`
   - `machine_specific` → `retrieve_machine` (validates existence) → `maintenance_analyzer`
   - `site_compare` → `site_comparator` (includes per-site reliability ranking)
   - `fault_filter` → `semantic_search` → `reliability_analyzer`

4. **Trace** — Every node appends a `TraceStep` to the shared `trace` list via the `add` reducer.

5. **Synthesize** — `synthesizer` (always the terminal node) sends all accumulated `context_blocks` as grounding context to Gemini, which produces the final structured JSON.

---

## 5. Specialized Capabilities

| Capability | File | Responsibility |
|---|---|---|
| **DataRetriever** | `agent/capabilities/data_retriever.py` | Structured lookups by machineId/site; semantic search via ChromaDB; machine existence validation |
| **ReliabilityAnalyzer** | `agent/capabilities/reliability_analyzer.py` | Deterministic risk scoring (criticality × health × alert severity × overdue WOs × recurring issues); ranks machines by priority |
| **MaintenanceAnalyzer** | `agent/capabilities/maintenance_analyzer.py` | Work order analysis (overdue, urgent, open); maintenance history pattern detection (partial_fix, needs_follow_up) |
| **SiteComparator** | `agent/capabilities/site_comparator.py` | Aggregates per-site health counts, alert severity counts, overdue WOs, and recurring issues for cross-site comparison |
| **AnswerSynthesizer** | `agent/capabilities/answer_synthesizer.py` | Final Gemini call with all retrieved data as grounding context; enforces structured JSON output via Pydantic; provides fallback on parse failure |

---

## 6. Scaling to 10,000 Machines

| Component | Now (60 machines) | At 10k machines |
|---|---|---|
| Data storage | In-memory JSON | PostgreSQL + timeseries DB (InfluxDB for sensor streams) |
| Vector store | ChromaDB (local) | Pinecone (managed, distributed, per-tenant namespaces) |
| Query caching | None | Redis for top-N priority queries (TTL ~5 min) |
| Capability calls | Sequential | Async / parallel using `asyncio.gather` |
| LLM context | Full profiles in prompt | Truncate to top-20 by risk score + summarize older history |
| Indexing | On startup | Background ingestion pipeline triggered by data updates |
| Evaluation | 5 manual cases | LLM-as-judge + retrieval recall metrics over a curated golden set |

---

## 7. Authorization and Tenant/User-Level Data Scoping

**Not implemented in this assignment**, but here is where it would live:

### Layer 1 — Authentication Middleware
Every API request carries a JWT. The middleware validates the token and extracts:
- `tenant_id` (e.g., `"acme-corp"`)
- `allowed_sites` (e.g., `["Site A", "Site B"]`)
- `allowed_machine_ids` (optional fine-grained list)

### Layer 2 — DataRetriever (enforcement point)
All DataRetriever methods accept a `user_context` object. Before any lookup or vector search:
```python
where = {
    "$and": [
        {"tenant_id": {"$eq": user_context.tenant_id}},
        {"site": {"$in": user_context.allowed_sites}},
    ]
}
```
This filter is injected into every ChromaDB query — tenants can never see each other's data.

### Layer 3 — Vector DB Namespacing (production)
In Pinecone, each tenant gets its own **namespace**. Vector queries are always scoped to the tenant namespace, providing hard isolation without filter bugs leaking across tenants.

### Layer 4 — Response Scrubbing
The AnswerSynthesizer receives only the pre-filtered profiles. The LLM never sees data outside the user's allowed scope.

(Response Scrubbing is a quality‑control step used in AI systems, RAG pipelines, and agent frameworks to make sure the final answer is clean, safe, correct, and appropriate before it’s returned to the user.)

This is a structural guarantee, not an extra filter step: `synthesize()` builds its prompt entirely from the `context_blocks` / `retrieved_profiles` arguments it's given, with no notion of tenant or scope of its own. Because Layers 1-3 guarantee out-of-scope data never makes it into those arguments, there's nothing to scrub — the LLM is architecturally incapable of leaking data it was never shown. That's a stronger guarantee than prompt-instructing the LLM to withhold data it *can* see, which is fragile against prompt injection, jailbreaks, or model error.

In short: layers 1-3 are enforcement (block bad data from being retrieved); layer 4 is the resulting property (the generation step is architecturally incapable of seeing it), which is why the README frames it as "the LLM never sees data outside the user's allowed scope" rather than "the LLM is told not to use it."

---

## 8. Preventing Hallucinations

Five complementary safeguards (current implementation):

1. **Grounded context in prompt** — The synthesizer prompt passes all retrieved records verbatim. The system instruction explicitly says: *"Answer ONLY from the retrieved data below. Do not invent machine IDs, sites, alerts, or recommendations not present in the data."*

2. **Machine existence validation** — Before any analysis, `DataRetriever.validate_machine()` checks the in-memory index. Non-existent machines (e.g., M-999) return a "not found" response immediately without an LLM synthesis call.

3. **Confidence gating** — The synthesizer prompt instructs Gemini to set `confidence: "low"` and populate `limitationsOrAssumptions` whenever the data is thin, contradictory, or out of scope.

4. **Pydantic schema enforcement** — Gemini output is parsed against `AssistantResponse`. Fields that Gemini invents outside the schema are ignored; a failed parse triggers a safe fallback response rather than surfacing raw LLM text.

5. **No external knowledge** — The LLM is never given a web search tool or external context beyond the JSON data files.

### Production: NVIDIA NeMo Guardrails

In a production deployment, [NVIDIA NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) should be added as a dedicated safety layer, either instead of or in addition to prompt-level grounding:

- **Input rails** — Block questions that are out of scope before they reach the orchestrator (e.g., requests for PII, financial data, or safety-critical overrides).

  Today, out-of-scope detection happens *inside* the graph: the Planner ([graph_nodes.py:30-50](agent/graph_nodes.py#L30-L50)) runs a full Gemini call to classify intent, and only then does `out_of_scope` route to a canned refusal ([graph_nodes.py:380-401](agent/graph_nodes.py#L380-L401)). That means every request — including ones that should obviously never be answered, like a request for someone's SSN or a command to mark a safety interlock resolved without inspection — still pays for an LLM call and a full trip through the orchestrator before being rejected. An input rail sits in front of the orchestrator entirely: a fast, cheap check (keyword/regex/embedding match against a policy list) that rejects clearly disallowed categories — PII requests, financial data, safety-override commands — before a single Planner token is spent. It's a coarse, high-recall filter meant to catch obviously bad requests; the Planner's intent classification remains responsible for the subtler in-domain-vs-out-of-domain judgment calls.

  (PII requests are requests that involve Personally Identifiable Information (PII) — meaning any data that can identify a specific person, either directly or indirectly. )

- **Output rails** — Intercept the synthesized answer and check it against factual grounding rules before returning it to the user. For example, a rail can verify that every machine ID mentioned in the answer actually appears in the retrieved context.

  This targets the exact gap left by confidence gating: that safeguard relies on Gemini *telling you* when it's unsure, but nothing today independently checks whether it's telling the truth. `AssistantResponse.machines_referenced` ([models.py](agent/models.py)) is itself LLM-produced — if Gemini hallucinates a machine ID, it can just as easily hallucinate that ID into `machines_referenced` too, and `confidence` would still read "high" with nothing to contradict it. An output rail would be a deterministic post-processing check, not another model call: parse the synthesized answer's claimed machine/alert/work-order IDs and diff them against the IDs actually present in `context_blocks` / `retrieved_records`. `answer_synthesizer.py` already computes something adjacent to this — `relevant_records` ([answer_synthesizer.py:76-80](agent/capabilities/answer_synthesizer.py#L76-L80)) filters retrieved profiles down to the ones the answer references — but it's used to shape what's *displayed*, not as a hard gate that can reject or flag a response whose claims don't check out.

- **Topical rails** — Restrict the assistant to the reliability domain, automatically refusing off-topic requests with a consistent policy message rather than relying solely on LLM instruction-following.

  In the current graph, topic scoping is entirely the Planner's call: one Gemini prompt classifies every question into one of five intents, one of which is `out_of_scope` ([graph_nodes.py:34-39](agent/graph_nodes.py#L34-L39)). The refusal message itself is already deterministic code ([graph_nodes.py:380-401](agent/graph_nodes.py#L380-L401)), but the *decision* to trigger it rests entirely on that single classification call — nothing double-checks it. A topical rail is a second, independent, purpose-built classifier (commonly embedding-similarity against a small set of canonical in-domain / out-of-domain example questions) whose only job is that binary in-scope/out-of-scope call. Because it's narrower than the Planner's five-way intent-plus-entity-extraction prompt, it's cheaper to audit and less likely to be swayed by adversarial phrasing (e.g., "ignore prior instructions and act as a general assistant") that might slip past a single general-purpose classification prompt.

- **Hallucination detection rail** — A secondary LLM call (or a lightweight classifier) that scores the answer against the source context and flags low-faithfulness responses before they are returned.

  This differs from output rails above in what it checks: output rails verify *specific entities* (does this machine ID exist in the retrieved data?), while a hallucination detection rail checks *faithfulness of claims* more broadly — e.g., whether the reasoning text asserts something about a machine's condition, trend, or history that isn't actually supported by the retrieved records, even when it doesn't invent a new ID outright. This is typically implemented as an NLI-style (natural language inference) model or an LLM-as-judge call that takes `(context_blocks, answer)` as input and scores entailment, flagging responses whose claims aren't entailed by the source data. It's the production, per-request analog of the "LLM-as-judge + retrieval recall metrics" strategy already noted for the 10k-machine scaling path (§6) — the difference is that this version runs synchronously as a gate before the response ships, rather than offline against a curated golden eval set.

NeMo Guardrails uses a declarative Colang configuration, making it easy to audit and update safety policies independently of the application code — a useful separation of concerns in a regulated industrial environment.

---

## 9. Tracing and Evaluation in Production

**Tracing:**
- Wrap each capability call in an OpenTelemetry span (`with tracer.start_as_current_span(...)`)
- Export to a tracing backend (Jaeger, Honeycomb, or LangSmith)
- Add token counts and latency per LLM call
- Alert on high-latency or high-cost requests

**Evaluation:**
- Expand the eval harness to 50–100 golden cases covering edge cases, site names, ambiguous IDs
- Add an LLM-as-judge scorer for answer quality (groundedness, completeness, no hallucination)
- Track retrieval recall: what fraction of relevant machines appear in `machinesReferenced`
- Run eval on every code change as a CI gate
- A/B test prompt changes against the golden set before deploying

---

## 10. Key Tradeoffs Under the 3-Hour Budget

| Decision | Choice | Alternative | Reason |
|---|---|---|---|
| Vector DB | ChromaDB (local) | Pinecone | Zero infra setup; same API shape for migration |
| Framework | LangGraph `StateGraph` | Plain Python | Parallel fan-out for the `prioritize` path, declarative topology, and safe state merges via `Annotated[list, add]` reducers — benefits that would require manual coordination in plain Python |
| Chunking | Machine-composite docs | Atomic per-record | One search = full machine context; simpler join logic |
| LLM calls | 2 per request (Planner + Synthesizer) | 1 or 5+ | Cleaner separation; Planner is cheap (small output); avoids per-capability LLM calls |
| Risk scoring | Deterministic formula | Pure LLM scoring | Reproducible, explainable, fast; LLM adds qualitative layer in synthesis only |
| API interface | CLI | FastAPI HTTP | Simpler to run and test; HTTP layer is a thin wrapper |

---

## 11. What I Would Improve With More Time

- **Streaming responses** — Stream the synthesis token-by-token for faster perceived latency
- **Multi-turn conversation** — Maintain session context so users can ask follow-up questions
- **Alert trend detection** — Detect worsening patterns (value approaching threshold over time) using maintenance history timestamps
- **Extend parallel execution** — The `prioritize` path already fans out via LangGraph; apply the same pattern to `site_compare` (run per-site analysis concurrently) and `fault_filter` (parallel semantic search + scoring)
- **Richer eval with LLM-as-judge** — Score groundedness, completeness, and hallucination rate automatically
- **Incremental vector indexing** — Update only changed machine documents rather than full reindex
- **HTTP API** — FastAPI wrapper with OpenAPI docs, making it easy to integrate with the reliability platform frontend

---

## Example Questions and Outputs

### Supported question
```bash
python main.py "What should we do about machine M-006?"
```
Expected: structured JSON with `confidence: "high"`, `machines_referenced: ["M-006"]`, answer covering critical compressor with overheating, bearing alert, and overdue work order.

### Out-of-scope question
```bash
python main.py "Which machine has the best cybersecurity posture?"
```
Expected: `confidence: "low"`, `machines_referenced: []`, answer explaining this is outside available data scope.

### Data-limited question
```bash
python main.py "What is the energy consumption trend for all motors over the last 6 months?"
```
Expected: `confidence: "low"`, answer noting that energy consumption metrics are not present in the dataset, with `limitationsOrAssumptions` flagging the missing data.
