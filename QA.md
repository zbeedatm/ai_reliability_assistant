# Q&A on the AI Reliability Assistant

Prep notes for discussing the technical decisions behind this project, organized by the
prep areas called out for the interview: Python/backend fundamentals, production GenAI
systems, agentic AI/RAG/tool-calling, LLM orchestration/eval/tracing, API and distributed
systems design, secure data access and tenant separation, and production tradeoffs
(reliability, latency, scalability, fallback, cost).

---

## Architecture & orchestration (LangGraph)

**Q: Why `StateGraph` over a plain function pipeline?**
A: The original version was a sequential if/elif in one `run()` method. Three things broke
down:
- **No parallelism** — independent steps ran serially even when they didn't depend on each other.
- **Implicit topology** — routing logic was buried in branching code instead of being inspectable.
- **Manual joins** — merging outputs from independent branches required hand-written coordination code.

LangGraph, built in [`_build_graph`](agent/orchestrator.py#L98-L171), fixes all three: a
declared topology (`add_node`/`add_edge`), fan-out (scatter)/fan-in (gather) for free, and `Annotated[list,
add]` reducers so parallel branches can write to shared state without clobbering each other.

**Q: Walk through the parallel fan-out for `prioritize` — how does `Annotated[list, add]`
prevent a race?**
A:
- `retrieve_all` produces `retrieved_profiles`, then the graph fires `reliability_analyzer`
  and `maintenance_analyzer` in the same superstep.
- Both read the same profiles and independently return a partial state dict with their own
  `context_blocks` and `trace` entries, e.g. `reliability_analyzer` returns
  `{"context_blocks": ["reliability text..."]}` and `maintenance_analyzer` returns
  `{"context_blocks": ["maintenance text..."]}`.
- **Default behavior without a reducer is last-write-wins**: if two nodes update the same key
  in the same step, LangGraph just overwrites one with the other. Since parallel nodes don't
  finish in a guaranteed order, whichever one "wins" is non-deterministic — that's the race:
  not a low-level threading bug, but silent data loss where one analyzer's output vanishes
  depending on timing.
- Because those fields are typed `Annotated[list, add]` in
  [agent/graph_state.py:15-18](agent/graph_state.py#L15-L18), LangGraph instead *concatenates*
  the two nodes' output lists (`operator.add`, i.e. list `+`) rather than overwriting:
  `context_blocks = [] + ["reliability text..."] + ["maintenance text..."]`. Order between the
  two doesn't affect correctness — concatenation gives you both either way, it only changes
  which appears first.
- `synthesizer` only fires once both predecessors in that superstep have completed, and it's
  guaranteed to see both contributions merged, not just whichever finished last.
- Without the reducer, whichever node's dict got applied last would silently discard the
  other's context block.

**Q: Two Gemini calls (planner + synthesizer) — why not one, why not five+?**
A:
- **Not one call** — that would force the model to simultaneously classify intent, decide
  what data to pull, and reason over data it doesn't have yet, hurting both steps.
- **Splitting them** lets the planner be cheap (small structured output, no context) and
  lets synthesis get clean grounding context assembled deterministically by capability code.
- **Not five+ calls** — a separate LLM call per capability (reliability, maintenance, site
  comparison) would multiply latency and cost for logic that's actually deterministic and
  doesn't need an LLM at all (see the risk-scoring question below).

**Q: The planner used to fall back to `fault_filter` on a parse failure — why not
`out_of_scope`?**
A: That was a real gap and I've since fixed it —
[agent/graph_nodes.py:70-80](agent/graph_nodes.py#L70-L80). Defaulting to `fault_filter` meant
a malformed planner response still triggered semantic search and a full synthesis call rather
than failing safe. The fallback now sets `intent="out_of_scope"` so
`route_after_planner` short-circuits straight to `synthesizer` with no capability calls, and
sets a separate `planner_parse_failed` flag so the failure isn't silently relabeled as "this
question is out of scope" — those are different claims. A parse failure means "the planner
choked," not "this question can't be answered," so
[synthesizer_node](agent/graph_nodes.py#L355-L378) gives it a distinct, honest message
("I couldn't confidently interpret that question — please try rephrasing it.") instead of
reusing the generic out-of-scope refusal. Conflating the two would mislabel the failure both
for the user and in the trace — you'd lose the ability to tell "genuinely off-topic" apart
from "the LLM had a bad JSON day" when debugging later.

---

## RAG pipeline & retrieval design

**Q: Why machine-composite documents instead of atomic per-record chunks?**
A: Each embedded document ([`_machine_doc_text`](vector_store.py#L13)) bundles the machine,
its active alerts, open work orders, and recent history into one text block.
- **Benefit** — a question like "machines with bearing issues" only needs one retrieval hit
  per machine to get full context; atomic chunking (one doc per alert, one per work order)
  would need N chunks joined back to a machine, adding a second lookup layer and risking a
  partially retrieved machine context.
- **Cost** — bigger embeddings, coarser (blurry, averaged‑out meaning) relevance signal per document. Negligible at 60
  machines, but worth flagging as a tradeoff that changes at scale.
  - "Bigger" isn't the vector growing (embedding dimensionality is fixed regardless of
    input length) — it's that more text goes into producing each one. A machine with 5
    alerts, 3 open work orders, and a maintenance history all get compressed into a
    single fixed-size vector alongside boilerplate metadata (manufacturer, install
    date, etc.), instead of one short, focused chunk per fact.
  - The coarser signal is the real cost: cosine similarity against the composite doc
    reflects how well a query matches the *entire* blob, not any one fact inside it. One
    bearing-related alert buried alongside four unrelated alerts and a verbose history
    gets diluted across all that surrounding text — the embedding is a blend, not a
    precise match to the one relevant sentence. An atomic per-alert chunk wouldn't have
    that dilution.
  - Negligible today because `semantic_search` pulls `n=15` out of a 60-document corpus
    ([graph_nodes.py:201](agent/graph_nodes.py#L201)) — already 25% of the entire
    dataset per query, so a blurrier signal still tends to surface the right machine.
    Embedding cost is trivial at this scale regardless of per-document verbosity.
  - Gets worse at scale on two fronts: precision matters more once there are thousands
    of plausible candidates instead of 60 (a diluted signal has more opportunities to
    lose to a false positive), and reindex cost compounds — updating one alert on one
    machine means re-embedding that machine's *entire* composite document rather than
    just the one changed fact. This is exactly why the hybrid-retrieval answer later in
    this section ("What would this embedding approach retrieve poorly?" — structured
    metadata filtering + vector search) is the proposed fix rather than pure semantic
    search over ever-larger composite blobs.

**Q: `semantic_search` fetches by ID a second time — why not return the vector doc
directly?**
A: [agent/graph_nodes.py:189-212](agent/graph_nodes.py#L189-L212)
- The vector doc's job is retrieval *ranking*, not being the source of truth.
- I re-fetch the full structured profile from `DataRetriever` so the reliability analyzer
  downstream gets typed fields (`healthStatus`, `criticality`, alert `status`) to score on,
  not a flattened text blob.
- It also means the synthesized answer reflects current data even if the embedding is stale
  until the next reindex.
- "The vector doc" is the flattened text string ChromaDB stores and returns on a query
  hit — literally the output of `_machine_doc_text(profile)`, stored as `document`
  ([vector_store.py:110-127](vector_store.py#L110-L127), `include=["documents", ...]`).
  Since the query response already contains that full text for every hit, reusing it
  directly looks like the obvious shortcut — skip the second lookup entirely.
- Two reasons it doesn't take that shortcut:
  1. **Staleness** — the stored text reflects whatever the machine looked like at last
     index time ([vector_store.py:91-108](vector_store.py#L91-L108), `_ensure_indexed`).
     If an alert got resolved or a work order closed since then and no `reindex()` ran,
     the stored text is out of date. Re-fetching from `DataRetriever` reads live JSON, so
     the answer reflects current state regardless of when the index was last refreshed.
  2. **Type mismatch, not just staleness** — even ignoring freshness, the stored
     `document` is a flattened *string*, while `reliability_analyzer_node` (which runs
     right after `semantic_search_node` in the `fault_filter` path) needs structured
     fields to score with — `profile["machine"]["criticality"]`, `["healthStatus"]`,
     `profile["alerts"]`, etc. ([reliability_analyzer.py:12-34](agent/capabilities/reliability_analyzer.py#L12-L34),
     `_score_machine`). You can't index into a paragraph of prose for `criticality` — a
     structured re-fetch is required either way, not just a freshness nicety.
- Net effect: the vector store's role here is narrowly "which machine IDs are relevant"
  — only `machine_id` from each hit's `metadata` is actually used; the `document` text
  itself is discarded and rebuilt from a fresh, structured source.

**Q: How would you measure retrieval quality at scale vs. today?**
A: Today I don't measure recall at all — 60 machines and `n=15` results means semantic
search rarely misses. At 10k machines I'd want a golden set of (query, expected machine IDs)
pairs and track recall@k — what fraction of expected machines actually appear in
`machinesReferenced`, which is exactly the metric I called out in the README as missing.

Recall = how many of the correct items your system actually retrieved.
It answers the question:
“Out of all the machines that should have been returned, how many did we actually get?”

It measures completeness.

**Q: What would this embedding approach retrieve poorly?**
A: Numeric/threshold queries — "machines with vibration over 8mm/s" — because cosine
similarity on natural-language text doesn't reason about numeric comparisons well. That's a
case for hybrid retrieval: structured filtering on the metadata (which I already have —
`criticality`, `health_status`, `fault_types` in
[`_machine_metadata`](vector_store.py#L61)) combined with vector search, rather than relying
on embeddings alone.

- **Why cosine similarity specifically fails at numeric reasoning** — "vibration over
  8mm/s" and "vibration over 3mm/s" produce nearly identical embeddings, since the text
  differs by one token and embedding space has no notion that "8 > threshold" is a
  true/false condition to evaluate against a machine's actual reading. The model measures
  lexical/topical closeness, not numeric truth — a semantic search for "over 8mm/s" could
  easily surface a machine whose vibration alert is well under 8, just because its
  composite document happens to mention "vibration" near similar-sounding phrasing.
- **What "hybrid retrieval" means concretely here** — combining ChromaDB's `where`
  metadata filter (exact/structured matching, including numeric operators like `$gt`)
  with the vector similarity query in the same call, instead of expecting the embedding
  to encode both semantic relevance *and* numeric truth.
- **What's already in place** — `VectorStore.semantic_search` already accepts an
  `extra_filter` parameter ([vector_store.py:110](vector_store.py#L110)) merged into the
  `where` clause via `$and` alongside the tenant filter
  ([vector_store.py:111-113](vector_store.py#L111-L113)) — that's the actual
  hybrid-filtering mechanism, and it already works today for whatever's in
  `_machine_metadata`: `criticality`, `health_status`, `fault_types`, `machine_type`,
  `site`. Filtering to `criticality=critical` combined with a semantic query would work
  right now.
- **Two gaps, not one, before the numeric example actually works:**
  1. *The plumbing is dead code.* `DataRetriever.semantic_search`
     ([data_retriever.py:24-25](agent/capabilities/data_retriever.py#L24-L25)) doesn't
     accept or forward a filter — `retriever.semantic_search(query, n=n)` drops straight
     through with no `extra_filter`. `semantic_search_node`
     ([graph_nodes.py:189-212](agent/graph_nodes.py#L189-L212)) never passes one either,
     so `extra_filter` exists on `VectorStore` but nothing upstream ever populates it.
  2. *The numeric field doesn't exist yet, even if wired up.* `_machine_metadata` only
     captures `fault_types` as a categorical label (a comma-joined string of active
     fault-type names) — no numeric value like "max active vibration reading" per
     machine. ChromaDB's `$gt`/`$gte` operators need an actual numeric metadata field to
     compare against; `_machine_metadata` would need something like
     `"vibration_max_value": 8.4`, computed from the machine's active alerts at indexing
     time, before a `{"vibration_max_value": {"$gt": 8}}` filter would mean anything.
- **To make the example actually work** you'd need three things together: (a) extend
  `_machine_metadata` with per-fault-type numeric fields at indexing time, (b) thread a
  filter parameter through `DataRetriever.semantic_search` → `semantic_search_node` down
  to `VectorStore`'s existing `extra_filter`, and (c) something to parse "over 8mm/s" out
  of the question into a structured `{fault_type, operator, threshold}` — most naturally
  an extension of the Planner's entity extraction, since it already pulls `fault_types`
  out of free text ([graph_nodes.py:34-51](agent/graph_nodes.py#L34-L51)).

---

## Agentic design / tool-calling workflows

**Q: Why a closed 5-intent taxonomy instead of open-ended LLM tool selection?**
A:
- **Closed taxonomy** makes routing deterministic and testable — I can write an eval case
  per intent and know exactly which capabilities should fire.
- **Open-ended function calling** is more flexible but less predictable — the model could
  skip a required capability or chain tools in an order that produces incomplete context.
- For a 5-intent domain under a hard 3-hour budget, deterministic routing was the right
  tradeoff; if the domain grew to dozens of capabilities with combinatorial paths, I'd
  reconsider and let the model plan more of the routing itself.
- **What "open-ended tool selection" actually means** — also called agentic tool-use or
  the ReAct pattern: instead of a human pre-defining fixed routes, you hand the LLM a set
  of tool/function definitions (name, description, parameters — one per capability) and
  let the model decide at inference time which tool(s) to call, in what order, with what
  arguments, potentially looping through multiple calls and re-evaluating after each
  result before deciding it's done. This is what OpenAI/Anthropic-style function calling,
  LangChain agents, and most "agent" frameworks mean by tool use.
- **Concretely, for this project**, that would mean giving the LLM function schemas for
  `DataRetriever`, `ReliabilityAnalyzer`, `MaintenanceAnalyzer`, `SiteComparator`, and
  `AnswerSynthesizer`, then letting it plan per-question — "call `get_machine_profile`
  first, look at the result, then decide whether to call `reliability_analyzer` next,"
  etc. The model would be doing the sequencing, not just the classification.
- **That's the opposite of what the code actually does.** The Planner
  ([graph_nodes.py:30-51](agent/graph_nodes.py#L30-L51)) does exactly one thing —
  classify the question into 1 of 5 fixed intents — and from there, a hardcoded
  LangGraph topology (`_build_graph`,
  [orchestrator.py:98-171](agent/orchestrator.py#L98-L171)) deterministically decides
  which capability nodes fire and in what order, based purely on which intent bucket the
  classification landed in. The LLM never decides "should I call `reliability_analyzer`
  now" — the graph's edges decide that; the model's only degree of freedom is which of 5
  labels to assign.
- **Why "I can write an eval case per intent" specifically depends on this choice** —
  with 5 fixed intents, `eval.py`'s cases can assert "this question should invoke
  exactly these capabilities" ahead of time. With open-ended tool selection there's no
  fixed "correct path" to assert against in advance — you'd need process-level evals
  (did it call a reasonable sequence of tools) or outcome-level LLM-as-judge evals
  instead.

**Q: Why short-circuit to the synthesizer on a not-found machine without calling the LLM?**
A: [agent/graph_nodes.py:112-114](agent/graph_nodes.py#L112-L114) — if `validate_machine()`
fails, there's no reliability data to reason over, so an LLM call would either hallucinate or
just restate "not found" — paying for a call that adds no value. It's cheaper and more
reliable to build that response deterministically in code.

**Q: Where's the boundary between deterministic and LLM capabilities?**
A:
- **Deterministic Python** — anything that's a pure function of structured data: risk
  scoring, overdue detection, site aggregation. This needs to be reproducible and auditable
  (a maintenance planner needs to trust the ranking is the same every time).
- **LLM** — used only where natural language understanding is actually required:
  classifying intent from free text, and synthesizing a coherent narrative answer from the
  structured context.
- I never let the LLM compute a score or decide a fact — it only explains facts I've already
  computed.

---

## LLM orchestration, evaluation, tracing, observability

**Q: What does the current trace actually capture, and what's missing?**
A: Every node appends a `TraceStep` — `step`, `input`, `output`, `data_sources`
([agent/models.py:6-10](agent/models.py#L6-L10)) — so I can show which capabilities ran and
why.

Missing for production observability:
- Latency per node
- Token counts and cost per LLM call
- Correlation IDs to stitch a trace across a distributed deployment

Today it's an in-memory list returned in the response; in production I'd wrap each node in
an OpenTelemetry span and export to something like Honeycomb or LangSmith so I can alert on
latency/cost outliers instead of just inspecting individual traces after the fact.

**Q: [`eval.py`](eval.py) uses substring `must_contain` checks — what's weak about that, and
how would LLM-as-judge fix it?**
A: The check itself is [`evaluate_case`](eval.py#L71-L99), scoring the 6
[`CASES`](eval.py#L17-L66) against `response.answer`/`response.reasoning`.
- **Weakness** — substring matching only catches missing facts, not incorrect ones, and
  can't judge whether reasoning is actually grounded in the retrieved data versus
  coincidentally containing the right words.
- **LLM-as-judge fix** — score groundedness (does every claim trace back to a retrieved
  record), completeness, and hallucination rate on a rubric, which scales better than
  hand-writing string checks per case.
- **Why I didn't build it** — under a 3-hour budget, 6 deterministic checks that run
  without an extra LLM call were the highest-value thing to ship first; it's explicitly
  called out as the first thing I'd add with more time.

**Q: How often would you expect Gemini to violate the JSON schema, and how would you
monitor it?**
A:
- In practice, low single-digit percent with a clear "return ONLY valid JSON" instruction
  and markdown-fence stripping — but non-zero.
- That's exactly why [`_fallback_response`](agent/capabilities/answer_synthesizer.py#L95-L115)
  exists rather than letting a raw parse error propagate as a 500.
- In production I'd track the fallback-triggered rate as a metric and alert if it spikes,
  since that's a leading indicator of a prompt regression or a model version change
  upstream.

---

## API / microservice design, distributed systems

**Q: How would you turn this into an HTTP service?**
A:
- [`Orchestrator.run(question)`](agent/orchestrator.py#L173-L186) is already a clean
  synchronous function boundary — wrap it in a FastAPI endpoint that takes a `user_context`
  (tenant, allowed sites) plus the question.
- Return the [`AssistantResponse`](agent/models.py#L20-L29) model directly since it's
  already Pydantic.
- The [`Orchestrator`](agent/orchestrator.py#L90-L96) and `VectorStore` currently get
  constructed once and reused (that's why [`main.py --interactive`](main.py#L39-L66) loads
  data once) — in a service I'd hold that as app state initialized on startup, not
  per-request, since reloading ChromaDB and re-parsing JSON per request would be wasteful.

**Q: What breaks when you horizontally scale to multiple instances?**
A:
- The in-memory indices in [`DataLoader`](data_loader.py) /
  [`DataRetriever`](agent/capabilities/data_retriever.py) are per-process — fine for one
  instance, but multiple replicas each load their own copy of the JSON data and their own
  [`ChromaDB` client](vector_store.py#L80-L89) pointed at the same local path, which isn't
  safe for concurrent writes.
- That's exactly why the 10k-scale plan (README section 6) moves structured data to
  Postgres and vectors to a managed, distributed Pinecone — both support concurrent
  reads/writes from many stateless app instances, and the in-memory index becomes a
  per-instance cache rather than the source of truth.

**Q: Why four separate stores (Postgres/InfluxDB/Redis/Pinecone) instead of one?**
A: Different access patterns want different engines:
- **Postgres** — structured relational lookups (machine/work order joins).
- **InfluxDB** — high-frequency sensor time-series, much better fit than rows in Postgres.
- **Pinecone** — vector similarity search needs a vector index.
- **Redis** — top-N priority queries that get asked repeatedly (e.g., "prioritize this
  week"), short TTL cache since risk scores don't change second-to-second.
- **Consistency risk** — if a machine record updates in Postgres, the corresponding
  Pinecone doc has to be reindexed or search returns stale context. I'd handle that with an
  event-driven reindex (a queue triggered on write) rather than polling.

---

## Secure data access, tenant separation

**Q: Walk through the 4-layer scoping design — where's it actually enforced?**
A:
1. **Auth middleware** validates a JWT and extracts `tenant_id`/`allowed_sites`.
2. **[`DataRetriever`](agent/capabilities/data_retriever.py) (enforcement point)** — every
   method would take that `user_context` and inject a `where` filter before any lookup or
   vector query. This is the actual enforcement point, not the API layer, because it's the
   last place before data leaves the system.
3. **Vector DB namespacing (production)** — in Pinecone, each tenant gets its own
   namespace, giving hard isolation instead of relying on a filter being applied correctly
   every time.
4. **Response scrubbing** — the synthesizer only ever sees (It sees only this, and it sees nothing else, at any time) pre-filtered profiles, so the
   LLM physically cannot see data outside scope. I'm not asking the model to respect a
   boundary; I'm not giving it the data in the first place.
   - Concretely: there are exactly two network calls to Gemini per request — the
     Planner's and the Synthesizer's. The Planner's payload is just
     `_PLANNER_SYSTEM` + the raw question ([graph_nodes.py:67](agent/graph_nodes.py#L67))
     — zero machine data, because no dataset variable is even in scope yet when that
     string is built. The Synthesizer's payload is instructions + question +
     `context_blocks`, and `context_blocks` for a `machine_specific` question contains
     exactly one thing: `_machine_doc_text()` of the single profile
     `retrieve_machine_node` fetched for the ID actually asked about
     ([graph_nodes.py:150](agent/graph_nodes.py#L150)). I verified this directly: of the
     60 machines sitting in server memory, 0 of the other 59 appear anywhere in that
     string — not because a filter stripped them, but because the code that builds
     `context_blocks` never touched their records to begin with.
   - The mechanism is about what Python concatenates into a string *before* the network
     call, not an instruction inside that string. Contrast with the alternative design:
     fetch all 60 machines every time, put all of them in `context_blocks`, and add a
     system-prompt line saying "only discuss the one the user asked about." In that
     design all 60 machines' data would physically be part of the HTTP payload sent to
     Google's servers — present in the model's context window, extractable via a
     jailbreak or prompt-injected follow-up. This codebase never constructs that payload
     in the first place.
   - "At any time" holds because there's no third call and no session Gemini retains
     state across — `generate_content()` returns once and the interaction is over. A
     follow-up question re-runs this entire trace from scratch with a freshly built,
     freshly narrow `context_blocks`.

**Q: Filter-based isolation (`where={"tenant_id": ...}`) vs. namespace isolation — what's
the realistic bug?**
A: [vector_store.py:110-127](vector_store.py#L110-L127) (the `where` filter built in
`semantic_search`) shows exactly this pattern today, single-tenant.
- **Realistic bug** — a new capability added later calls `semantic_search` directly with a
  hand-rolled `where` clause and forgets the tenant filter, or an `$or` that's supposed to
  be an `$and`. That's a one-line regression that silently leaks data cross-tenant, and it
  wouldn't necessarily fail loudly.
- **Why namespacing is stronger** — Pinecone namespace isolation makes that class of bug
  structurally impossible: the query physically cannot see another tenant's namespace
  regardless of what filter was or wasn't applied, so it's defense that doesn't depend on
  someone remembering to write the filter correctly.

**Q: Where would you enforce authorization — edge or data layer?**
A: Both, layered.
- **Edge/API gateway** — rejects unauthenticated requests early (cheap, avoids wasted
  work).
- **Data layer (`DataRetriever`)** — what actually protects against leakage if some
  internal code path bypasses the top-level check.
- Defense in depth: the edge check is an optimization, the data-layer check is the actual
  guarantee.

---

## Production tradeoffs: reliability, latency, cost, fallback

**Q: What's your latency estimate and biggest lever to cut it?**
A: Two sequential Gemini 2.5 Flash calls (planner then synthesizer) plus retrieval in
between — realistically low seconds end-to-end today, dominated by the two LLM round trips
since Flash is already the low-latency tier. The single biggest lever would be streaming the
synthesis response token-by-token (called out in README section 11) — it doesn't reduce
total time but cuts perceived latency substantially since the user sees the answer forming
instead of waiting for the full JSON to arrive.

**Q: Why deterministic risk scoring instead of asking the LLM to rank machines?**
A: [`_score_machine`](agent/capabilities/reliability_analyzer.py#L12-L34) is a
straightforward weighted formula:
- Criticality × 3
- Health × 2
- Active-alert severity, summed
- +5 per urgent open work order
- +6 per overdue work order
- +1 per recurring maintenance issue

Deterministic scoring is reproducible — the same inputs always produce the same ranking,
which matters when a maintenance planner is trusting this to prioritize their week and needs
to trust it won't reorder machines between two runs on identical data. It's also free (no
LLM cost) and explainable — I can show the exact score breakdown in the trace. The LLM's
role is layered on top, in synthesis, to explain the ranking in natural language, not to
compute it.

**Q: The weights (3, 2, 5, 6, 1) look hand-tuned — how would you validate them?**
A: Fair pushback — they're my best judgment, not calibrated against data. To validate, I'd
want historical data: given past prioritization decisions maintenance teams actually made,
does the formula's ranking correlate with what experienced planners chose? That's a
"question it" opportunity — I'd want to backtest the formula against real outcomes (e.g.,
which machines that were ranked highly actually failed or needed emergency intervention)
rather than trusting the weights by inspection.

**Q: What's your fallback if the Gemini API is down or rate-limited?**
A: Honestly, there isn't one today — a failed API call would propagate as an exception. In
production I'd add:
- **Retry-with-backoff** for transient failures.
- **A circuit breaker** so a sustained outage fails fast instead of queuing requests.
- **A degraded-mode response** from the synthesizer that still returns the deterministic
  risk ranking and retrieved records without the natural-language narrative — better to
  give the planner raw data than a hard error.

---

## Hallucination prevention / guardrails

**Q: Which of your 5 safeguards actually *prevents* hallucination vs. just labels it?**
A:
- **Actually prevents** — the [`_SYNTHESIS_SYSTEM`](agent/capabilities/answer_synthesizer.py#L9-L27)
  grounding instruction, and [`validate_machine`](agent/capabilities/data_retriever.py#L12-L13)
  (used in [`retrieve_machine_node`](agent/graph_nodes.py#L121-L162)) — validation stops a
  request before it ever reaches the LLM, so there's nothing to hallucinate about.
- **Detects/labels, doesn't prevent** — confidence gating and the
  [`AssistantResponse`](agent/models.py#L20-L29) Pydantic schema enforcement. They don't
  stop the model from inventing a fact, they just structure the failure so it's visible
  (`confidence: low`) or route to a safe fallback if the shape is wrong.
- Prompt-level "don't invent data" is instruction-following, which models don't obey 100% of
  the time — that's the real gap.

**Q: What would an output rail checking machine IDs actually look like as code?**
A:
1. After synthesis, parse `machines_referenced` out of the [parsed
   response](agent/capabilities/answer_synthesizer.py#L76-L80).
2. Check each ID against the set of machine IDs that were actually in `retrieved_profiles`
   for that request.
3. Any ID that appears in the answer text or `machines_referenced` but wasn't in the
   retrieved context gets flagged or stripped before the response returns.

That's stronger than prompt instructions because it's a code-level invariant check running
after generation — it doesn't rely on the model choosing to comply, it structurally verifies
the output regardless of what the model did.

---

## Behavioral

- **Own it** — I'd talk about the explicit tradeoff table (README section 10): under a
  3-hour budget I chose ChromaDB over Pinecone, CLI over HTTP API, deterministic scoring
  over LLM scoring, and I can defend each choice on its own terms rather than treating them
  as "things I didn't get to." Owning a scoped decision under a deadline, not apologizing
  for it.
- **Question it** — the risk-scoring weights are the clearest example: I picked numbers by
  judgment, not data, and I can say plainly that I'd want to backtest them against real
  prioritization outcomes rather than assume they're right. Pull one from real work history
  too if there's a stronger example with actual before/after evidence.
- **People first** — not really native to a solo take-home; use a real example from work
  history here, not this project.
- **Embrace it / Grow with it** — building the LangGraph parallel fan-out and
  structured-output enforcement under a hard time constraint, and making explicit cut calls
  (no streaming, no multi-turn, no HTTP layer) rather than half-building all of them — a
  concrete "operating well under uncertainty/constraints" story tied to this project.

---

## One gap to have a crisp answer ready for

This is CLI-only with no HTTP API. Given the role's emphasis on API/microservice design,
expect a direct "how would you turn this into a service" follow-up — the FastAPI wrapper
answer above is the one to have ready before walking in.
