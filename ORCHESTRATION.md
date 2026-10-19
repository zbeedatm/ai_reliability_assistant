# Orchestrator & Graph Nodes — Deep Dive

Companion doc to [orchestrator.py](agent/orchestrator.py) and [graph_nodes.py](agent/graph_nodes.py).
Both files already carry inline comments/docstrings; this doc pulls the non-obvious
mechanics together in one place — the things that aren't clear from reading a single
node in isolation.

---

## 0. Graph topology

Reproduced from the module docstring in [orchestrator.py:29-64](agent/orchestrator.py#L29-L64):

```
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
```

\* Same edge, two triggers: this short-circuit fires both for genuine
out-of-scope questions and for planner JSON-parse failures (see [§4](#4-synthesizer--the-terminal-node-four-branches) below for why they still get distinct user-facing messages).

The `prioritize` path is the one true parallel branch:

```
  retrieve_all ──► reliability_analyzer ─┐
               └──► maintenance_analyzer ─┴──► synthesizer
```

Both analyzer nodes run concurrently; `synthesizer` waits for both (see [§3](#3-parallel-fan-out-and-the-annotatedlist-add-reducer) below for how their outputs merge safely).

> **Where does "waits for both" actually come from?** Not from anything written in
> this repo — there's no explicit wait/join code anywhere in `orchestrator.py`. It's
> a property of LangGraph's own execution model (Pregel-style supersteps): `synthesizer`
> has two incoming edges, `reliability_analyzer → synthesizer` and
> `maintenance_analyzer → synthesizer`
> ([orchestrator.py:153](agent/orchestrator.py#L153) and
> [orchestrator.py:144](agent/orchestrator.py#L144)). When a node has multiple
> incoming edges from nodes active in the same superstep, LangGraph's runtime
> guarantees it only fires once all of them have completed, rather than firing once
> per predecessor. This project's code just declares the two edges; the join
> behavior itself is the framework's guarantee, not application logic. The module
> docstring calls this out directly at
> [orchestrator.py:23-27](agent/orchestrator.py#L23-L27) ("LangGraph knows that
> synthesizer sits downstream of both parallel branches...").

---

## 1. The five intents and where they route

[`route_after_planner`](agent/graph_nodes.py#L97-L109) maps the planner's classified
intent to the first node in that intent's path:

| Intent              | First node          | Retrieval style                              |
|----------------------|----------------------|-----------------------------------------------|
| `machine_specific`   | `retrieve_machine`   | exact ID lookup                                |
| `prioritize`         | `retrieve_all`        | full scan (every machine)                      |
| `fault_filter`       | `semantic_search`     | vector similarity over embedded text           |
| `site_compare`       | `site_comparator`     | exact site-name filter                         |
| `out_of_scope`       | `synthesizer`         | none — short-circuit, no retrieval at all      |

The pattern: **semantic search is used only when the query has no exact key to look
up.** Every other intent already knows precisely what data it needs (an ID, "all of
it", or a site name), so running those through embedding + nearest-neighbor search
would be strictly worse — slower, and less precise, since vector search could return a
similar-but-wrong result instead of the exact one asked for. See the inline note at
[orchestrator.py:127-134](agent/orchestrator.py#L127-L134).

> **Worked example — `prioritize`.** "Which machines should maintenance prioritize
> this week, and why?" classifies as `prioritize`, firing
> `planner → retrieve_all → {reliability_analyzer, maintenance_analyzer} (parallel) →
> synthesizer`. It skips `semantic_search` for the same reason: `semantic_search`
> returns a *ranked subset* (`n=15`, [graph_nodes.py:201](agent/graph_nodes.py#L201))
> based on textual similarity to the query — a machine could be the single highest-risk
> one in the fleet and still get excluded just because its alert/history text doesn't
> textually resemble "prioritize this week." `retrieve_all` instead does a full,
> unfiltered scan of every machine ([graph_nodes.py:166-186](agent/graph_nodes.py#L166-L186)),
> and `reliability_analyzer` applies its deterministic weighted formula across the
> *entire* fleet to rank them (see [QA.md](QA.md) for the formula). That's the only way
> to guarantee the top-N by score are actually the top-N — semantic similarity is the
> wrong tool when the ranking criterion is a structured formula over known fields, not
> "which machines are textually similar to this question."

### `retrieve_all` vs. `semantic_search` — the boundary is a judgment call, not a keyword rule

It's tempting to assume `fault_filter` fires whenever a question names a fault keyword
("bearing", "vibration", "heat") and `prioritize` fires otherwise. That's not the actual
test, and the two intents genuinely overlap on the same words — the planner is making a
semantic judgment about *what the question is asking for*, not pattern-matching
vocabulary:

- **`prioritize` → `retrieve_all`** fires when answering requires *ranking the whole
  fleet against each other* — "which machines should we work on first," "what needs
  urgent attention this week." That requires scoring every machine with the same
  formula and comparing them, so the full unfiltered fleet is the only correct input
  ([graph_nodes.py:167-188](agent/graph_nodes.py#L167-L188)) — see the worked example
  above for why a filtered/ranked-by-similarity subset would silently drop the actual
  highest-risk machine.
- **`fault_filter` → `semantic_search`** fires when the question is *about a
  topic/category* without asking for a fleet-wide ranking — "what's the energy trend
  for motors," "any temperature issues on the pumps." That's a textual/topical match,
  so nearest-neighbor search over the embedded documents is the right tool.

Two real saved runs show the same fault vocabulary landing on opposite sides of that
line:

| Question | Classified as | Why |
|---|---|---|
| "Are there any machines with bearing or vibration issues that need urgent attention?" | `prioritize` (`retrieve_all`) | "need urgent attention" reads as a ranking/triage ask — the fault words describe *which* machines are at risk, but the question is still "who should we act on first" |
| "What is the energy consumption trend for all motors over the last 6 months?" | `fault_filter` (`semantic_search`) | asking about a trend/topic across motors, not "which machines should we fix first" |

Both name specific fault-adjacent terms; only one asks for a fleet-wide priority
ranking. Since this distinction is made by the Planner's Gemini call rather than a
deterministic rule, the boundary can be genuinely fuzzy at the edges — a question
phrased slightly differently than either example above could plausibly land on either
side, and that's an inherent property of LLM-based intent classification, not a bug in
this codebase.

### Why `entities` exists at all

Two of the five intents — `machine_specific` and `site_compare` — need an *exact lookup
key* to do their job at all. Knowing the intent tells you *what kind* of lookup to run,
not *which* machine or site to run it on. Something has to pull "M-006" out of "What
should we do about machine M-006?" — that's `entities`. Without it, `retrieve_machine_node`
would know it's supposed to fetch a machine but have no ID to fetch.

There's already a regex fallback for both cases — `re.findall(r"[Mm]-\d+", ...)` for
machines, `re.findall(r"Site\s+[A-Za-z]", ...)` for sites
([graph_nodes.py:134-136](agent/graph_nodes.py#L134-L136) and
[graph_nodes.py:268-273](agent/graph_nodes.py#L268-L273)) — so the system *could* survive
on regex alone. Two reasons LLM extraction is the primary path instead:
1. Regex only catches the exact patterns it's written for; LLM extraction is more
   robust to phrasing the pattern didn't anticipate.
2. It's essentially free — the Planner already makes one Gemini call to classify
   intent, so piggybacking entity extraction onto the same structured JSON output costs
   no extra round-trip, versus a separate parsing step or second LLM call.

The regex fallback exists as a safety net for when the LLM's extraction misses, not as
the primary path — the same defense-in-depth pattern as `validate_machine()`
re-checking a claim elsewhere in this codebase rather than trusting it outright.

**Honest exception**: `fault_types` doesn't earn its keep the same way. `semantic_search_node`
runs the *raw question text* through vector search — it never reads `entities.fault_types`
at all ([graph_nodes.py:200-201](agent/graph_nodes.py#L200-L201)). Unlike `machines`/`sites`,
nothing downstream is currently blocked without it; it's captured for trace/observability
and as an unused hook for a future hybrid-retrieval extension (where it could become a
`where`-filter input to `semantic_search`'s existing but unwired `extra_filter`
parameter — see [QA.md](QA.md) on hybrid retrieval), not because current control flow
needs it.

### What `entities` actually looks like per intent

`entities` ([models.py:13-17](agent/models.py#L13-L17)) is populated by the Planner's
Gemini call from words in the question text — before any retrieval happens. Pulled
directly from saved runs in `output/*.json` (each row is one real `trace` entry, not a
constructed example):

| Question (paraphrased) | `intent` | `entities` |
|---|---|---|
| "What should we do about machine M-006?" | `machine_specific` | `{'machines': ['M-006'], 'sites': [], 'fault_types': []}` |
| "What should we do about machine M-007?" | `machine_specific` | `{'machines': ['M-007'], ...}` |
| "What should we do about machine MMM-000?" | `machine_specific` | `{'machines': ['MMM-000'], ...}` |
| "Which machines should maintenance prioritize this week?" | `prioritize` | `{'machines': [], 'sites': [], 'fault_types': []}` |
| "Are there machines with bearing or vibration issues needing urgent attention?" | `prioritize` | `{'machines': [], 'sites': [], 'fault_types': ['bearing', 'vibration']}` |
| "Energy consumption trend for motors?" | `fault_filter` | `{'machines': [], 'sites': [], 'fault_types': ['energy consumption', 'motors']}` |
| "Which machine has the best cybersecurity posture?" | `out_of_scope` | `{'machines': [], 'sites': [], 'fault_types': []}` |
| "Which maintenance technician is best qualified?" | `out_of_scope` | `{'machines': ['M-021'], 'sites': [], 'fault_types': []}` |

Three things worth noticing in real data rather than in theory:

1. **`entities` fills in whatever field matches the intent's lookup key** —
   `machine_specific` questions populate `machines`; the fault-related questions
   populate `fault_types`. `sites` is empty in every saved example because none of
   these test questions named a site.
2. **Entity extraction runs independently of intent classification.** The "best
   qualified technician" row classifies as `out_of_scope` (correctly — this assistant
   has no technician data) but still pulls `'M-021'` into `entities.machines`, because
   that ID literally appears in the question text. The Planner extracts what it sees
   in the sentence regardless of which of the 5 buckets it files the question under —
   two separate judgments in the same JSON output, not two stages of one judgment.
3. **`fault_types` is a loose bag of words, not a validated enum.** "energy
   consumption" and "motors" got extracted as fault types even though neither is an
   actual fault category in the dataset (the real ones are `bearing`, `vibration`,
   `temperature_high`, `seal_leak`, etc.). Nothing downstream validates this list
   against real fault types — `semantic_search_node` runs the raw question through
   vector search regardless of what's in `entities.fault_types`
   ([graph_nodes.py:200-201](agent/graph_nodes.py#L200-L201)) — so a made-up fault type
   here doesn't break anything, but `entities.fault_types` isn't reliable ground truth,
   just Gemini's best guess at which words in the question look fault-related.

---

## 2. Full path map (which analyzer nodes actually run per intent)

This is the part that's easy to get wrong just by skimming `add_edge` calls, because
the edges are declared in several places across `_build_graph`.

| Intent              | Path                                                                 | reliability_analyzer? | maintenance_analyzer? |
|-----------------------|-----------------------------------------------------------------------|:---:|:---:|
| `machine_specific`   | `retrieve_machine → maintenance_analyzer → synthesizer`             | ❌ | ✅ |
| `prioritize`         | `retrieve_all ⇉ {reliability_analyzer, maintenance_analyzer} → synthesizer` | ✅ | ✅ |
| `fault_filter`       | `semantic_search → reliability_analyzer → synthesizer`               | ✅ | ❌ |
| `site_compare`       | `site_comparator → synthesizer` (scoring done inline via `ra.rank_machines_by_priority`, not the graph node) | — | ❌ |
| `out_of_scope`       | `synthesizer` only (canned response)                                  | ❌ | ❌ |

Two things worth calling out explicitly:

- **`prioritize` is the only path with true parallel fan-out.** `retrieve_all` fires
  both analyzer nodes in the same LangGraph superstep
  ([orchestrator.py:151-152](agent/orchestrator.py#L151-L152)). Every other path is a
  straight line — one analyzer, or none.
- **`fault_filter` never runs `maintenance_analyzer`.** A question like "machines with
  bearing issues" gets a risk ranking but no maintenance/work-order summary in the
  synthesis context. This isn't obviously intentional — flagged inline at
  [orchestrator.py:159-164](agent/orchestrator.py#L159-L164) — and is worth revisiting
  if fault_filter answers ever need maintenance history.
- **`site_comparator` calls `ra.rank_machines_by_priority` directly**, as a plain
  function call, rather than routing through the `reliability_analyzer` graph node —
  so it shows up in `capabilities_used` but never touches that edge of the graph. See
  [graph_nodes.py:258-293](agent/graph_nodes.py#L258-L293).

  **Why not just route through the node?** `reliability_analyzer_node` is a thin
  `GraphState` adapter shaped for one input → one output: it scores a single
  `retrieved_profiles` list and writes one `top_n=10` context block
  ([graph_nodes.py:233-238](agent/graph_nodes.py#L233-L238)). `site_comparator_node`
  needs something the node doesn't produce — two independent rankings (site A, site B)
  combined into a single paired block at `top_n=5` each, sitting alongside the
  site-vs-site comparison text. Getting that from the graph node would mean adding two
  more edges/supersteps just to invoke the same node twice with different slices, then
  manually re-merging what the `Annotated[list, add]` reducer would otherwise
  concatenate as two separate, unpaired blocks — more graph complexity for a pure
  formatting need. Since `ra.rank_machines_by_priority` /
  `ra.build_risk_summary` ([reliability_analyzer.py:37](agent/capabilities/reliability_analyzer.py#L37),
  [reliability_analyzer.py:47](agent/capabilities/reliability_analyzer.py#L47)) are
  already plain, state-free functions — the node is just one caller of them, not the
  only legitimate entry point — calling them directly twice and composing the result
  the way this path needs is simpler and equally correct. That's also why
  `capabilities_used` still records `"reliability_analyzer"` even though the graph edge
  is untouched: the *capability* (scoring logic) is reused; only the *node wrapper* is
  bypassed.

---

## 3. Parallel fan-out and the `Annotated[list, add]` reducer

For `prioritize`, `retrieve_all` produces `retrieved_profiles`, then both
`reliability_analyzer` and `maintenance_analyzer` run **in the same superstep** and
each returns its own partial state update:

```python
# reliability_analyzer_node returns:
{"context_blocks": ["reliability analysis text..."]}

# maintenance_analyzer_node returns:
{"context_blocks": ["maintenance analysis text..."]}
```

**Without a reducer, LangGraph state updates are last-write-wins.** If two nodes
update the same key in the same step and there's no merge function, one write just
overwrites the other — and since parallel nodes don't finish in a guaranteed order,
whichever one "wins" is non-deterministic. That's the race condition: not a low-level
threading bug, but silent data loss where one analyzer's contribution vanishes
depending on timing.

[`graph_state.py:15-18`](agent/graph_state.py#L15-L18) types `context_blocks`,
`trace`, `retrieved_profiles`, and `capabilities_used` as `Annotated[list, add]`. This
tells LangGraph: when multiple nodes write to this key in the same step, concatenate
the lists (`operator.add`) instead of overwriting:

```python
context_blocks = [] + ["reliability text..."] + ["maintenance text..."]
```

Order between the two doesn't affect correctness — concatenation gives you both either
way, it only changes which appears first. `synthesizer` only fires once both
predecessors in that superstep complete, and it's guaranteed to see both
contributions merged, not just whichever finished last.

`response` and `not_found_machine` are **not** `Annotated[list, add]` — they're
single-writer fields (exactly one node ever sets them per run), so last-write-wins is
fine and no reducer is needed.

### What actually goes into a context block

`context_blocks: Annotated[list[str], add]` ([graph_state.py:15](agent/graph_state.py#L15))
is the grounding context that gets handed to the Synthesizer's Gemini call — it's the
`Retrieved data:` section of the final prompt. Each entry is one human-readable text
block; which node appends what depends on which capability ran:

| Node | What it appends |
|---|---|
| `retrieve_machine_node` | `[_machine_doc_text(p) for p in profiles]` — one composite text block per looked-up machine, via [`_machine_doc_text`](vector_store.py#L13) |
| `semantic_search_node` | Same — one [`_machine_doc_text`](vector_store.py#L13) block per semantically-matched machine |
| `reliability_analyzer_node` | One block: `"Machine priority ranking:\n" + <formatted risk-ranking table>` — a *summary* it built, not raw retrieved data |
| `maintenance_analyzer_node` | One block: `"Maintenance context:\n" + <work-order/history summary>` |
| `site_comparator_node` | Two blocks: the site-vs-site comparison, plus per-site risk rankings |
| `retrieve_all_node` | **None** — deliberately populates `retrieved_profiles` only; the analyzer nodes build their own summaries from that |

`synthesize()` ([answer_synthesizer.py:38](agent/capabilities/answer_synthesizer.py#L38))
does `"\n\n---\n\n".join(context_blocks)` and drops that straight into the prompt. You
can see this concretely in a real saved run: the `machine_specific` path for M-006 has
`"input": "question + 2 context blocks"` in its trace
([output/20260628_185817_...json:213](output/20260628_185817_what_should_we_do_about_machine_m_006.json#L213))
— one block from `retrieve_machine_node` (the machine's profile) and one from
`maintenance_analyzer_node` (the work-order/history summary), matching the
`machine_specific → maintenance_analyzer` path exactly.

**The `N` in `"question + N context blocks"` isn't a `prioritize`-specific number —
it's just `len(context_blocks)` for whatever path ran**
([answer_synthesizer.py:69](agent/capabilities/answer_synthesizer.py#L69)), so it
varies per intent based on the node table above:

| Intent | Nodes writing to `context_blocks` | `N` reported in this trace step |
|---|---|:---:|
| `prioritize` | `reliability_analyzer` + `maintenance_analyzer` | 2 |
| `machine_specific` | `retrieve_machine` (machine doc) + `maintenance_analyzer` | 2 |
| `site_compare` | `site_comparator` (2 blocks: comparison + per-site rankings) | 2 |
| `fault_filter` | `reliability_analyzer` only — this path never runs `maintenance_analyzer` | 1 |
| `prioritize`, fault-shaped (misclassified `fault_filter` question) | `reliability_analyzer` + `maintenance_analyzer` | 2 |
| `out_of_scope` | none — `synthesize()`/Gemini never called, canned response | *(step doesn't exist)* |

2 is the most common value (four of six paths), which makes it easy to mistake for a
constant — but a genuine `fault_filter` question should show `"question + 1 context
blocks"` here, since that route only ever produces the risk-ranking block.

---

## 4. `synthesizer` — the terminal node, four branches

Every path in the graph converges on `synthesizer → END`
([orchestrator.py:169](agent/orchestrator.py#L169)). Inside
[`synthesizer_node`](agent/graph_nodes.py#L309-L414) there are four mutually
exclusive branches, checked in order:

1. **`not_found_machine` is set** → canned "machine not found" response
   ([graph_nodes.py:335-353](agent/graph_nodes.py#L335-L353)). Reached when
   `retrieve_machine` couldn't validate a machine ID
   ([graph_nodes.py:137-148](agent/graph_nodes.py#L137-L148)) and
   `route_after_retrieve_machine` short-circuits past `maintenance_analyzer`
   ([graph_nodes.py:112-114](agent/graph_nodes.py#L112-L114)).
2. **`planner_parse_failed` is set** → canned "couldn't interpret" response
   ([graph_nodes.py:355-378](agent/graph_nodes.py#L355-L378)). Reached when the
   planner's Gemini call returned unparseable JSON
   ([graph_nodes.py:70-80](agent/graph_nodes.py#L70-L80)); the planner still sets
   `intent="out_of_scope"` so routing fails safe with zero capability calls, but this
   branch is checked *before* branch 3 so the response text stays honest about what
   actually went wrong (a parse failure, not necessarily an out-of-scope question).
3. **`plan.intent == "out_of_scope"`** → canned refusal response
   ([graph_nodes.py:380-401](agent/graph_nodes.py#L380-L401)). Reached directly from
   the planner via `route_after_planner`
   ([graph_nodes.py:104](agent/graph_nodes.py#L104)) — no retrieval or analysis nodes
   run at all for this path.
4. **Otherwise** → real Gemini synthesis over the collected `context_blocks`
   ([graph_nodes.py:404-411](agent/graph_nodes.py#L404-L411)).

Branches 1-3 skip the LLM call entirely and are built deterministically in code:
there's no reliability data to reason over, so a model call would either hallucinate
or just restate the same conclusion — paying for a call that adds no value. Branches 2
and 3 both route through the same `intent == "out_of_scope"` short-circuit but are kept
as separate branches specifically so their user-facing messages don't get conflated —
see the [QA.md](QA.md) entry on the planner's parse-failure fallback for why that
distinction matters. Branch 4 is the only path where the model actually writes the
answer text, and it's grounded only in what upstream nodes retrieved — the model never
sees data outside `context_blocks`.

---

## 5. Quick reference — every node's role

| Node | Path(s) | Role |
|---|---|---|
| `planner` | entry, all paths | Classifies intent + extracts entities via one Gemini call. Drives all routing. On JSON parse failure, fails safe as `out_of_scope` (flagged via `planner_parse_failed` for an honest downstream message) instead of guessing `fault_filter`. |
| `retrieve_machine` | `machine_specific` | Exact-ID lookup + existence validation. Short-circuits to `synthesizer` if the ID isn't found. |
| `retrieve_all` | `prioritize` | Fetches every machine profile. Fan-out point for parallel analyzers. |
| `semantic_search` | `fault_filter` | Vector similarity search, then re-fetches structured profiles per hit (embedding text is for ranking, not scoring). |
| `reliability_analyzer` | `prioritize` (parallel), `fault_filter` | Deterministic, no-LLM risk scoring/ranking. |
| `maintenance_analyzer` | `prioritize` (parallel), `machine_specific` | Deterministic, no-LLM work-order/maintenance-history summary. |
| `site_comparator` | `site_compare` | Resolves two site names, pulls both sites' profiles, combines comparison + inline risk ranking. |
| `synthesizer` | terminal, all paths | Four branches: not-found short-circuit, planner-parse-failure short-circuit, out-of-scope short-circuit, or real LLM synthesis. |

---

## 6. Scaling `retrieve_all` past a few thousand machines

`retrieve_all_node` ([graph_nodes.py:167-188](agent/graph_nodes.py#L167-L188)) backs the
`prioritize` path — the one branch that scans the *entire* fleet instead of an exact
key lookup (see [§1](#1-the-five-intents-and-where-they-route)). That raises an obvious
question: what happens as the fleet grows?

### At ~10k machines: current design holds up fine

- [`data_loader.py`](data_loader.py) loads all JSON into memory once at startup as
  plain dicts (`machines_by_id`, `alerts_by_machine`, …) — O(1) lookups, a few MB of
  RAM at 10k machines.
- `get_all_machines()` → `get_machine_profile()` per machine is O(n) dict lookups, no
  LLM calls involved — sub-second at 10k.
- `rank_machines_by_priority` (reliability_analyzer.py:52)
  is pure Python scoring + one sort, O(n log n) — still sub-second at 10k.
- Gemini never sees all 10k machines: ([`build_risk_summary(scored, top_n=10)`](agent/capabilities/reliability_analyzer.py#L62)) only
  sends the top 10 into the synthesis prompt, so LLM cost/latency doesn't scale with
  fleet size. This is a deliberate and valuable property of the current design. Called
  from [graph_nodes.py:235](agent/graph_nodes.py#L235) (`reliability_analyzer_node`,
  `top_n=10`) and twice from [graph_nodes.py:290-291](agent/graph_nodes.py#L290-L291)
  (`site_comparator_node`, `top_n=5` each, once per site).
- `builder.compile()` ([orchestrator.py:171](agent/orchestrator.py#L171)) is called
  with no checkpointer, so there's no per-superstep state serialization to a DB —
  `retrieved_profiles` just lives in process memory for the duration of one request.

**What's wasteful but not yet breaking:** `retrieve_all_node` eagerly builds *full*
profiles (machine + all alerts + all work orders + all history) for every one of the
10k machines even though only the top 10 ever reach the synthesizer — the other 9,990
get scored and discarded. And every `prioritize` request recomputes all of this from
scratch; there's no caching or incremental update between requests.

### At ~100k+: move the ranking out of the request path

The fix isn't to make the in-memory scan faster — it's to stop doing a full-fleet scan
per request at all:

1. **Two-phase fetch.** Split `get_all_machines()` into a lightweight scoring fetch
   (just `criticality`, `healthStatus`, alert severities, WO due dates/priorities) and a
   hydration step that pulls full profiles (alerts, work orders, history) only for the
   winning top-N *after* ranking. Full profiles for 99,990 discarded machines never get
   built.
2. **Push ranking into a real datastore.** Replace the static JSON files with
   Postgres/DuckDB and a `priority_score` column; `retrieve_all_node` becomes
   `SELECT ... ORDER BY priority_score DESC LIMIT N` instead of loading and sorting the
   whole fleet in Python on every call.
3. **Compute scores incrementally, not per-query.** Update `priority_score` when an
   alert or work order changes (write-time trigger / event hook), rather than
   recomputing `_score_machine` for the entire fleet synchronously inside the request.
   Turns an O(n) rescan per `prioritize` question into an O(1) leaderboard read.

**Tradeoff:** this trades the current design's simplicity (no schema, no DB round-trip,
pure dict lookups) for the ability to scale past a single process's RAM/CPU and to
serve concurrent requests without each one re-scanning the entire fleet. Worth doing
once fleet size or concurrent load actually gets there — for a fleet in the low tens of
thousands with single-user/demo-style traffic, the current in-memory approach is likely
still fine as-is, and #1 (two-phase fetch) is the one change worth making early since
it's a small refactor with no infra change required.

---

## See also

- [QA.md](QA.md) — interview-prep framing of these same decisions (why LangGraph, why
  deterministic scoring, tradeoffs, etc.)
- [README.md](README.md) — project overview and setup
