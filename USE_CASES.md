# Use Cases

The assistant supports five question intents, classified by the Planner
(`agent/graph_nodes.py`, `route_after_planner`). Each intent routes to a
different subset of capability nodes. This doc walks through each one with a
concrete example, includes a deep dive on the `prioritize` path since its
data flow (fetch all → rank → narrow to cited machines) is the least obvious
one to read from the output JSON alone, and documents a real misclassification
edge case (§5) where a fault-shaped question was routed as `prioritize`
instead of `fault_filter`.

---

## 1. `prioritize` — "Which machines need attention this week?"

**Example:**
```bash
python main.py "Which machines should maintenance prioritize this week, and why?"
```

**Route:** `planner → retrieve_all → [reliability_analyzer ‖ maintenance_analyzer] → synthesizer`

**Organized flow:**
```
"Which machines should maintenance prioritize this week, and why?"
                │
                ▼
          [planner]            intent=prioritize
                │
                ▼
    [retrieve_all_node]        fetches ALL 60 machine profiles
                │              retrieved_profiles = 60 profiles
                ▼
        ┌───────┴────────┐                          ← same superstep,
        ▼                ▼                            run IN PARALLEL
[reliability_analyzer] [maintenance_analyzer]
  scores all 60           summarizes WOs/history
  by weighted formula     across all 60
  writes top 10 as text
  (top_n=10)
        │                │
        └───────┬────────┘
                ▼
   context_blocks = [
     "Machine priority ranking:\n<top 10 lines, one per machine>",
     "Maintenance context:\n<summary spanning all 60>"
   ]
   retrieved_profiles = still all 60 — untouched by the top-10 text cap
                │
                ▼
   [synthesizer node] — two separate sub-steps, not one:
                │
                ├─(a) Gemini LLM call: prompt = context_blocks TEXT ONLY.
                │      retrieved_profiles is never sent to Gemini — the model
                │      only ever sees the 10 named candidates in the text, and
                │      writes the answer + machines_referenced (5 here) from
                │      what it read.
                │
                └─(b) synthesize() Python code, AFTER Gemini responds: filters
                       the untouched 60-profile retrieved_profiles list down to
                       just the IDs in machines_referenced → retrieved_records (5)
```

**Data flow (step-by-step):**
1. `retrieve_all_node` ([graph_nodes.py:168-189](agent/graph_nodes.py#L168-L189)) fetches
   **every** machine profile in the dataset into `retrieved_profiles` (60 machines today).
2. `reliability_analyzer_node` and `maintenance_analyzer_node` run **in parallel** on those
   60 profiles:
   - `reliability_analyzer` scores all 60 with a deterministic weighted formula and writes a
     text summary of the **top 10** into `context_blocks` (`top_n=10`,
     [graph_nodes.py:236](agent/graph_nodes.py#L236)) — this is what the LLM actually reads.
   - `maintenance_analyzer` summarizes work-order/history status across all 60 —
     [graph_nodes.py:242-254](agent/graph_nodes.py#L242-L254) passes the full,
     uncapped `retrieved_profiles` list to
     `summarize_maintenance_context` ([maintenance_analyzer.py:36-55](agent/capabilities/maintenance_analyzer.py#L36-L55)),
     which loops over every profile and writes one line per open urgent/overdue
     work order into `context_blocks[1]` — unlike `reliability_analyzer`, there's
     no `top_n` cap anywhere in this file, so all 60 machines' WOs can surface here.
3. `synthesizer` sends that text summary to Gemini, which writes the answer and reports
   `machines_referenced` — the machine IDs it actually discussed.
4. `answer_synthesizer.synthesize` ([answer_synthesizer.py:76-86](agent/capabilities/answer_synthesizer.py#L76-L86))
   filters the full 60-profile list down to just the machines in `machines_referenced`. That
   filtered list is what ends up in the output JSON's `retrieved_records`.

**Worked example** — [output/20260628_183647_which_machines_should_maintenance_priori.json](output/20260628_183647_which_machines_should_maintenance_priori.json):
- Trace confirms 60 fetched: `"Retrieved 60 machine profiles"`.
- Trace shows `reliability_analyzer` output: `"Top priority machines: [('M-021', 55),
  ('M-006', 49), ('M-011', 33), ('M-005', 31), ('M-034', 31)]"` — only **5** entries. This is
  a red herring if you read it as "5 machines were considered": `reliability_analyzer_node`
  ([graph_nodes.py:234-238](agent/graph_nodes.py#L234-L238)) scores all 60 machines once into
  `scored`, then builds two independent slices of that same list —
  `build_risk_summary(scored, top_n=10)` (10 explicitly, feeds `context_blocks` — what Gemini
  actually reads) and `build_trace_step(scored)` (no `top_n` passed, so it falls back to that
  function's own default of 5 — [reliability_analyzer.py:79](agent/capabilities/reliability_analyzer.py#L79),
  feeds only the trace log). The LLM was shown **10** candidates; the trace line just happens
  to print 5 of them because of an unrelated hardcoded default in a different helper function.
- The LLM's answer discussed 5 of those 10 candidates, so `machines_referenced` had 5 entries.
- `retrieved_records` in the output therefore has **5** entries — not because of a fixed
  top-N cutoff anywhere upstream, but because that's how many machines the answer actually
  cited out of the 10 it was shown. If the answer had named 3 or 8 of the 10, `retrieved_records`
  would have 3 or 8 instead.

**Key point:** "retrieve all" really does load everything into memory (step 1). The number
you see in `retrieved_records` is a *post-hoc narrowing* to what the final answer references
(step 4), not the size of any intermediate retrieval or ranking step. Don't read
`len(retrieved_records)` as "how many records were retrieved" — read the `trace` array for
that; `retrieved_records` answers "which records does this answer actually rely on."

---

## 2. `machine_specific` — "What's going on with M-006?"

**Example:**
```bash
python main.py "What should we do about machine M-006?"
```

**Route:** `planner → retrieve_machine → maintenance_analyzer → synthesizer`
(short-circuits to `synthesizer` directly if the machine ID doesn't exist —
`route_after_retrieve_machine`, [graph_nodes.py:114-116](agent/graph_nodes.py#L114-L116))

**Organized flow:**
```
"What should we do about machine M-006?"
                │
                ▼
          [planner]            intent=machine_specific
                │               entities.machines = ["M-006"]
                ▼
  [retrieve_machine_node]      validate_machine("M-006")
                │
        ┌───────┴────────┐
        ▼                ▼
   not found          found → fetch ONLY M-006's profile
        │                │
        ▼                ▼
  [synthesizer]    [maintenance_analyzer_node]
  canned "not         summarizes WOs/history for M-006 only
  found" response
  (no Gemini call)         │
        │                  ▼
        │        context_blocks = [
        │          "<M-006 machine-doc text>",
        │          "Maintenance context:\n<M-006 only>"
        │        ]
        │        retrieved_profiles = [M-006]   ← only ever 1 profile
        │                  │
        │                  ▼
        │            [synthesizer]        Gemini reads only M-006's data
        │                  │
        │                  ▼
        │        synthesize() narrows retrieved_profiles (1)
        │            → machines_referenced (["M-006"])
        │            → retrieved_records   (nothing extra to narrow away)
        ▼                  ▼
   response          response
```

**Data flow (step-by-step):**
- `retrieve_machine_node` resolves machine IDs from the planner's entities (falling back to a
  regex scan of the question for `M-\d+`), validates each against the dataset, and fetches
  only those profiles — never the full 60.
- `context_blocks` is built from just the requested machine(s)
  ([graph_nodes.py:156](agent/graph_nodes.py#L156)). The same universal `machines_referenced`
  narrowing described in §1 still runs here — every non-short-circuited path ends in the same
  `synthesize()` call — it's just invisible on this path because `retrieved_profiles` was
  already exactly the requested machine(s); there's nothing extra to narrow away.

**Expected output:** `confidence: "high"`, `machines_referenced: ["M-006"]`.

---

## 3. `site_compare` — "Compare Site A and Site B"

**Example:**
```bash
python main.py "How does Site A compare to Site B on reliability?"
```

**Route:** `planner → site_comparator → synthesizer`

**Organized flow:**
```
"How does Site A compare to Site B on reliability?"
                │
                ▼
          [planner]            intent=site_compare
                │               entities.sites = ["Site A", "Site B"]
                ▼
   [site_comparator_node]
                │
                ├─ get_site_profiles("Site A")  →  8 full profiles  ┐
                ├─ get_site_profiles("Site B")  → 14 full profiles  ┤  22 total
                │                                                   │  (retrieved_profiles)
                ├─ sc.compare_sites(profiles_a, profiles_b)         │
                │     aggregate health/alert/WO/recurring counts    │  scope: ALL
                │     computed over the FULL roster of each site    │  22 machines
                │                                                   │
                └─ ra.rank_machines_by_priority(profiles_a)         │
                   ra.rank_machines_by_priority(profiles_b)         │  scope: only
                     sorted by priority_score, top 5 per site kept  │  top 5 + 5 = 10
                     (ra.build_risk_summary(top_n=5))               │  of the 22
                │
                ▼
   context_blocks = [
     "Site comparison: <aggregate text — full 8+14 roster>",
     "Site A machines: <5 named lines>\nSite B machines: <5 named lines>"
   ]
   retrieved_profiles = full union of all 22 profiles (not capped to the top 10)
                │
                ▼
   [synthesizer node] — two separate sub-steps, not one:
                │
                ├─(a) Gemini LLM call: prompt = context_blocks TEXT ONLY.
                │      retrieved_profiles is never sent to Gemini — it can only
                │      name IDs it read in the text (the 10 candidates), never
                │      the other 12. Writes machines_referenced (7 here).
                │
                └─(b) synthesize() Python code, AFTER Gemini responds: filters
                       the untouched 22-profile retrieved_profiles list down to
                       just the IDs in machines_referenced → retrieved_records (7)
```

**Why this matters:** the aggregate block and the per-machine block have *different scopes* —
one covers all 22 machines, the other only the top 10 by priority score. The model can only
name IDs it can see, and it can only see 10 of the 22. See the worked example below for a
case where this caused the model to report a correct aggregate count ("3 warning-health
machines") while naming only 2 IDs, because the third machine fell outside the top-5 cutoff
and was never shown to it.

**Data flow (step-by-step):**
1. `site_comparator_node` ([graph_nodes.py:261-305](agent/graph_nodes.py#L261-L305)) resolves
   two site names (planner entities → regex scan of the question → hardcoded default), then
   calls `retriever.get_site_profiles(site)` for each — every machine at that site, not a
   sample.
2. `sc.compare_sites` aggregates health/alert/work-order counts per site
   ([site_comparator.py:50-53](agent/capabilities/site_comparator.py#L50-L53)), and
   `ra.rank_machines_by_priority` is called **directly** on each site's profile list
   ([graph_nodes.py:284-285](agent/graph_nodes.py#L284-L285)) — this reuses
   `reliability_analyzer`'s scoring function but bypasses the `reliability_analyzer` graph
   node entirely, so this is the one path where that capability runs without its own node
   firing.
3. Only the **top 5 per site** (`top_n=5`, [graph_nodes.py:291-292](agent/graph_nodes.py#L291-L292))
   get written into the text `context_blocks` the LLM reads — 5 + 5 = 10 candidate machines
   total, each with its own line (criticality, health, alert/WO counts, priority_score). This
   is the same *pattern* as `prioritize`'s fleet-wide `top_n=10`
   ([graph_nodes.py:236](agent/graph_nodes.py#L236)) — "only show the LLM the top N so the
   prompt doesn't balloon" — but it's a different setting in a different function; the two
   `10`s are unrelated numbers that happen to coincide here (5 per site × 2 sites), not the
   same `top_n=10` constant reused.
4. `retrieved_profiles` is set to the **full union** of both sites' profiles
   ([graph_nodes.py:288](agent/graph_nodes.py#L288)) — everything fetched in step 1, not just
   the top 5 per side.
5. Same as every other non-short-circuited path, `synthesize()` then narrows that full union
   down to whatever the LLM's `machines_referenced` names
   ([answer_synthesizer.py:76-86](agent/capabilities/answer_synthesizer.py#L76-L86)) before it
   becomes `retrieved_records`.

**Worked example** — [output/20260704_210252_how_does_site_a_compare_to_site_b_on_rel.json](output/20260704_210252_how_does_site_a_compare_to_site_b_on_rel.json):
- Trace: `site_comparator` step reports `"Site A: 1 critical, 11 overdue WOs | Site B: 2
  critical, 11 overdue WOs"` — the aggregate counts from `compare_sites`.
- Trace: `reliability_analyzer` step reports **`"Ranked 8 + 14 machines"`**. That's not a
  cutoff or a bug — it's literally how many machines exist at each site in `data/machines.json`:
  Site A has 8 machines (`M-001, M-006, M-011, M-016, M-036, M-046, M-051, M-056`), Site B has
  14 (`M-002, M-007, M-012, M-017, M-021, M-022, M-026, M-027, M-032, M-037, M-042, M-047,
  M-052, M-057`). The custom `TraceStep` built inline in `site_comparator_node`
  ([graph_nodes.py:296-301](agent/graph_nodes.py#L296-L301)) reports
  `len(profiles_a) + len(profiles_b)` — the full fetched count for each site, before the
  `top_n=5` text summary and before the final `machines_referenced` narrowing. If you asked
  about two other sites, you'd see different numbers reflecting that pair's actual machine
  counts (13 for Site C, 13 for Site D, 12 for Site E).
- Despite 8 + 14 = 22 profiles being fetched and scored, the LLM's answer only discussed 7
  machines. Tracing it further: only 10 of those 22 (the top 5 per site) ever appeared in the
  text the LLM read — Site A's top 5 by priority score are `M-006, M-011, M-016, M-046,
  M-051`; Site B's are `M-021, M-012, M-057, M-002, M-017`. Of those 10 candidates, the model
  named only 7 in its prose (`M-006, M-011, M-016, M-021, M-057, M-012, M-002`) — the ones it
  called out under "Critical Health Status" and "Warning Health Status" in its reasoning. The
  3 it left unnamed (`M-046, M-051, M-017`) are exactly the `monitor`-health machines in each
  site's top 5 — the model reported other metrics (overdue/urgent WO counts, alert severity
  counts) as plain numbers with no IDs attached, and simply never wrote those 3 IDs into the
  answer text. `retrieved_records` in the output has exactly those 7 entries, not 22 and not
  10, because of the same universal narrowing step described in §1: it keeps only whatever
  IDs the model actually typed into `machines_referenced`, not everything it was shown.
- **The aggregate-vs-detail scope gap, proven:** Site A actually has **3** warning-health
  machines — `M-011, M-016, M-036` (confirmed against `data/machines.json`) — and
  `compare_sites` correctly counts all 3 into the aggregate `context_blocks[0]` text ("3
  warning health"). But `M-036` isn't in Site A's top-5-by-priority-score list, so it never
  appears in `context_blocks[1]` at all — the model has no ID for it. That's exactly why the
  answer's reasoning says *"Site A has 3 machines with warning health (M-011, M-016)"* — a
  correct count paired with an incomplete ID list, not a mistake in arithmetic. The gap is
  structural: `compare_sites` scopes over all 8/14 machines, `build_risk_summary(top_n=5)`
  scopes over only the top 5 — the model is working from two blocks with different
  populations and can only name what the narrower one shows it.

**Key point:** the "8 + 14" you see in the trace is the true per-site machine count at the
moment of scoring — a useful sanity check that the right site was resolved and that nothing
was silently truncated. It is *not* the same number as `len(retrieved_records)`, which
reflects only the subset the final answer cites (see the note in §1 and the corrected claim
in the quick-reference table below — this path narrows too, it just wasn't obvious from the
code comments alone).

---

## 4. `fault_filter` — "Which machines have bearing issues?"

**Example:**
```bash
python main.py "Which machines are showing signs of bearing wear?"
```

**Route:** `planner → semantic_search → reliability_analyzer → synthesizer`

**Organized flow:**
```
"Which machines are showing signs of bearing wear?"
                │
                ▼
          [planner]            intent=fault_filter
                │               entities.fault_types = ["bearing"]
                ▼
  [semantic_search_node]       ChromaDB vector similarity query, n=15
                │              → up to 15 (machine_id, similarity) hits
                │              re-fetch each hit's FULL structured profile
                │              from DataRetriever (embedding text is only
                │              used for ranking, then discarded)
                ▼
   retrieved_profiles = up to 15 profiles
                │
                ▼
 [reliability_analyzer_node]   scores those ≤15 profiles, writes a
                │              top-N text summary — top_n=10 (SAME hardcoded
                │              call as §1, graph_nodes.py:236) — this DOES cap
                │              the pool whenever semantic_search returns >10
                ▼
   context_blocks = ["Machine priority ranking:\n<ranked lines from the top
                       ≤10 of the ≤15, by priority_score — NOT all ≤15>"]
   retrieved_profiles = still ≤15 — untouched by the text-summary cap
                │
                ▼
   [synthesizer node] — two separate sub-steps, not one:
                │
                ├─(a) Gemini LLM call: prompt = context_blocks TEXT ONLY (the
                │      ranking text for only the top ≤10 of the ≤15
                │      semantic-search candidates — the other ≤5 are fetched
                │      and scored but never appear in the text).
                │      retrieved_profiles is never sent to Gemini.
                │
                └─(b) synthesize() Python code, AFTER Gemini responds: filters
                       the untouched ≤15-profile retrieved_profiles list down
                       to just the IDs in machines_referenced → retrieved_records
```

**Data flow (step-by-step):**
- `semantic_search_node` ([graph_nodes.py:192-216](agent/graph_nodes.py#L192-L216)) runs a
  ChromaDB vector similarity query (`n=15`) over the embedded machine composite documents,
  then re-fetches each hit's full structured profile from `DataRetriever` (the embedding is
  used only for ranking relevance — scoring downstream needs typed fields, not flattened
  text; see [QA.md](QA.md) for why).
- `reliability_analyzer_node` then scores those ≤15 candidates and writes a text summary
  capped at `top_n=10` — the exact same hardcoded call used on the `prioritize` path
  ([graph_nodes.py:236](agent/graph_nodes.py#L236)). Unlike §1 (where 10 ≤ the 60-machine
  pool, so the cap is the whole story), here the cap can bite *twice*: `semantic_search`
  already narrowed 60 → ≤15, and then this `top_n=10` narrows that ≤15 down further to
  ≤10 whenever the semantic search actually returns more than 10 hits (the normal case for
  `n=15`). Concretely: if all 15 hits come back, only the top 10 by `priority_score` reach
  `context_blocks` — the other 5, however semantically relevant they were to the fault
  query, never appear in the text Gemini reads at all.
- `retrieved_profiles` stays at whatever `semantic_search` returned (up to 15) — this list is
  untouched by the `top_n=10` text cap, so it still holds all ≤15 for the final narrowing
  step. But Gemini's actual visible candidate pool for this path is ≤10, not ≤15.
- The same universal `machines_referenced` narrowing from §1 still applies at the final
  `synthesize()` call — it's just a smaller before/after gap (≤10 candidates shown → however
  many the answer cites) than the 60 → 5 gap on the `prioritize` path, not a path that skips
  narrowing entirely.

---

## 5. Edge case: a fault-shaped question classified as `prioritize`, not `fault_filter`

**Example:**
```bash
python main.py "Are there any machines with bearing or vibration issues that need urgent attention?"
```

This question *reads* like `fault_filter` (it names fault types: bearing, vibration) — but
the trace in
[output/20260628_190137_are_there_any_machines_with_bearing_or_v.json](output/20260628_190137_are_there_any_machines_with_bearing_or_v.json)
shows the Planner actually classified it as `prioritize`:

```
intent=prioritize, entities={'machines': [], 'sites': [], 'fault_types': ['bearing', 'vibration']}.
The user is asking for machines that need 'urgent attention', which falls under
prioritization, and is filtering this request by specific fault types.
```

The Planner reads "need urgent attention" as the dominant signal (→ `prioritize`) even though
it correctly extracted `fault_types: ['bearing', 'vibration']` as entities. Since
`route_after_planner` ([graph_nodes.py:99-111](agent/graph_nodes.py#L99-L111)) dispatches on
`intent` alone — the extracted `fault_types` entities are never read by any router — this
request took the **`retrieve_all` path**, not `semantic_search`:

**Organized flow:**
```
"Are there any machines with bearing or vibration issues that need urgent attention?"
                │
                ▼
          [planner]            intent=prioritize          ← NOT fault_filter
                │               entities.fault_types = ["bearing","vibration"]
                │               (extracted correctly, but never read by any router)
                ▼
   route_after_planner reads ONLY `intent` → dispatches to retrieve_all,
   exactly like §1 — semantic_search / ChromaDB is never invoked
                ▼
    [retrieve_all_node]        fetches ALL 60 machine profiles  (same as §1)
                │
                ▼
        ┌───────┴────────┐
        ▼                ▼
[reliability_analyzer] [maintenance_analyzer]
  top-10 by priority_score, full WO/history text — the words "bearing" and
  unrelated to fault type   "vibration" live only in here, inside raw
  (same node/call as §1)    work-order titles/descriptions
        │                │
        └───────┬────────┘
                ▼
   [synthesizer node] — two separate sub-steps, not one:
                │
                ├─(a) Gemini LLM call: prompt = context_blocks TEXT ONLY.
                │      Gemini pattern-matches "bearing"/"vibration" against WO
                │      titles in that text — a keyword read, not a vector-search
                │      hit. retrieved_profiles is never sent to Gemini.
                │
                └─(b) synthesize() Python code, AFTER Gemini responds: filters
                       the untouched 60-profile retrieved_profiles list down to
                       just the IDs in machines_referenced (5 — only 2 overlap
                       with the top-priority list) → retrieved_records
```

- `retrieve_all_node` fetched all 60 machine profiles (trace: `"Retrieved 60 machine profiles"`).
- `reliability_analyzer` and `maintenance_analyzer` ran in parallel over all 60, exactly as in
  the `prioritize` walkthrough above — it's the literal same node
  ([graph_nodes.py:225-239](agent/graph_nodes.py#L225-L239)), so the same §1 caveat applies
  here: the trace line prints the same top-5
  `[('M-021', 55), ('M-006', 49), ('M-011', 33), ('M-005', 31), ('M-034', 31)]` as the earlier
  "prioritize this week" example only because `build_trace_step(scored)`'s own hardcoded
  default (5) is unrelated to what Gemini reads. `context_blocks` still gets
  `build_risk_summary(scored, top_n=10)` — Gemini was shown the **top 10**, not 5, same as §1.
- No ChromaDB vector search ever ran for this question, despite it being the most
  fault-filter-shaped question in this doc. "Bearing or vibration" was matched by the LLM at
  synthesis time, reading urgent work-order titles/descriptions in the maintenance-analyzer
  text summary — a keyword match inside the prompt, not a semantic search hit.
- The final answer cites 5 machines (M-006, M-021, M-040, M-045, M-053) — only 2 of which
  (M-021, M-006) overlap with the top-5 priority ranking. M-040, M-045, M-053 aren't
  high-priority-score machines at all; they surfaced because their urgent work orders happen
  to mention bearing/vibration-adjacent keywords ("worn bearing," "balance rotating
  assembly," "schedule alignment") in the maintenance context text, and Gemini picked them out
  correctly by reading that text — not because any capability node filtered specifically for
  fault type.

**Why this matters:** the same narrowing behavior described in the `prioritize` deep dive
applies here too (`retrieved_records` = only the 5 `machines_referenced`, filtered from all
60 fetched) — but the *path* that got the data in front of the LLM was the general
"prioritize" summary, not a targeted fault search. It happened to work because Gemini is
reading full work-order titles, not because the system deliberately filtered by fault type.
A rarer fault type not showing up in the top-priority machines' work orders, or one that
uses different wording than the `fault_types` entity, could easily be missed since
`fault_types` is extracted but never actually used to route or filter.

---

## 6. `out_of_scope` — "What's the weather like today?"

**Example:**
```bash
python main.py "Which machine has the best cybersecurity posture?"
```

**Route:** `planner → synthesizer` (no capability nodes run at all)

**Organized flow:**
```
"Which machine has the best cybersecurity posture?"
                │
                ▼
          [planner]            intent=out_of_scope
                │
                ▼
   route_after_planner short-circuits straight to synthesizer —
   no retrieval node, no analyzer node, no capability runs at all
                │
                ▼
          [synthesizer]        canned refusal built in plain Python —
                │               NO Gemini call, NO context_blocks
                ▼
   response = {
     confidence: "low",
     machines_referenced: [],
     retrieved_records: [],
     capabilities_used: []
   }
```

**Data flow:**
- The planner classifies intent as `out_of_scope` and `route_after_planner`
  ([graph_nodes.py:99-111](agent/graph_nodes.py#L99-L111)) sends the request straight to
  `synthesizer`, bypassing every retrieval/analysis node.
- `synthesizer_node`'s out-of-scope branch ([graph_nodes.py:385-405](agent/graph_nodes.py#L385-L405))
  builds a canned refusal in code — no second Gemini call, no `context_blocks`.

**Expected output:** `confidence: "low"`, `machines_referenced: []`, `retrieved_records: []`,
`capabilities_used: []`.

A near-identical but distinct branch handles planner JSON-parse failures (also routes here,
but with an honest "couldn't interpret that" message instead of a scope refusal — see
[QA.md](QA.md) for why those two are kept separate).

---

## Quick reference

`synthesize()` ([answer_synthesizer.py:76-86](agent/capabilities/answer_synthesizer.py#L76-L86))
narrows `retrieved_profiles` down to `machines_referenced` on **every** path that reaches it —
that step isn't `prioritize`-specific. What differs per intent is how many profiles were
fetched *before* that narrowing, which determines how visible the effect is:

| Intent | Entry node | Parallel? | Profiles fetched before narrowing | Narrowing visible in practice? |
|---|---|---|---|---|
| `prioritize` | `retrieve_all` | Yes (reliability + maintenance analyzers) | All machines (60) | Yes — dramatic (e.g. 60 → 5) |
| `machine_specific` | `retrieve_machine` | No | Just the requested machine(s) | No — fetched set already == cited set |
| `site_compare` | `site_comparator` | No | Union of both sites' machines (e.g. 8 + 14 = 22) | Yes — real (e.g. 22 → 7, see §3) |
| `fault_filter` | `semantic_search` | No | Up to 15 (ChromaDB `n=15`) | Sometimes — smaller gap than `prioritize` |
| `prioritize` *(fault-shaped question, §5)* | `retrieve_all` | Yes | All machines (60) | Yes — dramatic (e.g. 60 → 5) |
| `out_of_scope` | *(none — direct to synthesizer)* | No | None — `synthesize()` never runs | N/A — canned response, no narrowing step at all |
