# Issues to Fix

Known inconsistencies/dead code found while documenting the system's data flow
(see [USE_CASES.md](USE_CASES.md)). Not bugs in the sense of wrong answers today,
but each is either misleading or silently doing nothing.

---

## 1. `build_risk_summary` and `build_trace_step` disagree on default `top_n`

**Where:**
- [`reliability_analyzer.py:62`](agent/capabilities/reliability_analyzer.py#L62) —
  `def build_risk_summary(scored: list[dict], top_n: int = 10) -> str`
- [`reliability_analyzer.py:79`](agent/capabilities/reliability_analyzer.py#L79) —
  `def build_trace_step(scored: list[dict], top_n: int = 5) -> TraceStep`

**The mismatch:** two independently-defaulted function signatures in the same
file, never reconciled by any caller.

- `ra.build_trace_step(scored)` is called in exactly one place —
  [`graph_nodes.py:237`](agent/graph_nodes.py#L237) — always with **no** `top_n`
  argument, so it silently falls back to its own default of 5.
- `ra.build_risk_summary` is called twice, always with an **explicit** `top_n`:
  `top_n=10` on the `prioritize` / fault-shaped path
  ([`graph_nodes.py:236`](agent/graph_nodes.py#L236)), `top_n=5` on the
  `site_compare` path ([`graph_nodes.py:291-292`](agent/graph_nodes.py#L291-L292)).
- Nothing ever passes `top_n=10` to `build_trace_step` to keep it in sync with
  `build_risk_summary`.

This reads as an oversight, not a deliberate design choice — most likely
`build_trace_step` was written with its own "just show a handful for the trace
log" default and nobody updated it when `build_risk_summary`'s default (or its
call-site override) was set to 10.

**Effect on the LLM answer:**
- Gemini's prompt is built from `context_blocks`, which contains
  `build_risk_summary`'s **top 10** — that's the actual candidate pool Gemini
  reasons over.
- The trace log (never sent to Gemini) shows `build_trace_step`'s **top 5** —
  purely a logging artifact of the mismatched default.
- Gemini then writes `machines_referenced`, citing however many of those 10
  candidates it chose to discuss (5, in the worked examples in
  [USE_CASES.md](USE_CASES.md) §1 and §5) — that number is coincidentally equal
  to the trace's hardcoded 5, but for an unrelated reason (how many the model
  happened to cite, not a cap).

**No correctness bug for the answer itself** (Gemini does get 10 to work with),
but the trace log is misleading — anyone reading `"Top priority machines: [...]"`
(5 entries) and assuming that's what Gemini saw would undercount by half.

**Suggested fix:** make `build_trace_step`'s default `10` (or have the
[`graph_nodes.py:237`](agent/graph_nodes.py#L237) call site pass `top_n=10`
explicitly) so the trace and the prompt agree.

---

## 2. `analyze_maintenance_history` is dead code

**Where:** [`maintenance_analyzer.py:25-31`](agent/capabilities/maintenance_analyzer.py#L25-L31)

**The issue:** this function is never called anywhere in the codebase — not
from `summarize_maintenance_context`, not from `build_trace_step`, not from any
graph node. Confirmed by searching the whole repo for its name; the only match
is its own `def`.

**Effect:** `maintenance_analyzer`'s actual output (both `context_blocks` and
the trace) is built entirely from `analyze_work_orders` — work orders only. The
`maintenance_history` detail this dead function computes (recurring-issue count,
5 most recent records) never reaches Gemini through this module. The only place
`maintenance_history` data reaches the LLM at all is indirectly, via
`reliability_analyzer`'s `_score_machine` — the `recurring` count is folded into
`priority_score` as a single number
([`reliability_analyzer.py:43-47`](agent/capabilities/reliability_analyzer.py#L43-L47)),
not surfaced as the detailed history this function would provide.

**Suggested fix:** either wire `analyze_maintenance_history`'s output into
`summarize_maintenance_context` (if the intent was to show recurring-issue
detail per machine), or delete the dead function if it was superseded and
never cleaned up.



---

# Known Limitations

Structural gaps that follow from deliberate design choices — nothing here is
broken code, and each behaves exactly as written. Listed separately from
"Issues to Fix" above because there's no clear-cut fix that isn't a tradeoff
(e.g. prompt size vs. completeness), whereas the issues above are
straightforward inconsistencies to reconcile.

---

## 1. `site_compare` answers can cite a count that doesn't match its own ID list

**Where:** [`graph_nodes.py:283-292`](agent/graph_nodes.py#L283-L292) —
`site_comparator_node` builds two `context_blocks` with different scopes:
`sc.compare_sites(profiles_a, profiles_b)` aggregates over the **full** site
roster (8 + 14 machines), while `ra.build_risk_summary(scored_a, top_n=5)` /
`(scored_b, top_n=5)` only names the **top 5 per site** by priority score.

**Not a code bug** — each function does exactly what it's written to do, and
the model faithfully reports both the aggregate count and the IDs it was
given. The gap is a byproduct of design, not broken logic.

**Worked example** —
[output/20260704_210252_how_does_site_a_compare_to_site_b_on_rel.json](output/20260704_210252_how_does_site_a_compare_to_site_b_on_rel.json):
Site A actually has 3 warning-health machines — `M-011, M-016, M-036`
(confirmed against `data/machines.json`) — and `compare_sites` correctly
counts all 3 into the aggregate text ("3 warning health"). But `M-036` isn't
in Site A's top-5-by-priority-score list, so it never appears in the
per-machine block at all — the model has no ID for it. The answer's reasoning
ends up saying *"Site A has 3 machines with warning health (M-011, M-016)"* —
a correct count paired with an incomplete ID list. See [USE_CASES.md](USE_CASES.md)
§3 for the full trace-by-trace walkthrough.

**Why it matters:** to a reader who doesn't know the internals, "3 machines
(2 named)" reads like a counting error even though it isn't one. Nothing in
the response — no `limitations_or_assumptions` entry, no caveat — flags that
the ID list may be incomplete relative to the count. A maintenance planner has
no way to know a third warning-health machine exists but wasn't named.

**Possible mitigation:** either (a) prompt Gemini to explicitly flag when a
count exceeds the number of IDs it was given for that category, or (b) raise
`top_n` in the per-site `build_risk_summary` calls to cover the full roster
when a site's machine count is small enough that prompt size isn't a concern
(e.g. use the aggregate counts from `compare_sites` to decide whether top-5 is
actually a safe truncation for that request).
