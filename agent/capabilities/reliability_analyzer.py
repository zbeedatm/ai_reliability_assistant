from __future__ import annotations
from datetime import date
from ..models import TraceStep

TODAY = date.today().isoformat()

_CRITICALITY_SCORE = {"critical": 4, "high": 3, "medium": 2, "low": 1}
_HEALTH_SCORE = {"critical": 4, "warning": 3, "monitor": 2, "healthy": 0}
_SEVERITY_SCORE = {"critical": 4, "high": 3, "medium": 2, "low": 1}



def _score_machine(profile: dict) -> int:
    """
    Deterministic weighted-sum risk score for one machine (higher = more urgent).
    Weights criticality x3 and health status x2 (baseline risk), adds severity points
    per active alert, +5 for any urgent open work order and +6 per overdue one, and
    +1 per maintenance-history record that didn't fully resolve the issue. No LLM
    call — used by rank_machines_by_priority to order the whole fleet.

    Criticality × 3
    Health × 2
    Active-alert severity, summed
    +5 per urgent open work order
    +6 per overdue work order
    +1 per recurring maintenance issue
    """
    m = profile["machine"]
    score = _CRITICALITY_SCORE.get(m.get("criticality", "low"), 1) * 3
    score += _HEALTH_SCORE.get(m.get("healthStatus", "healthy"), 0) * 2

    active_alerts = [a for a in profile["alerts"] if a["status"] == "active"]
    for alert in active_alerts:
        score += _SEVERITY_SCORE.get(alert.get("severity", "low"), 1)

    open_wos = [wo for wo in profile["work_orders"] if wo["status"] != "completed"]
    for wo in open_wos:
        if wo.get("priority") == "urgent":
            score += 5
        if wo.get("dueDate", "9999") < TODAY:
            score += 6  # overdue penalty

    recurring = sum(
        1 for h in profile["maintenance_history"]
        if h.get("outcome") in ("partial_fix", "needs_follow_up")
    )
    score += recurring

    return score


def rank_machines_by_priority(profiles: list[dict]) -> list[dict]:
    scored = [
        {"profile": p, "priority_score": _score_machine(p)}
        for p in profiles
        if p is not None
    ]
    scored.sort(key=lambda x: x["priority_score"], reverse=True)
    return scored


def build_risk_summary(scored: list[dict], top_n: int = 10) -> str:
    lines = []
    for rank, item in enumerate(scored[:top_n], 1):
        p = item["profile"]
        m = p["machine"]
        active_alerts = [a for a in p["alerts"] if a["status"] == "active"]
        open_wos = [wo for wo in p["work_orders"] if wo["status"] != "completed"]
        overdue_wos = [wo for wo in open_wos if wo.get("dueDate", "9999") < TODAY]
        lines.append(
            f"{rank}. {m['machineId']} ({m['machineType']}, {m['site']}): "
            f"criticality={m['criticality']}, health={m['healthStatus']}, "
            f"active_alerts={len(active_alerts)}, open_WOs={len(open_wos)}, "
            f"overdue_WOs={len(overdue_wos)}, priority_score={item['priority_score']}"
        )
    return "\n".join(lines)


def build_trace_step(scored: list[dict], top_n: int = 5) -> TraceStep:
    top = scored[:top_n]
    machine_ids = [x["profile"]["machine"]["machineId"] for x in top]
    scores = [x["priority_score"] for x in top]
    return TraceStep(
        step="reliability_analyzer",
        input="all machine profiles with alerts and work orders",
        output=f"Top priority machines: {list(zip(machine_ids, scores))}",
        data_sources=["machines", "alerts", "work_orders", "maintenance_history"],
    )
