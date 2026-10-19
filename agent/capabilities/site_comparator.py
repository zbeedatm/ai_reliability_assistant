from __future__ import annotations
from datetime import date
from ..models import TraceStep

TODAY = date.today().isoformat()


def _site_stats(profiles: list[dict]) -> dict:
    health_counts: dict[str, int] = {}
    alert_severity_counts: dict[str, int] = {}
    overdue_wos = 0
    urgent_wos = 0
    recurring_issues = 0

    for p in profiles:
        if p is None:
            continue
        m = p["machine"]
        hs = m.get("healthStatus", "unknown")
        health_counts[hs] = health_counts.get(hs, 0) + 1

        for a in p["alerts"]:
            if a["status"] == "active":
                sev = a.get("severity", "low")
                alert_severity_counts[sev] = alert_severity_counts.get(sev, 0) + 1

        for wo in p["work_orders"]:
            if wo["status"] == "completed":
                continue
            if wo.get("priority") == "urgent":
                urgent_wos += 1
            if wo.get("dueDate", "9999") < TODAY:
                overdue_wos += 1

        recurring_issues += sum(
            1 for h in p["maintenance_history"]
            if h.get("outcome") in ("partial_fix", "needs_follow_up")
        )

    return {
        "machine_count": len(profiles),
        "health_counts": health_counts,
        "active_alerts_by_severity": alert_severity_counts,
        "overdue_work_orders": overdue_wos,
        "urgent_work_orders": urgent_wos,
        "recurring_issues": recurring_issues,
    }


def compare_sites(site_a_profiles: list[dict], site_b_profiles: list[dict]) -> dict:
    stats_a = _site_stats(site_a_profiles)
    stats_b = _site_stats(site_b_profiles)
    return {"site_a": stats_a, "site_b": stats_b}


def format_comparison(site_a: str, site_b: str, comparison: dict) -> str:
    def fmt(name: str, stats: dict) -> str:
        critical_health = stats["health_counts"].get("critical", 0)
        warning_health = stats["health_counts"].get("warning", 0)
        critical_alerts = stats["active_alerts_by_severity"].get("critical", 0)
        high_alerts = stats["active_alerts_by_severity"].get("high", 0)
        return (
            f"{name}: {stats['machine_count']} machines, "
            f"{critical_health} critical health / {warning_health} warning health, "
            f"{critical_alerts} critical alerts / {high_alerts} high alerts, "
            f"{stats['overdue_work_orders']} overdue WOs, "
            f"{stats['urgent_work_orders']} urgent WOs, "
            f"{stats['recurring_issues']} recurring maintenance issues"
        )

    return fmt(site_a, comparison["site_a"]) + "\n" + fmt(site_b, comparison["site_b"])


def build_trace_step(site_a: str, site_b: str, comparison: dict) -> TraceStep:
    a = comparison["site_a"]
    b = comparison["site_b"]
    return TraceStep(
        step="site_comparator",
        input=f"profiles for {site_a} and {site_b}",
        output=(
            f"{site_a}: {a['health_counts'].get('critical', 0)} critical, "
            f"{a['overdue_work_orders']} overdue WOs | "
            f"{site_b}: {b['health_counts'].get('critical', 0)} critical, "
            f"{b['overdue_work_orders']} overdue WOs"
        ),
        data_sources=["machines", "alerts", "work_orders", "maintenance_history"],
    )
