from __future__ import annotations
from datetime import date
from ..models import TraceStep

TODAY = date.today().isoformat()


def analyze_work_orders(profiles: list[dict]) -> dict:
    overdue, urgent, open_wos = [], [], []
    for p in profiles:
        if p is None:
            continue
        mid = p["machine"]["machineId"]
        for wo in p["work_orders"]:
            if wo["status"] == "completed":
                continue
            open_wos.append({**wo, "_machineId": mid})
            if wo.get("priority") == "urgent":
                urgent.append({**wo, "_machineId": mid})
            if wo.get("dueDate", "9999") < TODAY:
                overdue.append({**wo, "_machineId": mid})
    return {"open": open_wos, "urgent": urgent, "overdue": overdue}


def analyze_maintenance_history(profile: dict) -> dict: # Seems like a left-over function that is not used anywhere in the codebase!
    history = profile.get("maintenance_history", [])
    recurring = [h for h in history if h.get("outcome") in ("partial_fix", "needs_follow_up")]
    recent = sorted(history, key=lambda h: h["performedAt"], reverse=True)[:5]
    return {
        "total_records": len(history),
        "recurring_issues": len(recurring),
        "recent": recent,
    }


def summarize_maintenance_context(profiles: list[dict]) -> str:
    wo_data = analyze_work_orders(profiles)
    lines = [
        f"Open work orders: {len(wo_data['open'])}",
        f"Urgent work orders: {len(wo_data['urgent'])}",
        f"Overdue work orders: {len(wo_data['overdue'])}",
    ]
    if wo_data["overdue"]:
        for wo in wo_data["overdue"]:
            lines.append(
                f"  OVERDUE: {wo['workOrderId']} for {wo['_machineId']} — "
                f"'{wo['title']}' due {wo['dueDate']}, priority={wo['priority']}"
            )
    if wo_data["urgent"]:
        for wo in wo_data["urgent"]:
            lines.append(
                f"  URGENT: {wo['workOrderId']} for {wo['_machineId']} — "
                f"'{wo['title']}' due {wo['dueDate']}"
            )
    return "\n".join(lines)


def build_trace_step(profiles: list[dict]) -> TraceStep:
    wo_data = analyze_work_orders(profiles)
    machine_ids = [p["machine"]["machineId"] for p in profiles if p]
    return TraceStep(
        step="maintenance_analyzer",
        input=f"work orders and history for {len(machine_ids)} machines",
        output=(
            f"Open WOs: {len(wo_data['open'])}, "
            f"Urgent: {len(wo_data['urgent'])}, "
            f"Overdue: {len(wo_data['overdue'])}"
        ),
        data_sources=["work_orders", "maintenance_history"],
    )
