from __future__ import annotations
import json
from pathlib import Path
from datetime import date


DATA_DIR = Path(__file__).parent / "data"
TODAY = date.today().isoformat()


class DataLoader:
    """
    Loads all four JSON datasets once at startup and builds in-memory dict indices
    over them for O(1) structured lookup — no re-parsing or re-scanning per request.

    Indices actually built here: 
    - machineId -> machine (`machines_by_id`), 
    - site -> machines (`machines_by_site`), 
    - machineId -> [alerts/work_orders/history] (`*_by_machine`). 
    
    There is deliberately no index by alert/work-order *status* —
    active vs. resolved, overdue vs. on-time, etc. are cheap enough to filter with a
    linear scan at call time (see `get_overdue_work_orders` below, and
    `reliability_analyzer._score_machine`'s `active_alerts` filter), so a dedicated
    status index isn't worth the extra bookkeeping at this dataset size.
    """

    def __init__(self, data_dir: Path = DATA_DIR):
        raw_machines = self._load(data_dir / "machines.json")
        raw_alerts = self._load(data_dir / "alerts.json")
        raw_work_orders = self._load(data_dir / "work_orders.json")
        raw_history = self._load(data_dir / "maintenance_history.json")

        # machineId -> machine dict, O(1) lookup for machine_exists()/get_machine_profile()
        self.machines_by_id: dict[str, dict] = {m["machineId"]: m for m in raw_machines}
        self.all_machines: list[dict] = raw_machines

        # machineId -> [alerts], joined into get_machine_profile()'s "alerts" field
        self.alerts_by_machine: dict[str, list[dict]] = {}
        for a in raw_alerts:
            self.alerts_by_machine.setdefault(a["machineId"], []).append(a)

        # machineId -> [work_orders], joined into get_machine_profile()'s "work_orders" field
        self.work_orders_by_machine: dict[str, list[dict]] = {}
        for wo in raw_work_orders:
            self.work_orders_by_machine.setdefault(wo["machineId"], []).append(wo)

        # machineId -> [maintenance_history], joined into get_machine_profile()'s "maintenance_history" field
        self.history_by_machine: dict[str, list[dict]] = {}
        for h in raw_history:
            self.history_by_machine.setdefault(h["machineId"], []).append(h)

        # site -> [machines], used by get_site_profiles() for site_compare
        self.machines_by_site: dict[str, list[dict]] = {}
        for m in raw_machines:
            self.machines_by_site.setdefault(m["site"], []).append(m)

        # Unindexed flat lists — kept alongside the indices above for callers that need
        # to scan/filter across the whole dataset rather than look up by one key.
        # Actual usage as of this writing:
        #   all_work_orders -> read by get_overdue_work_orders() below (same file),
        #     which is exposed via DataRetriever.get_overdue_work_orders()
        #     (agent/capabilities/data_retriever.py) — but no graph node currently
        #     calls it, so it's a capability that exists but isn't wired into any
        #     live orchestrator flow yet.
        #   all_alerts / all_history -> not read anywhere else in the codebase right
        #     now; every alert/history access goes through alerts_by_machine /
        #     history_by_machine via get_machine_profile() instead. Kept here for
        #     symmetry with all_work_orders and as a hook for a future flow that
        #     needs a fleet-wide alert or history scan.
        self.all_alerts: list[dict] = raw_alerts
        self.all_work_orders: list[dict] = raw_work_orders
        self.all_history: list[dict] = raw_history

    def _load(self, path: Path) -> list:
        with open(path) as f:
            return json.load(f)

    def machine_exists(self, machine_id: str) -> bool:
        return machine_id in self.machines_by_id

    def get_machine_profile(self, machine_id: str) -> dict | None:
        if machine_id not in self.machines_by_id:
            return None
        return {
            "machine": self.machines_by_id[machine_id],
            "alerts": self.alerts_by_machine.get(machine_id, []),
            "work_orders": self.work_orders_by_machine.get(machine_id, []),
            "maintenance_history": self.history_by_machine.get(machine_id, []),
        }

    def get_site_profiles(self, site: str) -> list[dict]:
        machines = self.machines_by_site.get(site, [])
        return [self.get_machine_profile(m["machineId"]) for m in machines]

    def get_overdue_work_orders(self) -> list[dict]:
        return [
            wo for wo in self.all_work_orders
            if wo.get("dueDate", "9999") < TODAY and wo.get("status") != "completed"
        ]

    def known_sites(self) -> list[str]:
        return list(self.machines_by_site.keys())
