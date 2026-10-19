from __future__ import annotations
from data_loader import DataLoader
from vector_store import VectorStore
from ..models import TraceStep


class DataRetriever:
    def __init__(self, data_loader: DataLoader, vector_store: VectorStore):
        self._dl = data_loader
        self._vs = vector_store

    def validate_machine(self, machine_id: str) -> bool:
        return self._dl.machine_exists(machine_id)

    def get_machine_profile(self, machine_id: str) -> dict | None:
        return self._dl.get_machine_profile(machine_id)

    def get_site_profiles(self, site: str) -> list[dict]:
        return self._dl.get_site_profiles(site)

    def get_overdue_work_orders(self) -> list[dict]:
        return self._dl.get_overdue_work_orders()

    def semantic_search(self, query: str, n: int = 10) -> list[dict]:
        return self._vs.semantic_search(query, n=n)

    def get_all_machines(self) -> list[dict]:
        return [
            self._dl.get_machine_profile(m["machineId"])
            for m in self._dl.all_machines
        ]

    def build_retrieval_trace(
        self, query: str, retrieved: list[dict]
    ) -> TraceStep:
        machine_ids = [r["metadata"]["machine_id"] for r in retrieved]
        return TraceStep(
            step="data_retriever",
            input=query,
            output=f"Retrieved {len(retrieved)} machine documents: {machine_ids}",
            data_sources=["machines", "alerts", "work_orders", "maintenance_history"],
        )
