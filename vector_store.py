from __future__ import annotations
import chromadb
from chromadb.config import Settings
from pathlib import Path
from data_loader import DataLoader


CHROMA_PATH = Path(__file__).parent / ".chroma_db"
COLLECTION_NAME = "reliability_data"
TENANT_ID = "default"  # single tenant for assignment; production would be per-request


def _machine_doc_text(profile: dict) -> str:
    m = profile["machine"]
    lines = [
        f"Machine {m['machineId']}: {m['name']} ({m['machineType']}) at {m['site']}, {m['area']}.",
        f"Manufacturer: {m['manufacturer']} model {m['model']}. Installed: {m['installDate']}.",
        f"Criticality: {m['criticality']}. Health status: {m['healthStatus']}. Operating hours: {m['operatingHours']}.",
        f"Notes: {m.get('notes', 'None')}.",
    ]

    active_alerts = [a for a in profile["alerts"] if a["status"] == "active"]
    if active_alerts:
        lines.append("Active alerts:")
        for a in active_alerts:
            lines.append(
                f"  [{a['severity'].upper()}] {a['faultType']} — {a['description']} "
                f"(value={a['value']}, threshold={a['threshold']}). "
                f"Action: {a.get('recommendedAction', '')}."
            )
    else:
        lines.append("No active alerts.")

    open_wos = [wo for wo in profile["work_orders"] if wo["status"] != "completed"]
    if open_wos:
        lines.append("Open/in-progress work orders:")
        for wo in open_wos:
            lines.append(
                f"  [{wo['priority'].upper()}] {wo['workOrderId']}: {wo['title']} — due {wo['dueDate']}, "
                f"status={wo['status']}, team={wo['assignedTeam']}. {wo['description']}"
            )
    else:
        lines.append("No open work orders.")

    recent_history = sorted(
        profile["maintenance_history"], key=lambda h: h["performedAt"], reverse=True
    )[:3]
    if recent_history:
        lines.append("Recent maintenance history:")
        for h in recent_history:
            lines.append(
                f"  {h['performedAt']} — {h['action']}: outcome={h['outcome']}. "
                f"Findings: {h['findings']} Notes: {h.get('technicianNotes', '')}."
            )
    else:
        lines.append("No maintenance history.")

    return "\n".join(lines)


def _machine_metadata(profile: dict, tenant_id: str) -> dict:
    m = profile["machine"]
    active_alerts = [a for a in profile["alerts"] if a["status"] == "active"]
    fault_types = list({a["faultType"] for a in active_alerts})
    return {
        "machine_id": m["machineId"],
        "site": m["site"],
        "criticality": m["criticality"],
        "health_status": m["healthStatus"],
        "machine_type": m["machineType"],
        "fault_types": ",".join(fault_types) if fault_types else "",
        "tenant_id": tenant_id,
        "document_type": "machine_composite",
    }


class VectorStore:
    def __init__(self, data_loader: DataLoader, tenant_id: str = TENANT_ID):
        self.tenant_id = tenant_id
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_PATH),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._data_loader = data_loader
        self._ensure_indexed()

    def _ensure_indexed(self):
        existing = self._collection.get(
            where={"tenant_id": {"$eq": self.tenant_id}},
            limit=1,
        )
        if existing["ids"]:
            return  # already indexed for this tenant

        docs, ids, metadatas = [], [], []
        for machine in self._data_loader.all_machines:
            mid = machine["machineId"]
            profile = self._data_loader.get_machine_profile(mid)
            docs.append(_machine_doc_text(profile))
            ids.append(f"{self.tenant_id}::{mid}")
            metadatas.append(_machine_metadata(profile, self.tenant_id))

        if docs:
            self._collection.add(documents=docs, ids=ids, metadatas=metadatas)

    def semantic_search(self, query: str, n: int = 10, extra_filter: dict | None = None) -> list[dict]:
        where: dict = {"tenant_id": {"$eq": self.tenant_id}}
        if extra_filter:
            where = {"$and": [where, extra_filter]}

        results = self._collection.query(
            query_texts=[query],
            n_results=min(n, self._collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out = []
        if results["ids"] and results["ids"][0]:
            for doc, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            ):
                out.append({"document": doc, "metadata": meta, "distance": dist})
        return out

    def reindex(self):
        existing_ids = self._collection.get(
            where={"tenant_id": {"$eq": self.tenant_id}}
        )["ids"]
        if existing_ids:
            self._collection.delete(ids=existing_ids)
        self._ensure_indexed()
