"""Versioned experience bank.

Storage model
-------------
* `events.jsonl` is the append-only source of truth: every add / revise / merge /
  split / retire / restore / usage / evidence event is recorded with a timestamp,
  the affected ids, and the payload.
* A *snapshot* is an immutable JSON file holding the full item set at a point in
  time, plus the event sequence number and a content hash.  Snapshots are never
  edited; a new one is written for each evolution checkpoint.
* Replaying the event log from any snapshot reconstructs the bank exactly.

Nothing is deleted: `retire` flips status and keeps the item for audit.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

from .schema import MemoryItem, dedup_key, now, new_id


class MemoryBank:
    def __init__(self, root=None):
        self.root = Path(root) if root else None
        self.items: dict[str, MemoryItem] = {}
        self.seq = 0
        self.version = 0
        self._lock = threading.RLock()
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)
            self.events_path = self.root / "events.jsonl"
        else:
            self.events_path = None

    # ------------------------------------------------------------------ events
    def _event(self, kind, payload):
        self.seq += 1
        ev = {"seq": self.seq, "kind": kind, "at": now(), **payload}
        if self.events_path:
            with self.events_path.open("a") as f:
                f.write(json.dumps(ev) + "\n")
        return ev

    # ------------------------------------------------------------------ mutation
    def add(self, item: MemoryItem, reason="", actor="teacher"):
        with self._lock:
            if item.memory_id in self.items:
                item.memory_id = new_id()
            errs = item.validate()
            if errs:
                raise ValueError(f"invalid memory item: {errs}")
            self.items[item.memory_id] = item
            self._event("add", {"memory_id": item.memory_id, "reason": reason,
                                "actor": actor, "item": item.to_dict()})
            return item.memory_id

    def revise(self, memory_id, patch: dict, reason="", actor="curator"):
        with self._lock:
            it = self.items[memory_id]
            old = it.to_dict()
            for k, v in patch.items():
                if not hasattr(it, k):
                    raise KeyError(k)
                setattr(it, k, v)
            it.version += 1
            it.updated_at = now()
            it.lineage.setdefault("revisions", []).append(
                {"from_version": it.version - 1, "at": it.updated_at, "reason": reason,
                 "before": {k: old.get(k) for k in patch}})
            errs = it.validate()
            if errs:
                raise ValueError(f"invalid revision: {errs}")
            self._event("revise", {"memory_id": memory_id, "patch": patch,
                                   "reason": reason, "actor": actor})
            return memory_id

    def merge(self, memory_ids, new_item: MemoryItem, reason="", actor="curator"):
        with self._lock:
            src = [self.items[i] for i in memory_ids]
            new_item.evidence = dict(new_item.evidence or {})
            new_item.evidence.setdefault("merged_from", list(memory_ids))
            new_item.lineage["merged_from"] = list(memory_ids)
            new_item.usage_statistics = {
                k: sum((s.usage_statistics or {}).get(k, 0) for s in src) + v
                for k, v in (new_item.usage_statistics or {}).items()}
            self.add(new_item, reason=reason or "merge", actor=actor)
            for i in memory_ids:
                self.items[i].status = "retired"
                self.items[i].updated_at = now()
                self.items[i].lineage.setdefault("superseded_by", []).append(new_item.memory_id)
            self._event("merge", {"sources": list(memory_ids),
                                  "target": new_item.memory_id, "reason": reason})
            return new_item.memory_id

    def split(self, memory_id, parts, reason="", actor="curator"):
        """Retire one item and add k narrowed items, each inheriting evidence."""
        with self._lock:
            src = self.items[memory_id]
            new_ids = []
            for p in parts:
                it = MemoryItem.from_dict({**src.to_dict(), **p})
                it.memory_id = new_id()
                it.version = 1
                it.status = "candidate"
                it.created_at = now()
                it.updated_at = now()
                it.lineage = {"split_from": memory_id}
                it.evidence = dict(it.evidence or {})
                it.evidence["split_from"] = memory_id
                self.add(it, reason=reason or "split", actor=actor)
                new_ids.append(it.memory_id)
            src.status = "retired"
            src.updated_at = now()
            src.lineage.setdefault("split_into", []).extend(new_ids)
            self._event("split", {"source": memory_id, "targets": new_ids,
                                  "reason": reason})
            return new_ids

    def retire(self, memory_id, reason="", actor="curator"):
        with self._lock:
            it = self.items[memory_id]
            it.status = "retired"
            it.updated_at = now()
            it.lineage.setdefault("retired_reason", reason)
            self._event("retire", {"memory_id": memory_id, "reason": reason,
                                   "actor": actor})

    def restore(self, memory_id, reason="", actor="curator"):
        with self._lock:
            it = self.items[memory_id]
            it.status = "verified"
            it.updated_at = now()
            self._event("restore", {"memory_id": memory_id, "reason": reason})

    def set_status(self, memory_id, status, reason=""):
        with self._lock:
            if status not in {"candidate", "verified", "useful", "restricted",
                              "contradicted", "retired"}:
                raise ValueError(status)
            self.items[memory_id].status = status
            self.items[memory_id].updated_at = now()
            self._event("status", {"memory_id": memory_id, "status": status,
                                   "reason": reason})

    def record_evidence(self, memory_id, evidence: dict, reason=""):
        with self._lock:
            it = self.items[memory_id]
            for k, v in evidence.items():
                if k in ("source_task_ids", "source_trajectory_ids",
                         "simulation_result_ids", "supporting_topologies"):
                    cur = it.evidence.setdefault(k, [])
                    for x in (v if isinstance(v, list) else [v]):
                        if x not in cur:
                            cur.append(x)
                else:
                    it.evidence[k] = v
            it.updated_at = now()
            self._event("evidence", {"memory_id": memory_id, "evidence": evidence,
                                     "reason": reason})

    def add_counterexample(self, memory_id, example: dict, reason=""):
        with self._lock:
            it = self.items[memory_id]
            it.counterexamples.append({"at": now(), **example})
            if len(it.counterexamples) >= 2 and it.status == "verified":
                it.status = "restricted"
            self._event("counterexample", {"memory_id": memory_id,
                                           "example": example, "reason": reason})

    def record_usage(self, memory_ids, outcome: str, task_id=None, used_ids=None):
        """outcome: 'success' | 'failure'.  `used_ids` are the memories whose
        action template actually appears in the final model; only those can
        accrue transfer credit."""
        with self._lock:
            used_ids = set(used_ids or [])
            for mid in memory_ids:
                it = self.items.get(mid)
                if not it:
                    continue
                u = it.usage_statistics
                u["retrieved"] = u.get("retrieved", 0) + 1
                u["tasks_seen"] = u.get("tasks_seen", 0) + 1
                if mid in used_ids:
                    u["used"] = u.get("used", 0) + 1
                    if outcome == "success":
                        u["positive_transfer"] = u.get("positive_transfer", 0) + 1
                    else:
                        u["negative_transfer"] = u.get("negative_transfer", 0) + 1
            self._event("usage", {"memory_ids": list(memory_ids), "outcome": outcome,
                                  "task_id": task_id, "used_ids": list(used_ids)})

    # ------------------------------------------------------------------ views
    def active(self):
        return [i for i in self.items.values() if i.status != "retired"]

    def by_status(self, status):
        return [i for i in self.items.values() if i.status == status]

    def size(self):
        return len(self.active())

    def duplicate_groups(self):
        groups = {}
        for it in self.active():
            groups.setdefault(dedup_key(it), []).append(it.memory_id)
        return {k: v for k, v in groups.items() if len(v) > 1}

    # ------------------------------------------------------------------ snapshots
    def snapshot(self, path=None, label="", meta=None):
        with self._lock:
            self.version += 1
            payload = {"version": self.version, "seq": self.seq, "label": label,
                       "created_at": now(), "meta": meta or {},
                       "items": [i.to_dict() for i in self.items.values()]}
            body = json.dumps(payload, sort_keys=True)
            payload["sha1"] = hashlib.sha1(body.encode()).hexdigest()
            if path:
                p = Path(path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(payload, indent=1))
                (p.parent / "LATEST").write_text(str(p) + "\n")
            self._event("snapshot", {"version": self.version, "label": label,
                                     "sha1": payload["sha1"]})
            return payload

    @staticmethod
    def load_snapshot(path):
        d = json.loads(Path(path).read_text())
        bank = MemoryBank(root=None)
        bank.version = d.get("version", 0)
        bank.seq = d.get("seq", 0)
        for raw in d.get("items", []):
            it = MemoryItem.from_dict(raw)
            bank.items[it.memory_id] = it
        return bank

    def stats(self):
        from collections import Counter
        st = Counter(i.status for i in self.items.values())
        ty = Counter(i.type for i in self.items.values() if i.status != "retired")
        lv = Counter((i.evidence or {}).get("verification_level", "hypothesis")
                     for i in self.items.values() if i.status != "retired")
        return {"active": self.size(), "total": len(self.items),
                "status": dict(st), "type": dict(ty), "verification": dict(lv),
                "version": self.version}
