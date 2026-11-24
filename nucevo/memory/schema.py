"""Memory item schema and validation.

An experience is not a paragraph in a vector store.  It carries its own
applicability conditions, evidence, counterexamples, usage statistics and
revision lineage, so it can be verified, restricted, split, merged or retired.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone

TYPES = {"success", "failure", "rule", "skill"}
STATUSES = {"candidate", "verified", "useful", "restricted", "contradicted", "retired"}
VERIFICATION_LEVELS = {"hypothesis", "structural", "single_point", "multi_point",
                       "counterfactual"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_usage():
    return {"retrieved": 0, "used": 0, "positive_transfer": 0,
            "negative_transfer": 0, "tasks_seen": 0}


@dataclass
class MemoryItem:
    memory_id: str
    type: str
    title: str
    description: str
    applicability: dict = field(default_factory=dict)
    problem_signature: str = ""
    recommended_strategy: str = ""
    action_template: list = field(default_factory=list)
    known_pitfalls: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    counterexamples: list = field(default_factory=list)
    usage_statistics: dict = field(default_factory=_default_usage)
    status: str = "candidate"
    version: int = 1
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)
    lineage: dict = field(default_factory=dict)
    probe: dict = field(default_factory=dict)

    def validate(self):
        errs = []
        if self.type not in TYPES:
            errs.append(f"bad type {self.type}")
        if self.status not in STATUSES:
            errs.append(f"bad status {self.status}")
        lvl = (self.evidence or {}).get("verification_level", "hypothesis")
        if lvl not in VERIFICATION_LEVELS:
            errs.append(f"bad verification_level {lvl}")
        if not self.title.strip():
            errs.append("empty title")
        if not self.description.strip():
            errs.append("empty description")
        if not isinstance(self.applicability, dict):
            errs.append("applicability must be an object")
        return errs

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        known = {f for f in MemoryItem.__dataclass_fields__}
        clean = {k: v for k, v in d.items() if k in known}
        clean.setdefault("memory_id", new_id())
        clean["usage_statistics"] = {**_default_usage(),
                                     **(clean.get("usage_statistics") or {})}
        clean["evidence"] = clean.get("evidence") or {}
        clean["evidence"].setdefault("verification_level", "hypothesis")
        return MemoryItem(**clean)

    def text(self, budget_chars=None):
        """Rendered form injected into the prompt."""
        app = self.applicability or {}
        cond = []
        for k in ("task_types", "component_types", "topology_constraints",
                  "physical_conditions"):
            v = app.get(k)
            if v:
                cond.append(f"{k}: {', '.join(map(str, v)) if isinstance(v, list) else v}")
        parts = [f"[{self.memory_id} | {self.type} | {self.status} | v{self.version}]",
                 f"{self.title}",
                 self.description]
        if self.recommended_strategy:
            parts.append(f"Strategy: {self.recommended_strategy}")
        if self.action_template:
            parts.append("Actions: " + " ; ".join(str(a) for a in self.action_template))
        if self.known_pitfalls:
            parts.append("Pitfalls: " + " ; ".join(map(str, self.known_pitfalls)))
        if cond:
            parts.append("Applies when: " + " | ".join(cond))
        out = "\n".join(parts)
        if budget_chars and len(out) > budget_chars:
            out = out[:budget_chars].rsplit(" ", 1)[0] + " ..."
        return out


def new_id():
    return "MEM_" + uuid.uuid4().hex[:8].upper()


def dedup_key(item: MemoryItem):
    """Cheap near-duplicate key for merge decisions."""
    return json.dumps([item.type,
                       sorted((item.applicability or {}).get("component_types", []) or []),
                       (item.applicability or {}).get("topology_constraints", []),
                       item.title.strip().lower()[:80]],
                      sort_keys=True)
