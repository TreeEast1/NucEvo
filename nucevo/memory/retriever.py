"""Memory retrieval and NucEvo task-aware dual-bank context assembly.

Core retrieval modes
--------------------
- `full`       : 0.5 * semantic + 0.3 * structural_applicability + 0.2 * historical_utility
- `embedding`  : semantic similarity only (frozen BAAI/bge-small-en-v1.5)
- `bm25`       : lexical BM25 only
- `sim_app`    : semantic + structural applicability
- `sim_util`   : semantic + historical utility
- `random`     : uniform random control

NucEvo dual-bank policy
-----------------------
Combines two complementary frozen memory banks built from the same evolution stream:
1. `verified_bank`: simulator-verified rules, structural invariants, and assembly skills.
2. `episodic_bank`: broad-coverage episodic/repair experiences from prior trajectories.

For `repair` tasks, NucEvo pairs the top-ranked verified rule note with the top-ranked
complementary episodic repair note under a fixed 4,096-character budget (split evenly).
For non-repair tasks (`assemble`, `tune`, `diagnose`, `multistep`), NucEvo uses full
multi-factor retrieval from `verified_bank` within the same 4,096-character budget.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import sys
import threading
from collections import Counter

from nucevo.config import cache_subdir

DEFAULT_BUDGET_CHARS = 4096
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

GENERIC_GUIDE = """Generic engineering guidance (control condition, no experience):
The plant is described by causal blocks; each block has inputs and outputs.
Every input connector must be driven exactly once by an output of a compatible
signal kind: power, temperature, mass flow, pressure, or reactivity.
At steady state each component must satisfy its own conservation law. A heated
channel obeys Q = m*Cp*(Tout - Tin). A fuel pin transfers fission heat through a
gap conductance to the cladding and from the cladding to the coolant; the fuel
temperature exceeds the clad temperature, which exceeds the coolant temperature.
A steam generator removes heat from the primary side; its outlet is colder than
its inlet. Mixers conserve mass and enthalpy; splitters divide mass by a fixed
fraction and pass temperature through. Pumps follow a head-flow curve; pipes
follow a quadratic pressure-drop law. A controller output is clipped between
zero and its saturation value.
Check the design point after every structural change, and remember that a model
which compiles and runs can still violate conservation. Prefer small, verified
edits; simulate after each one; never invent simulator output.
"""

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*|\d+(?:\.\d+)?")
STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "is", "be", "must",
    "should", "with", "for", "on", "that", "this", "it", "as", "at", "by",
}


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _TOKEN_RE.findall(text or "") if w.lower() not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = max(sum(self.lens) / max(len(self.lens), 1), 1e-9)
        self.k1, self.b = k1, b
        self.df: Counter[str] = Counter()
        for doc in self.docs:
            self.df.update(set(doc))
        self.n_docs = max(len(self.docs), 1)

    def idf(self, term: str) -> float:
        return math.log(1.0 + (self.n_docs - self.df[term] + 0.5) / (self.df[term] + 0.5))

    def scores(self, query: str) -> list[float]:
        query_counts = Counter(tokenize(query))
        out: list[float] = []
        for idx, doc in enumerate(self.docs):
            score = 0.0
            for term in query_counts:
                if term not in doc:
                    continue
                tf = doc[term]
                score += (
                    self.idf(term)
                    * tf
                    * (self.k1 + 1.0)
                    / (tf + self.k1 * (1.0 - self.b + self.b * self.lens[idx] / self.avg))
                )
            out.append(score)
        peak = max(out) if out else 0.0
        return [val / peak if peak > 0.0 else 0.0 for val in out]


class Embedder:
    """Thread-safe frozen embedding encoder with on-disk content caching."""

    _lock = threading.Lock()
    _model = None
    _tokenizer = None
    _model_name = None

    @classmethod
    def get(cls, model_name: str | None = None):
        target = model_name or os.environ.get("NUCEVO_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        with cls._lock:
            if cls._model is None or cls._model_name != target:
                os.environ.setdefault("HF_HOME", str(cache_subdir("hf")))
                from transformers import AutoModel, AutoTokenizer

                cls._tokenizer = AutoTokenizer.from_pretrained(target)
                cls._model = AutoModel.from_pretrained(target).eval()
                cls._model_name = target
            return cls._tokenizer, cls._model

    @classmethod
    def embed(cls, texts: list[str], model_name: str | None = None) -> list[list[float]]:
        target = model_name or os.environ.get("NUCEVO_EMBED_MODEL", DEFAULT_EMBED_MODEL)
        key = hashlib.sha1(("\n".join(texts) + f"|{target}").encode()).hexdigest()
        cache_path = cache_subdir("embeddings") / f"{key}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        import torch

        tokenizer, model = cls.get(target)
        with torch.no_grad():
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
        vectors = pooled.tolist()
        cache_path.write_text(json.dumps(vectors), encoding="utf-8")
        return vectors


def cosine(vec_a: list[float], vec_b: list[float]) -> float:
    return sum(x * y for x, y in zip(vec_a, vec_b))


def build_public_query(task: dict) -> dict:
    """Extract only observable task metadata for leakage-free retrieval."""
    return {
        "instruction": task["instruction"],
        "template": task["template"],
        "family": task["family"],
        "topology": {
            "instances": [{"type": inst["type"]} for inst in task["topology"]["instances"]]
        },
    }


def task_structure(task: dict) -> dict:
    comps = Counter(inst["type"] for inst in task["topology"]["instances"])
    return {
        "components": set(comps),
        "template": task["template"],
        "family": task["family"],
        "n_connects": task.get("n_connects", 0),
        "component_types": comps,
    }


def applicability_score(item, struct: dict) -> float:
    app = item.applicability or {}
    score, count = 0.0, 0
    task_types = set(app.get("task_types") or [])
    if task_types:
        count += 1
        score += 1.0 if struct["family"] in task_types else 0.0
    comp_types = set(app.get("component_types") or [])
    if comp_types:
        count += 1
        overlap = len(comp_types & struct["components"])
        score += overlap / max(len(comp_types | struct["components"]), 1)
    topo_constraints = app.get("topology_constraints") or []
    if topo_constraints:
        count += 1
        names: set[str] = set()
        for entry in topo_constraints:
            names |= set(re.findall(r"[a-z_]+[0-9]*", str(entry).lower()))
        tpl_lower = struct["template"].lower()
        joined = " ".join(map(str, topo_constraints)).lower()
        score += 1.0 if (tpl_lower in names or tpl_lower in joined) else 0.0
    return 0.5 if count == 0 else score / count


def utility_score(item) -> float:
    stats = item.usage_statistics or {}
    pos = stats.get("positive_transfer", 0)
    neg = stats.get("negative_transfer", 0)
    return (pos + 1.0) / (pos + neg + 2.0)


def memory_text(item) -> str:
    return "\n".join(
        filter(
            None,
            [
                item.title,
                item.description,
                item.recommended_strategy,
                " ".join(map(str, item.action_template or [])),
                " ".join(map(str, item.known_pitfalls or [])),
            ],
        )
    )


def fit_budget(items: list, budget_chars: int) -> list:
    selected, used = [], 0
    for item in items:
        rendered = item.text()
        if used + len(rendered) > budget_chars and selected:
            break
        selected.append(item)
        used += len(rendered)
    return selected


def render_items(items: list, budget_chars: int = DEFAULT_BUDGET_CHARS) -> str:
    parts, used = [], 0
    for item in items:
        chunk = item.text(budget_chars=budget_chars - used)
        parts.append(chunk)
        used += len(chunk)
        if used >= budget_chars:
            break
    return "\n\n".join(parts)


def pair_notes(primary_ranked: list, secondary_ranked: list, budget_chars: int = DEFAULT_BUDGET_CHARS) -> tuple[str, list[str]]:
    """Pair the top primary note with the top non-duplicate secondary note under an equal char budget."""
    chosen = list(primary_ranked[:1])
    for candidate in secondary_ranked:
        if not chosen or candidate.memory_id != chosen[0].memory_id:
            chosen.append(candidate)
            break
    per_note = (budget_chars - 2) // 2
    chunks = [item.text(budget_chars=per_note)[:per_note] for item in chosen]
    return "\n\n".join(chunks), [item.memory_id for item in chosen]


class Retriever:
    """Single-bank multi-factor experience retriever."""

    def __init__(self, bank, mode: str = "full", weights: tuple[float, float, float] = (0.5, 0.3, 0.2), seed: int = 0):
        self.bank = bank
        self.mode = mode
        self.weights = weights
        self.rng = random.Random(seed)

    def retrieve(self, task: dict, k: int = 6, budget_chars: int = DEFAULT_BUDGET_CHARS) -> list:
        items = [
            item
            for item in self.bank.active()
            if item.status not in ("contradicted", "retired")
            and (
                (item.evidence or {}).get("verification_level") != "hypothesis"
                or item.status in ("verified", "useful", "candidate")
            )
        ]
        if not items:
            return []

        struct = task_structure(task)
        query = " ".join(
            [
                task["instruction"],
                task["family"],
                task["template"],
                " ".join(sorted(struct["components"])),
            ]
        )
        if self.mode == "random":
            picked = self.rng.sample(items, min(k, len(items)))
            return fit_budget(picked, budget_chars)

        bm25_scores = BM25([memory_text(it) for it in items]).scores(query)
        emb_scores = None
        if self.mode in ("embedding", "sim_app", "sim_util", "full"):
            q_vec = Embedder.embed([query])[0]
            m_vecs = Embedder.embed([memory_text(it) for it in items])
            emb_scores = [cosine(q_vec, vec) for vec in m_vecs]

        scored = []
        w_rel, w_app, w_util = self.weights
        for idx, item in enumerate(items):
            rel = emb_scores[idx] if emb_scores is not None else bm25_scores[idx]
            app = applicability_score(item, struct)
            util = utility_score(item)
            if self.mode == "bm25":
                score = bm25_scores[idx]
            elif self.mode == "embedding":
                score = rel
            elif self.mode == "sim_app":
                denom = max(w_rel + w_app, 1e-9)
                score = (w_rel * rel + w_app * app) / denom
            elif self.mode == "sim_util":
                score = rel + w_util * (util - 0.5) * 2.0
            elif self.mode == "type_routed":
                preferred = {
                    "repair": {"rule", "failure"},
                    "diagnose": {"rule", "failure"},
                    "assemble": {"skill", "success"},
                    "tune": {"rule", "failure"},
                    "multistep": {"skill", "rule", "failure"},
                }.get(task["family"], set())
                score = rel + (0.5 if item.type in preferred else 0.0) + 0.1 * (util - 0.5) * 2.0
            else:
                score = w_rel * rel + w_app * app + w_util * (util - 0.5) * 2.0
            scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], pair[1].memory_id))
        return fit_budget([item for _, item in scored[:k]], budget_chars)


class NucEvoRetriever:
    """Task-aware dual-bank retriever for NucEvo and all benchmark comparison baselines."""

    def __init__(
        self,
        verified_bank,
        episodic_bank,
        budget_chars: int = DEFAULT_BUDGET_CHARS,
        weights: tuple[float, float, float] = (0.5, 0.3, 0.2),
    ):
        self.budget_chars = budget_chars
        self.verified_retriever = Retriever(verified_bank, mode="full", weights=weights)
        self.episodic_retriever = Retriever(episodic_bank, mode="full", weights=weights)
        self.episodic_semantic = Retriever(episodic_bank, mode="embedding", weights=weights)

    def retrieve_context(self, task: dict) -> tuple[str, list[str]]:
        """Default NucEvo retrieval: dual-bank paired context on repair, verified-bank full retrieval otherwise."""
        return self.retrieve_all_policies(task)["nucevo"]

    def retrieve_all_policies(self, task: dict) -> dict[str, tuple[str, list[str]]]:
        """Compute contexts for NucEvo and all matched control/ablation policies on the same task."""
        public = build_public_query(task)
        ranked_verified = self.verified_retriever.retrieve(public, budget_chars=sys.maxsize)
        ranked_episodic = self.episodic_retriever.retrieve(public, budget_chars=sys.maxsize)

        verified_fitted = fit_budget(ranked_verified, self.budget_chars)
        episodic_fitted = fit_budget(ranked_episodic, self.budget_chars)

        def pack(items: list) -> tuple[str, list[str]]:
            return render_items(items, budget_chars=self.budget_chars), [m.memory_id for m in items]

        out: dict[str, tuple[str, list[str]]] = {
            "none": ("", []),
            "generic": (GENERIC_GUIDE[: self.budget_chars], []),
            "reasoningbank": pack(episodic_fitted),
            "verified_only": pack(verified_fitted),
        }

        if task["family"] == "repair":
            semantic_items = self.episodic_semantic.retrieve(public, budget_chars=self.budget_chars)
            out["nucevo"] = pair_notes(ranked_verified, ranked_episodic, budget_chars=self.budget_chars)
            out["paired_verified"] = pair_notes(ranked_verified, ranked_verified[1:], budget_chars=self.budget_chars)
            out["semantic_episodic"] = pack(semantic_items)
        else:
            out["nucevo"] = out["verified_only"]
            out["paired_verified"] = out["verified_only"]
            out["semantic_episodic"] = out["reasoningbank"]

        return out


def _type_connections(final_text: str) -> set[tuple[str, str, str, str]]:
    from nucevo.benchmark.verifier import phys

    instances = phys.parse_instances(final_text or "")
    types = {inst["name"]: inst["type"] for inst in instances}
    connections: set[tuple[str, str, str, str]] = set()
    for port_a, port_b in phys.parse_connects(final_text or ""):
        if "." not in port_a or "." not in port_b:
            continue
        inst_a, name_a = port_a.split(".", 1)
        inst_b, name_b = port_b.split(".", 1)
        type_a, type_b = types.get(inst_a), types.get(inst_b)
        if not type_a or not type_b:
            continue
        if name_a in phys.COMPONENT_PORTS[type_a]["out"] and name_b in phys.COMPONENT_PORTS[type_b]["in"]:
            connections.add((type_a, name_a, type_b, name_b))
        elif name_b in phys.COMPONENT_PORTS[type_b]["out"] and name_a in phys.COMPONENT_PORTS[type_a]["in"]:
            connections.add((type_b, name_b, type_a, name_a))
    return connections


def used_memory_ids(items: list, final_text: str, bank=None) -> list[str]:
    """Identify which retrieved items' recommended actions actually appear in the final Modelica model."""
    present = _type_connections(final_text)
    used: list[str] = []
    for item in items:
        hit = False
        probe = item.probe or {}
        if probe.get("kind") == "connection_driver":
            src = probe.get("claimed_source") or {}
            key = (src.get("component"), src.get("port"), probe.get("component"), probe.get("port"))
            hit = key in present
        if not hit:
            for action in item.action_template or []:
                for _i1, p1, _i2, p2 in re.findall(
                    r"<?([A-Za-z_]\w*)>?\.(\w+)\s*[,;-]+\s*<?([A-Za-z_]\w*)>?\.(\w+)",
                    str(action),
                ):
                    if any((pa, pb) == (p1, p2) or (pb, pa) == (p1, p2) for _t1, pa, _t2, pb in present):
                        hit = True
                        break
                if hit:
                    break
        if hit:
            used.append(item.memory_id)
    return used
