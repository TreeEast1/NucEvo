"""Training-free experience evolution loop.

One step:
  retrieve -> act -> verify -> record usage -> teacher reflects -> candidate
  experiences -> simulator probe -> admission/verification -> bank CRUD

The solver, the embedding model and the teacher are all frozen.  Only the bank
changes.  Every checkpoint writes an immutable snapshot, so a curve point is
always reconstructible and re-evaluable.
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from nucevo.benchmark.verifier import phys
from nucevo.memory.bank import MemoryBank
from nucevo.memory.schema import MemoryItem, dedup_key, new_id, now
from nucevo.memory.verifier import evaluate_probe, apply_verification
from nucevo.memory.retriever import Retriever, render_items as render, used_memory_ids, Embedder, cosine, memory_text
from nucevo.solver.agent import run_episode
from nucevo.teacher.client import parse_json
from nucevo.teacher.prompts import reflect_messages, reflect_messages_multi, validate_experiences


@dataclass
class EvoConfig:
    name: str
    arm: str = "nucevolve"           # no_memory|append|reasoningbank|expel|ace|nucevolve
    retrieval: str = "full"
    memory_budget_tokens: int = 1024
    teacher_mode: str = "per_task"   # per_task|every_k|failures_only
    batch_k: int = 1
    verification: str = "counterfactual"   # none|structural|single_point|multi_point|counterfactual
    crud: str = "full"               # append|dedup|revise|merge|full
    utility: bool = True
    snapshot_every: int = 25
    teacher_temperature: float = 0.0
    max_steps: int | None = None
    parallel: int = 1                # episodes run concurrently against a frozen bank
    allow_types: list | None = None  # experience-type ablation
    novel_only: bool = False         # call the teacher only on new information
    max_items: int | None = None     # memory-size ablation (active item cap)


VERIFY_ORDER = ["hypothesis", "structural", "single_point", "multi_point",
                "counterfactual"]


def _rank(level):
    try:
        return VERIFY_ORDER.index(level)
    except ValueError:
        return 0


class EvolutionRunner:
    def __init__(self, cfg: EvoConfig, bank: MemoryBank, solver, teacher,
                 probe_pool, out_dir, log=print):
        self.cfg = cfg
        self.bank = bank
        self.solver = solver
        self.teacher = teacher
        self.probe_pool = probe_pool
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "trajectories").mkdir(exist_ok=True)
        (self.out / "snapshots").mkdir(exist_ok=True)
        self.logf = self.out / "evolution_log.jsonl"
        _sink = log

        def log(ev, _p=self.logf, _s=_sink):
            try:
                _s(ev)
            except Exception:
                pass
            try:
                with open(_p, "a") as fh:
                    fh.write(json.dumps(ev) + "\n")
            except Exception:
                pass
        self.state_path = self.out / "state.json"
        self.log = log
        self.processed = []
        self.rejected = 0
        self.teacher_calls = 0
        self.exp_added = 0
        self.exp_merged = 0
        self.exp_revised = 0
        self.exp_retired = 0
        self.seen_sigs = set()

    # ------------------------------------------------------------------ persistence
    def save_state(self):
        self.state_path.write_text(json.dumps({
            "config": asdict(self.cfg), "processed": self.processed,
            "bank_version": self.bank.version, "bank_size": self.bank.size(),
            "counters": {"rejected": self.rejected,
                         "teacher_calls": self.teacher_calls,
                         "added": self.exp_added, "merged": self.exp_merged,
                         "revised": self.exp_revised, "retired": self.exp_retired},
            "teacher_stats": self.teacher.stats.to_dict()}, indent=1))

    def snapshot(self, label=""):
        p = self.out / "snapshots" / f"v{self.bank.version + 1:03d}_{len(self.processed):04d}.json"
        payload = self.bank.snapshot(p, label=label,
                                     meta={"arm": self.cfg.arm, "n_tasks": len(self.processed),
                                           "config": asdict(self.cfg),
                                           "teacher": self.teacher.stats.to_dict()})
        self.log({"event": "snapshot", "path": str(p), "sha1": payload["sha1"],
                  "size": self.bank.size(), "n_tasks": len(self.processed)})
        return p

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, task):
        if self.cfg.arm == "no_memory":
            return []
        r = Retriever(self.bank, mode=self.cfg.retrieval, seed=0)
        return r.retrieve(task, budget_chars=self.cfg.memory_budget_tokens * 4)

    # ------------------------------------------------------------------ teacher
    def reflect(self, episodes, retrieved):
        if len(episodes) == 1:
            msgs = reflect_messages(episodes[0][0], episodes[0][1], retrieved)
        else:
            msgs = reflect_messages_multi(episodes, retrieved)
        try:
            out = self.teacher.chat(msgs, temperature=self.cfg.teacher_temperature,
                                    use_cache=True)
            self.teacher_calls += 1
        except Exception as e:                       # noqa: BLE001
            self.log({"event": "teacher_error", "error": str(e)[:300]})
            return [], []
        obj = parse_json(out)
        if obj is None:
            return [], [{"reason": "unparseable teacher output"}]
        return validate_experiences(obj, max_items=3 if len(episodes) == 1 else 5)

    # ------------------------------------------------------------------ probes
    def probe_context(self, exp, k=3):
        """Context tasks for probe evaluation.

        Restricted to the evolution pool (never dev/test) and, when the
        experience states a topology scope, to tasks matching that scope.  A rule
        that says "in a series plant" must not be falsified by a parallel plant.
        """
        probe = exp.get("probe") or {}
        comp = probe.get("component")
        src = (probe.get("claimed_source") or {}).get("component")
        need = {c for c in (comp, src) if c}
        if not need:
            return []
        cands = [t for t in self.probe_pool
                 if need <= {i["type"] for i in t["topology"]["instances"]}]
        app = exp.get("applicability") or {}
        tc = " ".join(map(str, app.get("topology_constraints") or [])).lower()
        if tc:
            import re as _re
            words = {w for w in _re.findall(r"[a-z_]{3,}", tc)}
            scoped = [t for t in cands
                      if t["template"].lower() in words
                      or any(t["template"].lower().startswith(w) for w in words)]
            if scoped:
                cands = scoped
        fams = set(app.get("task_types") or [])
        if fams:
            fam_pref = [t for t in cands if t["family"] in fams]
            if len(fam_pref) >= k:
                cands = fam_pref
        rng = random.Random(hash(json.dumps(probe, sort_keys=True)) & 0xffff)
        rng.shuffle(cands)
        return cands[:k]

    def verify_candidate(self, exp, result=None):
        """Simulator-grounded verification.  `rule` claims need a probe.  A
        `failure`/`skill`/`success` item is evidence-graded against what the
        episode's own verifier report actually shows, so a lesson about tool
        errors is only admitted if the trajectory contains one."""
        if self.cfg.verification == "none":
            return {"level": "hypothesis", "passed": None, "details": []}
        probe = exp.get("probe")
        if not probe:
            if result is not None and exp.get("type") in ("failure", "skill", "success"):
                cls = self._class_of(exp)
                ok = self._episode_supports(cls, result)
                return {"level": "structural" if ok else "hypothesis",
                        "passed": bool(ok),
                        "reason": f"evidence class {cls}: "
                                  f"{'observed' if ok else 'not observed'}",
                        "details": []}
            return {"level": "hypothesis", "passed": None, "details": [],
                    "reason": "no probe"}
        ctx = self.probe_context(exp)
        if not ctx:
            return {"level": "structural", "passed": None, "details": [],
                    "reason": "no context task"}
        min_multi = {"none": 1, "structural": 1, "single_point": 1,
                     "multi_point": 3, "counterfactual": 3}[self.cfg.verification]
        return evaluate_probe(probe, ctx, min_tasks_for_multi=min_multi)

    @staticmethod
    def _class_of(exp):
        txt = (exp.get("title", "") + " " + exp.get("description", "")).lower()
        if any(w in txt for w in ("tool", "json", "malformed", "format", "duplicate",
                                  "parameter is", "invalid call", "parse")):
            return "tool"
        if any(w in txt for w in ("connect", "wiring", "port", "unconnected",
                                  "topology", "routing")):
            return "structure"
        if any(w in txt for w in ("balance", "conserv", "energy", "enthalpy",
                                  "mass flow", "physics")):
            return "physics"
        if any(w in txt for w in ("design point", "target", "off-spec", "goal",
                                  "temperature offset")):
            return "goal"
        return "success"

    @staticmethod
    def _episode_supports(cls, result):
        if cls == "tool":
            return any((t.get("obs") or "").startswith("ERROR") or t.get("name") is None
                       for t in (result.get("tool_log") or []))
        if cls == "structure":
            return result.get("structural_ok") is False
        if cls == "physics":
            return result.get("physics_ok") is False
        if cls == "goal":
            return result.get("goal_ok") is False
        return bool(result.get("success"))

    def admit(self, exp, ver):
        """Verification-mode admission policy.

        Rules need their full probe tier.  Failure/skill/success items need only
        structural evidence: the episode's own verifier report must show the
        error class the lesson is about (tool error, structural fault, physics
        violation, or goal miss).  This keeps observed procedural lessons while
        refusing unverified physical claims.
        """
        mode = self.cfg.verification
        if mode == "none":
            return True
        if exp.get("probe") is None:
            need = "structural"
        else:
            need = {"structural": "structural", "single_point": "single_point",
                    "multi_point": "multi_point",
                    "counterfactual": "counterfactual"}[mode]
        if ver.get("passed") is False:
            return False
        lvl = ver.get("level", "hypothesis")
        return _rank(lvl) >= _rank(need)

    # ------------------------------------------------------------------ CRUD
    def _similar(self, item):
        best, best_id = 0.0, None
        act = [i for i in self.bank.active()
               if i.type == item.type and i.status != "contradicted"]
        if not act:
            return 0.0, None
        try:
            v = Embedder.embed([memory_text(item)] + [memory_text(i) for i in act])
            for i, it in enumerate(act):
                s = cosine(v[0], v[i + 1])
                if s > best:
                    best, best_id = s, it.memory_id
        except Exception:
            return 0.0, None
        return best, best_id

    def integrate(self, exp, ver, source_task, used_items):
        cfg = self.cfg
        item = MemoryItem(
            memory_id=new_id(),
            type=exp["type"],
            title=exp["title"][:120],
            description=exp["description"][:1200],
            applicability=exp.get("applicability") or {},
            problem_signature=(exp.get("problem_signature") or "")[:400],
            recommended_strategy=(exp.get("recommended_strategy") or "")[:800],
            action_template=[str(a)[:200] for a in (exp.get("action_template") or [])[:10]],
            known_pitfalls=[str(a)[:200] for a in (exp.get("known_pitfalls") or [])[:8]],
            evidence={"source_task_ids": [source_task["task_id"]],
                      "source_trajectory_ids": [],
                      "simulation_result_ids": [],
                      "verification_level": ver.get("level", "hypothesis"),
                      "probe_result": {"passed": ver.get("passed"),
                                       "reason": ver.get("reason")},
                      "teacher_confidence": exp.get("confidence")},
            status="verified" if ver.get("passed") else "candidate",
            probe=exp.get("probe") or {},
        )
        # duplicate / refine handling
        if cfg.crud in ("dedup", "merge", "revise", "full"):
            sim, near_id = self._similar(item)
            if sim > 0.93 and near_id:
                near = self.bank.items[near_id]
                self.bank.record_evidence(near_id, {
                    "source_task_ids": [source_task["task_id"]],
                    "supporting_topologies": [source_task["template"]]},
                    reason="duplicate support")
                if cfg.crud in ("merge", "full"):
                    self.exp_merged += 1
                    self.log({"event": "merge_evidence", "into": near_id, "sim": round(sim, 3)})
                return near_id
        ref = exp.get("refines")
        if (cfg.crud in ("revise", "full") and exp.get("novelty") == "refines_existing"
                and ref in self.bank.items and self.bank.items[ref].status != "retired"):
            patch = {}
            if exp.get("description"):
                patch["description"] = exp["description"][:1200]
            if exp.get("applicability"):
                patch["applicability"] = exp["applicability"]
            if exp.get("recommended_strategy"):
                patch["recommended_strategy"] = exp["recommended_strategy"][:800]
            if patch:
                try:
                    self.bank.revise(ref, patch, reason=f"task {source_task['task_id']}")
                    self.bank.record_evidence(ref, {
                        "source_task_ids": [source_task["task_id"]]}, reason="revision support")
                    self.exp_revised += 1
                    return ref
                except Exception:
                    pass
        mid = self.bank.add(item, reason=f"task {source_task['task_id']}", actor="teacher")
        self.exp_added += 1
        return mid

    def housekeeping(self):
        if self.cfg.crud not in ("merge", "full"):
            return
        # retire memories with repeated negative transfer and no positive evidence
        for it in self.bank.active():
            u = it.usage_statistics
            if u.get("negative_transfer", 0) >= 3 and u.get("positive_transfer", 0) == 0:
                self.bank.retire(it.memory_id, "negative transfer >=3 with no positive")
                self.exp_retired += 1
        # merge near-identical verified items
        act = [i for i in self.bank.active() if i.status in ("verified", "useful")]
        if len(act) >= 2:
            try:
                vecs = Embedder.embed([memory_text(i) for i in act])
            except Exception:
                return
            for a in range(len(act)):
                for b in range(a + 1, len(act)):
                    if act[a].type != act[b].type:
                        continue
                    if cosine(vecs[a], vecs[b]) > 0.97:
                        best = act[a] if _rank(
                            (act[a].evidence or {}).get("verification_level", "hypothesis")) >= \
                            _rank((act[b].evidence or {}).get(
                                "verification_level", "hypothesis")) else act[b]
                        new = MemoryItem.from_dict(best.to_dict())
                        new.memory_id = new_id()
                        new.evidence = dict(best.evidence or {})
                        new.usage_statistics = {
                            k: (act[a].usage_statistics or {}).get(k, 0) +
                               (act[b].usage_statistics or {}).get(k, 0)
                            for k in ("retrieved", "used", "positive_transfer",
                                      "negative_transfer", "tasks_seen")}
                        self.bank.merge([act[a].memory_id, act[b].memory_id], new,
                                        reason="near-duplicate verified items")
                        self.exp_merged += 1
                        return

    # ------------------------------------------------------------------ main loop
    def _client(self, task):
        if isinstance(self.solver, (list, tuple)):
            return self.solver[abs(hash(task["task_id"])) % len(self.solver)]
        return self.solver

    def _episode(self, task):
        items = self.retrieve(task)
        mem_text = render(items, budget_chars=self.cfg.memory_budget_tokens * 4)
        res = run_episode(task, self._client(task), memory_text=mem_text,
                          max_steps=self.cfg.max_steps,
                          save_dir=self.out / "trajectories")
        res["retrieved_ids"] = [m.memory_id for m in items]
        res["memory_tokens_est"] = len(mem_text) // 4
        return task, items, mem_text, res

    def run(self, tasks, start_index=0, on_checkpoint=None):
        from concurrent.futures import ThreadPoolExecutor
        chunk_size = max(1, int(self.cfg.parallel))
        idx = start_index
        while idx < len(tasks):
            chunk = tasks[idx:idx + chunk_size]
            if len(chunk) == 1:
                out = [self._episode(chunk[0])]
            else:
                with ThreadPoolExecutor(max_workers=len(chunk)) as ex:
                    out = list(ex.map(self._episode, chunk))

            episodes = []
            for task, items, mem_text, res in out:
                self.processed.append(task["task_id"])
                if self.cfg.arm != "no_memory":
                    try:
                        used = used_memory_ids(items, res.get("final_text", ""), self.bank)
                    except Exception as e:          # noqa: BLE001
                        used = []
                        self.log({"event": "usage_error", "error": str(e)[:200]})
                    self.bank.record_usage([m.memory_id for m in items],
                                           "success" if res.get("success") else "failure",
                                           task_id=task["task_id"], used_ids=used)
                    res["used_memory_ids"] = used
                self.log({"event": "episode", "task_id": task["task_id"],
                          "template": task["template"], "family": task["family"],
                          "success": bool(res.get("success")),
                          "structural": bool(res.get("structural_ok")),
                          "physics": bool(res.get("physics_ok")),
                          "goal": bool(res.get("goal_ok")),
                          "retrieved": res["retrieved_ids"],
                          "used": res.get("used_memory_ids", []),
                          "steps": res.get("episode_steps"),
                          "n_sim": res.get("n_sim"),
                          "mem_chars": len(mem_text)})
                episodes.append((task, res))

            # teacher reflection: one call per chunk when batching, else per task.
            # `novel_only` skips the call when the batch adds no new information:
            # every episode failed in the same way as a pattern already seen.
            if self.cfg.arm != "no_memory":
                if self.cfg.teacher_mode == "failures_only":
                    episodes = [(t, r) for t, r in episodes if not r.get("success")]
                if episodes and self.cfg.novel_only:
                    sigs = {(t["template"], t["family"], r.get("structural_ok"),
                             r.get("physics_ok"), r.get("goal_ok"),
                             bool(r.get("success"))) for t, r in episodes}
                    if all(s in self.seen_sigs for s in sigs) and \
                            not any(r.get("success") for _, r in episodes):
                        self.log({"event": "teacher_skipped", "reason": "no new information",
                                  "n": len(episodes)})
                        episodes = []
                    self.seen_sigs |= sigs
                if episodes:
                    self._learn(episodes, out[0][1])

            prev_idx = idx
            idx += len(chunk)
            crossed = (idx // self.cfg.snapshot_every) != (prev_idx // self.cfg.snapshot_every)
            if crossed or idx >= len(tasks):
                self.housekeeping()
                p = self.snapshot(label=f"after {idx} tasks")
                self.save_state()
                if on_checkpoint:
                    on_checkpoint(idx, p)
                self.log({"event": "stats", "n": idx, **self.bank.stats()})
        return self.bank

    def _learn(self, episodes, retrieved):
        accepted, rejected = self.reflect(episodes, retrieved)
        self.rejected += len(rejected)
        src_task = episodes[0][0]
        for exp in accepted:
          try:
            if self.cfg.allow_types and exp.get("type") not in self.cfg.allow_types:
                self.rejected += 1
                self.log({"event": "reject_exp", "title": exp.get("title"),
                          "reason": "type not allowed"})
                continue
            if self.cfg.max_items and self.bank.size() >= self.cfg.max_items:
                self.rejected += 1
                self.log({"event": "reject_exp", "title": exp.get("title"),
                          "reason": "memory capacity reached"})
                continue
            ver = self.verify_candidate(exp, episodes[0][1])
            if not self.admit(exp, ver):
                self.rejected += 1
                self.log({"event": "reject_exp", "title": exp.get("title"),
                          "reason": ver.get("reason") or "admission policy",
                          "level": ver.get("level")})
                continue
            mid = self.integrate(exp, ver, src_task, [])
            self.log({"event": "accept_exp", "memory_id": mid,
                      "type": exp["type"], "title": exp["title"][:100],
                      "level": ver.get("level"), "passed": ver.get("passed")})
          except Exception as e:                      # noqa: BLE001
            self.rejected += 1
            self.log({"event": "reject_exp", "title": exp.get("title"),
                      "reason": f"exception: {e}"})
        self.save_state()


def order_tasks(tasks: list[dict], order: str = "shuffled", seed: int = 11) -> list[dict]:
    ordered = list(tasks)
    if order == "shuffled":
        random.Random(seed).shuffle(ordered)
    elif order == "difficulty":
        ordered.sort(key=lambda t: (t.get("difficulty", 0), t["task_id"]))
    elif order == "interleaved":
        buckets: dict[tuple[str, str], list[dict]] = {}
        for task in ordered:
            buckets.setdefault((task["template"], task["family"]), []).append(task)
        keys = sorted(buckets)
        interleaved, idx = [], 0
        while any(buckets[k] for k in keys):
            key = keys[idx % len(keys)]
            if buckets[key]:
                interleaved.append(buckets[key].pop(0))
            idx += 1
        ordered = interleaved
    return ordered


def sample_probe_pool(tasks: list[dict], k: int = 80, seed: int = 11) -> list[dict]:
    """Sample simulator probe contexts strictly from the evolution pool (never validation/test)."""
    rng = random.Random(seed)
    by_template: dict[str, list[dict]] = {}
    for task in tasks:
        by_template.setdefault(task["template"], []).append(task)
    pool: list[dict] = []
    for _tpl, group in sorted(by_template.items()):
        group_copy = list(group)
        rng.shuffle(group_copy)
        pool.extend(group_copy[: max(1, k // max(len(by_template), 1))])
    return pool[:k]
