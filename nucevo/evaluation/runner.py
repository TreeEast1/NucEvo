"""Frozen-bank evaluation runner with SHA-256 prompt deduplication and atomic resumption."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nucevo.memory.retriever import DEFAULT_BUDGET_CHARS, NucEvoRetriever
from nucevo.solver.agent import run_episode
from .metrics import summarize_evaluation, write_csv_table

POLICY_KEYS = (
    "none",
    "generic",
    "reasoningbank",
    "verified_only",
    "nucevo",
    "paired_verified",
    "semantic_episodic",
)

POLICY_LABELS = {
    "none": "No Memory",
    "generic": "Generic Guide",
    "reasoningbank": "ReasoningBank (Adapted)",
    "verified_only": "NucEvo (Verified-Only)",
    "nucevo": "NucEvo (Full)",
    "paired_verified": "Ablation: Paired Verified Notes",
    "semantic_episodic": "Ablation: Semantic Episodic on Repair",
}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class EvalRunner:
    """Runs deduplicated, resumable frozen-memory evaluations across one or more policies."""

    def __init__(
        self,
        verified_bank,
        episodic_bank,
        clients: list,
        out_dir: Path,
        policies: tuple[str, ...] = ("nucevo",),
        budget_chars: int = DEFAULT_BUDGET_CHARS,
        order_seed: int = 771122,
    ):
        self.retriever = NucEvoRetriever(verified_bank, episodic_bank, budget_chars=budget_chars)
        self.clients = clients if isinstance(clients, (list, tuple)) else [clients]
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.policies = tuple(policies)
        self.order_seed = order_seed

    def build_index(self, tasks: list[dict]) -> dict:
        prompts: dict[str, str] = {}
        task_plans: dict[str, dict] = {}
        for task in tasks:
            all_ctx = self.retriever.retrieve_all_policies(task)
            plan = {}
            for pol in self.policies:
                text, ids = all_ctx[pol]
                key = sha256_text(text)
                prompts[key] = text
                plan[pol] = {"prompt_sha256": key, "retrieved_ids": ids}
            task_plans[task["task_id"]] = plan
        return {"policies": list(self.policies), "prompts": prompts, "tasks": task_plans}

    def _run_one(self, task: dict, prompt_key: str, memory_text: str, client) -> dict:
        ep_path = self.out_dir / "episodes" / task["task_id"] / f"{prompt_key}.json"
        if ep_path.exists():
            record = json.loads(ep_path.read_text(encoding="utf-8"))
            if record.get("task_id") == task["task_id"] and record.get("prompt_sha256") == prompt_key:
                return record
        result = run_episode(task, client, memory_text=memory_text)
        record = {
            "task_id": task["task_id"],
            "prompt_sha256": prompt_key,
            "memory_text": memory_text,
            "result": result,
        }
        atomic_write_json(ep_path, record)
        return record

    def run(self, tasks: list[dict], workers: int = 16) -> dict:
        index_path = self.out_dir / "index.json"
        index = self.build_index(tasks)
        atomic_write_json(index_path, index)

        task_map = {t["task_id"]: t for t in tasks}
        jobs = []
        for idx, task in enumerate(tasks):
            tid = task["task_id"]
            seen_keys = dict.fromkeys(index["tasks"][tid][pol]["prompt_sha256"] for pol in self.policies)
            for key in seen_keys:
                jobs.append((task_map[tid], key, index["prompts"][key], self.clients[idx % len(self.clients)]))

        random.Random(self.order_seed).shuffle(jobs)
        print(f"Evaluating {len(tasks)} tasks × {len(self.policies)} policies ({len(jobs)} distinct episodes)", flush=True)

        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self._run_one, *job) for job in jobs]
            for future in as_completed(futures):
                future.result()
                completed += 1
                if completed % 25 == 0 or completed == len(jobs):
                    print(f"Progress: {completed}/{len(jobs)} distinct episodes completed", flush=True)

        return self.report(tasks, index, distinct_episodes=len(jobs))

    def report(self, tasks: list[dict], index: dict, distinct_episodes: int | None = None) -> dict:
        rows = []
        for task in tasks:
            tid = task["task_id"]
            row = {
                "task_id": tid,
                "split": task.get("split", ""),
                "family": task.get("family", ""),
                "template": task.get("template", ""),
                "difficulty_tier": task.get("difficulty_tier", ""),
            }
            for pol in self.policies:
                key = index["tasks"][tid][pol]["prompt_sha256"]
                ep_path = self.out_dir / "episodes" / tid / f"{key}.json"
                rec = json.loads(ep_path.read_text(encoding="utf-8"))
                row[pol] = int(bool(rec["result"].get("success")))
            rows.append(row)

        primary = "nucevo" if "nucevo" in self.policies else self.policies[0]
        groups = summarize_evaluation(rows, self.policies, primary_policy=primary)

        with (self.out_dir / "per_task.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        write_csv_table(
            self.out_dir / "main_table.csv",
            groups,
            [
                ("all", "All Tasks"),
                ("split/id_test", "ID Test"),
                ("split/ood_near", "OOD-Near"),
                ("split/ood_far", "OOD-Far"),
                ("split/stress", "Stress"),
                ("split/dev", "Validation (Dev)"),
            ],
            self.policies,
            POLICY_LABELS,
        )
        write_csv_table(
            self.out_dir / "difficulty_table.csv",
            groups,
            [("tier/Easy", "Easy"), ("tier/Medium", "Medium"), ("tier/Hard", "Hard"), ("tier/Challenge", "Challenge")],
            self.policies,
            POLICY_LABELS,
        )
        write_csv_table(
            self.out_dir / "family_table.csv",
            groups,
            [
                ("family/repair", "Repair"),
                ("family/assemble", "Assemble"),
                ("family/tune", "Tune"),
                ("family/diagnose", "Diagnose"),
                ("family/multistep", "Multi-step"),
            ],
            self.policies,
            POLICY_LABELS,
        )

        summary = {
            "num_tasks": len(rows),
            "policies": list(self.policies),
            "logical_scores": len(rows) * len(self.policies),
            "distinct_episodes": distinct_episodes,
            "groups": groups,
        }
        atomic_write_json(self.out_dir / "summary.json", summary)
        return summary
