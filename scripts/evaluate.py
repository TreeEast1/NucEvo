#!/usr/bin/env python3
"""Evaluate NucEvo (and matched baselines) on the frozen NucEvo-Bench v1 validation or test partition.

Examples
--------
  # Evaluate NucEvo only on the 160-task validation split
  python scripts/evaluate.py --partition validation --policies nucevo --out runs/eval_val

  # Reproduce the full 800-task official test benchmark across all 7 policies
  python scripts/evaluate.py --config configs/eval_benchmark_v1.json --out runs/eval_test_full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nucevo.benchmark import load_partition, load_split
from nucevo.config import REPO_ROOT
from nucevo.evaluation import EvalRunner, POLICY_KEYS
from nucevo.memory import MemoryBank
from nucevo.solver import SolverClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/eval_benchmark_v1.json"))
    parser.add_argument("--partition", choices=("validation", "test"), default=None)
    parser.add_argument("--split", choices=("dev", "id_test", "ood_near", "ood_far", "stress"), default=None)
    parser.add_argument("--policies", default=None, help=f"Comma-separated subset of {','.join(POLICY_KEYS)}")
    parser.add_argument("--verified-bank", default=None, help="Path to frozen verified bank JSON")
    parser.add_argument("--episodic-bank", default=None, help="Path to frozen episodic bank JSON")
    parser.add_argument("--ports", default=None, help="Comma-separated vLLM solver ports")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", required=True, help="Output directory for episodes, tables, and summary.json")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    verified_path = Path(args.verified_bank or (REPO_ROOT / cfg["verified_bank"]))
    episodic_path = Path(args.episodic_bank or (REPO_ROOT / cfg["episodic_bank"]))
    ports = [int(p) for p in args.ports.split(",")] if args.ports else cfg.get("ports", [8011, 8012, 8013, 8014])
    workers = args.workers or cfg.get("workers", 16)
    policies = tuple(args.policies.split(",")) if args.policies else tuple(cfg.get("policies", ["nucevo"]))

    if args.split:
        tasks = load_split(args.split)
    else:
        tasks = load_partition(args.partition or cfg.get("partition", "test"))

    verified_bank = MemoryBank.load_snapshot(verified_path)
    episodic_bank = MemoryBank.load_snapshot(episodic_path)

    clients = [
        SolverClient(
            base_url=f"http://127.0.0.1:{port}/v1",
            model=cfg.get("served_model_name", "solver"),
            max_tokens=cfg.get("max_tokens", 700),
            stop=["</tool_call>"],
            seed=cfg.get("solver_seed", 0),
            temperature=0.0,
        )
        for port in ports
    ]

    runner = EvalRunner(
        verified_bank=verified_bank,
        episodic_bank=episodic_bank,
        clients=clients,
        out_dir=Path(args.out),
        policies=policies,
        budget_chars=cfg.get("budget_chars", 4096),
        order_seed=cfg.get("order_seed", 771122),
    )
    summary = runner.run(tasks, workers=workers)
    print("\nEvaluation finished. Overall summary:")
    print(json.dumps(summary["groups"]["all"], indent=2))


if __name__ == "__main__":
    main()
