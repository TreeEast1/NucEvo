#!/usr/bin/env python3
"""Run training-free experience evolution on the NucEvo-Bench v1 evolution split.

Examples
--------
  # Evolve the simulator-verified rule/skill bank
  python scripts/evolve.py --config configs/evolution_verified.json --out runs/verified_bank

  # Evolve the broad-coverage episodic repair bank
  python scripts/evolve.py --config configs/evolution_episodic.json --out runs/episodic_bank
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nucevo.benchmark import load_split
from nucevo.evolution import EvoConfig, EvolutionRunner, order_tasks, sample_probe_pool
from nucevo.memory import MemoryBank
from nucevo.solver import SolverClient
from nucevo.teacher import TeacherClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to evolution JSON config")
    parser.add_argument("--out", required=True, help="Output directory for snapshots, logs, and trajectories")
    parser.add_argument("--ports", default=None, help="Comma-separated solver vLLM ports (overrides config)")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    ports = [int(p) for p in args.ports.split(",")] if args.ports else cfg.get("ports", [8011, 8012, 8013, 8014])

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    evo_all = load_split("evolution")
    evo_tasks = order_tasks(evo_all, order=cfg.get("order", "shuffled"), seed=cfg.get("seed", 11))[: cfg["n_evolution"]]
    probe_tasks = sample_probe_pool(evo_tasks, k=cfg.get("probe_pool", 80), seed=cfg.get("seed", 11))

    solvers = [
        SolverClient(
            base_url=f"http://127.0.0.1:{port}/v1",
            model=cfg.get("served_model_name", "solver"),
            max_tokens=cfg.get("max_tokens", 700),
            stop=["</tool_call>"],
            seed=cfg.get("solver_seed", 0),
            temperature=0.0,
            log_path=out_dir / f"solver_calls_{port}.jsonl",
        )
        for port in ports
    ]
    teacher = TeacherClient(
        max_tokens=cfg.get("teacher_max_tokens", 10000),
        log_path=out_dir / "teacher_calls.jsonl",
    )

    bank = MemoryBank(root=out_dir / "bank")
    evo_cfg = EvoConfig(
        name=cfg["name"],
        arm=cfg.get("arm", "nucevolve"),
        retrieval=cfg.get("retrieval", "full"),
        memory_budget_tokens=cfg.get("memory_budget_tokens", 1024),
        teacher_mode=cfg.get("teacher_mode", "per_task"),
        verification=cfg.get("verification", "counterfactual"),
        crud=cfg.get("crud", "full"),
        utility=cfg.get("utility", True),
        snapshot_every=cfg.get("snapshot_every", 20),
        parallel=cfg.get("parallel", 6),
        novel_only=cfg.get("novel_only", True),
    )

    runner = EvolutionRunner(evo_cfg, bank, solvers, teacher, probe_tasks, out_dir)
    runner.run(evo_tasks, start_index=0)
    final_snapshot = runner.snapshot(label="final")
    runner.save_state()
    print(f"Evolution complete. Final bank snapshot saved to: {final_snapshot}")
    print("Bank stats:", json.dumps(bank.stats(), indent=2))


if __name__ == "__main__":
    main()
