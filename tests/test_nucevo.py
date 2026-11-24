"""Offline test suite for NucEvo benchmark integrity, parser, memory banks, and evaluation metrics."""
from __future__ import annotations

import unittest
from pathlib import Path

from nucevo.benchmark import (
    load_manifest,
    load_partition,
    load_split,
    verify_benchmark_integrity,
)
from nucevo.config import BANKS_DIR
from nucevo.evaluation.metrics import exact_mcnemar, summarize_evaluation
from nucevo.memory import (
    DEFAULT_BUDGET_CHARS,
    MemoryBank,
    MemoryItem,
    NucEvoRetriever,
    pair_notes,
)
from nucevo.solver.parser import parse_tool_call


class TestBenchmarkIntegrity(unittest.TestCase):
    def test_full_benchmark_hashes_and_anti_leakage(self):
        report = verify_benchmark_integrity()
        self.assertEqual(report["total_tasks"], 1560)
        self.assertEqual(report["unique_signatures"], 1560)
        self.assertEqual(report["partitions"], {"evolution": 600, "validation": 160, "test": 800})

    def test_partitions_disjoint(self):
        evo_ids = {t["task_id"] for t in load_partition("evolution")}
        val_ids = {t["task_id"] for t in load_partition("validation")}
        test_ids = {t["task_id"] for t in load_partition("test")}
        self.assertEqual(len(evo_ids), 600)
        self.assertEqual(len(val_ids), 160)
        self.assertEqual(len(test_ids), 800)
        self.assertTrue(evo_ids.isdisjoint(val_ids))
        self.assertTrue(evo_ids.isdisjoint(test_ids))
        self.assertTrue(val_ids.isdisjoint(test_ids))

    def test_manifest_has_no_absolute_paths(self):
        manifest = load_manifest()
        for meta in manifest["splits"].values():
            self.assertFalse(Path(meta["path"]).is_absolute())


class TestToolParser(unittest.TestCase):
    def test_standard_envelope(self):
        name, args = parse_tool_call(
            '<tool_call>{"name": "replace_connect", "args": {"old": ["a.x", "b.y"], "new": ["a.z", "b.y"]}}</tool_call>'
        )
        self.assertEqual(name, "replace_connect")
        self.assertEqual(args, {"old": ["a.x", "b.y"], "new": ["a.z", "b.y"]})

    def test_flat_top_level_fields(self):
        name, args = parse_tool_call(
            '<tool_call>{"name": "set_parameter", "instance": "mdot", "parameter": "k", "value": 19500}</tool_call>'
        )
        self.assertEqual(name, "set_parameter")
        self.assertEqual(args, {"instance": "mdot", "parameter": "k", "value": 19500})

    def test_fenced_json_arguments_key(self):
        name, args = parse_tool_call(
            '```json\n{"name": "add_connect", "arguments": {"pairs": [["ps.P", "fp.Q_fission"]]}}\n```'
        )
        self.assertEqual(name, "add_connect")
        self.assertEqual(args, {"pairs": [["ps.P", "fp.Q_fission"]]})

    def test_invalid_returns_none(self):
        self.assertEqual(parse_tool_call("I think we should simulate next."), (None, {}))


class TestMemoryAndDualBankPairing(unittest.TestCase):
    def test_pre_evolved_banks_load(self):
        verified = MemoryBank.load_snapshot(BANKS_DIR / "verified_bank.json")
        episodic = MemoryBank.load_snapshot(BANKS_DIR / "episodic_bank.json")
        self.assertEqual(verified.size(), 40)
        self.assertEqual(episodic.size(), 87)

    def test_pair_notes_respects_budget_and_deduplicates(self):
        m1 = MemoryItem(memory_id="MEM_A", type="rule", title="Rule A", description="A " * 1500)
        m2 = MemoryItem(memory_id="MEM_B", type="failure", title="Case B", description="B " * 1500)
        text, ids = pair_notes([m1], [m1, m2], budget_chars=DEFAULT_BUDGET_CHARS)
        self.assertEqual(ids, ["MEM_A", "MEM_B"])
        self.assertLessEqual(len(text), DEFAULT_BUDGET_CHARS)

    def test_dual_bank_routing_on_repair_vs_assemble(self):
        m1 = MemoryItem(
            memory_id="MEM_VER",
            type="rule",
            title="FuelPin Tcool uses Channel Tavg",
            description="FuelPin.Tcool must be driven by Channel.Tavg rather than Channel.Tin for steady energy balance.",
            status="verified",
            evidence={"verification_level": "counterfactual"},
        )
        m2 = MemoryItem(
            memory_id="MEM_EPI",
            type="failure",
            title="Repair swapped channel connection",
            description="When repairing swapped connections, inspect unconnected inputs and replace the wrong connect pair directly.",
            status="candidate",
            evidence={"verification_level": "hypothesis"},
        )
        b_ver = MemoryBank()
        b_ver.add(m1)
        b_epi = MemoryBank()
        b_epi.add(m2)

        retriever = NucEvoRetriever(b_ver, b_epi, budget_chars=DEFAULT_BUDGET_CHARS)
        # Use lexical BM25 mode in offline unit test so no HF download is needed
        retriever.verified_retriever.mode = "bm25"
        retriever.episodic_retriever.mode = "bm25"
        retriever.episodic_semantic.mode = "bm25"

        dev_tasks = load_split("dev")
        repair_task = next(t for t in dev_tasks if t["family"] == "repair")
        assemble_task = next(t for t in dev_tasks if t["family"] == "assemble")

        repair_ctx = retriever.retrieve_all_policies(repair_task)
        self.assertEqual(repair_ctx["nucevo"][1], ["MEM_VER", "MEM_EPI"])
        self.assertEqual(repair_ctx["verified_only"][1], ["MEM_VER"])

        assemble_ctx = retriever.retrieve_all_policies(assemble_task)
        self.assertEqual(assemble_ctx["nucevo"], assemble_ctx["verified_only"])


class TestMetrics(unittest.TestCase):
    def test_exact_mcnemar_matches_official_800(self):
        # 247 vs 177 with 118 only_first and 48 only_second
        first = [True] * 129 + [True] * 118 + [False] * 48 + [False] * 505
        second = [True] * 129 + [False] * 118 + [True] * 48 + [False] * 505
        res = exact_mcnemar(first, second)
        self.assertEqual(res["n"], 800)
        self.assertEqual(res["only_first"], 118)
        self.assertEqual(res["only_second"], 48)
        self.assertAlmostEqual(res["delta_pp"], 8.75, places=3)
        self.assertLess(res["p_two_sided"], 1e-7)

    def test_summarize_evaluation_groups(self):
        rows = [
            {"split": "id_test", "family": "repair", "difficulty_tier": "Hard", "nucevo": 1, "none": 0},
            {"split": "ood_near", "family": "assemble", "difficulty_tier": "Easy", "nucevo": 1, "none": 1},
        ]
        summary = summarize_evaluation(rows, ("none", "nucevo"), primary_policy="nucevo")
        self.assertEqual(summary["all"]["correct"]["nucevo"], 2)
        self.assertEqual(summary["all"]["correct"]["none"], 1)
        self.assertIn("split/id_test", summary)
        self.assertIn("tier/Hard", summary)


if __name__ == "__main__":
    unittest.main()
