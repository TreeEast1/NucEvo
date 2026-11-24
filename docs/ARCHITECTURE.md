# NucEvo Code Architecture & Developer Guide

This guide walks collaborators through the internal design, data flow, and extension points of the `nucevo` package.

---

## 1. Design Philosophy

NucEvo separates **reasoning execution**, **physical ground truth**, and **external experience memory** into strictly decoupled modules:

1. **Frozen Parameters**: Neither the solver LLM (`Qwen3-4B-Instruct-2507`) nor the retrieval encoder (`BAAI/bge-small-en-v1.5`) is ever fine-tuned.
2. **Physics as Oracle**: The external teacher never sees ground-truth reference connections. Every proposed physical rule must pass simulator-grounded multi-point and counterfactual probing in OpenModelica 1.27.
3. **Dual-Bank Specialization**: Empirical audits showed that strict counterfactual verification produces high-precision assembly & structural rules (`+30.9 pp` on `assemble`), whereas broad trajectory distillation retains diverse fault-repair hints (`+10.0 pp` on `repair`). Combining both banks via a **task-aware retrieval policy** captures both gains under the exact same 4,096-character budget.

---

## 2. Module-by-Module Walkthrough

### `nucevo/benchmark/` — Executable Benchmark & Verifier

| File | Responsibility |
| :--- | :--- |
| `nuclear_edu/NuclearEdu.mo` | 12 causal Modelica blocks sharing one `Sig` connector (`Real u`) so semantically wrong wiring compiles and must be caught by physics. |
| `simulator.py` | Compiles and integrates models with `omc` in isolated temp dirs; content-addressed disk cache (`~/.cache/nucevo/sim`). |
| `loader.py` | Loads `evolution` (600), `validation` (160), and `test` (800) partitions, attaches difficulty tiers, and audits SHA-1 integrity. |
| `generators/topologies.py` | Compositional graph builders for all 15 plant topologies (`single` through `kinetics_pumped`). |
| `generators/tasks.py` | Programmatic task instantiation (assemble, repair, tune, diagnose, multistep) with fault injection and signature hashing. |
| `verifier/phys.py` | Reference-free component conservation equations (fuel pin energy/conduction, channel enthalpy, SG NTU, mixer/splitter balance, pump/pipe hydraulics, kinetics, controller saturation). |
| `verifier/v3.py` | 6-layer verification gate: build -> simulate/steady -> structural -> component physics -> multi-point perturbation -> design goal & transient. |

### `nucevo/solver/` — Agent Runtime & Tool Sandbox

- **`parser.py`**: Parses `<tool_call>{...}</tool_call>`, fenced JSON blocks, legacy `tool_name {...}`, and raw JSON objects. Supports both nested `{"name": "...", "args": {...}}` and flat top-level fields (e.g., `{"name": "set_parameter", "instance": "mdot", "parameter": "k", "value": 19500}`).
- **`env.py` (`TaskEnv`)**: Maintains `TaskModel.mo` in memory and executes 13 tools. Provides rich port-direction diagnostics (distinguishing parameters like `mdot.k` from ports like `mdot.y`). Optional guards (`NUCEVO_STRICT_EDIT`, `NUCEVO_REPEAT_GUARD`) default to OFF for exact official reproducibility.
- **`agent.py` (`run_episode`)**: Runs the multi-turn loop with structure-aware step budgets (`assemble`: `min(30, 10 + n_connects)`, `multistep`: `22`, others: `16`) and rolling history truncation (`keep=8`, fallback `keep=4` when exceeding 48,000 prompt chars).

### `nucevo/memory/` — Versioned Memory, Probing, and Retrieval

- **`schema.py` (`MemoryItem`)**: Structured schema with `type` (`rule | failure | skill | success`), `status` (`candidate | verified | useful | restricted | contradicted | retired`), `applicability` (`task_types`, `component_types`, `topology_constraints`, `physical_conditions`), `action_template`, `known_pitfalls`, `probe`, `evidence`, `counterexamples`, `usage_statistics`, and `lineage`.
- **`bank.py` (`MemoryBank`)**: Thread-safe, append-only event-sourced store (`events.jsonl`). Supports `add`, `revise`, `merge`, `split`, `retire`, `restore`, `record_evidence`, `add_counterexample`, `record_usage`, and immutable SHA-1 `snapshot()` exports. Nothing is deleted.
- **`verifier.py`**: Evaluates structured probes on sampled evolution tasks:
  - `connection_driver`: Rewires target port to the claimed source and checks `v3.verify()`. Also runs an `alternative_source` (or initial faulted driver) to confirm counterfactual failure.
  - `parameter_effect`: Perturbs a component parameter by `±15%` and checks monotonic response direction.
  - Grades evidence as `structural` -> `single_point` -> `multi_point` -> `counterfactual`. Failed probes mark items `contradicted`.
- **`retriever.py` (`Retriever` & `NucEvoRetriever`)**:
  - Multi-factor scoring for each active non-contradicted item $m$ and observable task query $q$:
    $$S(m, q) = 0.5 \cdot \text{cos}(e_q, e_m) + 0.3 \cdot \text{App}(m, q) + 0.2 \cdot \bigl(2 \cdot \text{Util}(m) - 1\bigr)$$
    where $\text{Util}(m) = \frac{\text{pos} + 1}{\text{pos} + \text{neg} + 2}$ is the Beta(1, 1) posterior mean of verified action-level transfer credit.
  - **`NucEvoRetriever`** routes by observable `task["family"]`:
    - `family == "repair"`: Pairs top-1 note from `verified_bank` + top-1 non-duplicate note from `episodic_bank`, each truncated to `(4096 - 2) // 2 = 2047` chars.
    - All other families: Standard budget-fitted ranking from `verified_bank` (`4096` chars).

### `nucevo/evolution/runner.py` — Training-Free Evolution Loop

Each iteration processes a parallel chunk (`parallel=6`) against the current frozen bank:
1. Retrieve context -> run episodes concurrently -> evaluate via `v3.verify()`.
2. Detect actual memory usage (`used_memory_ids`: checks if the item's probed connection or action template appears in `final_text`) and update `positive_transfer` / `negative_transfer`.
3. If `novel_only=True`, skip teacher calls when every episode in the batch repeats an already-seen failure signature with zero successes.
4. Teacher reflects on the batch -> `validate_experiences()` rejects vague advice -> `verify_candidate()` executes OpenModelica probes.
5. `admit()` enforces verification tier (`counterfactual` for probed rules; episode-grounded `structural` evidence for procedural lessons).
6. `integrate()` performs semantic dedup (`cosine > 0.93`), targeted revisions (`refines_existing`), or insertion.
7. `housekeeping()` at snapshot boundaries retires items with `negative_transfer >= 3` and `positive_transfer == 0`, and merges near-identical verified items (`cosine > 0.97`).

### `nucevo/evaluation/` — Frozen Evaluation & Paired Statistics

- **`runner.py` (`EvalRunner`)**: Pre-computes SHA-256 prompt hashes for every `(task_id, policy)` pair, deduplicates identical prompts across policies so each unique `(task_id, prompt_sha256)` runs only once, writes atomic JSON episode files for crash-safe resumption, and outputs `main_table.csv`, `difficulty_table.csv`, `family_table.csv`, `per_task.csv`, and `summary.json`.
- **`metrics.py`**: Computes stratified success rates and exact two-sided binomial McNemar tests (`exact_mcnemar`) on paired task outcomes.

---

## 3. Common Extension Points for Collaborators

- **Adding a new retrieval policy**: Add an entry inside `NucEvoRetriever.retrieve_all_policies()` in `nucevo/memory/retriever.py`, register its key/label in `nucevo/evaluation/runner.py`, and run `scripts/evaluate.py --partition validation --policies nucevo,<your_policy>`.
- **Testing a new solver backbone**: Serve any OpenAI-compatible model via vLLM with `--served-model-name solver` and point `--ports` to it; no code changes required.
- **Adding new plant topologies or fault types**: Extend `nucevo/benchmark/generators/topologies.py` and `tasks.py`, verify new templates with `python -m nucevo.benchmark.validate_templates`, and keep `data/MANIFEST.json` untouched for v1 comparability.
