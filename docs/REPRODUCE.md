# Reproduction Guide

This checklist covers exact reproduction of both the **frozen test evaluation** (no teacher API needed) and **end-to-end experience evolution** from scratch.

---

## 1. Prerequisites

- **OS**: Linux (x86_64) recommended for OpenModelica + vLLM GPU inference.
- **Python**: `>= 3.10`
- **OpenModelica**: `1.27.1` (conda-forge `openmodelica` or official system packages).
- **GPU**: 1–4 NVIDIA GPUs (24 GB+ VRAM each; 4× GPUs with 16 concurrent workers finish the full 800-task × 7-policy evaluation in ~2 hours).
- **Models**:
  - Solver: `Qwen/Qwen3-4B-Instruct-2507`
  - Retriever encoder: `BAAI/bge-small-en-v1.5` (auto-downloaded via HuggingFace)

---

## 2. Environment Setup

```bash
# 1. Install Python dependencies
pip install -r requirements.txt
pip install vllm  # on GPU node

# 2. Verify OpenModelica compiler
export NUCEVO_OMC=$(which omc)
export OPENMODELICALIBRARY=/path/to/omlibrary
export OMP_NUM_THREADS=2
"$NUCEVO_OMC" --version

# 3. Verify NucEvo-Bench v1 integrity and offline unit tests
python scripts/verify_benchmark.py
python -m unittest discover -s tests -v
```

*(Optional)* Validate all 15 Modelica topology templates against OpenModelica:

```bash
python -m nucevo.benchmark.validate_templates
```

---

## 3. Serve the Frozen Solver (`Qwen3-4B-Instruct-2507`)

Launch one vLLM replica per allocated GPU (default ports `8011–8014`):

```bash
MODEL=/path/to/Qwen3-4B-Instruct-2507 GPUS=0,1,2,3 BASE_PORT=8011 bash scripts/serve_vllm.sh
```

Verify readiness:

```bash
curl -s http://127.0.0.1:8011/v1/models
```

Ensure `served-model-name` is `solver` and `max-model-len` is `16384`.

---

## 4. Reproduce the Official 800-Task Test Benchmark (Frozen Banks)

This uses the pinned banks in `assets/banks/verified_bank.json` and `assets/banks/episodic_bank.json` (no external teacher API required):

```bash
python -u scripts/evaluate.py \
  --config configs/eval_benchmark_v1.json \
  --ports 8011,8012,8013,8014 \
  --workers 16 \
  --out runs/repro_test_v1
```

### Resuming After Interruption

`scripts/evaluate.py` saves every deduplicated `(task_id, prompt_sha256)` episode atomically under `runs/repro_test_v1/episodes/`. Re-running the exact same command skips completed episodes and resumes cleanly.

### Output Artifacts

- `runs/repro_test_v1/main_table.csv` — Overall and per-split scores
- `runs/repro_test_v1/difficulty_table.csv` — Easy / Medium / Hard / Challenge breakdown
- `runs/repro_test_v1/family_table.csv` — Repair / Assemble / Tune / Diagnose / Multi-step breakdown
- `runs/repro_test_v1/per_task.csv` — Binary per-task matrix for paired statistical analysis
- `runs/repro_test_v1/summary.json` — Full metrics and exact two-sided McNemar p-values

---

## 5. Reproduce Dual-Bank Evolution from Scratch

Copy `.env.example` to `.env` and fill in your OpenAI-compatible reflection teacher endpoint:

```bash
cp .env.example .env
# Edit NUCEVO_TEACHER_BASE_URL, NUCEVO_TEACHER_API_KEY, NUCEVO_TEACHER_MODEL
```

Run both evolution streams on the same 120-task shuffled evolution order (`seed=11`):

```bash
# Verified Experience Bank (counterfactual simulator probing + full CRUD)
python -u scripts/evolve.py \
  --config configs/evolution_verified.json \
  --ports 8011,8012,8013,8014 \
  --out runs/evolve_verified

# Episodic Repair Bank (trajectory distillation + semantic dedup)
python -u scripts/evolve.py \
  --config configs/evolution_episodic.json \
  --ports 8011,8012,8013,8014 \
  --out runs/evolve_episodic
```

Then evaluate your newly evolved banks on the validation or test partition:

```bash
python -u scripts/evaluate.py \
  --partition test \
  --verified-bank runs/evolve_verified/snapshots/v*_0120.json \
  --episodic-bank runs/evolve_episodic/snapshots/v*_0120.json \
  --policies none,reasoningbank,verified_only,nucevo \
  --ports 8011,8012,8013,8014 \
  --workers 16 \
  --out runs/eval_custom_banks
```
