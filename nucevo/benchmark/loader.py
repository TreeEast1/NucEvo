"""Dataset loading and integrity verification for NucEvo-Bench v1.

Partition layout:
  - evolution   : evolution (600 tasks)
  - validation  : dev (160 tasks)
  - test        : id_test (240) + ood_near (240) + ood_far (200) + stress (120) = 800 tasks
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nucevo.config import DATA_DIR, REPO_ROOT

SPLIT_PATHS = {
    "evolution": DATA_DIR / "evolution" / "evolution.jsonl",
    "dev": DATA_DIR / "validation" / "dev.jsonl",
    "id_test": DATA_DIR / "test" / "id_test.jsonl",
    "ood_near": DATA_DIR / "test" / "ood_near.jsonl",
    "ood_far": DATA_DIR / "test" / "ood_far.jsonl",
    "stress": DATA_DIR / "test" / "stress.jsonl",
}

PARTITIONS = {
    "evolution": ("evolution",),
    "validation": ("dev",),
    "test": ("id_test", "ood_near", "ood_far", "stress"),
}


def load_manifest(data_dir: Path | None = None) -> dict:
    root = Path(data_dir) if data_dir else DATA_DIR
    return json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))


def load_difficulty_labels(data_dir: Path | None = None) -> dict[str, dict]:
    root = Path(data_dir) if data_dir else DATA_DIR
    payload = json.loads((root / "benchmark_difficulty.json").read_text(encoding="utf-8"))
    return payload["per_task"]


def load_split(name: str, limit: int | None = None, attach_difficulty: bool = True) -> list[dict]:
    if name not in SPLIT_PATHS:
        raise KeyError(f"Unknown split '{name}'. Expected one of {sorted(SPLIT_PATHS)}")
    path = SPLIT_PATHS[name]
    tasks = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if attach_difficulty:
        labels = load_difficulty_labels()
        for task in tasks:
            info = labels.get(task["task_id"])
            if info:
                task["difficulty_tier"] = info["difficulty"]
    return tasks[:limit] if limit is not None else tasks


def load_partition(partition: str, attach_difficulty: bool = True) -> list[dict]:
    if partition not in PARTITIONS:
        raise KeyError(f"Unknown partition '{partition}'. Expected one of {sorted(PARTITIONS)}")
    tasks: list[dict] = []
    for split_name in PARTITIONS[partition]:
        tasks.extend(load_split(split_name, attach_difficulty=attach_difficulty))
    return tasks


def verify_benchmark_integrity() -> dict:
    """Verify library SHA-1, split SHA-1 hashes, difficulty labels, and zero cross-split leakage."""
    manifest = load_manifest()
    lib_path = REPO_ROOT / manifest["library_path"]
    lib_sha1 = hashlib.sha1(lib_path.read_bytes()).hexdigest()
    if lib_sha1 != manifest["library_sha1"]:
        raise ValueError(f"NuclearEdu.mo SHA-1 mismatch: {lib_sha1} != {manifest['library_sha1']}")

    labels = load_difficulty_labels()
    seen_ids: set[str] = set()
    seen_signatures: set[str] = set()
    split_counts: dict[str, int] = {}

    for split_name, meta in manifest["splits"].items():
        path = REPO_ROOT / meta["path"]
        raw = path.read_bytes()
        sha1 = hashlib.sha1(raw).hexdigest()
        if sha1 != meta["sha1"]:
            raise ValueError(f"Split '{split_name}' SHA-1 mismatch: {sha1} != {meta['sha1']}")
        tasks = [json.loads(line) for line in raw.splitlines() if line.strip()]
        if len(tasks) != meta["n"]:
            raise ValueError(f"Split '{split_name}' count mismatch: {len(tasks)} != {meta['n']}")
        for task in tasks:
            tid = task["task_id"]
            sig = task["signature"]
            if tid in seen_ids:
                raise ValueError(f"Duplicate task_id across benchmark: {tid}")
            if sig in seen_signatures:
                raise ValueError(f"Duplicate task signature across benchmark: {sig}")
            if not task.get("reference_validation", {}).get("v3_success"):
                raise ValueError(f"Reference validation failed for {tid}")
            if tid not in labels:
                raise ValueError(f"Missing difficulty label for {tid}")
            seen_ids.add(tid)
            seen_signatures.add(sig)
        split_counts[split_name] = len(tasks)

    return {
        "benchmark": manifest["benchmark"],
        "library_sha1": lib_sha1,
        "total_tasks": len(seen_ids),
        "unique_signatures": len(seen_signatures),
        "splits": split_counts,
        "partitions": {
            part: sum(split_counts[s] for s in splits)
            for part, splits in PARTITIONS.items()
        },
    }
