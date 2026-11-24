"""Stratified evaluation aggregation and exact two-sided McNemar paired tests."""
from __future__ import annotations

import csv
from collections import defaultdict
from math import comb
from pathlib import Path


def exact_mcnemar(first: list[bool], second: list[bool]) -> dict:
    """Exact two-sided binomial (McNemar) paired test between two binary outcome vectors."""
    if len(first) != len(second):
        raise ValueError(f"Unpaired samples: {len(first)} vs {len(second)}")
    only_first = sum(bool(a) and not bool(b) for a, b in zip(first, second))
    only_second = sum(bool(b) and not bool(a) for a, b in zip(first, second))
    discordant = only_first + only_second
    if discordant == 0:
        p_val = 1.0
    else:
        tail = sum(comb(discordant, k) for k in range(min(only_first, only_second) + 1))
        p_val = min(1.0, 2.0 * tail / (2**discordant))
    n = len(first)
    first_sum = sum(bool(a) for a in first)
    second_sum = sum(bool(b) for b in second)
    return {
        "n": n,
        "first": first_sum,
        "second": second_sum,
        "only_first": only_first,
        "only_second": only_second,
        "delta_pp": round(100.0 * (first_sum - second_sum) / n, 3) if n else 0.0,
        "p_two_sided": p_val,
    }


def summarize_evaluation(
    rows: list[dict],
    policies: tuple[str, ...],
    primary_policy: str = "nucevo",
) -> dict:
    """Aggregate outcomes overall, by split, by task family, and by difficulty tier."""
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets["all"].append(row)
        if row.get("split"):
            buckets[f"split/{row['split']}"].append(row)
        if row.get("family"):
            buckets[f"family/{row['family']}"].append(row)
        if row.get("difficulty_tier"):
            buckets[f"tier/{row['difficulty_tier']}"].append(row)

    summary = {}
    for group_name, group_rows in sorted(buckets.items()):
        n = len(group_rows)
        vectors = {pol: [bool(r[pol]) for r in group_rows] for pol in policies}
        correct = {pol: sum(vec) for pol, vec in vectors.items()}
        rates = {pol: round(100.0 * correct[pol] / n, 2) if n else 0.0 for pol in policies}
        paired_vs_primary = {}
        if primary_policy in vectors:
            for pol in policies:
                if pol != primary_policy:
                    paired_vs_primary[pol] = exact_mcnemar(vectors[primary_policy], vectors[pol])
        summary[group_name] = {
            "n": n,
            "correct": correct,
            "rate_pct": rates,
            "paired_vs_primary": paired_vs_primary,
        }
    return summary


def write_csv_table(
    path: Path,
    groups: dict,
    row_keys: list[tuple[str, str]],
    policies: tuple[str, ...],
    policy_labels: dict[str, str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Group", "Tasks", *[policy_labels.get(p, p) for p in policies]])
        for group_key, display_name in row_keys:
            if group_key not in groups:
                continue
            entry = groups[group_key]
            n = entry["n"]
            row = [display_name, n]
            for pol in policies:
                c = entry["correct"][pol]
                row.append(f"{c}/{n} ({100.0 * c / n:.1f}%)")
            writer.writerow(row)
