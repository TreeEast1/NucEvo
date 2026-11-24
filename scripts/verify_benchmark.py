#!/usr/bin/env python3
"""Verify NucEvo-Bench v1 integrity: library SHA-1, split hashes, difficulty labels, and zero leakage."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nucevo.benchmark import verify_benchmark_integrity


def main() -> None:
    report = verify_benchmark_integrity()
    print(json.dumps(report, indent=2))
    print("\nOK: NucEvo-Bench v1 passed all integrity and anti-leakage checks.")


if __name__ == "__main__":
    main()
