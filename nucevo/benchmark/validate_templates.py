"""Validate every topology template: reference must simulate and pass all
component-level physics residuals.  Run once before any task generation."""
from __future__ import annotations

import json
import sys

from .generators.topologies import TEMPLATES, build, fingerprint
from .generators.model import reference_text
from .simulator import simulate
from .verifier import phys


def steady(series, tail=200):
    out = {}
    for k, ys in series.items():
        if not ys:
            continue
        v = ys[-tail:] if len(ys) > tail else ys
        out[k] = sum(v) / len(v)
    return out


def run(topo, stop=1200.0, n=None):
    txt = reference_text(topo, {})
    r = simulate({"TaskModel.mo": txt}, "TaskModel", stop_time=stop,
                 n_intervals=n or int(stop), variables=None,
                 with_series=True, timeout=120.0)
    if not r.ok:
        return None, r
    return steady(r.series), r


def main():
    rows, bad = [], []
    for name in TEMPLATES:
        topo = build(name)
        ss, r = run(topo)
        if ss is None:
            rows.append({"template": name, "ok": False, "stage": "simulate",
                         "errors": (r.errors or [])[:3]})
            bad.append(name)
            continue
        ok, det = phys.check(reference_text(topo, {}), ss)
        rows.append({"template": name, "ok": ok, "fingerprint": fingerprint(topo),
                     "failures": det["failures"][:8],
                     "scalars": {k: round(v, 4) for k, v in det["values"].items()
                                 if isinstance(v, float)}})
        if not ok:
            bad.append(name)
    print(json.dumps(rows, indent=1))
    print("\nFAILED:", bad if bad else "none", file=sys.stderr)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
