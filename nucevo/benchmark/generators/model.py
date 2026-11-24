"""Model-text construction for NucEvolve-Bench tasks.

Three views of the same topology:
  reference  the hidden correct plant (used only to produce design-point targets
             and to validate the task before it is admitted to a split)
  skeleton   assembly task: instances present, all/some connects missing
  faulted    repair/diagnose/tune task: complete plant with injected faults

Every task in the benchmark is admitted only after its reference passes the full
verifier; every repair task is admitted only after its faulted text *fails*.
"""
from __future__ import annotations

from .topologies import I


def _fmt(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    return f"{v:.8g}"


def instance_line(inst, overrides=None):
    ov = dict(inst.get("params") or {})
    if overrides:
        ov.update(overrides)
    if ov:
        ps = ", ".join(f"{k} = {_fmt(v)}" for k, v in ov.items())
        return f"  NuclearEdu.{inst['type']} {inst['name']}({ps});"
    return f"  NuclearEdu.{inst['type']} {inst['name']};"


def _body(topo, overrides=None):
    overrides = overrides or {}
    lines = [f'model TaskModel "{topo["template"]} plant"']
    for inst in topo["instances"]:
        lines.append(instance_line(inst, overrides.get(inst["name"], {})))
    lines.append("equation")
    return lines


def reference_text(topo, overrides=None):
    lines = _body(topo, overrides)
    for a, b in topo["connects"]:
        lines.append(f"  connect({a}, {b});")
    lines.append("end TaskModel;")
    return "\n".join(lines) + "\n"


def skeleton_text(topo, overrides=None, keep_connects=0, keep_frac=0.0):
    """Assembly view.  `keep_connects` keeps a prefix; `keep_frac` keeps a random
    (deterministically seeded) subset, which changes task difficulty without
    changing the topology."""
    lines = _body(topo, overrides)
    conns = list(topo["connects"])
    if keep_frac:
        import random as _r
        rng = _r.Random(int(keep_frac * 1e6) ^ len(conns))
        k = int(round(keep_frac * len(conns)))
        keep = [c for c in conns if rng.random() < k / max(len(conns), 1)]
        keep = keep[:k]
    else:
        keep = conns[:keep_connects]
    for a, b in keep:
        lines.append(f"  connect({a}, {b});")
    lines += [
        "  // TODO: complete the plant described in the task instruction.",
        "  // Every input connector must be driven exactly once; outputs may fan out.",
        "  // Port directions and types are listed in the task instruction.",
        "end TaskModel;",
    ]
    return "\n".join(lines) + "\n"


def faulted_text(topo, overrides=None, faults=None):
    """Repair view.  Supported faults:
      swap_connect   exchange the destinations of two connections
      replace_connect  rewire one destination to another legal input
      drop_connect   delete one connection
      param_scale    multiply a numeric parameter by a factor
      wrong_source   set a Const / PowerSource value to a wrong value
    """
    conns = list(topo["connects"])
    ov = {k: dict(v) for k, v in (overrides or {}).items()}
    for f in faults or []:
        kind = f["kind"]
        if kind == "swap_connect":
            p1, p2 = tuple(f["pairs"][0]), tuple(f["pairs"][1])
            if p1 not in conns or p2 not in conns:
                raise ValueError(f"swap pair absent: {p1} {p2}")
            i, j = conns.index(p1), conns.index(p2)
            a1, b1 = conns[i]
            a2, b2 = conns[j]
            conns[i] = (a1, b2)
            conns[j] = (a2, b1)
        elif kind == "replace_connect":
            old, new = tuple(f["old"]), tuple(f["new"])
            if old not in conns:
                raise ValueError(f"replace pair absent: {old}")
            conns[conns.index(old)] = new
        elif kind == "drop_connect":
            pr = tuple(f["pair"])
            if pr not in conns:
                raise ValueError(f"drop pair absent: {pr}")
            conns.remove(pr)
        elif kind == "param_scale":
            ov.setdefault(f["inst"], {})[f["param"]] = f["value"]
        elif kind == "wrong_source":
            ov.setdefault(f["inst"], {})[f["param"]] = f["value"]
        else:
            raise ValueError(kind)
    lines = _body(topo, ov)
    for a, b in conns:
        lines.append(f"  connect({a}, {b});")
    lines.append("end TaskModel;")
    return "\n".join(lines) + "\n"


def instance_table(topo):
    """Human-readable instance/port table handed to the agent.

    It lists the component instances and their port names and directions, but
    never the structure (which output drives which input)."""
    from ..verifier.phys import COMPONENT_PORTS
    rows = ["instance | type | inputs | outputs"]
    for inst in topo["instances"]:
        p = COMPONENT_PORTS[inst["type"]]
        rows.append(f"{inst['name']} | {inst['type']} | "
                    f"{', '.join(p['in']) or '-'} | {', '.join(p['out'])}")
    return "\n".join(rows)
