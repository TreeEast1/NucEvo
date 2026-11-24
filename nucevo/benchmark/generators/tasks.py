"""Task construction for NucEvolve-Bench.

A task = topology + operating point + family + (optional) faults + the hidden
reference.  Every admitted task satisfies:

  * its reference simulates, is steady, and passes the reference-free physics
    verifier;
  * every repair/tune task's faulted text fails the verifier (the task is
    actually broken);
  * its signature (topology, operating point, faults) is unique within a split.

Task families:
  assemble   complete the plant from a partially wired skeleton
  repair     fix injected structural faults
  tune       the plant runs but misses the design point; adjust parameters
  diagnose   repair + name the fault class
  multistep  several faults of mixed kinds; needs iterative simulation
"""
from __future__ import annotations

import hashlib
import json
import random

from . import topologies as T
from .model import reference_text, skeleton_text, faulted_text, instance_table
from ..verifier import phys, v3

DIAG_LABELS = ["swapped_connections", "wrong_parameter", "wrong_boundary_condition",
               "wrong_routing", "missing_connection"]

TEMPLATE_FAMILIES = ["assemble", "repair", "tune", "diagnose", "multistep"]


# --------------------------------------------------------------------------- operating point
def _const_roles(topo):
    """Which Const is the temperature source and which is the flow source, from
    the connect graph (survives renaming)."""
    roles = {}
    for inst in topo["instances"]:
        if inst["type"] != "Const":
            continue
        dsts = [b for a, b in topo["connects"] if a.split(".")[0] == inst["name"]]
        if any(d.endswith((".Tin", ".T")) for d in dsts):
            roles[inst["name"]] = "temperature"
        elif any(d.endswith((".mdot", ".m")) for d in dsts):
            roles[inst["name"]] = "flow"
        elif any(d.endswith("rho_ext") for d in dsts):
            roles[inst["name"]] = "reactivity"
    return roles


def randomize(topo, rng, wide=True):
    """Pick a physically sensible operating point.  Returns {inst: {param: value}}."""
    ov = {}
    roles = _const_roles(topo)
    for inst in topo["instances"]:
        typ, nm = inst["type"], inst["name"]
        if typ == "PowerSource":
            ov[nm] = {"P0": rng.choice([2.4e9, 2.7e9, 3.0e9, 3.2e9, 3.4e9])}
        elif typ == "PointKinetics":
            ov[nm] = {"P0": rng.choice([2.7e9, 3.0e9, 3.3e9])}
        elif typ == "Const":
            role = roles.get(nm)
            if role == "temperature":
                ov[nm] = {"k": rng.choice([555.0, 560.0, 565.0, 570.0])}
            elif role == "flow":
                ov[nm] = {"k": rng.choice([15000.0, 16500.0, 18000.0, 19500.0, 21000.0])}
            elif role == "reactivity":
                ov[nm] = {"k": 0.0}
        elif typ == "FuelPin":
            ov[nm] = {"G_gap": rng.choice([8e6, 1e7, 1.2e7]),
                      "G_cl": rng.choice([1.2e8, 1.5e8, 1.8e8])}
        elif typ == "SteamGenerator":
            ov[nm] = {"UA": rng.choice([2.8e7, 3.33e7, 3.8e7]),
                      "Tsec": rng.choice([485.0, 490.0, 495.0])}
        elif typ == "Splitter":
            ov[nm] = {"frac": rng.choice([0.4, 0.5, 0.6])}
        elif typ == "Pump":
            ov[nm] = {"Head0": rng.choice([2.8e6, 3.2e6, 3.6e6])}
        elif typ == "Pipe":
            ov[nm] = {"K": rng.choice([3.5e-3, 4.756e-3, 6.0e-3])}
        elif typ == "Controller":
            ov[nm] = {"Kp": rng.choice([8e7, 1e8, 1.2e8]),
                      "Tset": rng.choice([590.0, 600.0, 610.0])}
    return ov


# --------------------------------------------------------------------------- reference
def _tol(name, target):
    if name.startswith("T"):
        return 3.0, 0.05
    if name.startswith(("m", "M")):
        return None, 0.03
    return None, 0.05


# Parameters whose value is visible at steady state (changing one must move the
# design point).  Thermal capacitances and kinetics constants are excluded: they
# change only the transient, so "faulting" them would not create a broken task.
SENSITIVE = {
    "Const": ["k"], "PowerSource": ["P0"], "FuelPin": ["G_gap", "G_cl"],
    "SteamGenerator": ["UA", "Tsec"], "Pump": ["Head0", "Mdot0"], "Pipe": ["K"],
    "Splitter": ["frac"], "Controller": ["Kp", "Tset"], "PointKinetics": ["P0"],
}


def make_reference(task):
    topo = task["topology"]
    txt = reference_text(topo, task["overrides"])
    r = v3.run(txt, stop=task.get("stop", 1200.0), timeout=120.0)
    if not r.ok:
        return None
    ss = v3.steady(r.series)
    vals = {name: ss.get(var, float("nan")) for name, var in topo["measure"].items()}
    if not all(isinstance(v, float) and v == v for v in vals.values()):
        return None
    ok, det = phys.check(txt, ss)
    if not ok:
        return None
    task = dict(task)
    task["targets"] = vals
    task["tolerances"], task["abs_tolerances"] = {}, {}
    for name, val in vals.items():
        at, rt = _tol(name, val)
        if at is not None:
            task["abs_tolerances"][name] = at
        task["tolerances"][name] = rt
    task["reference_physics_ok"] = True
    return task


# --------------------------------------------------------------------------- faults
def _inputs_by_kind(topo):
    """(kind -> [(inst, port)]) for use as replacement destinations."""
    out = {}
    for inst in topo["instances"]:
        for p in phys.COMPONENT_PORTS[inst["type"]]["in"]:
            k = phys.KIND.get((inst["type"], p), "generic")
            out.setdefault(k, []).append((inst["name"], p))
    return out


def _outputs_by_kind(topo):
    out = {}
    for inst in topo["instances"]:
        for p in phys.COMPONENT_PORTS[inst["type"]]["out"]:
            k = phys.KIND.get((inst["type"], p), "generic")
            out.setdefault(k, []).append((inst["name"], p))
    return out


def _fault_value(param, v, rng):
    """Perturb a parameter into a clearly wrong but physically legal value."""
    if param == "frac":
        return min(0.9, max(0.1, v + rng.choice([-0.25, 0.25])))
    if param == "Tset":
        return v + rng.choice([-20.0, -15.0, 15.0, 20.0])
    if param == "Tsec":
        return v + rng.choice([-15.0, 15.0])
    f = rng.choice([0.1, 0.2, 5.0, 10.0])
    return v * f


def inject_faults(topo, rng, kinds, n_faults):
    conns = topo["connects"]
    used, faults = set(), []
    for _ in range(n_faults * 8):
        if len(faults) >= n_faults:
            break
        kind = rng.choice(kinds)
        if kind == "swap_connect" and len(conns) > 1:
            i = rng.randrange(len(conns))
            j = rng.randrange(len(conns))
            if i == j or i in used or j in used:
                continue
            used |= {i, j}
            faults.append({"kind": kind,
                           "pairs": [list(conns[i]), list(conns[j])],
                           "label": "swapped_connections"})
        elif kind == "replace_connect":
            i = rng.randrange(len(conns))
            if i in used:
                continue
            a_old, b_old = conns[i]
            dst_inst, dst_port = b_old.split(".", 1)
            dst_type = {x["name"]: x["type"] for x in topo["instances"]}[dst_inst]
            k = phys.KIND.get((dst_type, dst_port), "generic")
            cands = [f"{a}.{p}" for a, p in _outputs_by_kind(topo).get(k, [])
                     if f"{a}.{p}" != a_old]
            if not cands:
                continue
            used.add(i)
            faults.append({"kind": kind, "old": [a_old, b_old],
                           "new": [rng.choice(cands), b_old],
                           "label": "wrong_routing"})
        elif kind == "drop_connect":
            i = rng.randrange(len(conns))
            if i in used:
                continue
            used.add(i)
            faults.append({"kind": kind, "pair": list(conns[i]),
                           "label": "missing_connection"})
        elif kind in ("param_scale", "wrong_source"):
            numeric = [(x["name"], x["type"], p, v) for x in topo["instances"]
                       for p in SENSITIVE.get(x["type"], [])
                       for v in [phys.params_of(x).get(p)]
                       if isinstance(v, (int, float)) and v not in (0, 1)]
            if not numeric:
                continue
            nm, typ, p, v = rng.choice(numeric)
            if (nm, p) in used:
                continue
            nv = _fault_value(p, v, rng)
            if nv == v:
                continue
            used.add((nm, p))
            faults.append({"kind": kind, "inst": nm, "param": p, "value": nv,
                           "label": "wrong_parameter" if kind == "param_scale"
                           else "wrong_boundary_condition"})
    return faults


# --------------------------------------------------------------------------- instruction
def build_instruction(task):
    topo, fam = task["topology"], task["family"]
    lines = []
    if fam == "assemble":
        lines.append("TASK (assembly): the plant below is incompletely wired. Add the "
                     "missing connect(...) statements so that it realises the design.")
    elif fam == "repair":
        lines.append("TASK (repair): the plant below compiles but does not meet its "
                     "design point. Find and repair the fault(s).")
    elif fam == "tune":
        lines.append("TASK (tuning): the plant runs but its operating point is wrong. "
                     "Adjust the parameters that are out of specification.")
    elif fam == "diagnose":
        lines.append("TASK (diagnosis + repair): the plant is faulty. First call "
                     f"`diagnose` with exactly one of {DIAG_LABELS}, then repair it.")
    elif fam == "multistep":
        lines.append("TASK (multi-step repair): the plant has several independent "
                     "faults. Work iteratively: inspect, simulate, repair, re-check.")
    lines.append("")
    lines.append("Component instances and their ports (output -> input direction):")
    lines.append(instance_table(topo))
    lines.append("")
    lines.append("Design point (must be met within tolerance):")
    for name, val in task["targets"].items():
        at = task["abs_tolerances"].get(name)
        rt = task["tolerances"].get(name, 0.05)
        tol = f"+/-{at:g}" if at is not None else f"+/-{rt*100:g}%"
        lines.append(f"  {name} = {val:.6g}  ({tol})")
    lines.append("")
    n_in = sum(len(phys.COMPONENT_PORTS[i["type"]]["in"]) for i in topo["instances"])
    lines.append(f"Requirements: every one of the {n_in} input connectors must be driven "
                 "exactly once; the plant must be steady at the design point; mass and "
                 "energy must balance in every component, including at off-design "
                 "operating points. Do not modify the library or rename components.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- task factory
def make_task(task_id, template, family, rng, split, n_faults=1, rename_scheme=None):
    topo = T.build(template)
    if rename_scheme:
        topo = T.rename(topo, rng, rename_scheme)
    ov = randomize(topo, rng)
    task = {
        "task_id": task_id, "split": split, "template": template, "family": family,
        "stop": 1200.0, "overrides": ov,
        "fingerprint": T.fingerprint(topo),
        "provenance": "nucevolve_bench_v1",
    }
    task["topology"] = topo
    task = make_reference(task)
    if task is None:
        return None

    if family == "assemble":
        keep = rng.choice([0.0, 0.0, 0.15, 0.35])
        task["keep_fraction"] = keep
        text = skeleton_text(topo, ov, keep_frac=keep)
        task["difficulty"] = round(len(topo["connects"]) * (1 - keep), 1)
    else:
        kinds = {
            "repair": ["swap_connect", "replace_connect", "drop_connect"],
            "diagnose": ["swap_connect", "replace_connect", "drop_connect"],
            "tune": ["param_scale", "wrong_source"],
            "multistep": ["swap_connect", "replace_connect", "drop_connect",
                          "param_scale", "wrong_source"],
        }[family]
        faults = inject_faults(topo, rng, kinds, n_faults)
        if not faults:
            return None
        task["faults"] = faults
        task["difficulty"] = len(faults)
        text = faulted_text(topo, ov, faults)
        v = v3.verify(task, text, do_multipoint=False, do_transient=False)
        if v["success"]:
            return None                      # not actually broken -> not a task
        task["broken_verify"] = {k: v[k] for k in
                                 ("build_ok", "sim_ok", "structural_ok",
                                  "physics_ok", "goal_ok")}
    task["task_text_initial"] = text
    task["instruction"] = build_instruction(task)
    task["n_connects"] = len(topo["connects"])
    task["n_instances"] = len(topo["instances"])
    task["signature"] = signature(task)
    return task


def try_make_task(task_id, template, family, rng, split, n_faults=1, attempts=25,
                  rename_scheme=None):
    """Generation is rejection-sampled: most rejections are faults that turned out
    not to break the plant.  Returns None only if every attempt was rejected."""
    for _ in range(attempts):
        t = make_task(task_id, template, family, rng, split, n_faults=n_faults,
                      rename_scheme=rename_scheme)
        if t is not None:
            return t
    return None


def signature(task):
    payload = {"template": task["template"],
               "overrides": task["overrides"],
               "faults": [{k: v for k, v in f.items() if k != "label"}
                          for f in task.get("faults", [])],
               "keep": task.get("keep_fraction", 0)}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def serialise(task):
    """JSON-safe copy (topology is embedded as data)."""
    return json.loads(json.dumps(task))


def deserialise(d):
    return dict(d)
