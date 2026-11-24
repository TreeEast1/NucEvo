"""Verifier v3 — structural, physical, multi-operating-point, design-point.

Deliberately does **not** require the submitted connect list to equal the hidden
reference.  A different wiring that drives every input, conserves mass/energy in
every block, and reproduces the design point at several operating points is
accepted.  Conversely a reference-identical model that does not conserve is
rejected.

Criteria (reported separately, never collapsed):
  build/sim      compiles and integrates to the requested horizon
  steady         the measured signals stop moving
  structural     every input driven exactly once, out->in, kinds compatible
  physics        every block's own conservation law closes (reference-free)
  multipoint     physics still closes after boundary perturbation
  transient      kinetics direction test (kinetics topologies only)
  goal           design point reproduced within tolerance
"""
from __future__ import annotations

import math
import re

from ..simulator import simulate
from . import phys


def steady(series, tail=200):
    out = {}
    for k, ys in series.items():
        if not ys:
            continue
        v = ys[-tail:] if len(ys) > tail else ys
        out[k] = sum(v) / len(v)
    return out


def run(text, stop=1200.0, timeout=90.0):
    return simulate({"TaskModel.mo": text}, "TaskModel", stop_time=stop,
                    n_intervals=int(stop), variables=None, with_series=True,
                    timeout=timeout)


def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


# --------------------------------------------------------------------------- structure
def structural_check(text, topo):
    d = {"checks": {}, "errors": [], "unconnected": [], "double_driven": [],
         "bad_direction": [], "incompatible": []}
    inst = {i["name"]: i["type"] for i in phys.parse_instances(text)}
    expected = {i["name"]: i["type"] for i in topo["instances"]}
    d["extra_instances"] = sorted(set(inst) - set(expected))
    d["missing_instances"] = sorted(set(expected) - set(inst))
    d["renamed_or_retyped"] = sorted(
        f"{n}:{expected[n]}->{inst[n]}" for n in set(inst) & set(expected)
        if inst[n] != expected[n])
    d["checks"]["instance_set"] = not (d["extra_instances"] or d["missing_instances"]
                                       or d["renamed_or_retyped"])

    conns = phys.parse_connects(text)
    seen, driven = set(), {}
    direction_ok, kind_ok, dup = True, True, False
    for a, b in conns:
        key = tuple(sorted((a, b)))
        if key in seen:
            dup = True
            d["errors"].append(f"duplicate connect {a} -> {b}")
            continue
        seen.add(key)
        ia, pa = a.split(".", 1) if "." in a else (a, "")
        ib, pb = b.split(".", 1) if "." in b else (b, "")
        if ia not in inst or ib not in inst:
            d["errors"].append(f"unknown instance in {a} -> {b}")
            direction_ok = False
            continue
        ta, tb = inst[ia], inst[ib]
        pa_in = pa in phys.COMPONENT_PORTS[ta]["in"]
        pa_out = pa in phys.COMPONENT_PORTS[ta]["out"]
        pb_in = pb in phys.COMPONENT_PORTS[tb]["in"]
        pb_out = pb in phys.COMPONENT_PORTS[tb]["out"]
        if not (pa_in or pa_out) or not (pb_in or pb_out):
            d["errors"].append(f"unknown port in {a} -> {b}")
            direction_ok = False
            continue
        if ia == ib:
            d["errors"].append(f"self connection {a} -> {b}")
            direction_ok = False
            continue
        if pa_out and pb_in:
            dst = b
        elif pb_out and pa_in:
            dst = a
        else:
            d["bad_direction"].append(f"{a} -> {b}")
            direction_ok = False
            continue
        driven[dst] = driven.get(dst, 0) + 1
        ka = phys.KIND.get((ta, pa), "generic")
        kb = phys.KIND.get((tb, pb), "generic")
        if ka != "generic" and kb != "generic" and ka != kb:
            kind_ok = False
            d["incompatible"].append(f"{a}({ka}) -> {b}({kb})")
    d["checks"]["connect_direction"] = direction_ok and not d["bad_direction"]
    d["checks"]["no_duplicates"] = not dup
    d["checks"]["signal_kinds"] = kind_ok

    missing, double = [], []
    for name, typ in expected.items():
        for p in phys.COMPONENT_PORTS[typ]["in"]:
            n = driven.get(f"{name}.{p}", 0)
            (missing if n == 0 else double if n > 1 else []).append(f"{name}.{p}")
    d["unconnected"], d["double_driven"] = missing, double
    d["checks"]["all_inputs_driven_once"] = not missing and not double
    d["n_connects"] = len(conns)
    d["ok"] = all(d["checks"].values())
    return d["ok"], d


# --------------------------------------------------------------------------- multipoint
KNOBS = {
    "Const": ["k"], "PowerSource": ["P0"], "Pump": ["Head0"], "Pipe": ["K"],
    "SteamGenerator": ["UA"], "Splitter": ["frac"], "Controller": ["Kp"],
}


def _set_param(text, inst, param, value):
    m = re.search(rf"(NuclearEdu\.\w+\s+{re.escape(inst)}\s*\()([^)]*)(\))", text)
    if not m:
        return None
    body = m.group(2)
    pm = re.search(rf"\b{re.escape(param)}\s*=\s*([-+0-9.eE]+)", body)
    if pm:
        nb = body[:pm.start(1)] + f"{float(value):.8g}" + body[pm.end(1):]
    else:
        nb = body + (", " if body.strip() else "") + f"{param} = {float(value):.8g}"
    return text[:m.start(2)] + nb + text[m.end(2):]


def multipoint_check(text, topo, timeout=60.0, factors=(0.88, 1.12), max_knobs=2):
    out = {"points": [], "ok": True}
    insts = phys.parse_instances(text)
    expected_types = {i["name"]: i["type"] for i in topo["instances"]}
    knobs = []
    for i in insts:
        if expected_types.get(i["name"]) != i["type"]:
            continue
        for p in KNOBS.get(i["type"], []):
            if p in (i.get("params") or {}):
                knobs.append((i["name"], p, float(i["params"][p])))
    # prefer boundary/source knobs over structural fractions
    knobs.sort(key=lambda k: (k[1] in ("frac",), k[0]))
    for name, par, base in knobs[:max_knobs]:
        for f in factors:
            if par == "frac" and not (0.05 < base * f < 0.95):
                continue
            txt = _set_param(text, name, par, base * f)
            if txt is None:
                continue
            r = run(txt, timeout=timeout)
            if not r.ok:
                out["points"].append({"knob": f"{name}.{par}", "factor": f,
                                      "ok": False, "why": "sim_failed"})
                out["ok"] = False
                continue
            ss = steady(r.series)
            pok, det = phys.check(txt, ss)
            out["points"].append({"knob": f"{name}.{par}", "factor": f,
                                  "ok": bool(pok), "failures": det["failures"][:4]})
            out["ok"] = out["ok"] and pok
    return out["ok"], out


# --------------------------------------------------------------------------- transient
def transient_check(text, topo, timeout=60.0, threshold=0.005):
    if topo["template"] not in ("kinetics", "kinetics_pumped"):
        return True, {"ok": True, "skipped": True}
    # locate the Const that drives rho_ext
    driver = {b: a for a, b in phys.parse_connects(text)}.get(
        next((f"{i['name']}.rho_ext" for i in topo["instances"]
              if i["type"] == "PointKinetics"), ""), None)
    if not driver or not driver.endswith(".y"):
        return False, {"ok": False, "why": "no rho source"}
    rho_inst = driver.split(".")[0]
    p_inst = next(i["name"] for i in topo["instances"] if i["type"] == "PointKinetics")
    vals = {}
    for sign, key in ((0, "base"), (1, "up"), (-1, "down")):
        txt = _set_param(text, rho_inst, "k", sign * 5e-4)
        if txt is None:
            return False, {"ok": False, "why": "rho param not numeric"}
        r = run(txt, stop=300.0, timeout=timeout)
        ys = r.series.get(f"{p_inst}.P.u", []) if r.ok else []
        vals[key] = ys[-1] if ys else float("nan")
    ok = (all(_finite(v) for v in vals.values())
          and vals["up"] > (1 + threshold) * vals["base"]
          and vals["down"] < (1 - threshold) * vals["base"])
    return ok, {"ok": bool(ok), "values": vals,
                "rel": {k: (v / vals["base"] - 1 if vals["base"] else None)
                        for k, v in vals.items()}}


# --------------------------------------------------------------------------- full verify
def verify(task, text, do_multipoint=True, do_transient=True, timeout=90.0, stop=None):
    topo = task["topology"]
    stop = stop or task.get("stop", 1200.0)
    r = run(text, stop=stop, timeout=timeout)
    out = {"build_ok": bool(r.build_ok), "sim_ok": bool(r.ok), "steady": False,
           "structural_ok": False, "physics_ok": False, "multipoint_ok": False,
           "transient_ok": True, "goal_ok": False, "success": False,
           "values": {}, "errors": [], "messages": (r.messages or "")[:1500]}
    if not r.ok:
        raw = getattr(r, "raw_stdout", "") or ""
        out["errors"] = [ln for ln in raw.splitlines() if "Error" in ln][-10:]
        return out

    ss = steady(r.series)
    sok, sd = structural_check(text, topo)
    out["structural_ok"], out["structural"] = sok, sd
    pok, pd = phys.check(text, ss)
    out["physics_ok"], out["physics"] = pok, pd

    # steadiness on every measured variable
    worst = 0.0
    for _, var in topo["measure"].items():
        ys = r.series.get(var, [])
        if len(ys) > 101:
            tail = ys[-100:]
            worst = max(worst, max(tail) - min(tail))
    out["steady"] = worst < 0.5
    out["steady_range"] = worst

    vals = {name: ss.get(var, float("nan")) for name, var in topo["measure"].items()}
    out["values"] = vals
    goal, details = True, {}
    for name, target in task.get("targets", {}).items():
        v = vals.get(name, float("nan"))
        tol = task.get("tolerances", {}).get(name, 0.05)
        abs_tol = task.get("abs_tolerances", {}).get(name)
        okv = False
        if _finite(v):
            okv = (abs(v - target) <= abs_tol) if abs_tol is not None else \
                (abs(v - target) <= tol * max(abs(target), 1e-9))
        details[name] = {"value": v, "target": target, "ok": bool(okv)}
        goal = goal and okv
    out["goal_detail"] = details
    out["goal_ok"] = bool(goal)

    if do_multipoint and pok and sok:
        mok, md = multipoint_check(text, topo, timeout=timeout)
    else:
        mok, md = False, {"ok": False, "skipped": True}
    out["multipoint_ok"], out["multipoint"] = bool(mok), md
    tok, td = transient_check(text, topo, timeout=timeout) if do_transient else (True, {"skipped": True})
    out["transient_ok"], out["transient"] = bool(tok), td

    out["success"] = bool(out["build_ok"] and out["sim_ok"] and out["steady"]
                          and sok and pok and out["goal_ok"] and out["transient_ok"]
                          and (mok if do_multipoint else True))
    return out
