"""Simulator-grounded verification of candidate experiences.

The teacher proposes a rule; the simulator decides whether it is true.  A
`probe` is a structured, testable claim attached to an experience:

  connection_driver   "in component X, port P must be driven by (component Y, port Q)"
                      with an optional `alternative_source` for a counterfactual.
  parameter_effect    "changing parameter p of component X moves measured value M
                      in direction D (relative to the design point)"

The verifier instantiates the claim on real tasks that contain the components,
simulates the rewired/perturbed plant, and grades the evidence:

  structural      the named components and ports exist and can be connected
  single_point    the claim reproduces the design point on one task
  multi_point     it reproduces the design point on >=3 tasks at different
                  operating points
  counterfactual  additionally, the named alternative does NOT reproduce it on
                  at least one task

A failed claim yields a stored counterexample and never becomes `verified`.
"""
from __future__ import annotations

import copy
import re

from nucevo.benchmark.generators.model import reference_text
from nucevo.benchmark.verifier import phys, v3


def _instances(task):
    return {i["name"]: i["type"] for i in task["topology"]["instances"]}


def _find_by_type(task, typ):
    return [i["name"] for i in task["topology"]["instances"] if i["type"] == typ]


def _driver_of(topo, port):
    for a, b in topo["connects"]:
        if b == port:
            return a
    return None


def _dst_of(conn, types):
    """The input side of a connect, whatever order it was written in."""
    a, b = conn
    if "." not in a or "." not in b:
        return None
    ia, pa = a.split(".", 1)
    ib, pb = b.split(".", 1)
    ta, tb = types.get(ia), types.get(ib)
    if not ta or not tb:
        return None
    if pa in phys.COMPONENT_PORTS[ta]["out"] and pb in phys.COMPONENT_PORTS[tb]["in"]:
        return b
    if pb in phys.COMPONENT_PORTS[tb]["out"] and pa in phys.COMPONENT_PORTS[ta]["in"]:
        return a
    return None


def _src_of(conn, types):
    d = _dst_of(conn, types)
    if d is None:
        return None
    return conn[0] if conn[1] == d else conn[1]


def _rewire_text(task, target_port, source_inst, source_port):
    topo = copy.deepcopy(task["topology"])
    types = {i["name"]: i["type"] for i in topo["instances"]}
    conns = [c for c in topo["connects"] if _dst_of(c, types) != target_port]
    conns.append((f"{source_inst}.{source_port}", target_port))
    topo["connects"] = conns
    return reference_text(topo, task["overrides"])


def _check_claim(task, target_port, source_inst, source_port):
    """Rewire the task's plant so target_port is driven by the proposed source,
    then require physics + design point to hold."""
    text = _rewire_text(task, target_port, source_inst, source_port)
    v = v3.verify(task, text, do_multipoint=False, do_transient=False)
    return {"ok": bool(v["success"]), "build": v["build_ok"], "sim": v["sim_ok"],
            "physics": v["physics_ok"], "goal": v["goal_ok"],
            "failures": (v.get("physics") or {}).get("failures", [])[:5]}


def evaluate_probe(probe, context_tasks, min_tasks_for_multi=3):
    """context_tasks: tasks that contain the probe's components, ideally drawn
    from the same template at several operating points."""
    kind = probe.get("kind")
    if kind == "connection_driver":
        return _eval_connection(probe, context_tasks, min_tasks_for_multi)
    if kind == "parameter_effect":
        return _eval_parameter(probe, context_tasks)
    return {"level": "structural", "passed": None, "reason": f"unknown probe kind {kind}",
            "details": {}}


def _eval_connection(probe, tasks, min_multi):
    comp, port = probe.get("component"), probe.get("port")
    src = probe.get("claimed_source") or {}
    alt = probe.get("alternative_source") or {}
    details, ok_tasks, alt_failed = [], 0, 0
    structural_ok = False
    for task in tasks:
        insts = _instances(task)
        targets = [n for n, t in insts.items() if t == comp]
        sources = [n for n, t in insts.items() if t == src.get("component")]
        alt_sources = [n for n, t in insts.items() if t == alt.get("component")]
        if not targets or not sources:
            details.append({"task": task["task_id"], "skipped": "components absent"})
            continue
        spec = phys.COMPONENT_PORTS.get(comp, {})
        src_spec = phys.COMPONENT_PORTS.get(src.get("component"), {})
        if port not in spec.get("in", []) or src.get("port") not in src_spec.get("out", []):
            return {"level": "structural", "passed": False,
                    "reason": "named port is not an input/output of the named type",
                    "details": details}
        structural_ok = True
        # A type-level rule can be instantiated by any matching pair; the claim is
        # supported if *some* pairing reproduces the design point.  In a plant with
        # three channels, fp2.Tcool must be fed by ch2, not by ch1.
        hit = None
        for tname in targets[:3]:
            target_port = f"{tname}.{port}"
            for sname in sources[:3]:
                res = _check_claim(task, target_port, sname, src["port"])
                if res["ok"]:
                    hit = (target_port, sname, res)
                    break
            if hit:
                break
        if hit is None:
            details.append({"task": task["task_id"], "claimed": {"ok": False}})
            continue
        target_port, sname, cres = hit
        ok_tasks += 1
        details.append({"task": task["task_id"], "claimed": cres,
                        "pair": [sname, target_port]})
        # counterfactual: an explicitly named alternative, else the driver that
        # the *initial* (agent-visible) model actually had.
        alt_failed_here = None
        if alt.get("component") and alt_sources:
            for s2 in alt_sources[:3]:
                ares = _check_claim(task, target_port, s2, alt["port"])
                if not ares["ok"]:
                    alt_failed_here = ares
                    break
        else:
            init = phys.parse_connects(task.get("task_text_initial", ""))
            init_types = {i["name"]: i["type"] for i in
                          phys.parse_instances(task.get("task_text_initial", ""))}
            drv = next((_src_of(c, init_types) for c in init
                        if _dst_of(c, init_types) == target_port), None)
            if drv and drv.split(".")[0] != sname:
                di, dp = drv.split(".", 1)
                ares = _check_claim(task, target_port, di, dp)
                if not ares["ok"]:
                    alt_failed_here = ares
        if alt_failed_here is not None:
            alt_failed += 1
            details.append({"task": task["task_id"], "alternative_failed": True})
    if not structural_ok:
        return {"level": "structural", "passed": None,
                "reason": "no context task contained the named components",
                "details": details}
    if ok_tasks == 0:
        return {"level": "structural", "passed": False,
                "reason": "claimed connection never reproduces the design point",
                "details": details}
    if ok_tasks >= min_multi and alt_failed >= 1:
        level = "counterfactual"
    elif ok_tasks >= min_multi:
        level = "multi_point"
    else:
        level = "single_point"
    return {"level": level, "passed": True,
            "reason": f"{ok_tasks}/{len(tasks)} tasks reproduced; "
                      f"counterfactual failed on {alt_failed}",
            "details": details}


def _eval_parameter(probe, tasks):
    comp = probe.get("component")
    param = probe.get("parameter")
    meas = probe.get("measure")
    direction = probe.get("direction", "increase")
    details, ok = [], 0
    for task in tasks:
        insts = _instances(task)
        cand = [n for n, t in insts.items() if t == comp]
        if not cand or meas not in task["targets"]:
            continue
        name = cand[0]
        cur = phys.params_of({"type": comp,
                              "params": task["overrides"].get(name, {})}).get(param)
        if cur is None:
            continue
        for factor in (1.15, 0.85):
            text = v3._set_param(reference_text(task["topology"], task["overrides"]),
                                 name, param, cur * factor)
            r = v3.run(text, stop=task.get("stop", 1200.0))
            if not r.ok:
                continue
            ss = v3.steady(r.series)
            got = ss.get(task["topology"]["measure"][meas])
            base = task["targets"][meas]
            if got is None:
                continue
            moved_up = got > base
            good = moved_up if (direction == "increase") == (factor > 1) else not moved_up
            details.append({"task": task["task_id"], "factor": factor,
                            "value": got, "base": base, "consistent": bool(good)})
            ok += bool(good)
    if not details:
        return {"level": "structural", "passed": None, "reason": "no probe executed",
                "details": details}
    level = "multi_point" if ok >= 2 else "single_point"
    return {"level": level, "passed": ok >= max(1, len(details) // 2),
            "reason": f"{ok}/{len(details)} perturbations consistent", "details": details}


def apply_verification(bank, memory_id, result, task_ids=()):
    """Write a probe result back into the bank (evidence + status)."""
    it = bank.items[memory_id]
    level = result.get("level", "structural")
    passed = result.get("passed")
    bank.record_evidence(memory_id, {
        "verification_level": level if passed else "structural",
        "probe_result": {"passed": passed, "reason": result.get("reason"),
                         "n_details": len(result.get("details", []))},
        "source_task_ids": list(task_ids),
    }, reason="probe")
    if passed is False:
        bank.add_counterexample(memory_id, {
            "probe": it.probe, "reason": result.get("reason")},
            reason="probe failed")
        bank.set_status(memory_id, "contradicted", "probe failed")
    elif passed is True and it.status == "candidate":
        bank.set_status(memory_id, "verified", f"probe:{level}")
    return bank
