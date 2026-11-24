"""Composable nuclear plant topologies for NucEvolve-Bench.

A topology is a *causal block graph*: every block has typed input/output ports,
and a legal topology drives every input exactly once.  Instead of four hand-
written plants, Phase 7 composes plants from templates parameterised by
structure (number of parallel channels, number of SG trains, series stages,
optional instrumentation / pump / controller).  This is what makes "a new task"
mean a genuinely new physical arrangement rather than a new random seed on the
same 5-connect list.

Template families
-----------------
single            power source -> fuel pin -> channel (open, once-through)
loop              recirculating primary loop with one steam generator
parallelN         N parallel channels, splitter/mixer chains (N = 2,3,4)
seriesN           N channels in series (N = 2,3), once-through
sensor_loop       loop + pass-through instrumentation on the hot leg
two_sg            hot leg split into two steam generators
kinetics          point kinetics with fuel/moderator feedback
pumped            loop with pump/pipe hydraulic circuit (flow is solved)
controlled        parallel channels under closed-loop power control
pumped_controlled pumped loop + sensor/controller power control
kinetics_pumped   kinetics loop with a pump/pipe circuit

Novelty is defined by the *template + structural parameters*, never by instance
names or parameter values.  `rename()` exists only as a nuisance factor.
"""
from __future__ import annotations

import copy
import hashlib
import json


def I(name, typ, **params):
    return {"name": name, "type": typ, "params": params}


# --------------------------------------------------------------------------- templates
def _single(rng=None):
    return {
        "template": "single", "n": 1,
        "instances": [
            I("ps", "PowerSource", P0=3.0e9),
            I("fp", "FuelPin"),
            I("ch", "Channel"),
            I("tin", "Const", k=565.0),
            I("mdot", "Const", k=18000.0),
        ],
        "connects": [
            ("ps.P", "fp.Q_fission"),
            ("fp.Qcool", "ch.Q_fuel"),
            ("fp.Tcool", "ch.Tavg"),
            ("tin.y", "ch.Tin"),
            ("mdot.y", "ch.mdot"),
        ],
        "measure": {"Tout": "ch.Tout.u", "Tavg": "ch.Tavg.u", "Tf": "fp.Tf.u",
                    "Tc": "fp.Tc.u", "Qfuel": "fp.Qcool.u", "Tin": "tin.s",
                    "mdot": "mdot.s"},
        "sensors": [],
    }


def _loop(rng=None):
    return {
        "template": "loop", "n": 1,
        "instances": [
            I("ps", "PowerSource", P0=3.0e9),
            I("fp", "FuelPin"),
            I("ch", "Channel"),
            I("sg", "SteamGenerator"),
            I("mdot", "Const", k=18000.0),
        ],
        "connects": [
            ("ps.P", "fp.Q_fission"),
            ("fp.Qcool", "ch.Q_fuel"),
            ("fp.Tcool", "ch.Tavg"),
            ("mdot.y", "ch.mdot"),
            ("mdot.y", "sg.mdot"),
            ("ch.Tout", "sg.Tin"),
            ("sg.Tout", "ch.Tin"),
        ],
        "measure": {"Thot": "ch.Tout.u", "Tcold": "sg.Tout.u", "Tf": "fp.Tf.u",
                    "Tc": "fp.Tc.u", "Qsg": "sg.Qsg.u", "Qfuel": "fp.Qcool.u",
                    "mdot": "mdot.s"},
        "sensors": [],
    }


def _parallel(n):
    """N parallel channels fed by a right-leaning splitter chain and recombined
    by a mirrored mixer chain.  Split fractions are set so the mass split is
    uniform; the mixer chain uses the same splitter mass outputs, so the mass
    and energy balances close exactly."""
    inst = [I("ps", "PowerSource", P0=3.0e9)]
    inst.append(I("tin", "Const", k=565.0))
    inst.append(I("mdot", "Const", k=18000.0))
    for k in range(1, n + 1):
        inst += [I(f"fp{k}", "FuelPin"), I(f"ch{k}", "Channel")]
    # splitter chain
    fracs = [1.0 / (n - k + 1) for k in range(1, n)]      # last splitter is 2-way
    for k in range(1, n):
        inst.append(I(f"split{k}", "Splitter", frac=fracs[k - 1]))
    for k in range(1, n):
        inst.append(I(f"mix{k}", "Mixer"))

    conns = [("tin.y", "split1.T"), ("mdot.y", "split1.m")]
    for k in range(1, n + 1):
        conns += [(f"ps.P", f"fp{k}.Q_fission"),
                  (f"fp{k}.Qcool", f"ch{k}.Q_fuel"),
                  (f"fp{k}.Tcool", f"ch{k}.Tavg")]
    # splitter tree
    for k in range(1, n - 1):
        conns += [(f"split{k}.T2", f"split{k+1}.T"), (f"split{k}.m2", f"split{k+1}.m")]
    # legs 1..n-1 take output 1 of splitter k; leg n takes output 2 of splitter n-1
    for k in range(1, n):
        conns += [(f"split{k}.T1", f"ch{k}.Tin"), (f"split{k}.m1", f"ch{k}.mdot")]
    conns += [(f"split{n-1}.T2", f"ch{n}.Tin"), (f"split{n-1}.m2", f"ch{n}.mdot")]
    # mixer chain: mix{k} combines leg k with the result of the deeper chain
    # mix_{n-1} combines leg n-1 and leg n; mix_k combines leg k and mix_{k+1}
    conns += [(f"ch{n-1}.Tout", f"mix{n-1}.T1"), (f"ch{n}.Tout", f"mix{n-1}.T2"),
              (f"split{n-1}.m1", f"mix{n-1}.m1"), (f"split{n-1}.m2", f"mix{n-1}.m2")]
    for k in range(n - 2, 0, -1):
        conns += [(f"ch{k}.Tout", f"mix{k}.T1"), (f"mix{k+1}.T", f"mix{k}.T2"),
                  (f"split{k}.m1", f"mix{k}.m1"), (f"split{k}.m2", f"mix{k}.m2")]

    measure = {"Tmix": "mix1.T.u", "mmix": "mix1.m.u", "Tin": "tin.s", "mdot": "mdot.s"}
    for k in range(1, n + 1):
        measure[f"T{k}"] = f"ch{k}.Tout.u"
        measure[f"Tf{k}"] = f"fp{k}.Tf.u"
    return {"template": f"parallel{n}", "n": n, "instances": inst,
            "connects": conns, "measure": measure, "sensors": []}


def _series(n):
    """N channels in series (once-through), each with its own fuel pin and
    power source.  Stage k inlet is stage k-1 outlet."""
    inst, conns = [], []
    inst.append(I("mdot", "Const", k=18000.0))
    for k in range(1, n + 1):
        inst += [I(f"ps{k}", "PowerSource", P0=3.0e9 / n),
                 I(f"fp{k}", "FuelPin"), I(f"ch{k}", "Channel")]
    if n >= 1:
        inst.append(I("tin", "Const", k=565.0))
        conns += [("tin.y", "ch1.Tin")]
    for k in range(1, n + 1):
        conns += [(f"ps{k}.P", f"fp{k}.Q_fission"),
                  (f"fp{k}.Qcool", f"ch{k}.Q_fuel"),
                  (f"fp{k}.Tcool", f"ch{k}.Tavg"),
                  ("mdot.y", f"ch{k}.mdot")]
        if k > 1:
            conns.append((f"ch{k-1}.Tout", f"ch{k}.Tin"))
    measure = {"Tin": "tin.s", "mdot": "mdot.s"}
    for k in range(1, n + 1):
        measure[f"T{k}"] = f"ch{k}.Tout.u"
        measure[f"Tf{k}"] = f"fp{k}.Tf.u"
    return {"template": f"series{n}", "n": n, "instances": inst,
            "connects": conns, "measure": measure, "sensors": []}


def _sensor_loop(rng=None):
    t = _loop()
    t["template"] = "sensor_loop"
    t["instances"].append(I("sens", "Sensor"))
    t["connects"] = [("ch.Tout", "sens.y_in"), ("sens.y", "sg.Tin")] + [
        c for c in t["connects"] if c != ("ch.Tout", "sg.Tin")]
    t["measure"]["Thot_s"] = "sens.y.u"
    t["sensors"] = ["sens"]
    return t


def _two_sg(rng=None):
    inst = [
        I("ps", "PowerSource", P0=3.0e9),
        I("fp", "FuelPin"),
        I("ch", "Channel"),
        I("split", "Splitter", frac=0.5),
        I("sg1", "SteamGenerator"), I("sg2", "SteamGenerator"),
        I("mix", "Mixer"),
        I("mdot", "Const", k=18000.0),
    ]
    conns = [
        ("ps.P", "fp.Q_fission"),
        ("fp.Qcool", "ch.Q_fuel"),
        ("fp.Tcool", "ch.Tavg"),
        ("mdot.y", "ch.mdot"),
        ("ch.Tout", "split.T"),
        ("mdot.y", "split.m"),
        ("split.T1", "sg1.Tin"), ("split.m1", "sg1.mdot"),
        ("split.T2", "sg2.Tin"), ("split.m2", "sg2.mdot"),
        ("sg1.Tout", "mix.T1"), ("sg2.Tout", "mix.T2"),
        ("split.m1", "mix.m1"), ("split.m2", "mix.m2"),
        ("mix.T", "ch.Tin"),
    ]
    measure = {"Thot": "ch.Tout.u", "Tcold": "mix.T.u", "Tf": "fp.Tf.u",
               "Qsg1": "sg1.Qsg.u", "Qsg2": "sg2.Qsg.u", "mdot": "mdot.s"}
    return {"template": "two_sg", "n": 2, "instances": inst, "connects": conns,
            "measure": measure, "sensors": []}


def _kinetics(rng=None):
    inst = [
        I("kin", "PointKinetics", P0=3.0e9),
        I("fp", "FuelPin"), I("ch", "Channel"), I("sg", "SteamGenerator"),
        I("mdot", "Const", k=18000.0), I("rho", "Const", k=0.0),
    ]
    conns = [
        ("rho.y", "kin.rho_ext"), ("kin.P", "fp.Q_fission"),
        ("fp.Qcool", "ch.Q_fuel"), ("fp.Tcool", "ch.Tavg"),
        ("fp.Tf", "kin.Tf"), ("ch.Tavg", "kin.Tm"),
        ("mdot.y", "ch.mdot"), ("mdot.y", "sg.mdot"),
        ("ch.Tout", "sg.Tin"), ("sg.Tout", "ch.Tin"),
    ]
    measure = {"P": "kin.P.u", "n": "kin.n.u", "Tf": "fp.Tf.u",
               "Thot": "ch.Tout.u", "Tcold": "sg.Tout.u", "Qsg": "sg.Qsg.u"}
    return {"template": "kinetics", "n": 1, "instances": inst, "connects": conns,
            "measure": measure, "sensors": []}


def _pumped(rng=None):
    inst = [
        I("ps", "PowerSource", P0=3.0e9),
        I("fp", "FuelPin"), I("ch", "Channel"), I("sg", "SteamGenerator"),
        I("pump", "Pump"), I("pipe", "Pipe"),
    ]
    conns = [
        ("ps.P", "fp.Q_fission"), ("fp.Qcool", "ch.Q_fuel"),
        ("fp.Tcool", "ch.Tavg"),
        ("pump.mdot", "ch.mdot"), ("pump.mdot", "sg.mdot"),
        ("ch.Tout", "sg.Tin"), ("sg.Tout", "ch.Tin"),
        ("pipe.dp", "pump.dp"), ("pump.mdot", "pipe.mdot"),
    ]
    measure = {"Thot": "ch.Tout.u", "Tcold": "sg.Tout.u", "Tf": "fp.Tf.u",
               "Qsg": "sg.Qsg.u", "mdot": "pump.mdot.u"}
    return {"template": "pumped", "n": 1, "instances": inst, "connects": conns,
            "measure": measure, "sensors": []}


def _controlled(rng=None):
    inst = [
        I("ctrl", "Controller", Kp=1.0e8, Tset=600.0, Qmax=4.5e9),
        I("fp1", "FuelPin"), I("fp2", "FuelPin"),
        I("ch1", "Channel"), I("ch2", "Channel"),
        I("split", "Splitter", frac=0.5), I("mix", "Mixer"),
        I("tin", "Const", k=565.0), I("mdot", "Const", k=18000.0),
        I("sens", "Sensor"),
    ]
    conns = [
        ("ctrl.Q", "fp1.Q_fission"), ("ctrl.Q", "fp2.Q_fission"),
        ("fp1.Qcool", "ch1.Q_fuel"), ("fp2.Qcool", "ch2.Q_fuel"),
        ("fp1.Tcool", "ch1.Tavg"), ("fp2.Tcool", "ch2.Tavg"),
        ("tin.y", "split.T"), ("mdot.y", "split.m"),
        ("split.m1", "ch1.mdot"), ("split.m2", "ch2.mdot"),
        ("split.T1", "ch1.Tin"), ("split.T2", "ch2.Tin"),
        ("ch1.Tout", "mix.T1"), ("ch2.Tout", "mix.T2"),
        ("split.m1", "mix.m1"), ("split.m2", "mix.m2"),
        ("mix.T", "sens.y_in"), ("sens.y", "ctrl.T"),
    ]
    measure = {"P": "ctrl.Q.u", "Tmix": "mix.T.u", "T1": "ch1.Tout.u",
               "T2": "ch2.Tout.u", "Tf1": "fp1.Tf.u", "Tf2": "fp2.Tf.u",
               "Tin": "tin.s", "mdot": "split.m.u"}
    return {"template": "controlled", "n": 2, "instances": inst, "connects": conns,
            "measure": measure, "sensors": ["sens"]}


def _pumped_controlled(rng=None):
    inst = [
        I("ctrl", "Controller", Kp=1.2e8, Tset=600.0, Qmax=4.5e9),
        I("fp", "FuelPin"), I("ch", "Channel"), I("sg", "SteamGenerator"),
        I("pump", "Pump"), I("pipe", "Pipe"), I("sens", "Sensor"),
    ]
    conns = [
        ("ctrl.Q", "fp.Q_fission"), ("fp.Qcool", "ch.Q_fuel"),
        ("fp.Tcool", "ch.Tavg"),
        ("pump.mdot", "ch.mdot"), ("pump.mdot", "sg.mdot"),
        ("ch.Tout", "sg.Tin"), ("sg.Tout", "ch.Tin"),
        ("pipe.dp", "pump.dp"), ("pump.mdot", "pipe.mdot"),
        ("ch.Tout", "sens.y_in"), ("sens.y", "ctrl.T"),
    ]
    measure = {"P": "ctrl.Q.u", "Thot": "ch.Tout.u", "Tcold": "sg.Tout.u",
               "Tf": "fp.Tf.u", "Qsg": "sg.Qsg.u", "mdot": "pump.mdot.u"}
    return {"template": "pumped_controlled", "n": 1, "instances": inst,
            "connects": conns, "measure": measure, "sensors": ["sens"]}


def _kinetics_pumped(rng=None):
    inst = [
        I("kin", "PointKinetics", P0=3.0e9),
        I("fp", "FuelPin"), I("ch", "Channel"), I("sg", "SteamGenerator"),
        I("pump", "Pump"), I("pipe", "Pipe"), I("rho", "Const", k=0.0),
    ]
    conns = [
        ("rho.y", "kin.rho_ext"), ("kin.P", "fp.Q_fission"),
        ("fp.Qcool", "ch.Q_fuel"), ("fp.Tcool", "ch.Tavg"),
        ("fp.Tf", "kin.Tf"), ("ch.Tavg", "kin.Tm"),
        ("pump.mdot", "ch.mdot"), ("pump.mdot", "sg.mdot"),
        ("ch.Tout", "sg.Tin"), ("sg.Tout", "ch.Tin"),
        ("pipe.dp", "pump.dp"), ("pump.mdot", "pipe.mdot"),
    ]
    measure = {"P": "kin.P.u", "n": "kin.n.u", "Tf": "fp.Tf.u",
               "Thot": "ch.Tout.u", "Tcold": "sg.Tout.u", "Qsg": "sg.Qsg.u",
               "mdot": "pump.mdot.u"}
    return {"template": "kinetics_pumped", "n": 1, "instances": inst,
            "connects": conns, "measure": measure, "sensors": []}


TEMPLATES = {
    "single": _single,
    "loop": _loop,
    "parallel2": lambda: _parallel(2),
    "parallel3": lambda: _parallel(3),
    "parallel4": lambda: _parallel(4),
    "series2": lambda: _series(2),
    "series3": lambda: _series(3),
    "series4": lambda: _series(4),
    "sensor_loop": _sensor_loop,
    "two_sg": _two_sg,
    "kinetics": _kinetics,
    "pumped": _pumped,
    "controlled": _controlled,
    "pumped_controlled": _pumped_controlled,
    "kinetics_pumped": _kinetics_pumped,
}

# Which topologies the agent may ever see during experience accumulation.
EVOLUTION_TEMPLATES = ["single", "loop", "parallel2", "parallel3", "series2",
                       "series3", "sensor_loop"]
# Same component vocabulary, new arrangement -> the fair OOD transfer test.
OOD_NEAR_TEMPLATES = ["parallel4", "series4", "two_sg"]
# Component types never seen during evolution -> capability boundary probe.
OOD_FAR_TEMPLATES = ["pumped", "controlled", "pumped_controlled", "kinetics",
                     "kinetics_pumped"]


def _normalize(topo):
    """Express every connect as (output_port, input_port).

    Modelica accepts either order, and the hand-written templates use both.  A
    single canonical order removes a whole class of downstream bugs (fault
    injection, probe rewiring, use detection)."""
    from ..verifier.phys import COMPONENT_PORTS
    types = {i["name"]: i["type"] for i in topo["instances"]}
    out = []
    for a, b in topo["connects"]:
        ia, pa = a.split(".", 1)
        ib, pb = b.split(".", 1)
        ta, tb = types[ia], types[ib]
        if pa in COMPONENT_PORTS[ta]["in"] and pb in COMPONENT_PORTS[tb]["out"]:
            a, b = b, a
        elif not (pa in COMPONENT_PORTS[ta]["out"] and pb in COMPONENT_PORTS[tb]["in"]):
            raise ValueError(f"illegal connect {a} -> {b} in {topo['template']}")
        out.append((a, b))
    topo["connects"] = out
    return topo


def build(template):
    if template not in TEMPLATES:
        raise KeyError(template)
    return _normalize(copy.deepcopy(TEMPLATES[template]()))


# --------------------------------------------------------------------------- naming
NAME_POOLS = {
    "PowerSource": ["ps", "src", "reactor_power", "pwr", "core_src"],
    "FuelPin": ["fp", "fuel", "pin", "rod"],
    "Channel": ["ch", "coolant", "duct", "chan"],
    "SteamGenerator": ["sg", "boiler", "steamgen", "hx"],
    "Const": ["tin", "mdot", "t_in", "flow_in"],
    "Splitter": ["split", "manifold_in", "tee"],
    "Mixer": ["mix", "manifold_out", "header"],
    "Pump": ["pump", "pmp", "circulator"],
    "Pipe": ["pipe", "line", "riser"],
    "PointKinetics": ["kin", "neutronics", "core_kin"],
    "Controller": ["ctrl", "regulator", "power_ctrl"],
    "Sensor": ["sens", "probe", "tdet"],
}


def rename(topo, rng, scheme="suffix"):
    """Nuisance-factor renaming.  Structure and physics are unchanged; only the
    identifier strings differ.  Used to test whether a retrieved template is
    being matched structurally or by string identity."""
    t = copy.deepcopy(topo)
    mapping = {}
    used = set()
    for inst in t["instances"]:
        typ = inst["type"]
        base = inst["name"]
        if scheme == "suffix":
            new = f"{base}_{rng.randrange(10, 99)}"
        elif scheme == "pool":
            pool = NAME_POOLS.get(typ, [base])
            new = rng.choice(pool)
            while new in used:
                new = new + str(rng.randrange(1, 10))
        else:
            new = base
        used.add(new)
        mapping[base] = new
    for inst in t["instances"]:
        inst["name"] = mapping[inst["name"]]
    t["connects"] = [(mapping[p.split(".")[0]] + "." + p.split(".", 1)[1], q)
                     for p, q in t["connects"]]
    t["connects"] = [(a, mapping[b.split(".")[0]] + "." + b.split(".", 1)[1])
                     for a, b in t["connects"]]
    t["measure"] = {k: (mapping[v.split(".")[0]] + "." + v.split(".", 1)[1])
                    for k, v in t["measure"].items()}
    t["sensors"] = [mapping[s] for s in t.get("sensors", [])]
    t["rename_map"] = mapping
    return t


def fingerprint(topo):
    """Structural identity: template, N, and the canonical type-level edge graph.
    Instance names are deliberately excluded."""
    types = {i["name"]: i["type"] for i in topo["instances"]}
    edges = sorted((types[a.split(".")[0]] + "." + a.split(".", 1)[1],
                    types[b.split(".")[0]] + "." + b.split(".", 1)[1])
                   for a, b in topo["connects"])
    payload = {"template": topo["template"], "n": topo["n"],
               "types": sorted(types.values()), "edges": edges}
    h = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
    return {"template": topo["template"], "n": topo["n"], "hash": h,
            "n_instances": len(topo["instances"]), "n_connects": len(topo["connects"])}


def summary(topo):
    from collections import Counter
    c = Counter(i["type"] for i in topo["instances"])
    return {"template": topo["template"], "n": topo["n"],
            "components": dict(sorted(c.items())), "n_connects": len(topo["connects"])}


if __name__ == "__main__":
    for name in TEMPLATES:
        t = build(name)
        print(f"{name:22s} inst={len(t['instances']):2d} conn={len(t['connects']):2d} "
              f"fp={fingerprint(t)['hash']}")
