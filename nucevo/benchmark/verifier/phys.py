"""Topology-independent physics verification.

Every block in NuclearEdu is causal, so at steady state each component must
satisfy its own conservation law *using the values on its connected ports*.
That makes a reference-free residual checker possible: we need not know the
intended wiring, only the plant's own simulated signals.

Two consequences drive the implementation:

1.  Parameters must be read from the **submitted model text**, not from the
    task spec -- in repair/tune tasks the agent is expected to change them.
2.  OpenModelica eliminates aliases, so a constant-driven input (e.g.
    `fp.Q_fission` fed by `ps.P`) often has no column in the result CSV.  We
    therefore resolve each port through the submitted connect graph and fall
    back to the block's own algebraic law.

"Physically self-consistent" and "meets the design point" are separate criteria
and are reported separately.
"""
from __future__ import annotations

import math
import re

CP = 5200.0

COMPONENT_PORTS = {
    "Const": {"in": [], "out": ["y"]},
    "PowerSource": {"in": [], "out": ["P"]},
    "FuelPin": {"in": ["Q_fission", "Tcool"], "out": ["Tf", "Tc", "Qcool"]},
    "Channel": {"in": ["Tin", "mdot", "Q_fuel"], "out": ["Tout", "Tavg"]},
    "SteamGenerator": {"in": ["Tin", "mdot"], "out": ["Tout", "Qsg"]},
    "Pump": {"in": ["dp"], "out": ["mdot"]},
    "Pipe": {"in": ["mdot"], "out": ["dp"]},
    "PointKinetics": {"in": ["rho_ext", "Tf", "Tm"], "out": ["P", "n"]},
    "Mixer": {"in": ["T1", "m1", "T2", "m2"], "out": ["T", "m"]},
    "Splitter": {"in": ["T", "m"], "out": ["T1", "m1", "T2", "m2"]},
    "Controller": {"in": ["T"], "out": ["Q"]},
    "Sensor": {"in": ["y_in"], "out": ["y"]},
}

KIND = {
    ("Const", "y"): "generic", ("PowerSource", "P"): "power",
    ("FuelPin", "Q_fission"): "power", ("FuelPin", "Tcool"): "temperature",
    ("FuelPin", "Tf"): "temperature", ("FuelPin", "Tc"): "temperature",
    ("FuelPin", "Qcool"): "power",
    ("Channel", "Tin"): "temperature", ("Channel", "mdot"): "massflow",
    ("Channel", "Q_fuel"): "power", ("Channel", "Tout"): "temperature",
    ("Channel", "Tavg"): "temperature",
    ("SteamGenerator", "Tin"): "temperature", ("SteamGenerator", "mdot"): "massflow",
    ("SteamGenerator", "Tout"): "temperature", ("SteamGenerator", "Qsg"): "power",
    ("Pump", "dp"): "pressure", ("Pump", "mdot"): "massflow",
    ("Pipe", "mdot"): "massflow", ("Pipe", "dp"): "pressure",
    ("PointKinetics", "rho_ext"): "reactivity",
    ("PointKinetics", "Tf"): "temperature",
    ("PointKinetics", "Tm"): "temperature", ("PointKinetics", "P"): "power",
    ("PointKinetics", "n"): "generic",
    ("Mixer", "T1"): "temperature", ("Mixer", "m1"): "massflow",
    ("Mixer", "T2"): "temperature", ("Mixer", "m2"): "massflow",
    ("Mixer", "T"): "temperature", ("Mixer", "m"): "massflow",
    ("Splitter", "T"): "temperature", ("Splitter", "m"): "massflow",
    ("Splitter", "T1"): "temperature", ("Splitter", "m1"): "massflow",
    ("Splitter", "T2"): "temperature", ("Splitter", "m2"): "massflow",
    ("Controller", "T"): "temperature", ("Controller", "Q"): "power",
    ("Sensor", "y_in"): "generic", ("Sensor", "y"): "generic",
}

# Library defaults (NuclearEdu.mo is frozen; sha1 pinned in MANIFEST).
DEFAULTS = {
    "Const": {"k": 1.0},
    "PowerSource": {"P0": 3.0e9},
    "FuelPin": {"Cf": 1.0e8, "Cc": 4.0e7, "G_gap": 1.0e7, "G_cl": 1.5e8,
                "Tf0": 901.0, "Tc0": 601.0},
    "Channel": {"Cp": 5200.0, "Cth": 4.0e7, "Tin0": 565.0, "Tout0": 597.0},
    "SteamGenerator": {"Cp": 5200.0, "UA": 3.33e7, "Tsec": 490.0,
                       "Cth": 4.0e7, "Tin0": 565.0},
    "Pump": {"Head0": 3.2e6, "Mdot0": 25000.0},
    "Pipe": {"K": 4.756e-3},
    "PointKinetics": {"Beta": 0.0065, "L": 1.0e-5, "Lambda": 0.0784,
                      "Alpha_f": -2.5e-5, "Alpha_m": -5.0e-4,
                      "Tf0": 901.0, "Tm0": 581.0, "P0": 3.0e9},
    "Mixer": {"Cp": 5200.0},
    "Splitter": {"frac": 0.5},
    "Controller": {"Kp": 1.0e8, "Tset": 600.0, "Qmax": 4.5e9},
    "Sensor": {},
}


# --------------------------------------------------------------------------- parsing
def mask_comments(text):
    text = text or ""
    text = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)


INST_RE = re.compile(r"NuclearEdu\.(\w+)\s+(\w+)\s*(?:\(([^)]*)\))?\s*;")


def parse_instances(text):
    """[{name, type, params}] in declaration order, from the submitted text."""
    out = []
    for m in INST_RE.finditer(mask_comments(text)):
        typ, name, body = m.group(1), m.group(2), m.group(3) or ""
        params = {}
        for pm in re.finditer(r"(\w+)\s*=\s*([-+0-9.eE]+)", body):
            try:
                params[pm.group(1)] = float(pm.group(2))
            except ValueError:
                pass
        out.append({"name": name, "type": typ, "params": params})
    return out


def parse_connects(text):
    return [(a, b) for a, b in
            re.findall(r"connect\(\s*([\w.]+)\s*,\s*([\w.]+)\s*\)", mask_comments(text))]


def params_of(inst):
    p = dict(DEFAULTS.get(inst["type"], {}))
    p.update(inst.get("params") or {})
    return p


# --------------------------------------------------------------------------- resolution
class Resolver:
    """Resolve the steady value of any port, following aliases through the
    submitted connect graph when the simulator did not emit the column."""

    def __init__(self, instances, connects, ss):
        self.inst = {i["name"]: i for i in instances}
        self.driver = {b: a for a, b in connects}      # input -> source output
        self.ss = ss
        self.cache = {}

    def direct(self, name, port):
        for key in (f"{name}.{port}.u", f"{name}.{port}", f"{name}.s"):
            if key in self.ss:
                v = self.ss[key]
                if isinstance(v, (int, float)) and math.isfinite(v):
                    return v
        return None

    def value(self, name, port, depth=0):
        if depth > 10:
            return None
        key = (name, port)
        if key in self.cache:
            return self.cache[key]
        v = self.direct(name, port)
        if v is not None:
            self.cache[key] = v
            return v
        # follow the wire backwards
        src = self.driver.get(f"{name}.{port}")
        if src:
            si, sp = src.split(".", 1)
            v = self.value(si, sp, depth + 1)
            if v is not None:
                self.cache[key] = v
                return v
        # fall back to the block's own algebraic law
        v = self.law(name, port, depth)
        self.cache[key] = v
        return v

    def inp(self, name, port, depth=0):
        return self.value(name, port, depth + 1)

    def law(self, name, port, depth=0):
        inst = self.inst.get(name)
        if inst is None:
            return None
        t, pr = inst["type"], params_of(inst)
        if t == "Const" and port == "y":
            return pr["k"]
        if t == "PowerSource" and port == "P":
            return pr["P0"]
        if t == "Controller" and port == "Q":
            temp = self.inp(name, "T", depth)
            if temp is None:
                return None
            return min(pr["Qmax"], max(0.0, pr["Kp"] * (pr["Tset"] - temp)))
        if t == "Sensor" and port == "y":
            return self.inp(name, "y_in", depth)
        if t == "Splitter":
            tv, mv = self.inp(name, "T", depth), self.inp(name, "m", depth)
            f = pr["frac"]
            if port == "T1" or port == "T2":
                return tv
            if port == "m1":
                return None if mv is None else f * mv
            if port == "m2":
                return None if mv is None else (1 - f) * mv
        if t == "Mixer":
            t1, m1 = self.inp(name, "T1", depth), self.inp(name, "m1", depth)
            t2, m2 = self.inp(name, "T2", depth), self.inp(name, "m2", depth)
            if None in (t1, m1, t2, m2):
                return None
            if port == "m":
                return m1 + m2
            if port == "T":
                return (m1 * t1 + m2 * t2) / max(m1 + m2, 1e-6)
        return None


def _rel(x, scale, floor=1e-9):
    return abs(x) / max(abs(scale), floor)


def _finite(*xs):
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in xs)


class Residuals:
    def __init__(self, instances, connects, ss):
        self.instances = instances
        self.r = Resolver(instances, connects, ss)
        self.checks = {}
        self.values = {}
        self.failures = []

    def g(self, name, port):
        return self.r.value(name, port)

    def add(self, name, ok, detail=None):
        self.checks[name] = bool(ok)
        if detail is not None:
            self.values[name] = detail
        if not ok:
            self.failures.append(name)

    def check(self):
        for inst in self.instances:
            fn = getattr(self, f"_c_{inst['type']}", None)
            if fn is None:
                self.add(f"{inst['name']}:known_type", False)
            else:
                fn(inst["name"], inst)
        self._global()

    # ------------------------------------------------------------------ blocks
    def _c_Const(self, nm, inst):
        k = params_of(inst)["k"]
        y = self.g(nm, "y")
        self.add(f"{nm}:const", y is not None and _rel(y - k, k, 1.0) < 1e-6)

    def _c_PowerSource(self, nm, inst):
        p0 = params_of(inst)["P0"]
        p = self.g(nm, "P")
        self.add(f"{nm}:powersource", p is not None and _rel(p - p0, p0, 1.0) < 1e-6)

    def _c_FuelPin(self, nm, inst):
        pr = params_of(inst)
        qf, qc = self.g(nm, "Q_fission"), self.g(nm, "Qcool")
        tf, tc, tcl = self.g(nm, "Tf"), self.g(nm, "Tc"), self.g(nm, "Tcool")
        if not _finite(qf, qc, tf, tc, tcl):
            self.add(f"{nm}:fuelpin_signals", False)
            return
        self.add(f"{nm}:energy", _rel(qc - qf, qf, 1e5) < 0.02, _rel(qc - qf, qf, 1e5))
        self.add(f"{nm}:clad_law", _rel(qc - pr["G_cl"] * (tc - tcl), qc, 1e5) < 0.05)
        self.add(f"{nm}:gap_law", _rel(qf - pr["G_gap"] * (tf - tc), qf, 1e5) < 0.05)
        self.add(f"{nm}:ordering", tf > tc > tcl)

    def _c_Channel(self, nm, inst):
        pr = params_of(inst)
        tin, tout = self.g(nm, "Tin"), self.g(nm, "Tout")
        tavg, m, q = self.g(nm, "Tavg"), self.g(nm, "mdot"), self.g(nm, "Q_fuel")
        if not _finite(tin, tout, tavg, m, q):
            self.add(f"{nm}:channel_signals", False)
            return
        res = q + m * pr["Cp"] * (tin - tout)
        scale = max(abs(q), abs(m * pr["Cp"] * (tout - tin)), 1e3)
        self.add(f"{nm}:energy", abs(res) / scale < 0.02, abs(res) / scale)
        self.add(f"{nm}:tavg", abs(tavg - 0.5 * (tin + tout)) < 0.02)
        self.add(f"{nm}:heating", tout > tin - 0.5)

    def _c_SteamGenerator(self, nm, inst):
        pr = params_of(inst)
        tin, tout = self.g(nm, "Tin"), self.g(nm, "Tout")
        m, q = self.g(nm, "mdot"), self.g(nm, "Qsg")
        if not _finite(tin, tout, m, q):
            self.add(f"{nm}:sg_signals", False)
            return
        eps = 1 - math.exp(-pr["UA"] / (pr["Cp"] * max(m, 1.0)))
        q_pred = eps * pr["Cp"] * m * (tin - pr["Tsec"])
        self.add(f"{nm}:ntu_law", _rel(q - q_pred, q, 1e5) < 0.02, _rel(q - q_pred, q, 1e5))
        self.add(f"{nm}:energy",
                 _rel(m * pr["Cp"] * (tin - tout) - q, q, 1e5) < 0.05)
        self.add(f"{nm}:cooling", tout < tin + 0.5)

    def _c_Pump(self, nm, inst):
        pr = params_of(inst)
        dp, m = self.g(nm, "dp"), self.g(nm, "mdot")
        if not _finite(dp, m):
            self.add(f"{nm}:pump_signals", False)
            return
        pred = pr["Mdot0"] * math.sqrt(max(0.0, 1.0 - dp / pr["Head0"]))
        self.add(f"{nm}:pump_curve", _rel(m - pred, m, 1.0) < 1e-3)

    def _c_Pipe(self, nm, inst):
        pr = params_of(inst)
        dp, m = self.g(nm, "dp"), self.g(nm, "mdot")
        if not _finite(dp, m):
            self.add(f"{nm}:pipe_signals", False)
            return
        self.add(f"{nm}:pipe_law", _rel(dp - pr["K"] * m * abs(m), dp, 1.0) < 1e-3)

    def _c_PointKinetics(self, nm, inst):
        pr = params_of(inst)
        p, n = self.g(nm, "P"), self.g(nm, "n")
        if not _finite(p, n):
            self.add(f"{nm}:kinetics_signals", False)
            return
        self.add(f"{nm}:p_n", _rel(p - pr["P0"] * n, p, 1.0) < 1e-6)
        self.add(f"{nm}:positive_power", p > 0.0)

    def _c_Mixer(self, nm, inst):
        t1, m1 = self.g(nm, "T1"), self.g(nm, "m1")
        t2, m2 = self.g(nm, "T2"), self.g(nm, "m2")
        t, m = self.g(nm, "T"), self.g(nm, "m")
        if not _finite(t1, m1, t2, m2, t, m):
            self.add(f"{nm}:mixer_signals", False)
            return
        self.add(f"{nm}:mass", _rel(m - (m1 + m2), m, 1.0) < 1e-6)
        self.add(f"{nm}:enthalpy",
                 _rel(t - (m1 * t1 + m2 * t2) / max(m, 1e-6), t, 1.0) < 1e-6)

    def _c_Splitter(self, nm, inst):
        pr = params_of(inst)
        t, m = self.g(nm, "T"), self.g(nm, "m")
        t1, m1 = self.g(nm, "T1"), self.g(nm, "m1")
        t2, m2 = self.g(nm, "T2"), self.g(nm, "m2")
        if not _finite(t, m, t1, m1, t2, m2):
            self.add(f"{nm}:splitter_signals", False)
            return
        f = pr["frac"]
        self.add(f"{nm}:mass1", _rel(m1 - f * m, m, 1.0) < 1e-6)
        self.add(f"{nm}:mass2", _rel(m2 - (1 - f) * m, m, 1.0) < 1e-6)
        self.add(f"{nm}:t1", abs(t1 - t) < 1e-6)
        self.add(f"{nm}:t2", abs(t2 - t) < 1e-6)

    def _c_Controller(self, nm, inst):
        pr = params_of(inst)
        t, q = self.g(nm, "T"), self.g(nm, "Q")
        if not _finite(t, q):
            self.add(f"{nm}:controller_signals", False)
            return
        pred = min(pr["Qmax"], max(0.0, pr["Kp"] * (pr["Tset"] - t)))
        self.add(f"{nm}:control_law", _rel(q - pred, pr["Qmax"], 1.0) < 1e-5)

    def _c_Sensor(self, nm, inst):
        yi, yo = self.g(nm, "y_in"), self.g(nm, "y")
        self.add(f"{nm}:passthrough", _finite(yi, yo) and abs(yo - yi) < 1e-6)

    # ------------------------------------------------------------------ global
    def _global(self):
        temps = []
        for inst in self.instances:
            for p in COMPONENT_PORTS.get(inst["type"], {"out": []})["out"]:
                if KIND.get((inst["type"], p)) == "temperature":
                    v = self.g(inst["name"], p)
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        temps.append(v)
        self.add("bounds:temperature", bool(temps) and all(250.0 < v < 1800.0 for v in temps),
                 min(temps) if temps else None)
        self.add("bounds:no_nan", bool(temps))


def check(text, ss):
    instances = parse_instances(text)
    connects = parse_connects(text)
    r = Residuals(instances, connects, ss)
    r.check()
    return all(r.checks.values()), {"checks": r.checks, "values": r.values,
                                    "failures": r.failures}
