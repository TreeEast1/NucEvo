"""Phase 7 tool environment for NucEvolve-Bench tasks (real OpenModelica).

Differences from the Phase 6 environment, driven by the Phase 5/6 audits:

*  Port diagnostics: when the agent names a *parameter* instead of a port
  (`mdot.k`), the error says so and lists the instance's real ports and their
  directions, instead of a bare "unknown port".
*  Batch connects are first-class (`add_connect` takes a list), because large
  plants need 10+ connections and a 12-step budget would make the task
  unsolvable for reasons unrelated to engineering.
*  The step budget is a property of the task, not a global constant.
*  `inspect` returns the component port table, so the model never has to guess
  which ports exist.
*  Verification uses v3 (structural + reference-free physics + multi-point +
  design point), never reference-connect equality.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from nucevo.benchmark.verifier import phys, v3
from .parser import parse_tool_call

# Phase 7.1 optional tool-interface guards.  Default OFF so that every Phase 7
# result stays exactly reproducible; experiment code turns them on explicitly.
import os  # noqa: E402
STRICT_EDIT = os.environ.get("NUCEVO_STRICT_EDIT") == "1"
REPEAT_GUARD = os.environ.get("NUCEVO_REPEAT_GUARD") == "1"

SYSTEM = """You are an expert nuclear thermal-hydraulics engineer working inside an execution environment.
You can inspect and edit a Modelica model and run the OpenModelica compiler/simulator.

Reply with EXACTLY ONE tool call and nothing else, using raw JSON (no markdown fences, no prose):

<tool_call>{"name": "read_file", "args": {"path": "TaskModel.mo"}}</tool_call>

Tools:
- read_file {"path": "TaskModel.mo"}
- inspect {}
  lists every component instance, its ports, and their direction (in/out).
- edit_file {"path": "TaskModel.mo", "old": "<exact existing text>", "new": "<replacement>"}
- write_file {"path": "TaskModel.mo", "content": "<entire new file>"}
- add_connect {"pairs": [["ps.P","fp.Q_fission"], ["fp.Qcool","ch.Q_fuel"]]}
  adds connect(...) statements. Every pair must be output -> input; every input driven once.
- replace_connect {"old": ["fp.Tcool","ch.Tin"], "new": ["fp.Tcool","ch.Tavg"]}
- remove_connect {"pair": ["fp.Tcool","ch.Tin"]}
- set_parameter {"instance": "mdot", "parameter": "k", "value": 19500}
  `k` is a PARAMETER of the Const block; it is not a port. Ports are listed by `inspect`.
- run_simulation {"stop_time": 1200}  compile + simulate; returns measured values.
- check_physics {"stop_time": 1200}   per-component conservation residuals, steadiness.
- compare_to_targets {"stop_time": 1200}  measured value vs design target.
- diagnose {"label": "<fault category>"}  diagnosis tasks only.
- submit {}

Rules:
- Never invent simulation output: call run_simulation to see results.
- A model that compiles and runs can still be physically wrong. Mass and energy
  must balance in every component; the design point must be met.
- Do not modify the library `NuclearEdu` and do not rename components.
- If a tool call fails, read the error and change the call. Submit when the model
  meets every requirement in the task instruction.
- The file already ends with `end TaskModel;`. Never add a second one.
"""

DIAG_LABELS = ["swapped_connections", "wrong_parameter", "wrong_boundary_condition",
               "wrong_routing", "missing_connection"]

def _short(s, n=1600):
    s = s or ""
    return s if len(s) <= n else s[:n] + "\n...[truncated]"


def _mask_comments(text):
    text = text or ""
    text = re.sub(r"/\*.*?\*/", lambda m: " " * len(m.group(0)), text, flags=re.S)
    return re.sub(r"//[^\n]*", lambda m: " " * len(m.group(0)), text)


class TaskEnv:
    def __init__(self, task, max_steps=16, timeout=60.0):
        self.task = task
        self.topo = task["topology"]
        self.files = {"TaskModel.mo": task["task_text_initial"]}
        self.max_steps = max_steps
        self.step_i = 0
        self.tool_log = []
        self.done = False
        self.submitted = False
        self.diagnosis = None
        self.timeout = timeout
        self.n_sim = 0
        self.n_edit = 0
        self.last_verify = None

    # ------------------------------------------------------------------ prompt
    def memory_header(self, memory_text=""):
        if not memory_text:
            return ""
        return f"=== EXPERIENCE MEMORY (retrieved, may be partially applicable) ===\n{memory_text}\n=== END MEMORY ===\n\n"

    def episode_header(self, episode_text=""):
        """Phase 7.2: episode memory of THIS task, injected directly.

        It is placed before the retrieved global memory so that, when the two
        disagree, the instruction is unambiguous and the task-local evidence
        wins.  Default empty => Phase 7 prompts are byte-identical.
        """
        if not episode_text:
            return ""
        return ("=== CURRENT TASK EXPERIENCE (from your own previous attempts on THIS "
                "task; highest priority) ===\n" + episode_text +
                "\n=== END CURRENT TASK EXPERIENCE ===\n\n")

    def initial_user_message(self, memory_text="", episode_text=""):
        return (self.episode_header(episode_text) + self.memory_header(memory_text)
                + self.task["instruction"]
                + f"\n\n=== TaskModel.mo ===\n{self.files['TaskModel.mo']}\n\n"
                "Begin by inspecting the file and running a simulation.")

    def messages(self, history, memory_text="", episode_text=""):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": self.initial_user_message(memory_text,
                                                                      episode_text)}]
        for a, o in history:
            msgs.append({"role": "assistant", "content": a})
            msgs.append({"role": "user", "content": "Observation: " + o})
        return msgs

    # ------------------------------------------------------------------ helpers
    def _instances(self):
        return phys.parse_instances(self.files["TaskModel.mo"])

    def _ports_of(self, inst_name):
        for i in self._instances():
            if i["name"] == inst_name:
                return i["type"], phys.COMPONENT_PORTS.get(i["type"])
        return None, None

    def _port_hint(self, port):
        """Explain why a port name is wrong and what the legal ports are."""
        if "." not in port:
            typ, spec = self._ports_of(port)
            if spec:
                return (f"'{port}' is an instance, not a port. Use '{port}.{spec['out'][0]}' "
                        f"(outputs: {spec['out']}; inputs: {spec['in']}).")
            return f"'{port}' is not a known instance."
        inst, p = port.split(".", 1)
        typ, spec = self._ports_of(inst)
        if spec is None:
            return f"'{inst}' is not a declared instance."
        pr = phys.params_of({"type": typ, "params": {}})
        if p in pr or p in {k for i in self._instances() if i["name"] == inst
                            for k in i["params"]}:
            return (f"'{inst}.{p}' is a PARAMETER, not a connector. "
                    f"Outputs: {spec['out']}; inputs: {spec['in']}. "
                    f"Parameters are set with set_parameter, not connect.")
        return (f"'{inst}' (type {typ}) has no port '{p}'. "
                f"Outputs: {spec['out']}; inputs: {spec['in']}.")

    def _unconnected_inputs(self):
        txt = _mask_comments(self.files["TaskModel.mo"])
        conns = phys.parse_connects(txt)
        driven = set()
        insts = phys.parse_instances(self.files["TaskModel.mo"])
        types = {i["name"]: i["type"] for i in insts}
        for a, b in conns:
            ia, pa = a.split(".", 1) if "." in a else (a, "")
            ib, pb = b.split(".", 1) if "." in b else (b, "")
            if ia in types and pa in phys.COMPONENT_PORTS[types[ia]]["out"]:
                driven.add(b)
            if ib in types and pb in phys.COMPONENT_PORTS[types[ib]]["out"]:
                driven.add(a)
        missing = []
        for i in insts:
            for p in phys.COMPONENT_PORTS.get(i["type"], {"in": []})["in"]:
                if f"{i['name']}.{p}" not in driven:
                    missing.append(f"{i['name']}.{p}")
        return missing

    def _dir(self, port):
        if "." not in port:
            return None, None
        inst, p = port.split(".", 1)
        typ, spec = self._ports_of(inst)
        if spec is None:
            return None, typ
        if p in spec["out"]:
            return "out", typ
        if p in spec["in"]:
            return "in", typ
        return None, typ

    # ------------------------------------------------------------------ simulation
    def _verify(self, stop_time=1200.0):
        return v3.verify(self.task, self.files["TaskModel.mo"], stop=stop_time,
                         do_multipoint=False, do_transient=False, timeout=self.timeout)

    def _obs_run(self, args, physics=False):
        stop = float(args.get("stop_time", self.task.get("stop", 1200.0)))
        stop = max(10.0, min(stop, 2000.0))
        self.n_sim += 1
        v = self._verify(stop)
        self.last_verify = v
        if not v["build_ok"]:
            miss = self._unconnected_inputs()
            return ("BUILD FAILED\n" + "\n".join(v.get("errors") or [])
                    + ("\nUnconnected inputs: " + ", ".join(miss) if miss else
                       "\nAll inputs connected."))
        if not v["sim_ok"]:
            return "BUILD OK, SIMULATION FAILED\n" + _short(v["messages"])
        if physics:
            ph = v.get("physics") or {}
            lines = [f"steady={v['steady']} (max drift {v.get('steady_range', 0):.4g})"]
            for k, ok in (ph.get("checks") or {}).items():
                if not ok:
                    lines.append(f"  FAIL {k}")
            if all((ph.get("checks") or {}).values()):
                lines.append("  all component conservation checks PASS")
            for k, val in (ph.get("values") or {}).items():
                if isinstance(val, float):
                    lines.append(f"  {k} = {val:.4g}")
            return "PHYSICS\n" + "\n".join(lines)
        lines = ["SIMULATION OK"]
        for k, val in v["values"].items():
            lines.append(f"  {k} = {val:.6g}")
        return "\n".join(lines)

    def _obs_compare(self, args):
        stop = float(args.get("stop_time", self.task.get("stop", 1200.0)))
        self.n_sim += 1
        v = self._verify(stop)
        self.last_verify = v
        if not v["sim_ok"]:
            return self._obs_run(args)
        lines = ["TARGET COMPARISON (value / target / rel.err)"]
        for k, d in (v.get("goal_detail") or {}).items():
            tgt = d["target"]
            rel = abs(d["value"] - tgt) / max(abs(tgt), 1e-9)
            lines.append(f"  {k}: {d['value']:.6g} / {tgt:.6g} / {rel*100:.2f}% "
                         f"{'OK' if d['ok'] else 'OFF'}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ editing
    def _structural_veto(self, text):
        """Phase 7.1: refuse edits that break structure the agent cannot see.

        `add_connect` already rejects duplicates and wrong directions; the
        free-text editors did not, so a model could introduce a duplicate and
        only discover it at the final evaluation.  Returns None when the text is
        acceptable, else an error string.
        """
        if not STRICT_EDIT:
            return None
        _ok, d = v3.structural_check(text, self.topo)
        bad = d.get("errors") or []
        if bad:
            return ("ERROR: the edit would introduce a structural fault that the "
                    "verifier rejects: " + "; ".join(bad[:3])
                    + ". Use add_connect/replace_connect, which refuse invalid "
                    "connections, or remove the duplicate line.")
        return None

    def _obs_edit(self, args):
        path = args.get("path", "TaskModel.mo")
        old, new = args.get("old", ""), args.get("new", "")
        if path != "TaskModel.mo":
            return "ERROR: only TaskModel.mo is editable."
        txt = self.files["TaskModel.mo"]
        n = txt.count(old) if old else -1
        if n != 1:
            return f"ERROR: old string occurs {n} times; it must occur exactly once."
        if STRICT_EDIT and old and old == new:
            return "ERROR: the replacement is identical to the old text; nothing would change."
        new_txt = txt.replace(old, new, 1)
        if new_txt.count("end TaskModel;") != 1 or new_txt.count("model TaskModel") != 1:
            return ("ERROR: the edit would leave a malformed file (duplicate or missing "
                    "`model TaskModel` / `end TaskModel;`).")
        veto = self._structural_veto(new_txt)
        if veto:
            return veto
        self.files["TaskModel.mo"] = new_txt
        self.n_edit += 1
        return "OK: edit applied."

    def _obs_write(self, args):
        content = args.get("content", "")
        if content.count("model TaskModel") != 1 or content.count("end TaskModel;") != 1:
            return "ERROR: content must contain exactly one `model TaskModel ... end TaskModel;`."
        if "package NuclearEdu" in content or "model NuclearEdu" in content:
            return "ERROR: do not redefine the library."
        veto = self._structural_veto(content)
        if veto:
            return veto
        self.files["TaskModel.mo"] = content
        self.n_edit += 1
        return "OK: file replaced."

    def _obs_inspect(self, args):
        insts = self._instances()
        if not insts:
            return "No NuclearEdu instances declared."
        out = ["instance | type | inputs | outputs"]
        for i in insts:
            spec = phys.COMPONENT_PORTS.get(i["type"])
            if not spec:
                out.append(f"{i['name']} | {i['type']} | UNKNOWN TYPE")
                continue
            out.append(f"{i['name']} | {i['type']} | {', '.join(spec['in']) or '-'} | "
                       f"{', '.join(spec['out'])}")
        miss = self._unconnected_inputs()
        out.append("")
        out.append("Unconnected inputs: " + (", ".join(miss) if miss else "none"))
        return "\n".join(out)

    def _obs_add_connect(self, args):
        pairs = args.get("pairs") or []
        if isinstance(pairs, (list, tuple)) and pairs and not isinstance(pairs[0], (list, tuple)):
            pairs = [pairs]
        if not isinstance(pairs, list) or not pairs:
            return "ERROR: pairs must be a non-empty list of [output_port, input_port]."
        txt = self.files["TaskModel.mo"]
        existing = {tuple(sorted(c)) for c in phys.parse_connects(txt)}
        lines, errs = [], []
        for pr in pairs:
            if not (isinstance(pr, (list, tuple)) and len(pr) == 2):
                errs.append(f"bad pair {pr}")
                continue
            a, b = str(pr[0]).strip(), str(pr[1]).strip()
            da, ta = self._dir(a)
            db, tb = self._dir(b)
            if da is None or db is None:
                bad = a if da is None else b
                errs.append(f"{bad}: {self._port_hint(bad)}")
                continue
            if da == db:
                errs.append(f"{a} -> {b}: both are {'inputs' if da == 'in' else 'outputs'}; "
                            "a connection must go output -> input.")
                continue
            if da == "in":
                a, b = b, a
            if tuple(sorted((a, b))) in existing:
                errs.append(f"duplicate: {a} -> {b} is already connected.")
                continue
            existing.add(tuple(sorted((a, b))))
            lines.append(f"  connect({a}, {b});")
        if not lines:
            return "ERROR: no valid connections added. " + " | ".join(errs)
        if "end TaskModel;" not in txt:
            return "ERROR: malformed file (no end TaskModel;)."
        txt = txt.replace("end TaskModel;", "\n".join(lines) + "\nend TaskModel;", 1)
        self.files["TaskModel.mo"] = txt
        self.n_edit += 1
        miss = self._unconnected_inputs()
        return (f"OK: added {len(lines)} connection(s)."
                + (" Rejected: " + " | ".join(errs) if errs else "")
                + ("\nUnconnected inputs: " + ", ".join(miss) if miss else
                   "\nAll inputs connected."))

    def _obs_replace_connect(self, args):
        old, new = args.get("old"), args.get("new")
        if not (isinstance(old, (list, tuple)) and len(old) == 2 and
                isinstance(new, (list, tuple)) and len(new) == 2):
            return "ERROR: old and new must each be [portA, portB]."
        if tuple(sorted(map(str, old))) == tuple(sorted(map(str, new))):
            return "ERROR: old and new are identical; nothing to change."
        for p in list(old) + list(new):
            if self._dir(str(p))[0] is None:
                return f"ERROR: {p}: {self._port_hint(str(p))}"
        txt = self.files["TaskModel.mo"]
        pat = re.compile(r"connect\(\s*" + re.escape(str(old[0])) + r"\s*,\s*" +
                         re.escape(str(old[1])) + r"\s*\)")
        pat2 = re.compile(r"connect\(\s*" + re.escape(str(old[1])) + r"\s*,\s*" +
                          re.escape(str(old[0])) + r"\s*\)")
        mask = _mask_comments(txt)
        matches = list(pat.finditer(mask)) + list(pat2.finditer(mask))
        if len(matches) != 1:
            return (f"ERROR: connect{tuple(old)} not found exactly once. "
                    f"Unconnected inputs: {self._unconnected_inputs()}")
        m = matches[0]
        self.files["TaskModel.mo"] = (txt[:m.start()] +
                                      f"connect({new[0]}, {new[1]})" + txt[m.end():])
        self.n_edit += 1
        miss = self._unconnected_inputs()
        return (f"OK: replaced connect{tuple(old)} with connect{tuple(new)}."
                + ("\nUnconnected inputs: " + ", ".join(miss) if miss else
                   "\nAll inputs connected."))

    def _obs_remove_connect(self, args):
        pr = args.get("pair")
        if not (isinstance(pr, (list, tuple)) and len(pr) == 2):
            return "ERROR: pair must be [portA, portB]."
        txt = self.files["TaskModel.mo"]
        pat = re.compile(r"\s*connect\(\s*" + re.escape(str(pr[0])) + r"\s*,\s*" +
                         re.escape(str(pr[1])) + r"\s*\)\s*;?")
        pat2 = re.compile(r"\s*connect\(\s*" + re.escape(str(pr[1])) + r"\s*,\s*" +
                          re.escape(str(pr[0])) + r"\s*\)\s*;?")
        mask = _mask_comments(txt)
        matches = list(pat.finditer(mask)) or list(pat2.finditer(mask))
        if not matches:
            return f"ERROR: connect{tuple(pr)} not found."
        m = matches[0]
        self.files["TaskModel.mo"] = txt[:m.start()] + txt[m.end():]
        self.n_edit += 1
        miss = self._unconnected_inputs()
        return (f"OK: removed connect{tuple(pr)}."
                + ("\nUnconnected inputs: " + ", ".join(miss) if miss else
                   "\nAll inputs connected."))

    def _obs_set_parameter(self, args):
        inst = str(args.get("instance", ""))
        par = str(args.get("parameter", ""))
        val = args.get("value")
        if not inst or not par or val is None:
            return "ERROR: instance, parameter and value are required."
        try:
            val_num = float(val)
        except (TypeError, ValueError):
            return (f"ERROR: value must be a numeric literal, got {val!r}. "
                    "Use the design-point number, not an expression.")
        txt = self.files["TaskModel.mo"]
        m = re.search(rf"(NuclearEdu\.\w+\s+{re.escape(inst)}\s*\()([^)]*)(\))", txt)
        if not m:
            if self._ports_of(inst)[0] is not None:
                return f"ERROR: {self._port_hint(inst + '.' + par)}"
            return f"ERROR: instance {inst} not found."
        body = m.group(2)
        pm = re.search(rf"\b{re.escape(par)}\s*=\s*([-+0-9.eE]+)", body)
        if pm:
            new_body = body[:pm.start(1)] + f"{val_num:.8g}" + body[pm.end(1):]
        else:
            new_body = (body + (", " if body.strip() else "") + f"{par} = {val_num:.8g}")
        self.files["TaskModel.mo"] = txt[:m.start(2)] + new_body + txt[m.end(2):]
        self.n_edit += 1
        return f"OK: set {inst}.{par} = {val_num:.8g}."

    # ------------------------------------------------------------------ step
    def step(self, action_text):
        name, args = parse_tool_call(action_text)
        self.step_i += 1
        if name is None:
            obs = ("ERROR: no valid tool call found. Reply with exactly one "
                   '<tool_call>{"name": "...", "args": {...}}</tool_call>.')
        elif name == "read_file":
            obs = self.files["TaskModel.mo"]
        elif name == "inspect":
            obs = self._obs_inspect(args)
        elif name == "edit_file":
            obs = self._obs_edit(args)
        elif name == "write_file":
            obs = self._obs_write(args)
        elif name == "add_connect":
            obs = self._obs_add_connect(args)
        elif name == "replace_connect":
            obs = self._obs_replace_connect(args)
        elif name == "set_parameter":
            obs = self._obs_set_parameter(args)
        elif name == "remove_connect":
            obs = self._obs_remove_connect(args)
        elif name == "run_simulation":
            obs = self._obs_run(args)
        elif name == "check_physics":
            obs = self._obs_run(args, physics=True)
        elif name == "compare_to_targets":
            obs = self._obs_compare(args)
        elif name == "diagnose":
            lab = str(args.get("label", "")).strip()
            if lab not in DIAG_LABELS:
                obs = f"ERROR: label must be one of {DIAG_LABELS}."
            else:
                self.diagnosis = lab
                obs = f"OK: recorded diagnosis '{lab}'."
        elif name == "submit":
            self.submitted = True
            self.done = True
            obs = "Submitted."
        else:
            obs = f"ERROR: unknown tool '{name}'. Available: " + ", ".join(
                ["read_file", "inspect", "edit_file", "write_file", "add_connect",
                 "replace_connect", "remove_connect", "set_parameter", "run_simulation",
                 "check_physics", "compare_to_targets", "diagnose", "submit"])
        if REPEAT_GUARD and obs and "ERROR" in obs and name is not None:
            key = (name, json.dumps(args, sort_keys=True, default=str))
            prev = getattr(self, "_failed_calls", None)
            if prev is None:
                prev = self._failed_calls = {}
            if key in prev:
                obs = (f"REPEATED FAILED CALL. You already issued exactly this call at "
                       f"step {prev[key]} and it failed with: {obs} "
                       f"Do NOT repeat it. Choose a different action: read the model, "
                       f"inspect the port table, or use a different tool.")
            else:
                prev[key] = self.step_i
        self.tool_log.append({"step": self.step_i, "name": name,
                              "args": args if isinstance(args, dict) else {},
                              "obs": (obs or "")[:600]})
        if self.step_i >= self.max_steps and not self.done:
            self.done = True
            obs = (obs or "") + "\n[step limit reached; episode ended]"
        return obs, self.done, {}

    # ------------------------------------------------------------------ scoring
    def evaluate(self):
        v = v3.verify(self.task, self.files["TaskModel.mo"], do_multipoint=True,
                      do_transient=True, timeout=max(self.timeout, 90.0))
        v["steps"] = self.step_i
        v["n_sim"] = self.n_sim
        v["n_edit"] = self.n_edit
        v["submitted"] = self.submitted
        v["diagnosis"] = self.diagnosis
        if self.task["family"] == "diagnose":
            truth = sorted({f["label"] for f in self.task.get("faults", [])})
            v["diagnosis_true"] = truth
            v["diagnosis_ok"] = bool(self.diagnosis is not None and
                                     (self.diagnosis in truth or len(truth) > 1))
        else:
            v["diagnosis_ok"] = True
        v["success"] = bool(v["success"] and v["diagnosis_ok"])
        v["final_text"] = self.files["TaskModel.mo"]
        return v
