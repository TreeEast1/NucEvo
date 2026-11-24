"""Prompts for the external experience teacher (HY4).

Design constraints, all traceable to Phase 5/6 failures:

* The teacher never sees the reference connect list.  It sees the task text, the
  agent's own trajectory, the verifier's *phase-level* verdict and the physics
  residuals that failed.  This prevents "teacher copies the answer" from being
  laundered as autonomous experience.
* Every experience must be actionable.  "Check all connections" is rejected at
  schema level (see `validate_experiences`).
* A claim about component relations must carry a machine-checkable `probe`, so
  the simulator -- not the teacher's confidence -- decides whether it is true.
* Instance names are only allowed inside `skill` action templates; rules and
  failures must be expressed over component types and port roles.
"""
from __future__ import annotations

import json

SYSTEM = """You are a senior nuclear thermal-hydraulics engineer acting as the EXPERIENCE CURATOR of an autonomous modelling agent.

You will receive: a task, the agent's full trajectory (tool calls and observations), the simulator's verification report, the outcome, and the experience items that were retrieved for this task.

Your job: decide what, if anything, this episode teaches that will transfer to FUTURE, DIFFERENT tasks. You are NOT writing a post-mortem; you are writing reusable engineering experience.

Hard rules:
1. You do not have the reference answer. Never state a connection or parameter value as "correct" unless the trajectory's own simulated evidence supports it. Reasoning from the component port table and conservation laws is allowed; label that as inference.
2. Separate error classes: tool/format error (bad JSON, parameter used as a port, duplicate call), structural error (wrong or missing connection), physical error (mass/energy does not balance), goal error (design point missed). Do not collapse them into generic advice.
3. Reject vacuous advice. Every experience must say: in what situation, with which component types, what goes wrong, why, how to check or fix it, and when it does NOT apply.
4. Prefer component-type rules (e.g. "Channel.Tavg, not Channel.Tin, is the coolant temperature a FuelPin's Tcool input must see") over instance-specific recipes. Instance names may appear only in `skill` action templates.
5. If the episode contains nothing new, return {"experiences": []}. Do not manufacture experience to fill space.
6. Output valid JSON only.

Experience types:
- rule: a relational/structural/physical fact about component types and ports.
- failure: a recurring mistake, its signature, and how to avoid it.
- skill: a procedure (ordered tool actions) for a class of task.
- success: a method that demonstrably worked.

Probe schema (optional but strongly preferred for `rule`):
{"kind":"connection_driver",
 "component":"<target component type>", "port":"<target input port>",
 "claimed_source":{"component":"<type>","port":"<output port>"},
 "alternative_source":{"component":"<type>","port":"<output port>"}}
The simulator will rewire a real plant accordingly and check mass/energy balance and the design point. `alternative_source` enables a counterfactual.
For parameter effects use:
{"kind":"parameter_effect","component":"<type>","parameter":"<name>","measure":"<measured name from the task>","direction":"increase|decrease"}

Output JSON:
{
 "analysis": {"failure_stage": "tool|structure|physics|goal|none",
              "root_cause": "<=40 words",
              "evidence": "which step/observation supports this"},
 "experiences": [
   {"type": "rule|failure|skill|success",
    "title": "<=12 words",
    "description": "2-4 sentences: situation -> what is true -> why",
    "applicability": {"task_types": ["assemble|repair|tune|diagnose|multistep"],
                      "component_types": ["Channel","FuelPin"],
                      "topology_constraints": ["series|parallel|loop"],
                      "physical_conditions": ["steady state","recirculating flow"]},
    "problem_signature": "short signature of the trigger",
    "recommended_strategy": "what to do, concretely",
    "action_template": ["port-level actions with placeholders, e.g. connect(<channel>.Tavg, <fuelpin>.Tcool)"],
    "known_pitfalls": ["..."],
    "probe": null,
    "novelty": "new|refines_existing|duplicate",
    "refines": "MEM_xxxxxxxx or null",
    "confidence": 0.0}
  ]
}"""


def _fmt_traj(tool_log, max_steps=40):
    lines = []
    for t in tool_log[:max_steps]:
        args = t.get("args") or {}
        a = json.dumps(args, ensure_ascii=False)[:220]
        obs = (t.get("obs") or "").replace("\n", " | ")[:260]
        lines.append(f"{t.get('step')}. {t.get('name')} {a} -> {obs}")
    if len(tool_log) > max_steps:
        lines.append(f"... {len(tool_log) - max_steps} more steps")
    return "\n".join(lines)


def reflect_messages(task, result, retrieved, max_traj_steps=40):
    v = result
    report = {
        "build_ok": v.get("build_ok"), "sim_ok": v.get("sim_ok"),
        "steady": v.get("steady"), "structural_ok": v.get("structural_ok"),
        "physics_ok": v.get("physics_ok"), "goal_ok": v.get("goal_ok"),
        "multipoint_ok": v.get("multipoint_ok"),
        "physics_failures": ((v.get("physics") or {}).get("failures") or [])[:12],
        "structural_failures": {k: val for k, val in
                                ((v.get("structural") or {}).get("checks") or {}).items()
                                if not val},
        "unconnected_inputs": ((v.get("structural") or {}).get("unconnected") or [])[:12],
        "goal_detail": {k: {"ok": d.get("ok")} for k, d in
                        (v.get("goal_detail") or {}).items()},
        "steps": v.get("episode_steps"), "submitted": v.get("submitted"),
        "diagnosis": v.get("diagnosis"),
    }
    mem = "\n".join(f"- {m.memory_id}: {m.title} :: {m.description[:220]}"
                    for m in retrieved) or "(none)"
    user = f"""TASK
id: {task['task_id']}
template: {task['template']}  family: {task['family']}  components: {task['n_instances']}  connections: {task['n_connects']}
instruction:
{task['instruction']}

INITIAL MODEL
{task['task_text_initial']}

AGENT TRAJECTORY
{_fmt_traj(v.get('tool_log', []), max_traj_steps)}

FINAL MODEL
{v.get('final_text', '(none)')[:4000]}

VERIFIER REPORT
{json.dumps(report, ensure_ascii=False, indent=1)}

OUTCOME: {'SUCCESS' if v.get('success') else 'FAILURE'}

EXPERIENCE RETRIEVED FOR THIS TASK
{mem}

Return the JSON object described in the system prompt."""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


def reflect_messages_multi(episodes, retrieved, max_traj_steps=18):
    """One teacher call over a batch of K episodes (teacher-frequency ablation)."""
    blocks = []
    for i, (task, v) in enumerate(episodes):
        report = {"build_ok": v.get("build_ok"), "sim_ok": v.get("sim_ok"),
                  "structural_ok": v.get("structural_ok"),
                  "physics_ok": v.get("physics_ok"), "goal_ok": v.get("goal_ok"),
                  "physics_failures": ((v.get("physics") or {}).get("failures") or [])[:6],
                  "unconnected_inputs": ((v.get("structural") or {}).get("unconnected") or [])[:6]}
        blocks.append(f"""--- EPISODE {i+1} ---
task: {task['task_id']} | template {task['template']} | family {task['family']}
instruction:
{task['instruction'][:1200]}
initial model:
{task['task_text_initial'][:1500]}
trajectory:
{_fmt_traj(v.get('tool_log', []), max_traj_steps)}
final model:
{(v.get('final_text') or '')[:1800]}
verifier: {json.dumps(report, ensure_ascii=False)}
outcome: {'SUCCESS' if v.get('success') else 'FAILURE'}""")
    mem = "\n".join(f"- {m.memory_id}: {m.title} :: {m.description[:160]}"
                    for m in retrieved) or "(none)"
    user = f"""EPISODES
{chr(10).join(blocks)}

EXPERIENCE CURRENTLY RETRIEVED
{mem}

Extract only experiences that generalise ACROSS these episodes and to future tasks. If they share no reusable lesson, return {{"experiences": []}}.
Return the JSON object from the system prompt."""
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- validation
VAGUE = [
    "check all connections", "pay attention to physics", "ensure the model is correct",
    "try to fix errors", "be careful", "double check", "verify everything",
    "make sure the model runs", "review the code", "ensure correctness",
]


def _is_vague(text):
    t = (text or "").lower()
    return any(p in t for p in VAGUE) and len(t) < 160


def validate_experiences(obj, max_items=4):
    """Schema + quality gate.  Returns (accepted, rejected_with_reason)."""
    accepted, rejected = [], []
    if not isinstance(obj, dict):
        return [], [{"reason": "not an object"}]
    exps = obj.get("experiences") or []
    if not isinstance(exps, list):
        return [], [{"reason": "experiences not a list"}]
    for e in exps[:max_items]:
        if not isinstance(e, dict):
            rejected.append({"reason": "item not an object", "item": str(e)[:120]})
            continue
        if e.get("type") not in ("rule", "failure", "skill", "success"):
            rejected.append({"reason": "bad type", "item": e})
            continue
        title, desc = (e.get("title") or "").strip(), (e.get("description") or "").strip()
        if not title or not desc:
            rejected.append({"reason": "missing title/description", "item": e})
            continue
        if _is_vague(title) or _is_vague(desc):
            rejected.append({"reason": "vague advice", "item": e})
            continue
        if len(desc.split()) < 12:
            rejected.append({"reason": "too thin", "item": e})
            continue
        app = e.get("applicability") or {}
        if not isinstance(app, dict):
            rejected.append({"reason": "applicability not an object", "item": e})
            continue
        probe = e.get("probe")
        if probe is not None and not isinstance(probe, dict):
            rejected.append({"reason": "probe not an object", "item": e})
            continue
        e = dict(e)
        e["description"] = desc
        e["title"] = title
        accepted.append(e)
    return accepted, rejected
